<!--
SPDX-FileCopyrightText: ASSUME Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Staggered-Training Availability Fixes

## Background

In **staggered learning mode**, both worlds register an identical *superset* of
RL agents so the shared replay buffer and MATD3 policy see a consistent agent
set every step. Units that belong to the superset but are *not* part of the
currently-active scenario are neutralised by setting their forecaster
`availability = 0` for the periods they should not act.

Two problems were found with this approach:

1. **Storage units ignored `availability`.** Unlike power plants and renewables,
   the `Storage` dispatch bounds never consulted `availability`, so a unit that
   was supposed to be "off" could still charge/discharge — an availability leak.
2. **Forced-off learning units received a spurious negative reward.** The
   power-plant reward computed opportunity cost against nameplate `max_power`,
   so a unit with zero available capacity was penalised for capacity it
   physically could not provide. Because MATD3 uses *parameter sharing*, this
   biased reward poisoned the shared policy for every agent.

The work below is grouped into three tiers:

- **Tier 0** — make storage honour `availability` (the availability leak).
- **Tier 1** — remove the forced-off reward bias for power plants.
- **Tier 2** — mask forced-off transitions out of the MATD3 policy update so
  "off" steps contribute no gradient at all.

(Tier 3 — neutralising an off-agent's contribution to *other* agents'
centralised-critic joint-action input — was intentionally **not** implemented.)

---

## Tier 0 — Storage honours `availability`

### `assume/units/storage.py`

**What changed:** Both `calculate_min_max_charge` and
`calculate_min_max_discharge` now read the forecaster availability for the
requested window and multiply the technical power limits by it:

```python
availability = self.forecaster.availability.loc[start:end_excl]

min_power_charge = availability * self.min_power_charge - (base_load + capacity_pos)
max_power_charge = availability * self.max_power_charge - (base_load + capacity_neg)
# (analogous for discharge)
```

**Why:** Default `availability = 1` leaves normal behaviour completely
unchanged, while `availability = 0` collapses both the min and max bounds to
zero, forcing the unit fully off. This closes the leak that previously let
"off" storage units still dispatch in staggered training.

### `tests/test_storage.py`

**What changed:** Added `test_availability_forces_off()`. It builds a `Storage`
unit with a `UnitForecaster(index, availability=0, ...)` and asserts that both
the charge and discharge min/max bounds are `0`.

**Why:** Locks in the Tier 0 behaviour and documents the contract that
staggered training depends on.

---

## Tier 1 — Remove forced-off reward bias (power plants)

### `assume/strategies/learning_strategies.py` — `EnergyLearningStrategy.calculate_reward`

**What changed:** The opportunity-cost term now uses the actually-offered
(availability-adjusted) capacity instead of nameplate `max_power`:

```python
available_power = offered_volume_total
opportunity_cost = (
    (market_clearing_price - marginal_cost)
    * (available_power - accepted_volume_total)
    * duration
)
```

**Why:** When a unit is forced off, `offered_volume_total = 0`, so the
opportunity cost becomes `0` and the reward becomes `0` instead of a spurious
negative value. This mirrors `RenewableEnergyLearningSingleBidStrategy`, which
already used `offered_volume_total`. A negative reward on a step the unit could
not influence would otherwise be propagated to the shared policy.

---

## Tier 2 — Mask forced-off transitions in the MATD3 update

Tier 1 zeroes the reward, but a zero-reward transition is still a *training
target*. Tier 2 goes further and excludes forced-off transitions from the
gradient entirely, via a per-agent activity mask threaded from the reward
functions through to the optimiser.

### `assume/strategies/learning_strategies.py` — activity flag

**What changed:** All three reward functions now compute an `active` flag and
pass it to the cache:

```python
active = 1.0 if unit.forecaster.availability.at[start] > 0 else 0.0
...
self.learning_role.add_reward_to_cache(unit.id, start, reward, regret, profit, active)
```

Affected strategies:
- `EnergyLearningStrategy` (power plants)
- `StorageEnergyLearningStrategy` (storage; `regret` arg stays `0`)
- `RenewableEnergyLearningSingleBidStrategy` (renewables)

**Why:** `active = 0` flags a step where the unit was forced off. This signal is
semantically correct for every unit type (it also cleanly distinguishes a
"forced off" step from a *valid idle* step for storage).

### `assume/reinforcement_learning/learning_role.py`

**What changed:**
- Added `import numpy as np`.
- Added an `all_active` cache (`defaultdict(lambda: defaultdict(list))`) in both
  the init and the per-flush reset blocks.
- `store_to_buffer_and_update` snapshots/reset `all_active` and adds an
  `"active"` entry to the cache dict it hands off.
- `_store_to_buffer_and_update_sync` passes the masks to the buffer:
  ```python
  mask=np.squeeze(
      transform_buffer_data(cache["active"], device, self.rl_strats.keys()),
      axis=-1,
  )
  ```
- `add_reward_to_cache(..., active=1.0)` gained an `active` parameter (default
  `1.0` keeps existing callers working) and appends it to `all_active`.

**Why:** Carries the per-agent activity mask alongside rewards through the same
caching/flushing pipeline so it lands in the replay buffer aligned with every
transition.

### `assume/reinforcement_learning/buffer.py`

**What changed:**
- `ReplayBufferSamples` gained a trailing `masks: th.Tensor` field.
- `__init__` allocates `self.masks = np.ones((buffer_size, n_rl_units))`
  (default `1.0`).
- `add(..., mask=None)` stores the mask (all-ones when omitted).
- `sample()` returns the masks for the sampled batch (now a 5-tuple).
- `save()` writes `masks` into the npz.
- `load()` reads `masks` if present, otherwise defaults to `np.ones_like(rews)`.

**Why:** Persists the mask with each transition. Every default (ones,
backward-compatible `load`) ensures buffers built or saved before this change
behave exactly as before — masking is opt-in via the data itself.

### `assume/reinforcement_learning/algorithms/matd3.py`

**What changed:**
- Removed the now-unused `from torch.nn import functional as F`.
- Unpacked `masks = transitions.masks` after sampling.
- **Critic loss** — replaced `F.mse_loss(...)` with a masked, normalised sum:
  ```python
  mask_i = masks[:, i].unsqueeze(1)
  mask_norm = mask_i.sum().clamp(min=1.0)
  critic_loss = sum(
      (mask_i * (current_q - target_Q_values) ** 2).sum() / mask_norm
      for current_q in current_Q_values
  )
  ```
- **Actor loss** — applied the same masking:
  ```python
  mask_i = masks[:, i].unsqueeze(1)
  q1 = critic.q1_forward(all_states_i, all_actions_clone)
  actor_loss = -(mask_i * q1).sum() / mask_i.sum().clamp(min=1.0)
  ```

**Why:** Forced-off transitions (`mask = 0`) contribute nothing to either loss,
so the shared policy never trains on steps where a unit could not act.
Normalising by `mask.sum().clamp(min=1.0)` keeps the gradient scale stable
regardless of how many samples in the batch are active (and avoids divide-by-zero
when none are).

### `tests/test_rl_buffer.py`

**What changed:** Updated both `sample()` unpacks to the new 5-tuple
`(observations, actions, next_observations, rewards, masks)` and added
`assert masks.shape == (1, 4)` / `(2, 4)`.

**Why:** Keeps the buffer tests in sync with the new `masks` field and verifies
its shape.

---

## Supporting test fix

### `tests/test_flexable_storage_strategies.py`

**What changed:** In `test_flexable_pos_crm_storage` and
`test_flexable_neg_crm_storage`, the forecaster index was extended from
`periods=4` to `periods=48`.

**Why:** These tests replace the unit's forecaster with a short 4-period one but
then query a window extending to `05:00`, which exceeds that horizon. Before
Tier 0 this was harmless, but now that storage slices `availability` over the
queried window, the short forecaster returned a mismatched-length array
(`operands could not be broadcast together with shapes (3,) (4,)`). Extending
the forecaster to cover the window fixes the **test artifact** — in production
the forecaster always spans the full simulation horizon, so this never occurs.
The two tests use scalar prices, so lengthening the index is safe; other tests
in the file that use 4-element price lists were left untouched.

---

## Verification

- Targeted suites (storage, RL buffer, MATD3, replay-buffer persistence, RL
  strategies, DRL storage strategy, learning role, flexable storage): **all
  pass** (55 + targeted tests green).
- Full fast suite (`-m "not require_learning and not require_network and not
  slow"`): **359 passed**. The single failure,
  `test_load_srmc_congestion_from_db`, is **pre-existing and unrelated**
  (confirmed failing on a clean `git stash` of these changes).
