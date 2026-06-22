<!--
SPDX-FileCopyrightText: ASSUME Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Staggered-Training Reward Regression — Diagnosis & Recommended Changes

## Context

Commit `88ac62d9955af590dc849b3c2d0af74e8e9925ed`
("fix(learning): honour availability for storage and mask forced-off RL
transitions") introduced three tiers of changes to support paired-scenario
staggered training. After this commit, training on the
`profit_FG_Hrly2040_GGR_*` scenarios showed:

- evaluation **reward ~10× higher** than before,
- **regret exploded**,
- **critic loss** climbing from ~1 → 8+ and **critic grad-norm** from ~0.25 → 125
  (monotonic, unbounded — the signature of Q-value divergence),
- periodic spikes aligned with `swap_order_per_episode` world swaps.

The scenario uses a single nodal EOM market; demand is naive; renewables use
`renewable_energy_learning_compatible_congestion`; thermal/hydro use
`powerplant_energy_learning_single_bid_congestion`; storage uses
`storage_energy_learning_congestion`. **Hydro is always 100 % available** (use is
managed via a pre-computed price curve), so availability-derating of hydro is
*not* a factor.

## Root cause (primary)

**Tier 2's activity mask conflates "native unit at zero availability" with
"foreign unit".**

Every reward function derives the training mask as:

```python
active = 1.0 if unit.forecaster.availability.at[start] > 0 else 0.0
```

The intent was to exclude *foreign* units — the paired scenario's superset of
agents, which the loader forces to `availability = 0` for the entire horizon
(`assume/scenario/loader_csv.py`, `for uid in all_foreign_ids: availability[uid] = 0.0`)
— from the shared MATD3 gradient.

But a **native renewable uses availability as its capacity factor**: solar is
legitimately `0` every night, wind in every lull, and its bid volume is
`max_power = availability × nameplate`. The predicate above cannot tell these
genuine, informative states apart from a foreign unit, so it **masks them out of
the policy update** (`active = 0.0`).

### Why this diverges

The critic still has to *predict* the value of zero-availability states because
they appear as `next_observations` in the bootstrap target
`y = r + γ·Q(s')`, but it is **never trained on them** as `(s, a)` pairs. The
ungrounded `Q(s')` extrapolates upward, inflating targets → critic loss and
grad-norm climb → the actor chases the inflated Q and bids up → the single nodal
clearing price rises → **every** unit's profit jumps (including thermal/hydro,
whose own reward formula is unchanged). The periodic spikes line up with
`swap_order_per_episode`, when the active/foreign split flips.

Note the commit message states this renewable strategy "was previously missed,
so its forced-off steps were never masked" — i.e. **this commit *added* masking
to exactly the strategy your renewables use.** Pre-commit, those night hours were
trained on as zero-reward transitions; post-commit they are deleted from every
gradient.

## Validation status

| Claim | Statement | Status |
|---|---|---|
| **A. Precondition (necessary)** | The mask must distinguish *native* zero-availability from *foreign* | ✅ **Fixed & proven** by `tests/test_rl_strategies.py::test_activity_mask_keys_off_foreign_flag_not_availability` (deterministic, ~0.2 s) |
| **B. Causal link (sufficient)** | Deleting those transitions drives the Q-divergence / 10× reward | ⏳ Not yet proven — requires a seeded micro-A/B training run (see below) |

## Recommended changes

### 1. Fix the mask to key off "is foreign", not availability (PRIMARY) — ✅ IMPLEMENTED

The mask must reflect a **static, per-unit fact** ("is this unit foreign in the
currently active scenario") rather than the dynamic `availability > 0`.

- A `is_foreign` flag (default `False`) was added to the `UnitForecaster` base class
  (`assume/common/forecaster.py`). All forecaster subclasses inherit it via
  `super().__init__`.
- `assume/scenario/loader_csv.py` now sets `unit_forecasts[uid].is_foreign = True`
  for every id in `all_foreign_ids` (the paired-scenario superset it forces off).
- `active` in **all four** reward functions now reads:
  ```python
  active = 0.0 if unit.forecaster.is_foreign else 1.0
  ```
  Affected methods in `assume/strategies/learning_strategies.py`:
  - `EnergyLearningStrategy.calculate_reward`
  - `StorageEnergyLearningStrategy.calculate_reward`
  - `RenewableEnergyLearningSingleBidStrategy.calculate_reward`
  - `RenewableEnergyLearningCompatibleStrategy.calculate_reward`
- For non-staggered runs nothing is foreign → all masks `1.0` → identical to
  pre-commit behaviour, and native renewables keep learning their night hours.
- The regression test now asserts the corrected behaviour: a native unit stays
  active (1.0) in both its generating and its zero-availability hour, while a unit
  with `is_foreign=True` is masked (0.0).

### 2. Apply the fix to all four reward functions, not just renewables — ✅ IMPLEMENTED

`StorageEnergyLearningStrategy` and the powerplant strategies used the identical
`availability.at[start] > 0` predicate. Any native unit with a genuine outage hour
was wrongly masked. The fix in (1) touches all four paths.

### 3. Tier 1 opportunity-cost reference (already applied; keep)

`EnergyLearningStrategy.calculate_reward` was changed by the commit to compute
opportunity cost against `offered_volume_total` (the *residual* headroom returned
by `calculate_min_max_power`, which is **endogenous** — it nets out the unit's own
committed dispatch/reserve). This has been reverted to an **exogenous** baseline:

```python
availability = unit.forecaster.availability.at[start]
available_power = availability * unit.max_power
```

- `availability == 1` reproduces the original nameplate baseline exactly.
- `availability == 0` collapses opportunity cost (and reward) to 0 for genuinely
  forced-off units without the endogenous reward-hacking hole.

This is a **no-op for the current scenario** (single-bid strategies don't set
volume via the action), but it is correct and should be kept. Expect it *not* to
move the GGR curves.

### 4. Replay-buffer mask back-compat is asymmetric (note / guard)

`ReplayBuffer.load()` defaults missing masks to all-ones, but resuming from a
pre-commit buffer mid-run (`load_replay_buffer: true`) means old transitions train
while new ones get masked — a silent distribution shift. Recommend documenting this
in the staggered-training docs and/or guarding against mixed-vintage buffers.

### 5. Scaling/comment inconsistency in `EnergyLearningStrategy` (cosmetic)

`scaling = 1 / (max_bid_price * unit.max_power)` divides by nameplate while the
reward comment describes normalisation against offered capacity. Cosmetic now that
`available_power` is back to nameplate, but worth tidying if this code is revisited.

## Proving Claim B (causal half) — proposed micro-A/B

A unit test cannot exhibit divergence (it is an emergent training dynamic), but a
small, **seeded, deterministic** A/B isolates the cause in minutes rather than hours:

- Fixed seed (42), 2 short episodes, on a cut-down scenario with ≥1 native solar
  profile and a single nodal market.
- Three arms, identical except one line:
  1. `active = availability > 0` (current code)
  2. `active = is_foreign` (proposed fix)
  3. masking removed entirely (pre-commit behaviour)
- **Metric (not eyeballing):** log per-update `critic_total_grad_norm` and the
  per-agent active fraction (`mask_norm`). Assert arm 1's critic grad-norm grows
  monotonically while arms 2 & 3 stay bounded, and that arm 1's masked fraction for
  the solar agent tracks its night hours (~50 %).

## Files touched so far

- `assume/common/forecaster.py` — added `is_foreign` flag (default `False`) to
  `UnitForecaster` (changes 1 & 2).
- `assume/scenario/loader_csv.py` — sets `is_foreign = True` on foreign forecasters
  (changes 1 & 2).
- `assume/strategies/learning_strategies.py` — mask in all four reward functions
  now keys off `unit.forecaster.is_foreign` (changes 1 & 2); Tier 1 opportunity-cost
  baseline reverted to `availability × nameplate` (change 3).
- `tests/test_rl_strategies.py` — added
  `test_activity_mask_keys_off_foreign_flag_not_availability`
  (deterministic regression test for the fix).
