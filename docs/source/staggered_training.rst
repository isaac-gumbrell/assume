.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

##################################
Staggered (Paired-Scenario) Training
##################################

Staggered training trains a single shared MATD3 policy across **two paired
scenarios** at once. Both scenarios share the same set of RL agents but
differ in some structural element (typically the underlying grid: e.g. a
*business-as-usual* network and an *investment* network with reinforced
lines). Each training episode advances both worlds in lock-step,
``train_freq``-sized chunks; transitions from both worlds are pushed into
**one shared replay buffer** and a single batch of gradient updates is
applied per chunk. The result is a policy that has been exposed to both
scenarios and will generalise to either at deployment time.

When to use it
==============

Use staggered training when you want a single bidding policy that is robust
across two structural variants of the same market — for example to study
how a transmission expansion changes the optimal bidding behaviour while
keeping the agents (and their observation/action spaces) identical. The
classic D3 use case is paired with the congestion-aware observations from
:doc:`learning` so the policy sees the same agents under different grid
constraints.

If you only have one scenario, use the regular single-world training entry
point :func:`assume.scenario.loader_csv.run_learning` instead.

Required scenario layout
========================

You need **two scenario folders** that share the same set of RL agents.
Below we use the example pair shipped under ``examples/inputs/`` —
``staggered_bau`` (the primary) and ``staggered_inv`` (the paired variant).

The primary scenario's ``config.yaml`` declares both scenarios in a new
``staggered_training`` block under ``learning_config``::

    staggered:
      start_date: 2019-03-01 00:00
      end_date: 2019-03-03 00:00
      time_step: 1h
      seed: 42

      learning_config:
        learning_mode: true
        algorithm: matd3
        training_episodes: 6
        episodes_collecting_initial_experience: 1
        train_freq: 24h
        gradient_steps: 5
        batch_size: 32
        validation_episodes_interval: 3

        staggered_training:
          enabled: true
          swap_order_per_episode: true
          scenarios:
            - path: "."                  # primary; resolved relative to this config
              name: "bau"
            - path: "../staggered_inv"   # paired variant
              name: "inv"

      markets_config:
        EOM:
          # ... standard market block, identical across both scenarios
          ...

Key points:

* The ``staggered_training.scenarios`` list **must contain exactly two
  entries**, each with a ``path`` (relative to the primary scenario folder
  or absolute) and a unique ``name``.
* The paired scenario (``staggered_inv`` here) does **not** need its own
  ``staggered_training`` block — only the primary scenario activates the
  paired flow.
* ``swap_order_per_episode: true`` alternates which world advances first
  every episode so neither side dominates the early training distribution.
* The simulation horizon (``end_date - start_date``) must be an integer
  multiple of ``train_freq``.

The two scenarios may differ in the grid (line capacities), powerplant
fleet, demand profiles, or fuel prices. Any unit that exists only in one
scenario is automatically injected into the other world with
``availability = 0`` (powerplants/storages) or ``demand = 0`` (demand
units), so it cannot bid. This guarantees both worlds register the **same
agent set** while remaining structurally distinct.

How to run
==========

Use :func:`assume.scenario.loader_csv.run_staggered_learning`:

.. code-block:: python

    from assume.scenario.loader_csv import run_staggered_learning

    run_staggered_learning(
        inputs_path="examples/inputs",
        scenario="staggered_bau",          # the primary scenario folder
        study_case="staggered",
        db_uri="sqlite:///examples/local_db/assume_staggered.db",
        export_csv_path="examples/outputs/staggered",
        verbose=True,
    )

The function builds two ``World`` instances, loads both scenarios with the
cross-scenario unit superset merged in, wires their MATD3 algorithms and
replay buffer to the same Python objects, and drives both worlds through
the staggered training loop. A complete runnable driver is provided at
``examples/world_script_staggered.py``.

The example pair runs in roughly 5–10 seconds and is the recommended
smoke test after any change to the staggered training code path.

Outputs and namespacing
=======================

Both worlds share the same database / CSV output directory, but each
world's ``simulation_id`` is set to **its own scenario folder name** so
rows never collide. After a run against the example fixture you can query::

    SELECT DISTINCT simulation FROM rl_params;
    -- staggered_bau
    -- staggered_inv

Per-world metrics also appear in TensorBoard under separate run
directories (``tensorboard/staggered_bau`` and ``tensorboard/staggered_inv``).
The evaluation reward reported by the trainer (``avg_reward``) is the mean
across both worlds; per-scenario averages are emitted alongside as
``avg_reward_bau`` / ``avg_reward_inv``.

A single shared policy is saved to the ``trained_policies_save_path``
declared in the primary scenario, just as for ordinary single-world
training.

Acceptance criteria the loader enforces
=======================================

The paired-scenario loader checks the following invariants up front and
raises a clear ``ValueError`` if any is violated:

* **G2 — RL agent set match.** After loading, both worlds must register
  exactly the same set of RL unit ids.
* **Line id parity.** If both scenarios declare a transmission grid, the
  set of line ids must match (only line *capacities* may differ).
* **Scenario name uniqueness.** The two ``name`` values must be distinct
  and DB-safe (alphanumeric / underscore).

The G4 invariant (foreign units contribute zero to local market
clearings) is enforced implicitly by setting their availability / demand
forecast series to zero across the entire horizon.

Limitations
===========

* Exactly two paired scenarios are supported.
* Both scenarios must share the same simulation horizon and time step.
* All scenarios must use the same ``algorithm`` (``matd3``) and consistent
  RL hyperparameters in their ``learning_config`` blocks.
* Distributed simulation is not yet supported in the staggered flow —
  both worlds are driven from the same Python process.

Further reading
===============

* Implementation overview: ``.isaac_docs/d3_staggered_training_implementation.md``
* Method note: ``.isaac_docs/congestion_staggered_training_method_overview.md``
* Example fixture: ``examples/inputs/staggered_bau`` and
  ``examples/inputs/staggered_inv``
* Driver script: ``examples/world_script_staggered.py``
