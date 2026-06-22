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
agent set** while remaining structurally distinct. The same neutralisation
is reused when evaluating the trained policy — see
`Running the superset in non-learning mode with learned policies`_.

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

The superset scenario folder
============================

During loading, the union of all units across both scenarios — the
**superset** — is computed so each world registers the same RL agent set.
As a side effect, :func:`assume.scenario.loader_csv.build_staggered_supersets`
also materialises a self-contained, directly-runnable ``_superset`` folder
inside each scenario directory:

.. code-block:: text

    examples/inputs/staggered_bau/
    └── _superset/
        ├── config.yaml              # copied from the scenario
        ├── powerplant_units.csv     # union of both scenarios' units
        ├── storage_units.csv        # union
        ├── demand_units.csv         # union
        ├── demand_df.csv            # copies of all other input files
        ├── ...                      # (profiles, fuel prices, .license, ...)
        └── foreign_units.json       # manifest of the foreign unit ids

The unit CSVs hold the **union** of both scenarios' units, so the folder
presents the full agent set the shared policy was trained on. The
``foreign_units.json`` manifest records which of those units are
*foreign* — i.e. native to the *other* paired scenario — so they can be
neutralised when the folder is run on its own:

.. code-block:: json

    {
      "version": 1,
      "scenario": "staggered_bau",
      "foreign_unit_ids": {
        "powerplant_units": ["pp_inv_only"],
        "demand_units": ["demand_inv_only"]
      }
    }

.. note::

   The ``_superset`` folders are regenerated on every training run and are
   git-ignored. They are not edited by hand; treat them as build artefacts.

Running the superset in non-learning mode with learned policies
===============================================================

To evaluate the trained policy you run the ``_superset`` folder as an
ordinary (non-learning) scenario that loads the saved policy. Because the
unit CSVs contain the **union** of both fleets, the foreign generators
would otherwise dispatch at full capacity and pollute the clearing. To
prevent this, the loader reads ``foreign_units.json`` and forces every
foreign unit to **zero output** for the entire horizon — exactly the same
neutralisation applied during paired training:

* foreign powerplants and storages → ``availability = 0`` (cannot dispatch);
* foreign demand units → ``demand = 0`` (no load contribution);
* every foreign unit's forecaster is flagged ``is_foreign = True``.

Native units are untouched, so each superset world reflects only its own
fleet while still registering the complete agent set the shared policy
expects. If the manifest references a unit id that is **not** present in
the loaded CSVs, the loader raises a ``ValueError`` (the folder is out of
sync with its manifest and should be regenerated).

The convenience helper
:func:`assume.scenario.loader_csv.run_staggered_evaluation` wires this up
for both scenarios in one call. It rebuilds the supersets, loads each
``_superset`` folder, points it at the shared trained policy, and runs it
in non-learning mode:

.. code-block:: python

    from assume.scenario.loader_csv import run_staggered_evaluation

    run_staggered_evaluation(
        inputs_path="examples/inputs",
        scenario="staggered_bau",          # the primary scenario folder
        study_case="staggered",
        # defaults to the anchor scenario's last_policies if omitted:
        trained_policies_path=None,
        db_uri="sqlite:///examples/local_db/assume_staggered_eval.db",
        export_csv_path="examples/outputs/staggered_eval",
    )

Each evaluated world's ``simulation_id`` is namespaced ``<scenario>_eval``
(e.g. ``staggered_bau_eval`` / ``staggered_inv_eval``) so its outputs stay
separable from the training rows. The same shared policy is loaded by every
paired world; by default it is read from
``<anchor>/learned_strategies/<anchor>_<study_case>/last_policies``. Pass
``trained_policies_path`` explicitly if your policy was saved elsewhere.

Alternatively, because the ``_superset`` folder is self-contained, you can
point the standard loader or the ``assume`` CLI directly at it — the
foreign-unit neutralisation happens automatically via the manifest.

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
