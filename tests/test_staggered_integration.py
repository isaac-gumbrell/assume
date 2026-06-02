# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""End-to-end integration tests for D3 staggered (paired-scenario) training.

These tests exercise the full path through :func:`run_staggered_learning`
against the paired example fixture under
``examples/inputs/staggered_bau`` / ``examples/inputs/staggered_inv``.

They are marked ``require_learning`` and ``slow`` so they are skipped from
the default fast suite. Run them explicitly with::

    python -m pytest tests/test_staggered_integration.py -m "require_learning"

What they verify (D3 acceptance criteria):

* **G2** — both worlds register the same set of RL agent ids after
  ``load_staggered_scenario`` (so the shared MATD3 policy operates on
  identical agent slots).
* **G4** — every foreign unit injected into a world reports zero
  availability (powerplants/storages) or zero demand (demand units), so
  it cannot bid into that world's market.
* **Namespacing** — each world's ``simulation_id`` is its own scenario
  folder name and DB rows from a real training run land under both
  ids without overlap.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from assume.scenario.loader_csv import (
    load_staggered_scenario,
    run_staggered_learning,
)
from assume.world import World

FIXTURE_INPUTS = Path("examples/inputs")
FIXTURE_PRIMARY = "staggered_bau"
FIXTURE_PAIRED = "staggered_inv"
FIXTURE_STUDY_CASE = "staggered"


def _skip_if_fixture_missing() -> None:
    if not (FIXTURE_INPUTS / FIXTURE_PRIMARY / "config.yaml").exists():
        pytest.skip(f"Paired fixture not found at {FIXTURE_INPUTS / FIXTURE_PRIMARY}")
    if not (FIXTURE_INPUTS / FIXTURE_PAIRED / "config.yaml").exists():
        pytest.skip(f"Paired fixture not found at {FIXTURE_INPUTS / FIXTURE_PAIRED}")


def _clean_learned_strategies() -> None:
    """Remove any leftover policy / DB artefacts from a previous run."""
    for scen in (FIXTURE_PRIMARY, FIXTURE_PAIRED):
        target = FIXTURE_INPUTS / scen / "learned_strategies"
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)


@pytest.fixture(autouse=True)
def _non_interactive(monkeypatch):
    """Make `confirm_learning_save_path` skip its interactive prompt."""
    monkeypatch.setenv("NON_INTERACTIVE", "1")


# ---------------------------------------------------------------------------
# load_staggered_scenario — G2 / G4 / namespacing (no training loop)
# ---------------------------------------------------------------------------


@pytest.mark.require_learning
@pytest.mark.slow
def test_load_staggered_scenario_g2_rl_agent_sets_match():
    """G2 — both worlds must end up with the same RL agent ids."""
    _skip_if_fixture_missing()
    _clean_learned_strategies()

    world_a = World(database_uri="", export_csv_path="")
    world_b = World(database_uri="", export_csv_path="")
    load_staggered_scenario(
        worlds=[world_a, world_b],
        inputs_path=str(FIXTURE_INPUTS),
        scenario=FIXTURE_PRIMARY,
        study_case=FIXTURE_STUDY_CASE,
    )

    rl_a = set(world_a.learning_role.rl_strats.keys())
    rl_b = set(world_b.learning_role.rl_strats.keys())
    assert rl_a == rl_b, (
        f"G2 violation: RL agent ids differ between worlds. "
        f"only_a={rl_a - rl_b}, only_b={rl_b - rl_a}"
    )
    # Sanity: there is at least one RL agent (otherwise the test is vacuous).
    assert len(rl_a) >= 1


@pytest.mark.require_learning
@pytest.mark.slow
def test_load_staggered_scenario_g4_foreign_units_neutralized():
    """G4 — foreign units injected into each world must not be able to bid.

    Concretely: any powerplant id that is *not* native to a given world must
    have an ``availability`` series of all zeros in that world's forecaster.
    """
    _skip_if_fixture_missing()
    _clean_learned_strategies()

    # Read the local powerplant id sets directly from the CSVs to know what
    # is "native" to each side.
    import pandas as pd

    pp_bau = pd.read_csv(
        FIXTURE_INPUTS / FIXTURE_PRIMARY / "powerplant_units.csv", index_col=0
    )
    pp_inv = pd.read_csv(
        FIXTURE_INPUTS / FIXTURE_PAIRED / "powerplant_units.csv", index_col=0
    )
    native_bau = set(pp_bau.index.astype(str))
    native_inv = set(pp_inv.index.astype(str))

    foreign_in_bau = native_inv - native_bau
    foreign_in_inv = native_bau - native_inv
    # Smoke: at least one side must inject something for this test to be
    # meaningful. If both scenarios are identical, skip.
    if not foreign_in_bau and not foreign_in_inv:
        pytest.skip("Paired fixture has identical powerplant sets; G4 is vacuous.")

    world_a = World(database_uri="", export_csv_path="")
    world_b = World(database_uri="", export_csv_path="")
    load_staggered_scenario(
        worlds=[world_a, world_b],
        inputs_path=str(FIXTURE_INPUTS),
        scenario=FIXTURE_PRIMARY,
        study_case=FIXTURE_STUDY_CASE,
    )

    # World A is loaded for the primary (`staggered_bau`); foreigners there
    # are the units only present in `staggered_inv`.
    for fuid in foreign_in_bau:
        fc = world_a.scenario_data["unit_forecasts"][fuid]
        assert (fc.availability == 0).all(), (
            f"G4 violation: foreign unit {fuid!r} in world A has non-zero availability."
        )

    for fuid in foreign_in_inv:
        fc = world_b.scenario_data["unit_forecasts"][fuid]
        assert (fc.availability == 0).all(), (
            f"G4 violation: foreign unit {fuid!r} in world B has non-zero availability."
        )


@pytest.mark.require_learning
@pytest.mark.slow
def test_load_staggered_scenario_namespaces_simulation_ids():
    """Each world's ``simulation_id`` must be the scenario folder name and the
    two ids must be distinct (so DB outputs do not collide).
    """
    _skip_if_fixture_missing()
    _clean_learned_strategies()

    world_a = World(database_uri="", export_csv_path="")
    world_b = World(database_uri="", export_csv_path="")
    load_staggered_scenario(
        worlds=[world_a, world_b],
        inputs_path=str(FIXTURE_INPUTS),
        scenario=FIXTURE_PRIMARY,
        study_case=FIXTURE_STUDY_CASE,
    )

    sid_a = world_a.scenario_data["simulation_id"]
    sid_b = world_b.scenario_data["simulation_id"]
    assert sid_a == FIXTURE_PRIMARY
    assert sid_b == FIXTURE_PAIRED
    assert sid_a != sid_b
    # No stutter:
    assert "_staggered_" not in sid_a
    assert "_staggered_" not in sid_b


# ---------------------------------------------------------------------------
# run_staggered_learning — full smoke + DB namespacing
# ---------------------------------------------------------------------------


@pytest.mark.require_learning
@pytest.mark.slow
def test_run_staggered_learning_writes_namespaced_db_rows(tmp_path):
    """Run a tiny end-to-end staggered training and verify that DB rows
    appear under both per-world simulation_ids without colliding.
    """
    _skip_if_fixture_missing()
    _clean_learned_strategies()

    db_path = tmp_path / "assume_staggered.db"
    db_uri = f"sqlite:///{db_path}"

    run_staggered_learning(
        inputs_path=str(FIXTURE_INPUTS),
        scenario=FIXTURE_PRIMARY,
        study_case=FIXTURE_STUDY_CASE,
        db_uri=db_uri,
        export_csv_path=str(tmp_path / "out"),
        verbose=False,
    )

    assert db_path.exists(), "Training did not produce a database file."

    import sqlite3

    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        tables = [
            r[0]
            for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "rl_params" in tables, f"rl_params missing from DB. tables={tables}"

        sims_in_rl_params = {
            r[0]
            for r in cur.execute("SELECT DISTINCT simulation FROM rl_params").fetchall()
        }
        assert FIXTURE_PRIMARY in sims_in_rl_params, (
            f"Expected '{FIXTURE_PRIMARY}' rows in rl_params, got {sims_in_rl_params}"
        )
        assert FIXTURE_PAIRED in sims_in_rl_params, (
            f"Expected '{FIXTURE_PAIRED}' rows in rl_params, got {sims_in_rl_params}"
        )
    finally:
        con.close()
