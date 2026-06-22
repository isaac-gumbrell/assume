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
    _check_staggered_input_file_parity,
    build_staggered_supersets,
    load_config_and_create_forecaster,
    load_staggered_scenario,
    run_staggered_evaluation,
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


def _clean_supersets() -> None:
    """Remove generated ``_superset`` folders from the example fixtures."""
    for scen in (FIXTURE_PRIMARY, FIXTURE_PAIRED):
        target = FIXTURE_INPUTS / scen / "_superset"
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


# ---------------------------------------------------------------------------
# _check_staggered_input_file_parity — file-presence validation
# ---------------------------------------------------------------------------


def test_input_file_parity_matching_scenarios_no_warning(tmp_path, caplog):
    """No warning when both scenario folders contain exactly the same CSV files."""
    import logging

    scen_a = tmp_path / "scen_a"
    scen_b = tmp_path / "scen_b"
    scen_a.mkdir()
    scen_b.mkdir()

    for scen in (scen_a, scen_b):
        (scen / "forecasts_df.csv").write_text("datetime\n2045-01-01\n")
        (scen / "fuel_prices_df.csv").write_text("datetime\n2045-01-01\n")

    with caplog.at_level(logging.WARNING, logger="assume.scenario.loader_csv"):
        _check_staggered_input_file_parity([str(scen_a), str(scen_b)], ["bau", "inv"])

    assert "mismatch" not in caplog.text.lower()


def test_input_file_parity_missing_forecasts_df_warns(tmp_path, caplog):
    """Warning is raised when one scenario is missing forecasts_df.csv."""
    import logging

    scen_a = tmp_path / "scen_a"
    scen_b = tmp_path / "scen_b"
    scen_a.mkdir()
    scen_b.mkdir()

    # scen_a has forecasts_df.csv; scen_b does not
    (scen_a / "forecasts_df.csv").write_text("datetime\n2045-01-01\n")
    (scen_a / "fuel_prices_df.csv").write_text("datetime\n2045-01-01\n")
    (scen_b / "fuel_prices_df.csv").write_text("datetime\n2045-01-01\n")

    with caplog.at_level(logging.WARNING, logger="assume.scenario.loader_csv"):
        _check_staggered_input_file_parity([str(scen_a), str(scen_b)], ["bau", "inv"])

    assert "mismatch" in caplog.text.lower()
    assert "forecasts_df.csv" in caplog.text


def test_input_file_parity_unit_csvs_are_ignored(tmp_path, caplog):
    """Unit-definition CSVs (powerplant_units etc.) do not trigger a warning."""
    import logging

    scen_a = tmp_path / "scen_a"
    scen_b = tmp_path / "scen_b"
    scen_a.mkdir()
    scen_b.mkdir()

    # scen_a has extra powerplant_units.csv; scen_b does not
    (scen_a / "powerplant_units.csv").write_text("name\npp1\n")
    (scen_a / "storage_units.csv").write_text("name\nst1\n")

    with caplog.at_level(logging.WARNING, logger="assume.scenario.loader_csv"):
        _check_staggered_input_file_parity([str(scen_a), str(scen_b)], ["bau", "inv"])

    assert "mismatch" not in caplog.text.lower()


# ---------------------------------------------------------------------------
# _superset folder + foreign_units.json manifest (standalone non-learning runs)
# ---------------------------------------------------------------------------


def _copy_fixture_pair(tmp_path) -> tuple[Path, Path]:
    """Copy the paired example fixtures into ``tmp_path`` (avoids repo pollution)."""
    _skip_if_fixture_missing()
    bau = tmp_path / FIXTURE_PRIMARY
    inv = tmp_path / FIXTURE_PAIRED
    shutil.copytree(FIXTURE_INPUTS / FIXTURE_PRIMARY, bau)
    shutil.copytree(FIXTURE_INPUTS / FIXTURE_PAIRED, inv)
    return bau, inv


def _foreign_powerplants() -> tuple[set[str], set[str]]:
    """Return (foreign-in-bau, foreign-in-inv) powerplant id sets."""
    import pandas as pd

    pp_bau = pd.read_csv(
        FIXTURE_INPUTS / FIXTURE_PRIMARY / "powerplant_units.csv", index_col=0
    )
    pp_inv = pd.read_csv(
        FIXTURE_INPUTS / FIXTURE_PAIRED / "powerplant_units.csv", index_col=0
    )
    native_bau = set(pp_bau.index.astype(str))
    native_inv = set(pp_inv.index.astype(str))
    return native_inv - native_bau, native_bau - native_inv


def test_build_supersets_writes_self_contained_folder_and_manifest(tmp_path):
    """``build_staggered_supersets`` materialises a runnable ``_superset`` folder."""
    import json

    bau, inv = _copy_fixture_pair(tmp_path)
    build_staggered_supersets([str(bau), str(inv)])

    foreign_in_bau, _ = _foreign_powerplants()

    superset = bau / "_superset"
    # Self-contained: union unit CSVs + copied config + manifest.
    assert (superset / "config.yaml").exists()
    assert (superset / "powerplant_units.csv").exists()
    assert (superset / "demand_df.csv").exists()
    assert (superset / "foreign_units.json").exists()

    manifest = json.loads((superset / "foreign_units.json").read_text())
    assert manifest["version"] == 1
    manifest_pp = set(manifest["foreign_unit_ids"].get("powerplant_units", []))
    assert manifest_pp == foreign_in_bau


def test_superset_manifest_neutralizes_foreign_units(tmp_path):
    """Loading a ``_superset`` folder standalone forces foreign units to zero output."""
    bau, inv = _copy_fixture_pair(tmp_path)
    build_staggered_supersets([str(bau), str(inv)])

    foreign_in_bau, _ = _foreign_powerplants()
    if not foreign_in_bau:
        pytest.skip("Paired fixture has identical powerplant sets; test is vacuous.")

    scenario_data = load_config_and_create_forecaster(
        str(bau), "_superset", FIXTURE_STUDY_CASE
    )
    unit_forecasts = scenario_data["unit_forecasts"]

    # Foreign powerplants: availability all zero and flagged as foreign.
    for fuid in foreign_in_bau:
        fc = unit_forecasts[fuid]
        assert (fc.availability == 0).all(), (
            f"foreign unit {fuid!r} should have zero availability in the superset run"
        )
        assert fc.is_foreign is True

    # Native powerplants remain active and are not flagged.
    import pandas as pd

    native_bau = set(
        pd.read_csv(bau / "powerplant_units.csv", index_col=0).index.astype(str)
    )
    for nuid in native_bau - foreign_in_bau:
        assert unit_forecasts[nuid].is_foreign is False


def test_superset_manifest_unknown_id_raises(tmp_path):
    """A manifest referencing a missing unit id is a hard error."""
    import json

    bau, inv = _copy_fixture_pair(tmp_path)
    build_staggered_supersets([str(bau), str(inv)])

    manifest_path = bau / "_superset" / "foreign_units.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["foreign_unit_ids"].setdefault("powerplant_units", []).append(
        "does_not_exist_pp"
    )
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="does_not_exist_pp"):
        load_config_and_create_forecaster(str(bau), "_superset", FIXTURE_STUDY_CASE)


def _make_synthetic_pair(tmp_path) -> tuple[Path, Path]:
    """Build two synthetic scenarios that diverge in every unit type.

    Starts from a copy of the example primary scenario (which provides a valid
    ``config.yaml``, ``demand_df.csv`` and ``fuel_prices_df.csv``) and injects
    divergent powerplant, storage and demand units so that loading either
    scenario's superset exercises foreign neutralisation for *all three* unit
    types.

    Returns ``(scen_a, scen_b)`` where, relative to ``scen_a``:
        * ``pp_foreign``      — a powerplant that exists only in ``scen_b``
        * ``st_foreign``      — a storage unit that exists only in ``scen_b``
        * ``demand_foreign``  — a demand unit that exists only in ``scen_b``
    and ``st_a_only`` is a storage unit that exists only in ``scen_a`` (so it is
    foreign from ``scen_b``'s perspective).
    """
    import pandas as pd

    _skip_if_fixture_missing()
    scen_a = tmp_path / "scen_a"
    scen_b = tmp_path / "scen_b"
    shutil.copytree(FIXTURE_INPUTS / FIXTURE_PRIMARY, scen_a)
    shutil.copytree(FIXTURE_INPUTS / FIXTURE_PRIMARY, scen_b)

    storage_cols = [
        "technology",
        "bidding_EOM",
        "max_power_charge",
        "max_power_discharge",
        "capacity",
        "max_soc",
        "min_soc",
        "efficiency_charge",
        "efficiency_discharge",
        "unit_operator",
    ]

    def _storage_row(name: str) -> pd.DataFrame:
        return pd.DataFrame(
            [["PSPP", "naive", 100, 100, 1000, 1000, 0, 0.9, 0.9, "Operator 1"]],
            columns=storage_cols,
            index=pd.Index([name], name="name"),
        )

    # scen_a gets a storage unit that only it owns.
    _storage_row("st_a_only").to_csv(scen_a / "storage_units.csv")

    # scen_b gets a divergent storage unit, an extra powerplant and an extra
    # demand unit (each foreign from scen_a's perspective).
    _storage_row("st_foreign").to_csv(scen_b / "storage_units.csv")

    pp_b = pd.read_csv(scen_b / "powerplant_units.csv", index_col=0)
    pp_extra = pp_b.iloc[[0]].copy()
    pp_extra.index = pd.Index(["pp_foreign"], name=pp_b.index.name)
    pd.concat([pp_b, pp_extra]).to_csv(scen_b / "powerplant_units.csv")

    dem_b = pd.read_csv(scen_b / "demand_units.csv", index_col=0)
    dem_extra = dem_b.iloc[[0]].copy()
    dem_extra.index = pd.Index(["demand_foreign"], name=dem_b.index.name)
    pd.concat([dem_b, dem_extra]).to_csv(scen_b / "demand_units.csv")

    return scen_a, scen_b


def test_superset_neutralizes_all_foreign_unit_types(tmp_path):
    """Foreign powerplants, storages and demand units are all forced to zero."""
    scen_a, scen_b = _make_synthetic_pair(tmp_path)
    build_staggered_supersets([str(scen_a), str(scen_b)])

    scenario_data = load_config_and_create_forecaster(
        str(scen_a), "_superset", FIXTURE_STUDY_CASE
    )
    unit_forecasts = scenario_data["unit_forecasts"]

    # Foreign powerplant and storage: availability all zero + flagged foreign.
    for fuid in ("pp_foreign", "st_foreign"):
        fc = unit_forecasts[fuid]
        assert (fc.availability == 0).all(), (
            f"foreign unit {fuid!r} should have zero availability"
        )
        assert fc.is_foreign is True

    # Foreign demand: demand profile forced to zero (no load contribution).
    demand_fc = unit_forecasts["demand_foreign"]
    assert demand_fc.is_foreign is True
    assert (demand_fc.availability == 0).all()
    assert (demand_fc.demand == 0).all(), (
        "foreign demand unit should contribute zero load"
    )

    # Native units across all types stay active and are not flagged foreign.
    for nuid in ("pp_1", "st_a_only", "demand_EOM"):
        assert unit_forecasts[nuid].is_foreign is False
    # A native demand unit keeps a non-zero load somewhere in the horizon.
    assert (unit_forecasts["demand_EOM"].demand != 0).any()


def test_superset_manifest_symmetry_between_scenarios(tmp_path):
    """Each scenario's manifest lists exactly the units native only to the other."""
    import json

    scen_a, scen_b = _make_synthetic_pair(tmp_path)
    build_staggered_supersets([str(scen_a), str(scen_b)])

    man_a = json.loads((scen_a / "_superset" / "foreign_units.json").read_text())[
        "foreign_unit_ids"
    ]
    man_b = json.loads((scen_b / "_superset" / "foreign_units.json").read_text())[
        "foreign_unit_ids"
    ]

    # Foreign in A == units only in B (pp_foreign, st_foreign, demand_foreign).
    assert set(man_a.get("powerplant_units", [])) == {"pp_foreign"}
    assert set(man_a.get("storage_units", [])) == {"st_foreign"}
    assert set(man_a.get("demand_units", [])) == {"demand_foreign"}

    # Foreign in B == units only in A (just the storage unit st_a_only).
    assert set(man_b.get("storage_units", [])) == {"st_a_only"}
    assert set(man_b.get("powerplant_units", [])) == set()
    assert set(man_b.get("demand_units", [])) == set()


@pytest.mark.require_learning
@pytest.mark.slow
def test_run_staggered_evaluation_end_to_end(tmp_path):
    """Train, then evaluate the supersets with the learned policy.

    Verifies the full non-learning pipeline: the shared policy is loaded, each
    superset world runs to completion under its own ``<scenario>_eval``
    simulation id, and a foreign generator dispatches zero energy.
    """
    import sqlite3

    _skip_if_fixture_missing()
    _clean_learned_strategies()
    _clean_supersets()

    foreign_in_bau, _ = _foreign_powerplants()
    if not foreign_in_bau:
        pytest.skip("Paired fixture has identical powerplant sets; test is vacuous.")

    train_db = f"sqlite:///{tmp_path / 'train.db'}"
    eval_db_path = tmp_path / "eval.db"
    eval_db = f"sqlite:///{eval_db_path}"

    try:
        run_staggered_learning(
            inputs_path=str(FIXTURE_INPUTS),
            scenario=FIXTURE_PRIMARY,
            study_case=FIXTURE_STUDY_CASE,
            db_uri=train_db,
            export_csv_path="",
            verbose=False,
        )

        worlds = run_staggered_evaluation(
            inputs_path=str(FIXTURE_INPUTS),
            scenario=FIXTURE_PRIMARY,
            study_case=FIXTURE_STUDY_CASE,
            db_uri=eval_db,
            export_csv_path="",
        )
        assert len(worlds) == 2
    finally:
        _clean_learned_strategies()
        _clean_supersets()

    assert eval_db_path.exists(), "Evaluation did not produce a database file."

    con = sqlite3.connect(eval_db_path)
    try:
        cur = con.cursor()
        tables = {
            r[0]
            for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "unit_dispatch" in tables, f"unit_dispatch missing. tables={tables}"

        sims = {
            r[0]
            for r in cur.execute(
                "SELECT DISTINCT simulation FROM unit_dispatch"
            ).fetchall()
        }
        assert f"{FIXTURE_PRIMARY}_eval" in sims, (
            f"Expected '{FIXTURE_PRIMARY}_eval' rows, got {sims}"
        )
        assert f"{FIXTURE_PAIRED}_eval" in sims, (
            f"Expected '{FIXTURE_PAIRED}_eval' rows, got {sims}"
        )

        # A foreign generator in the bau world must dispatch zero energy.
        foreign_uid = sorted(foreign_in_bau)[0]
        rows = cur.execute(
            "SELECT MAX(ABS(power)) FROM unit_dispatch "
            "WHERE simulation = ? AND unit = ?",
            (f"{FIXTURE_PRIMARY}_eval", foreign_uid),
        ).fetchall()
        assert rows and rows[0][0] is not None, (
            f"Foreign unit {foreign_uid!r} produced no dispatch rows."
        )
        assert rows[0][0] == 0, (
            f"Foreign unit {foreign_uid!r} dispatched non-zero power "
            f"(max abs = {rows[0][0]}) in the superset evaluation run."
        )
    finally:
        con.close()
