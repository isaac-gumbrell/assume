# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for D3 staggered-training building blocks.

These tests focus on the small, pure pieces of the staggered-training feature:

* :func:`_chunk_boundaries` — chunk-tiling math.
* :func:`build_staggered_supersets` — cross-scenario unit-union logic and the
  ``_superset/`` CSV emission.
* :class:`LearningConfig` — the ``staggered_training`` field is plumbed.
* :func:`run_learning` — the staggered-training gate redirects callers to the
  paired-scenario entry point.
* :func:`_ensure_persistent_loop` — secondary worlds share the anchor's loop.

Heavier end-to-end paired-scenario flows are covered separately once a real
fixture is available; these tests intentionally avoid scenario YAMLs and
mango containers so they stay in the fast suite.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from assume.common.base import LearningConfig
from assume.reinforcement_learning.staggered_trainer import (
    StaggeredTrainer,
    _chunk_boundaries,
    _ensure_persistent_loop,
)
from assume.scenario.loader_csv import (
    build_staggered_supersets,
    load_config_and_create_forecaster,
    load_staggered_scenario,
    run_learning,
)

# ---------------------------------------------------------------------------
# _chunk_boundaries
# ---------------------------------------------------------------------------


def test_chunk_boundaries_tiles_window_evenly():
    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2022-01-02")  # 24h
    chunks = _chunk_boundaries(start, end, "6h")
    assert len(chunks) == 4
    # contiguous
    for (_, end_a), (start_b, _) in zip(chunks, chunks[1:]):
        assert end_a == start_b
    # fully spans
    assert chunks[0][0] == start.timestamp()
    assert chunks[-1][1] == end.timestamp()


def test_chunk_boundaries_truncates_last_chunk():
    """If the horizon is not a multiple of train_freq, the last chunk is short
    rather than overshooting ``end``.
    """
    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2022-01-01 10:00")  # 10h
    chunks = _chunk_boundaries(start, end, "6h")
    assert len(chunks) == 2
    assert chunks[-1][1] == end.timestamp()
    # last chunk shorter than train_freq
    assert chunks[-1][1] - chunks[-1][0] < 6 * 3600


def test_chunk_boundaries_empty_when_start_at_end():
    ts = pd.Timestamp("2022-01-01")
    assert _chunk_boundaries(ts, ts, "1h") == []


def test_chunk_boundaries_single_chunk_when_window_le_train_freq():
    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2022-01-01 03:00")
    chunks = _chunk_boundaries(start, end, "6h")
    assert len(chunks) == 1
    assert chunks[0] == (start.timestamp(), end.timestamp())


# ---------------------------------------------------------------------------
# build_staggered_supersets
# ---------------------------------------------------------------------------


def _write_unit_csvs(folder, powerplants=None, storages=None, demands=None):
    """Helper to drop minimal unit CSVs into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    if powerplants is not None:
        powerplants.to_csv(folder / "powerplant_units.csv")
    if storages is not None:
        storages.to_csv(folder / "storage_units.csv")
    if demands is not None:
        demands.to_csv(folder / "demand_units.csv")


def test_build_staggered_supersets_unions_disjoint_scenarios(tmp_path):
    """With two scenarios that hold disjoint power plants, each scenario's
    ``extras`` should contain exactly the *other* scenario's rows, and the
    ``_superset/`` CSV should hold the union.
    """
    bau = tmp_path / "bau"
    inv = tmp_path / "inv"
    pp_bau = pd.DataFrame(
        {"max_power": [100, 200]}, index=pd.Index(["ppA", "ppB"], name="name")
    )
    pp_inv = pd.DataFrame({"max_power": [300]}, index=pd.Index(["ppC"], name="name"))
    _write_unit_csvs(bau, powerplants=pp_bau)
    _write_unit_csvs(inv, powerplants=pp_inv)

    local_ids, extras = build_staggered_supersets([str(bau), str(inv)])

    assert local_ids[0] == {"ppA", "ppB"}
    assert local_ids[1] == {"ppC"}

    # bau gets ppC as a foreign unit; inv gets ppA, ppB.
    assert set(extras[str(bau)]["powerplant_units"].index) == {"ppC"}
    assert set(extras[str(inv)]["powerplant_units"].index) == {"ppA", "ppB"}

    # _superset CSVs contain the union for both scenarios.
    union = {"ppA", "ppB", "ppC"}
    bau_super = pd.read_csv(bau / "_superset" / "powerplant_units.csv", index_col=0)
    inv_super = pd.read_csv(inv / "_superset" / "powerplant_units.csv", index_col=0)
    assert set(bau_super.index.astype(str)) == union
    assert set(inv_super.index.astype(str)) == union


def test_build_staggered_supersets_no_extras_for_identical_scenarios(tmp_path):
    """If both scenarios hold the same unit ids, there is nothing foreign to
    inject and the per-scenario extras DataFrames are empty.
    """
    a = tmp_path / "a"
    b = tmp_path / "b"
    pp = pd.DataFrame({"max_power": [100]}, index=pd.Index(["pp1"], name="name"))
    _write_unit_csvs(a, powerplants=pp)
    _write_unit_csvs(b, powerplants=pp)

    local_ids, extras = build_staggered_supersets([str(a), str(b)])

    assert local_ids[0] == local_ids[1] == {"pp1"}
    assert extras[str(a)]["powerplant_units"].empty
    assert extras[str(b)]["powerplant_units"].empty


def test_build_staggered_supersets_honors_unit_file_overrides(tmp_path):
    """Supersets must use the case-selected unit CSVs, not the defaults."""
    bau = tmp_path / "bau"
    inv = tmp_path / "inv"
    ignored_bau = pd.DataFrame(
        {"max_power": [100]}, index=pd.Index(["ignored_bau"], name="name")
    )
    ignored_inv = pd.DataFrame(
        {"max_power": [200]}, index=pd.Index(["ignored_inv"], name="name")
    )
    selected_bau = pd.DataFrame(
        {"max_power": [300]}, index=pd.Index(["pp_bau"], name="name")
    )
    selected_inv = pd.DataFrame(
        {"max_power": [400]}, index=pd.Index(["pp_inv"], name="name")
    )
    _write_unit_csvs(bau, powerplants=ignored_bau)
    _write_unit_csvs(inv, powerplants=ignored_inv)
    selected_bau.to_csv(bau / "selected_powerplants.csv")
    selected_inv.to_csv(inv / "selected_powerplants.csv")

    local_ids, extras = build_staggered_supersets(
        [str(bau), str(inv)],
        {
            str(bau): {"powerplant_units": "selected_powerplants.csv"},
            str(inv): {"powerplant_units": "selected_powerplants.csv"},
        },
    )

    assert local_ids == [{"pp_bau"}, {"pp_inv"}]
    assert set(extras[str(bau)]["powerplant_units"].index) == {"pp_inv"}
    assert set(extras[str(inv)]["powerplant_units"].index) == {"pp_bau"}


def test_build_staggered_supersets_handles_missing_unit_types(tmp_path):
    """A scenario without a given unit-type CSV (e.g. no storage) should not
    crash; foreign storages from the other scenario should still be emitted as
    extras for the scenario that lacks them.
    """
    a = tmp_path / "a"
    b = tmp_path / "b"
    pp_a = pd.DataFrame({"max_power": [100]}, index=pd.Index(["pp1"], name="name"))
    pp_b = pd.DataFrame({"max_power": [200]}, index=pd.Index(["pp2"], name="name"))
    storages_b = pd.DataFrame(
        {"max_power_charge": [50], "capacity": [200]},
        index=pd.Index(["s1"], name="name"),
    )
    _write_unit_csvs(a, powerplants=pp_a)  # no storages
    _write_unit_csvs(b, powerplants=pp_b, storages=storages_b)

    _, extras = build_staggered_supersets([str(a), str(b)])

    # Storage missing locally in `a` ⇒ s1 must appear in a's extras.
    assert set(extras[str(a)]["storage_units"].index) == {"s1"}
    # b already has s1 ⇒ no foreign storages for b.
    assert extras[str(b)]["storage_units"].empty


# ---------------------------------------------------------------------------
# LearningConfig
# ---------------------------------------------------------------------------


def test_learning_config_staggered_field_defaults_to_none():
    cfg = LearningConfig()
    assert cfg.staggered_training is None


def test_learning_config_staggered_field_round_trips():
    payload = {
        "enabled": True,
        "scenarios": [
            {"path": "scenarios/bau", "name": "bau"},
            {"path": "scenarios/inv", "name": "inv"},
        ],
        "swap_order_per_episode": False,
    }
    cfg = LearningConfig(staggered_training=payload)
    assert cfg.staggered_training == payload
    assert cfg.staggered_training["scenarios"][1]["name"] == "inv"


# ---------------------------------------------------------------------------
# run_learning gate
# ---------------------------------------------------------------------------


def test_run_learning_redirects_when_staggered_enabled():
    """When ``staggered_training.enabled`` is set, the single-world entry point
    must refuse to run and tell the caller to use ``run_staggered_learning``.
    """
    fake_world = SimpleNamespace(
        scenario_data={
            "config": {
                "learning_config": {
                    "staggered_training": {"enabled": True},
                },
            },
        },
    )
    with pytest.raises(ValueError, match="run_staggered_learning"):
        run_learning(fake_world)


def test_run_learning_does_not_redirect_when_staggered_disabled():
    """``enabled: false`` must fall through to the normal single-world path.
    We assert *no* ``ValueError`` mentioning ``run_staggered_learning`` is
    raised — it must fail later for an unrelated reason (missing learning_role
    on our minimal fake world) which proves the gate let it through.
    """
    fake_world = SimpleNamespace(
        scenario_data={
            "config": {
                "learning_config": {
                    "staggered_training": {"enabled": False},
                },
            },
        },
        export_csv_path="",
    )
    with pytest.raises(Exception) as excinfo:
        run_learning(fake_world)
    assert "run_staggered_learning" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# _ensure_persistent_loop
# ---------------------------------------------------------------------------


def test_ensure_persistent_loop_aligns_secondary_to_anchor():
    """Both worlds must end up sharing the anchor's event loop so the trainer
    can drive them with a single ``run_until_complete``.
    """
    primary = asyncio.new_event_loop()
    secondary_loop = asyncio.new_event_loop()
    try:
        anchor = SimpleNamespace(loop=primary)
        secondary = SimpleNamespace(loop=secondary_loop)
        result = _ensure_persistent_loop([anchor, secondary])
        assert result is primary
        assert anchor.loop is primary
        assert secondary.loop is primary
        assert asyncio.get_event_loop() is primary
    finally:
        primary.close()
        secondary_loop.close()
        # Restore a fresh loop for any subsequent tests.
        asyncio.set_event_loop(asyncio.new_event_loop())


def test_run_learning_resume_skips_training_and_keeps_tensorboard(
    tmp_path, monkeypatch
):
    """Resume mode should keep existing TensorBoard logs and skip training if
    loaded episodes already reach the configured training horizon.
    """

    class FakeLearningRole:
        def __init__(self, learning_config):
            self.learning_config = learning_config
            self.rl_algorithm = SimpleNamespace(
                initialize_policy=lambda: None,
                obs_dim=1,
                act_dim=1,
            )
            self.rl_strats = {"u1": SimpleNamespace()}
            self.device = "cpu"
            self.float_type = None
            self.buffer = None
            self.episodes_done = 0
            self.eval_episodes_done = 0

        def load_inter_episodic_data(self, data):
            self.buffer = data["buffer"]

        def determine_validation_interval(self):
            return 1

        def sync_train_freq_with_simulation_horizon(self):
            return "1h"

        def load_runtime_state(self, path):
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self.episodes_done = int(payload.get("episodes_done", 0))
            self.eval_episodes_done = int(payload.get("eval_episodes_done", 0))

        def get_inter_episodic_data(self):
            return {
                "buffer": self.buffer,
                "actors_and_critics": None,
                "max_eval": {},
                "all_eval": {},
                "avg_all_eval": [],
                "episodes_done": self.episodes_done,
                "eval_episodes_done": self.eval_episodes_done,
            }

    save_dir = tmp_path / "learned"
    save_dir.mkdir(parents=True, exist_ok=True)
    state_path = save_dir / "last_policies" / "learning_state.pt"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"episodes_done": 2, "eval_episodes_done": 4}),
        encoding="utf-8",
    )

    sim_id = "resume_single_world"
    tb_dir = Path("tensorboard") / sim_id
    tb_dir.mkdir(parents=True, exist_ok=True)
    marker = tb_dir / "keep_me.txt"
    marker.write_text("keep", encoding="utf-8")

    lc = LearningConfig(
        learning_mode=True,
        training_episodes=2,
        episodes_collecting_initial_experience=1,
        train_freq="1h",
        trained_policies_save_path=str(save_dir),
        load_learning_state=True,
        learning_state_load_path=str(state_path),
    )

    fake_world = SimpleNamespace(
        scenario_data={
            "simulation_id": sim_id,
            "config": {"learning_config": {}},
        },
        learning_role=FakeLearningRole(lc),
        export_csv_path="",
        run=lambda: None,
        reset=lambda: None,
        db_uri="",
    )

    monkeypatch.setattr(
        "assume.scenario.loader_csv.confirm_learning_save_path", lambda *_: None
    )
    monkeypatch.setattr("assume.scenario.loader_csv.setup_world", lambda **_: None)

    run_learning(fake_world)

    assert marker.exists(), "TensorBoard directory should be preserved in resume mode"


def test_staggered_trainer_resume_skips_training_and_keeps_tensorboard(
    tmp_path, monkeypatch
):
    """Staggered trainer in resume mode should preserve TensorBoard dirs and
    skip training when loaded episodes already complete the horizon.
    """

    class FakeLearningRole:
        def __init__(self, learning_config):
            self.learning_config = learning_config
            self.rl_algorithm = SimpleNamespace(
                initialize_policy=lambda: None,
                obs_dim=1,
                act_dim=1,
            )
            self.rl_strats = {"u1": SimpleNamespace()}
            self.device = "cpu"
            self.float_type = None
            self.buffer = None
            self.episodes_done = 0
            self.eval_episodes_done = 0

        def load_inter_episodic_data(self, data):
            self.buffer = data["buffer"]

        def determine_validation_interval(self):
            return 1

        def sync_train_freq_with_simulation_horizon(self):
            return "1h"

        def load_runtime_state(self, path):
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self.episodes_done = int(payload.get("episodes_done", 0))
            self.eval_episodes_done = int(payload.get("eval_episodes_done", 0))

        def get_inter_episodic_data(self):
            return {
                "buffer": self.buffer,
                "actors_and_critics": None,
                "max_eval": {},
                "all_eval": {},
                "avg_all_eval": [],
                "episodes_done": self.episodes_done,
                "eval_episodes_done": self.eval_episodes_done,
            }

        tensor_board_logger = SimpleNamespace(update_tensorboard=lambda: None)

    save_dir = tmp_path / "learned_staggered"
    save_dir.mkdir(parents=True, exist_ok=True)
    state_path = save_dir / "last_policies" / "learning_state.pt"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"episodes_done": 2, "eval_episodes_done": 3}),
        encoding="utf-8",
    )

    lc = LearningConfig(
        learning_mode=True,
        training_episodes=2,
        episodes_collecting_initial_experience=1,
        train_freq="1h",
        trained_policies_save_path=str(save_dir),
        load_learning_state=True,
        learning_state_load_path=str(state_path),
    )

    shared_loop = asyncio.new_event_loop()
    sim_a = "resume_staggered_a"
    sim_b = "resume_staggered_b"
    tb_a = Path("tensorboard") / sim_a
    tb_b = Path("tensorboard") / sim_b
    tb_a.mkdir(parents=True, exist_ok=True)
    tb_b.mkdir(parents=True, exist_ok=True)
    marker_a = tb_a / "keep_me.txt"
    marker_b = tb_b / "keep_me.txt"
    marker_a.write_text("keep", encoding="utf-8")
    marker_b.write_text("keep", encoding="utf-8")

    world_a = SimpleNamespace(
        scenario_data={"simulation_id": sim_a, "config": {"learning_config": {}}},
        simulation_id=sim_a,
        learning_role=FakeLearningRole(lc),
        export_csv_path="",
        reset=lambda: None,
        loop=shared_loop,
    )
    world_b = SimpleNamespace(
        scenario_data={"simulation_id": sim_b, "config": {"learning_config": {}}},
        simulation_id=sim_b,
        learning_role=FakeLearningRole(lc),
        export_csv_path="",
        reset=lambda: None,
        loop=shared_loop,
    )

    monkeypatch.setattr(
        "assume.reinforcement_learning.staggered_trainer._share_learning_state",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "assume.reinforcement_learning.staggered_trainer._ensure_persistent_loop",
        lambda worlds: worlds[0].loop,
    )
    monkeypatch.setattr(
        "assume.reinforcement_learning.staggered_trainer.confirm_learning_save_path",
        lambda *_: None,
    )
    monkeypatch.setattr("assume.scenario.loader_csv.setup_world", lambda **_: None)

    try:
        trainer = StaggeredTrainer([world_a, world_b], verbose=False)
        trainer.run()
    finally:
        shared_loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())

    assert marker_a.exists(), "Anchor TensorBoard directory should be preserved"
    assert marker_b.exists(), "Secondary TensorBoard directory should be preserved"


@pytest.mark.require_learning
def test_staggered_trainer_updates_tensorboard_for_both_worlds(tmp_path, monkeypatch):
    """After each staggered training episode, both worlds' TensorBoard loggers
    must be refreshed so each simulation_id produces visible scalar series.
    """

    class CountingLogger:
        def __init__(self):
            self.calls = 0

        def update_tensorboard(self):
            self.calls += 1

    class FakeLearningRole:
        def __init__(self, learning_config):
            self.learning_config = learning_config
            self.rl_algorithm = SimpleNamespace(
                initialize_policy=lambda: None,
                obs_dim=1,
                act_dim=1,
                save_params=lambda **_: None,
            )
            self.rl_strats = {"u1": SimpleNamespace()}
            self.device = "cpu"
            self.float_type = None
            self.buffer = None
            self.episodes_done = 0
            self.eval_episodes_done = 0
            self.tensor_board_logger = CountingLogger()

        def load_inter_episodic_data(self, data):
            self.buffer = data["buffer"]

        def determine_validation_interval(self):
            return 1

        def sync_train_freq_with_simulation_horizon(self):
            return "1h"

        def get_inter_episodic_data(self):
            return {
                "buffer": self.buffer,
                "actors_and_critics": None,
                "max_eval": {},
                "all_eval": {},
                "avg_all_eval": [],
                "episodes_done": self.episodes_done,
                "eval_episodes_done": self.eval_episodes_done,
            }

        def save_runtime_state(self, path):
            pass

    lc = LearningConfig(
        learning_mode=True,
        training_episodes=1,
        episodes_collecting_initial_experience=99,
        train_freq="1h",
        trained_policies_save_path=str(tmp_path / "learned"),
    )

    shared_loop = asyncio.new_event_loop()
    world_a = SimpleNamespace(
        scenario_data={"simulation_id": "sim_a", "config": {"learning_config": {}}},
        simulation_id="sim_a",
        learning_role=FakeLearningRole(lc),
        export_csv_path="",
        reset=lambda: None,
        loop=shared_loop,
    )
    world_b = SimpleNamespace(
        scenario_data={"simulation_id": "sim_b", "config": {"learning_config": {}}},
        simulation_id="sim_b",
        learning_role=FakeLearningRole(lc),
        export_csv_path="",
        reset=lambda: None,
        loop=shared_loop,
    )

    monkeypatch.setattr(
        "assume.reinforcement_learning.staggered_trainer._share_learning_state",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "assume.reinforcement_learning.staggered_trainer._ensure_persistent_loop",
        lambda worlds: worlds[0].loop,
    )
    monkeypatch.setattr(
        "assume.reinforcement_learning.staggered_trainer.confirm_learning_save_path",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "assume.reinforcement_learning.staggered_trainer.StaggeredTrainer._run_episode_chunked",
        lambda *_, **__: None,
    )
    monkeypatch.setattr("assume.scenario.loader_csv.setup_world", lambda **_: None)

    try:
        trainer = StaggeredTrainer([world_a, world_b], verbose=False)
        trainer.run()
    finally:
        shared_loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())

    assert world_a.learning_role.tensor_board_logger.calls == 1
    assert world_b.learning_role.tensor_board_logger.calls == 1


def test_load_staggered_scenario_rejects_duplicate_simulation_ids(
    tmp_path, monkeypatch
):
    """Staggered loader must fail fast when both worlds resolve to the same
    simulation_id to prevent TensorBoard/DB namespace collisions.
    """

    primary = tmp_path / "primary"
    scenario_a = tmp_path / "scenario_a"
    scenario_b = tmp_path / "scenario_b"
    primary.mkdir(parents=True, exist_ok=True)
    scenario_a.mkdir(parents=True, exist_ok=True)
    scenario_b.mkdir(parents=True, exist_ok=True)

    primary.joinpath("config.yaml").write_text(
        "\n".join(
            [
                "case:",
                "  learning_config:",
                "    staggered_training:",
                "      enabled: true",
                "      scenarios:",
                "        - path: ../scenario_a",
                "          name: bau",
                "        - path: ../scenario_a",
                "          name: inv",
            ]
        ),
        encoding="utf-8",
    )
    scenario_a.joinpath("config.yaml").write_text("case: {}\n", encoding="utf-8")
    scenario_b.joinpath("config.yaml").write_text("case: {}\n", encoding="utf-8")

    monkeypatch.setattr(
        "assume.scenario.loader_csv._check_staggered_input_file_parity",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "assume.scenario.loader_csv.build_staggered_supersets",
        lambda paths, unit_file_overrides=None: ([], {p: {} for p in paths}),
    )
    monkeypatch.setattr(
        "assume.scenario.loader_csv.load_config_and_create_forecaster",
        lambda *_, **__: {
            "simulation_id": "same_simulation_id",
            "config": {"learning_config": {}},
        },
    )

    def _fake_setup_world(world, **_):
        world.learning_role = SimpleNamespace(rl_strats={})
        world.markets = {}

    monkeypatch.setattr("assume.scenario.loader_csv.setup_world", _fake_setup_world)

    loop = asyncio.new_event_loop()
    worlds = [
        SimpleNamespace(loop=loop, scenario_data={}, markets={}),
        SimpleNamespace(loop=loop, scenario_data={}, markets={}),
    ]

    try:
        with pytest.raises(ValueError, match="distinct non-empty simulation_id"):
            load_staggered_scenario(
                worlds=worlds,
                inputs_path=str(tmp_path),
                scenario="primary",
                study_case="case",
            )
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# load_config_and_create_forecaster — extra_units injection (G2 / G4 loader side)
# ---------------------------------------------------------------------------


def test_extra_units_appended_with_zero_availability_and_demand(tmp_path):
    """When ``extra_units`` is passed, foreign powerplant rows must be appended
    to ``powerplant_units`` and their forecasters must report zeroed
    availability. Foreign demand rows must additionally have a zeroed demand
    series. This covers the loader-side guarantees for D3 acceptance criteria
    G2 (matching agent sets) and G4 (foreign units contribute zero).
    """
    foreign_pp = pd.DataFrame(
        {
            "technology": ["nuclear"],
            "bidding_EOM": ["powerplant_energy_naive"],
            "fuel_type": ["uranium"],
            "emission_factor": [0.0],
            "max_power": [500.0],
            "min_power": [100.0],
            "efficiency": [0.3],
            "additional_cost": [10.0],
            "unit_operator": ["Foreign Op"],
        },
        index=pd.Index(["foreign_pp"], name="name"),
    )
    foreign_demand = pd.DataFrame(
        {
            "technology": ["inflex_demand"],
            "bidding_EOM": ["demand_energy_naive"],
            "max_power": [10000.0],
            "min_power": [0.0],
            "unit_operator": ["foreign_eom"],
        },
        index=pd.Index(["foreign_demand"], name="name"),
    )

    extras = {
        "powerplant_units": foreign_pp,
        "demand_units": foreign_demand,
    }

    scenario_data = load_config_and_create_forecaster(
        inputs_path="examples/inputs",
        scenario="example_01a",
        study_case="tiny",
        extra_units=extras,
    )

    # Foreign rows are appended to the unit DataFrames…
    assert "foreign_pp" in scenario_data["powerplant_units"].index
    assert "foreign_demand" in scenario_data["demand_units"].index
    # …without dropping any local rows.
    assert "Unit 1" in scenario_data["powerplant_units"].index
    assert "demand_EOM" in scenario_data["demand_units"].index

    # The PowerplantForecaster for the foreign unit reports zero availability
    # over the entire horizon — this is what guarantees foreign units cannot
    # bid into the local market.
    foreign_pp_fc = scenario_data["unit_forecasts"]["foreign_pp"]
    assert (foreign_pp_fc.availability == 0).all()
    # Local units retain their default availability (== 1) since no override.
    local_pp_fc = scenario_data["unit_forecasts"]["Unit 1"]
    assert (local_pp_fc.availability > 0).any()

    # The DemandForecaster for the foreign demand unit reports zero demand.
    foreign_demand_fc = scenario_data["unit_forecasts"]["foreign_demand"]
    assert (foreign_demand_fc.demand == 0).all()


def test_extra_units_skipped_when_already_local():
    """If a foreign DataFrame contains a row whose id already exists locally,
    that row must not be duplicated and must keep its local (non-zero)
    availability. This guards against an over-eager merge wiping local units.
    """
    duplicate_pp = pd.DataFrame(
        {
            "technology": ["nuclear"],
            "bidding_EOM": ["powerplant_energy_naive"],
            "fuel_type": ["uranium"],
            "emission_factor": [0.0],
            "max_power": [9999.0],  # different value to detect overwrites
            "min_power": [0.0],
            "efficiency": [0.3],
            "additional_cost": [10.0],
            "unit_operator": ["Operator 1"],
        },
        index=pd.Index(["Unit 1"], name="name"),  # already in example_01a
    )

    scenario_data = load_config_and_create_forecaster(
        inputs_path="examples/inputs",
        scenario="example_01a",
        study_case="tiny",
        extra_units={"powerplant_units": duplicate_pp},
    )

    # Unit 1 appears exactly once.
    assert (scenario_data["powerplant_units"].index == "Unit 1").sum() == 1
    # Local row was not overwritten.
    assert scenario_data["powerplant_units"].loc["Unit 1", "max_power"] != 9999.0
    # Local availability untouched.
    assert (scenario_data["unit_forecasts"]["Unit 1"].availability > 0).any()
