# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Round trip of storage schedules through small two-node CSV scenarios.

Node ``north`` has cheap generation, node ``south`` has an expensive generator,
the demand and a 2h battery. The line between them is limited, so the south runs
short in the evening peak.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from assume import World
from assume.common.storage_schedule import export_storage_schedule
from assume.scenario.loader_csv import load_scenario_folder

START = "2019-01-01 00:00"
END = "2019-01-02 23:00"
INDEX = pd.date_range(START, END, freq="h")
SCHEDULE_PRICES = {"schedule_discharge_price": -100.0, "schedule_charge_price": 1000.0}

# Daily demand: low at night, morning shoulder, evening peak. North demand makes
# the system-wide (network-blind) price forecast high by day and low at night, so
# the existing heuristic charges overnight.
SOUTH_DAY = [40, 40, 40, 40, 40, 60, 120, 150, 160, 150, 140, 130,
             130, 130, 130, 140, 170, 200, 230, 240, 200, 150, 80, 50]  # fmt: skip
NORTH_DAY = [150, 150, 150, 150, 150, 170, 200, 230, 240, 230, 220, 210,
             210, 210, 210, 220, 230, 240, 240, 240, 230, 200, 170, 150]  # fmt: skip
DEMANDS = ["demand_north", "demand_south"]


def write_scenario(
    path: Path,
    storage_strategy: str,
    bidding_params: dict,
    demand: pd.DataFrame | None = None,
    forecasts: pd.DataFrame | None = None,
    line_capacity: float = 60.0,
    learning_config: dict | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "start_date": START,
        "end_date": END,
        "time_step": "1h",
        "save_frequency_hours": None,
        "bidding_strategy_params": bidding_params,
        "markets_config": {
            "EOM": {
                "start_date": START,
                "operator": "EOM_operator",
                "product_type": "energy",
                "products": [{"duration": "1h", "count": 1, "first_delivery": "1h"}],
                "opening_frequency": "1h",
                "opening_duration": "1h",
                "volume_unit": "MWh",
                "maximum_bid_volume": 100000,
                "maximum_bid_price": 3000,
                "minimum_bid_price": -500,
                "price_unit": "EUR/MWh",
                "market_mechanism": "complex_clearing",
                "additional_fields": ["node", "max_power"],
                "param_dict": {
                    "network_path": ".",
                    "solver_name": "highs",
                    "zones_identifier": "zone_id",
                    "pricing_mechanism": "pay_as_clear",
                },
            }
        },
    }
    if learning_config is not None:
        config["learning_config"] = learning_config
    with open(path / "config.yaml", "w") as f:
        yaml.safe_dump({"base": config}, f)

    pd.DataFrame(
        {
            "v_nom": [220.0, 220.0],
            "zone_id": ["north", "south"],
            "x": [0, 1],
            "y": [0, 0],
        },
        index=pd.Index(["north", "south"], name="name"),
    ).to_csv(path / "buses.csv")
    pd.DataFrame(
        {
            "bus0": ["north"],
            "bus1": ["south"],
            "s_nom": [line_capacity],
            "s_max_pu": [1.0],
            "x": [0.01],
            "r": [0.001],
        },  # fmt: skip
        index=pd.Index(["line_n_s"], name="line"),
    ).to_csv(path / "lines.csv")

    pd.DataFrame(
        {
            "technology": ["nuclear", "oil"],
            "bidding_EOM": ["powerplant_energy_naive"] * 2,
            "fuel_type": ["uranium", "oil"],
            "emission_factor": [0.0, 0.0],
            "max_power": [300.0, 100.0],
            "min_power": [0.0, 0.0],
            "efficiency": [1.0, 1.0],
            "additional_cost": [10.0, 80.0],
            "node": ["north", "south"],
            "unit_operator": ["operator", "operator"],
        },
        index=pd.Index(["gen_north", "gen_south"], name="name"),
    ).to_csv(path / "powerplant_units.csv")
    pd.DataFrame(
        {"fuel": ["price"], "uranium": [0.0], "oil": [0.0], "co2": [0.0]}
    ).to_csv(path / "fuel_prices_df.csv", index=False)

    pd.DataFrame(
        {
            "technology": ["battery"],
            "bidding_EOM": [storage_strategy],
            "max_power_charge": [50.0],
            "max_power_discharge": [50.0],
            "efficiency_charge": [0.95],
            "efficiency_discharge": [0.95],
            "min_soc": [0.0],
            "max_soc": [1.0],
            "initial_soc": [0.5],
            "capacity": [100.0],
            "additional_cost_charge": [0.0],
            "additional_cost_discharge": [0.0],
            "node": ["south"],
            "unit_operator": ["operator"],
        },
        index=pd.Index(["battery"], name="name"),
    ).to_csv(path / "storage_units.csv")

    pd.DataFrame(
        {
            "technology": ["inflex_demand"] * 2,
            "bidding_EOM": ["demand_energy_naive"] * 2,
            "max_power": [1000.0] * 2,
            "min_power": [0.0] * 2,
            "node": ["north", "south"],
            "unit_operator": ["demand_operator"] * 2,
        },
        index=pd.Index(DEMANDS, name="name"),
    ).to_csv(path / "demand_units.csv")
    if demand is None:
        demand = base_demand()
    demand.to_csv(path / "demand_df.csv", index_label="datetime")
    if forecasts is not None:
        forecasts.to_csv(path / "forecasts_df.csv", index_label="datetime")


def base_demand() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "demand_north": np.tile(NORTH_DAY, 2).astype(float),
            "demand_south": np.tile(SOUTH_DAY, 2).astype(float),
        },
        index=INDEX,
    )


def run_scenario(inputs: Path, scenario: str, output: Path) -> pd.DataFrame:
    world = World(database_uri=None, export_csv_path=str(output))
    load_scenario_folder(
        world, inputs_path=str(inputs), scenario=scenario, study_case="base"
    )
    world.run()
    dispatch = pd.read_csv(
        output / f"{scenario}_base" / "unit_dispatch.csv", parse_dates=["time"]
    )
    return dispatch


def series(dispatch: pd.DataFrame, unit: str, column: str = "power") -> pd.Series:
    rows = dispatch[dispatch["unit"] == unit].set_index("time")[column]
    return rows.reindex(INDEX[:-1])


def unserved(dispatch: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    """Unserved demand per demand unit for every dispatched hour."""
    served = pd.DataFrame({unit: -series(dispatch, unit) for unit in DEMANDS})
    return (demand.reindex(served.index) - served).clip(lower=0)


def production_cost(dispatch: pd.DataFrame) -> float:
    return float(
        series(dispatch, "gen_north").sum() * 10
        + series(dispatch, "gen_south").sum() * 80
    )


@pytest.fixture
def raw_run(tmp_path):
    inputs, output = tmp_path / "inputs", tmp_path / "output"
    write_scenario(
        inputs / "raw",
        "storage_energy_heuristic_flexable",
        {"eom_foresight": "6h", "heuristic_foresight": "6h"},
    )
    dispatch = run_scenario(inputs, "raw", output)
    schedule = export_storage_schedule(
        "raw_base", INDEX, csv_path=output, storage_unit_ids=["battery"]
    )
    return inputs, output, dispatch, schedule


def test_raw_run_has_deficits_and_exports_schedule(raw_run):
    _, _, dispatch, schedule = raw_run

    assert unserved(dispatch, base_demand()).to_numpy().sum() > 0
    battery = series(dispatch, "battery")
    assert (battery > 0).any() and (battery < 0).any()
    np.testing.assert_allclose(
        schedule["storage_schedule_battery"].iloc[:-1],
        series(dispatch, "battery"),
        atol=1e-9,
    )
    assert schedule["storage_schedule_battery"].iloc[-1] == 0


def test_fixed_schedule_round_trip_reproduces_physical_dispatch(raw_run):
    inputs, output, raw, schedule = raw_run
    write_scenario(
        inputs / "fixed",
        "storage_energy_fixed_schedule",
        SCHEDULE_PRICES,
        forecasts=schedule,
    )
    fixed = run_scenario(inputs, "fixed", output)

    for unit in ["battery", "gen_north", "gen_south", *DEMANDS]:
        np.testing.assert_allclose(series(fixed, unit), series(raw, unit), atol=1e-4)
    np.testing.assert_allclose(
        series(fixed, "battery", "soc"), series(raw, "battery", "soc"), atol=1e-4
    )
    assert production_cost(fixed) == pytest.approx(production_cost(raw), abs=1e-2)

    again = export_storage_schedule(
        "fixed_base", INDEX, csv_path=output, storage_unit_ids=["battery"]
    )
    # CSV outputs keep 5 significant digits, so a step that empties the battery
    # can be cut by ~1e-4 MW when the rounded schedule is executed.
    pd.testing.assert_frame_equal(again, schedule, atol=1e-3, check_freq=False)


def test_fixed_schedule_with_adjusted_demand_has_no_deficits(raw_run):
    inputs, output, raw, schedule = raw_run
    demand = base_demand()
    adjusted = demand.copy()
    adjusted.loc[INDEX[:-1]] -= unserved(raw, demand)
    write_scenario(
        inputs / "adjusted",
        "storage_energy_fixed_schedule",
        SCHEDULE_PRICES,
        demand=adjusted,
        forecasts=schedule,
    )
    fixed = run_scenario(inputs, "adjusted", output)

    assert unserved(fixed, adjusted).to_numpy().max() == pytest.approx(0.0, abs=1e-4)
    np.testing.assert_allclose(
        series(fixed, "battery"), series(raw, "battery"), atol=1e-4
    )
    assert production_cost(fixed) == pytest.approx(production_cost(raw), abs=1e-2)


@pytest.mark.require_learning
def test_schedule_dispatch_learning_smoke(raw_run):
    """Train a few episodes, then evaluate with the saved policy loaded."""
    from assume.scenario.loader_csv import run_learning

    inputs, output, raw, schedule = raw_run
    policies = inputs.parent / "policies"
    write_scenario(
        inputs / "learning",
        "storage_energy_learning_schedule_dispatch_congestion",
        {**SCHEDULE_PRICES, "n_lines": 1, "foresight": 4},
        forecasts=schedule,
        learning_config={
            "learning_mode": True,
            "max_bid_price": 3000,
            "algorithm": "matd3",
            "learning_rate": 0.001,
            "training_episodes": 3,
            "episodes_collecting_initial_experience": 1,
            "train_freq": "24h",
            "gradient_steps": 2,
            "batch_size": 8,
            "gamma": 0.99,
            "device": "cpu",
            "noise_sigma": 0.1,
            "noise_scale": 1,
            "noise_dt": 1,
            "validation_episodes_interval": 1,
            "trained_policies_save_path": str(policies),
        },
    )
    world = World(
        database_uri=f"sqlite:///{inputs.parent / 'learning.db'}",
        export_csv_path=str(output),
    )
    load_scenario_folder(
        world, inputs_path=str(inputs), scenario="learning", study_case="base"
    )
    run_learning(world)
    assert any((policies / "last_policies").rglob("*.pt"))

    world.run()  # evaluation run with the last saved policy loaded
    evaluated = pd.read_csv(
        output / "learning_base" / "unit_dispatch.csv", parse_dates=["time"]
    )
    battery = series(evaluated, "battery")
    scheduled = schedule["storage_schedule_battery"].reindex(battery.index)
    deviation = series(evaluated, "battery", "schedule_deviation")

    # Charging never exceeds the schedule; discharge only falls short by withholding.
    assert (battery[scheduled < 0] >= scheduled[scheduled < 0] - 1e-3).all()
    assert (battery[scheduled > 0] <= scheduled[scheduled > 0] + 1e-3).all()
    assert (battery[scheduled == 0].abs() <= 1e-3).all()
    soc = series(evaluated, "battery", "soc")
    assert soc.between(-1e-6, 1 + 1e-6).all()
    assert deviation.notna().all()


def test_fixed_schedule_fails_in_network_where_it_cannot_clear(raw_run):
    inputs, output, raw, schedule = raw_run
    write_scenario(
        inputs / "tight",
        "storage_energy_fixed_schedule",
        SCHEDULE_PRICES,
        forecasts=schedule,
        line_capacity=0.0,
    )
    with pytest.raises(ValueError, match="did not clear"):
        run_scenario(inputs, "tight", output)
