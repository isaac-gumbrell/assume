# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest

from assume.common.fast_pandas import FastSeries
from assume.common.forecaster import UnitForecaster
from assume.common.market_objects import MarketConfig
from assume.common.storage_schedule import (
    load_storage_schedule,
    replay_schedule_soc,
    schedule_from_unit_dispatch,
    validate_storage_schedule,
)
from assume.strategies.scheduled_storage import StorageEnergyFixedScheduleStrategy
from assume.units import Storage

INDEX = pd.date_range("2023-07-01", periods=6, freq="h")


def make_storage(schedule=None, availability=1.0, unit_id="battery") -> Storage:
    forecaster = UnitForecaster(
        INDEX, market_prices={"EOM": 50}, availability=availability
    )
    if schedule is not None:
        forecaster.storage_schedule = FastSeries(
            index=forecaster.index, value=np.asarray(schedule, dtype=float)
        )
    return Storage(
        id=unit_id,
        unit_operator="operator",
        technology="battery",
        bidding_strategies={},
        forecaster=forecaster,
        max_power_charge=-100,
        max_power_discharge=100,
        min_power_charge=-10,
        min_power_discharge=10,
        capacity=200,
        initial_soc=0.5,
        efficiency_charge=0.9,
        efficiency_discharge=0.8,
    )


@pytest.fixture
def market_config():
    mc = MarketConfig()
    mc.market_id = "EOM"
    mc.product_type = "energy"
    return mc


def products(start):
    return [(start, start + pd.Timedelta(hours=1), None)]


# Discharge 80 MW (-0.5 SoC), charge 100 MW (+0.45), idle, discharge 40 MW (-0.25)
FEASIBLE = [80.0, -100.0, 0.0, 40.0, 0.0, 0.0]


def test_replay_matches_storage_update():
    soc = replay_schedule_soc(FEASIBLE, 0.5, 200, 0.9, 0.8, 1.0)
    np.testing.assert_allclose(soc, [0.5, 0.0, 0.45, 0.45, 0.2, 0.2, 0.2])


def test_validate_accepts_feasible_schedule():
    validate_storage_schedule(make_storage(), np.array(FEASIBLE), 1e-3)


def test_replay_resets_rounding_overshoot_without_accumulating():
    # Each discharge empties the battery plus 1e-4 MW of rounding.
    schedule = np.array([80.0001, -1000 / 9] * 3)
    soc = replay_schedule_soc(schedule, 0.5, 200, 0.9, 0.8, 1.0, 0.0, 1.0, 1e-5)
    assert soc.min() == 0.0
    np.testing.assert_allclose(soc[-1], 0.5)

    soc = replay_schedule_soc(schedule, 0.5, 200, 0.9, 0.8, 1.0, 0.0, 1.0, 1e-9)
    assert soc.min() < 0.0


@pytest.mark.parametrize(
    "schedule, match",
    [
        ([np.nan, 0, 0, 0, 0, 0], "missing"),
        ([120, 0, 0, 0, 0, 0], "exceeds 100"),
        ([0, -120, 0, 0, 0, 0], "exceeds -100"),
        ([5, 0, 0, 0, 0, 0], "below min_power_discharge"),
        ([0, -5, 0, 0, 0, 0], "below min_power_charge"),
        ([80, 80, 0, 0, 0, 0], "SoC after this step"),
        ([-100, -100, 0, 0, 0, 0], "SoC after this step"),
        ([0, 0, 0], "3 steps, expected 6"),
    ],
)
def test_validate_rejects_infeasible_schedule(schedule, match):
    with pytest.raises(ValueError, match=match):
        validate_storage_schedule(make_storage(), np.array(schedule, float), 1e-3)


def test_validate_rejects_dispatch_while_unavailable():
    unit = make_storage(availability=0.0)
    with pytest.raises(ValueError, match="exceeds 0"):
        validate_storage_schedule(unit, np.array(FEASIBLE), 1e-3)


def test_load_storage_schedule_reads_own_column_and_rejects_unknown_units():
    unit = make_storage()
    other = make_storage(unit_id="other")
    df = pd.DataFrame({"storage_schedule_battery": FEASIBLE}, index=INDEX)

    schedule = load_storage_schedule(df, [unit, other], unit, unit.forecaster.index)
    np.testing.assert_allclose(schedule.data, FEASIBLE)
    assert (
        load_storage_schedule(df, [unit, other], other, other.forecaster.index) is None
    )
    assert load_storage_schedule(None, [unit], unit, unit.forecaster.index) is None

    df["storage_schedule_missing_unit"] = 0.0
    with pytest.raises(ValueError, match="unknown storage units"):
        load_storage_schedule(df, [unit, other], unit, unit.forecaster.index)


def test_forecaster_initialize_loads_schedule():
    unit = make_storage()
    df = pd.DataFrame({"storage_schedule_battery": FEASIBLE}, index=INDEX)
    unit.forecaster.forecast_algorithms = {
        "price": "price_default_test",
        "residual_load": "residual_load_default_test",
    }
    from assume.common.forecast_algorithms import get_forecast_registries

    unit.forecaster._registries = get_forecast_registries()
    unit.forecaster.initialize([unit], [], df, unit)
    np.testing.assert_allclose(unit.forecaster.storage_schedule.data, FEASIBLE)


def dispatch_table(power, soc, unit="battery", index=INDEX):
    return pd.DataFrame(
        {"time": index[: len(power)], "unit": unit, "power": power, "soc": soc}
    )


def test_schedule_from_unit_dispatch_pivots_and_pads_trailing_steps():
    soc = replay_schedule_soc(FEASIBLE, 0.5, 200, 0.9, 0.8, 1.0)
    generator = pd.DataFrame(
        {"time": INDEX[:4], "unit": "gen", "power": 50.0, "soc": np.nan}
    )
    dispatch = pd.concat(
        [dispatch_table(FEASIBLE[:4], soc[:4]), generator], ignore_index=True
    )

    schedule = schedule_from_unit_dispatch(dispatch, INDEX)

    assert list(schedule.columns) == ["storage_schedule_battery"]
    np.testing.assert_allclose(
        schedule["storage_schedule_battery"], FEASIBLE[:4] + [0.0, 0.0]
    )


@pytest.mark.parametrize(
    "dispatch, match",
    [
        (
            dispatch_table([0.0, 0.0], [0.5, 0.5], index=INDEX[[0, 2]]),
            "no power",
        ),
        (
            pd.concat([dispatch_table([80.0], [0.5])] * 2, ignore_index=True),
            "duplicate",
        ),
        (dispatch_table([0.0, 0.0], [0.5, 0.4]), "zero power"),
        (dispatch_table([80.0, 0.0], [0.5, 0.6]), "against the sign"),
        (
            dispatch_table([0.0], [0.5], index=pd.DatetimeIndex(["2024-01-01"])),
            "outside the target index",
        ),
    ],
)
def test_schedule_from_unit_dispatch_rejects_inconsistent_output(dispatch, match):
    with pytest.raises(ValueError, match=match):
        schedule_from_unit_dispatch(dispatch, INDEX)


def fixed_strategy(**kwargs):
    params = {"schedule_discharge_price": -10.0, "schedule_charge_price": 10_000.0}
    params.update(kwargs)
    return StorageEnergyFixedScheduleStrategy(**params)


@pytest.mark.parametrize(
    "missing", ["schedule_discharge_price", "schedule_charge_price"]
)
def test_fixed_schedule_requires_prices(missing):
    params = {"schedule_discharge_price": -10.0, "schedule_charge_price": 10_000.0}
    params.pop(missing)
    with pytest.raises(ValueError, match=missing):
        StorageEnergyFixedScheduleStrategy(**params)


def test_fixed_schedule_bids_schedule_at_fixed_prices(market_config):
    unit = make_storage(FEASIBLE)
    strategy = fixed_strategy()

    discharge = strategy.calculate_bids(unit, market_config, products(INDEX[0]))
    assert len(discharge) == 1
    assert discharge[0]["volume"] == pytest.approx(80.0)
    assert discharge[0]["price"] == -10.0

    discharge[0]["accepted_volume"] = 80.0
    discharge[0]["accepted_price"] = 30.0
    unit.set_dispatch_plan(market_config, discharge)
    unit.execute_current_dispatch(INDEX[0], INDEX[0])
    strategy.calculate_reward(unit, market_config, discharge)

    charge = strategy.calculate_bids(unit, market_config, products(INDEX[1]))
    assert charge[0]["volume"] == pytest.approx(-100.0)
    assert charge[0]["price"] == 10_000.0

    assert strategy.calculate_bids(unit, market_config, products(INDEX[2])) == []


def test_fixed_schedule_bids_multiple_products(market_config):
    unit = make_storage(FEASIBLE)
    bids = fixed_strategy().calculate_bids(
        unit,
        market_config,
        [(t, t + pd.Timedelta(hours=1), None) for t in INDEX],
    )
    assert [bid["volume"] for bid in bids] == pytest.approx([80.0, -100.0, 40.0])


def test_fixed_schedule_requires_column(market_config):
    unit = make_storage()
    with pytest.raises(ValueError, match="storage_schedule_battery"):
        fixed_strategy().calculate_bids(unit, market_config, products(INDEX[0]))


def test_fixed_schedule_foreign_unit_without_column_is_idle(market_config):
    unit = make_storage(availability=0.0)
    unit.forecaster.is_foreign = True
    assert (
        fixed_strategy().calculate_bids(unit, market_config, products(INDEX[0])) == []
    )


def test_fixed_schedule_rejects_infeasible_schedule_before_bidding(market_config):
    unit = make_storage([80.0, 80.0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="SoC after this step"):
        fixed_strategy().calculate_bids(unit, market_config, products(INDEX[0]))


def test_fixed_schedule_rejects_infeasible_step_at_actual_soc(market_config):
    unit = make_storage(FEASIBLE)
    unit.outputs["soc"].at[INDEX[0]] = 0.1  # actual SoC deviates from the replay
    with pytest.raises(ValueError, match="only"):
        fixed_strategy().calculate_bids(unit, market_config, products(INDEX[0]))


def test_fixed_schedule_raises_when_volume_does_not_clear(market_config):
    unit = make_storage(FEASIBLE)
    strategy = fixed_strategy()
    bids = strategy.calculate_bids(unit, market_config, products(INDEX[0]))
    bids[0]["accepted_volume"] = 50.0
    bids[0]["accepted_price"] = 30.0
    with pytest.raises(ValueError, match="did not clear"):
        strategy.calculate_reward(unit, market_config, bids)
