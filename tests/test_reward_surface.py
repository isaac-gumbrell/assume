# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Analytic assertions about the reward surface itself.

These evaluate ``calculate_reward`` directly on constructed orderbooks. No market
is cleared and no policy is trained, so nothing here depends on competitive
dynamics between agents - every assertion is a property of the reward function
alone. That is what makes them able to distinguish a reward defect from healthy
competitive convergence, which the training metrics cannot.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from assume.common.base import LearningConfig
from assume.common.forecaster import PowerplantForecaster, UnitForecaster

try:
    import torch as th

    from assume.reinforcement_learning import Learning
    from assume.strategies.learning_strategies import (
        EnergyLearningSingleBidStrategy,
        StorageEnergyLearningStrategy,
    )
except ImportError:
    th = None

from assume.common.market_objects import MarketConfig
from assume.units import PowerPlant, Storage

MARKET_ID = "test_market"
FORECAST_PRICE = 50.0
MARGINAL_COST = 20.0
MAX_POWER = 1000.0
MAX_BID_PRICE = 100.0
START = pd.Timestamp("2023-07-01 12:00")


@pytest.fixture
def market_config() -> MarketConfig:
    mc = MarketConfig()
    mc.market_id = MARKET_ID
    mc.product_type = "energy"
    return mc


def _make_generator(**strategy_kwargs) -> PowerPlant:
    """A zero-min-power generator, matching the production hydro units."""
    index = pd.date_range("2023-06-30 22:00:00", periods=48, freq="h")
    forecaster = PowerplantForecaster(
        index,
        fuel_prices={"lignite": 10},
        market_prices={MARKET_ID: FORECAST_PRICE},
        residual_load={MARKET_ID: 0},
    )
    config = {
        "unit_id": "test_pp",
        "learning_config": LearningConfig(
            algorithm="matd3",
            learning_mode=True,
            training_episodes=3,
            max_bid_price=MAX_BID_PRICE,
        ),
    }
    learning_role = Learning(config["learning_config"], index[0], index[-1])
    unit = PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="hydro",
        index=index,
        max_power=MAX_POWER,
        min_power=0.0,
        efficiency=0.5,
        fuel_type="lignite",
        bidding_strategies={
            MARKET_ID: EnergyLearningSingleBidStrategy(
                learning_role=learning_role, **(config | strategy_kwargs)
            )
        },
        forecaster=forecaster,
    )
    # Start-up costs would add a step change unrelated to the bidding question.
    unit.hot_start_cost = 0.0
    return unit


def _dispatch(unit, market_config, accepted_fraction: float, clearing_price: float):
    """Evaluate the reward for a single order dispatched at ``accepted_fraction``."""
    end = START + unit.index.freq
    accepted_volume = accepted_fraction * MAX_POWER
    rejected = accepted_volume == 0

    order = {
        "start_time": START,
        "end_time": end,
        "only_hours": None,
        "price": clearing_price,
        "volume": MAX_POWER,
        "node": unit.node,
        # The clearing algorithms zero both fields on a rejected order.
        "accepted_volume": accepted_volume,
        "accepted_price": 0.0 if rejected else clearing_price,
    }

    unit.outputs["energy"].loc[:] = 0.0
    unit.outputs["energy"].at[START] = accepted_volume

    strategy = unit.bidding_strategies[market_config.market_id]
    with patch.object(PowerPlant, "calculate_marginal_cost", return_value=MARGINAL_COST):
        strategy.calculate_reward(unit, market_config, orderbook=[order])

    role = strategy.learning_role
    return {
        "reward": role.all_rewards[START][unit.id][-1],
        "regret": role.all_regrets[START][unit.id][-1],
        "profit": role.all_profits[START][unit.id][-1],
    }


@pytest.mark.require_learning
def test_rejected_order_still_penalises_withholding(market_config):
    """Regression guard for the rejected-order clearing price defect.

    A rejected order carries ``accepted_price = 0``. Reading the clearing price
    off that order made the opportunity cost negative, which ``max(., 0)`` then
    clipped away, so complete withholding scored exactly zero - no penalty at
    all, at the one moment the regret term exists to penalise.
    """
    unit = _make_generator()
    withheld = _dispatch(unit, market_config, accepted_fraction=0.0, clearing_price=0.0)

    assert withheld["profit"] == 0.0
    assert withheld["regret"] > 0.0, (
        "a fully rejected order must still incur an opportunity cost when the "
        "market price exceeds marginal cost"
    )
    assert withheld["reward"] < 0.0


@pytest.mark.require_learning
def test_reward_is_monotone_in_dispatched_fraction(market_config):
    """No discontinuity at first dispatch, and more dispatch is never worse.

    The pre-fix reward dropped discontinuously at the first accepted MW, making
    idling strictly better than being slightly in the merit order.
    """
    unit = _make_generator()
    fractions = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
    rewards = [
        _dispatch(unit, market_config, f, FORECAST_PRICE)["reward"] for f in fractions
    ]

    assert rewards == sorted(rewards), f"reward not monotone in dispatch: {rewards}"
    assert rewards[0] < rewards[1], "idling must not beat marginal dispatch"


@pytest.mark.require_learning
def test_full_dispatch_maximises_reward(market_config):
    """The reward's argmax sits at full dispatch whenever the margin is positive.

    This is the closed-form statement behind M1: with the clearing price exogenous
    under pay-as-clear, the reward pushes the agent toward being dispatched, and
    therefore toward bidding down. Whether that is correct market behaviour or an
    artefact of the regret weight is what the best-response probe decides.
    """
    unit = _make_generator()
    rewards = {
        f: _dispatch(unit, market_config, f, FORECAST_PRICE)["reward"]
        for f in (0.0, 0.5, 1.0)
    }
    assert max(rewards, key=rewards.get) == 1.0


@pytest.mark.require_learning
def test_regret_weight_zero_leaves_profit_only(market_config):
    """The ablation knob must reproduce a pure profit reward exactly."""
    unit = _make_generator(regret_weight=0.0)
    scaling = 1 / (MAX_BID_PRICE * MAX_POWER)

    for fraction in (0.0, 0.5, 1.0):
        result = _dispatch(unit, market_config, fraction, FORECAST_PRICE)
        expected_profit = (FORECAST_PRICE - MARGINAL_COST) * fraction * MAX_POWER
        assert result["regret"] == 0.0
        assert result["profit"] == pytest.approx(expected_profit)
        assert result["reward"] == pytest.approx(scaling * expected_profit)


@pytest.mark.require_learning
def test_regret_scales_are_configurable(market_config):
    """Idle and dispatched regret scales must both be reachable from config."""
    unit = _make_generator(regret_scale_idle=0.25, regret_scale_dispatched=0.05)
    margin = FORECAST_PRICE - MARGINAL_COST

    idle = _dispatch(unit, market_config, 0.0, FORECAST_PRICE)
    assert idle["regret"] == pytest.approx(0.25 * margin * MAX_POWER)

    half = _dispatch(unit, market_config, 0.5, FORECAST_PRICE)
    assert half["regret"] == pytest.approx(0.05 * margin * 0.5 * MAX_POWER)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def _make_storage(**strategy_kwargs) -> Storage:
    index = pd.date_range("2023-06-30 22:00:00", periods=48, freq="h")
    config = {
        "unit_id": "test_storage",
        "learning_config": LearningConfig(
            algorithm="matd3",
            learning_mode=True,
            training_episodes=3,
            max_bid_price=MAX_BID_PRICE,
        ),
        "storage_bid_price_limit": MAX_BID_PRICE,
    }
    learning_role = Learning(config["learning_config"], index[0], index[-1])
    return Storage(
        id="test_storage",
        unit_operator="test_operator",
        technology="storage",
        bidding_strategies={
            MARKET_ID: StorageEnergyLearningStrategy(
                learning_role=learning_role, **(config | strategy_kwargs)
            )
        },
        max_power_charge=-500,
        max_power_discharge=500,
        capacity=1000,
        min_soc=0,
        max_soc=1,
        initial_soc=0.5,
        efficiency_charge=1.0,
        efficiency_discharge=1.0,
        forecaster=UnitForecaster(index, market_prices={MARKET_ID: FORECAST_PRICE}),
    )


def _storage_reward(unit, market_config, accepted_volume, accepted_price, next_soc):
    end = START + unit.index.freq
    order = {
        "start_time": START,
        "end_time": end,
        "only_hours": None,
        "price": accepted_price,
        "volume": accepted_volume,
        "node": unit.node,
        "accepted_volume": accepted_volume,
        "accepted_price": accepted_price if accepted_volume else 0.0,
    }
    unit.outputs["energy"].loc[:] = 0.0
    unit.outputs["energy"].at[START] = accepted_volume
    unit.outputs["soc"].at[START] = 0.5
    unit.outputs["soc"].at[end] = next_soc
    unit.outputs["cost_stored_energy"].at[START] = 0.0

    strategy = unit.bidding_strategies[market_config.market_id]
    with patch.object(Storage, "calculate_marginal_cost", return_value=0.0):
        strategy.calculate_reward(unit, market_config, orderbook=[order])

    role = strategy.learning_role
    return {
        "reward": role.all_rewards[START][unit.id][-1],
        "profit": role.all_profits[START][unit.id][-1],
    }


@pytest.mark.require_learning
def test_charging_is_not_dominated_by_idling(market_config):
    """Charging must not be strictly worse than doing nothing.

    Charging is a cash outflow, so a cash-only reward ranked it below idling in
    every state. No policy can learn to arbitrage a battery it is punished for
    filling. Crediting the value of the stored energy makes charging neutral;
    the gain then comes from discharging later, which is what the critic is for.
    """
    unit = _make_storage()
    idle = _storage_reward(unit, market_config, 0, 0.0, next_soc=0.5)
    charge = _storage_reward(unit, market_config, -400, 20.0, next_soc=0.9)

    assert idle["reward"] == pytest.approx(0.0)
    assert charge["profit"] < 0.0, "charging is still a cash outflow"
    assert charge["reward"] >= idle["reward"] - 1e-9, (
        f"charging ({charge['reward']}) must not be dominated by idling "
        f"({idle['reward']})"
    )


@pytest.mark.require_learning
def test_discharging_earns_the_margin(market_config):
    unit = _make_storage()
    unit.outputs["cost_stored_energy"].at[START] = 0.0
    discharge = _storage_reward(unit, market_config, 400, 60.0, next_soc=0.1)

    assert discharge["profit"] > 0.0
    assert discharge["reward"] > 0.0


@pytest.mark.require_learning
def test_soc_value_weight_zero_restores_cash_only_reward(market_config):
    """The ablation knob must reproduce the previous, cash-only behaviour."""
    unit = _make_storage(soc_value_weight=0.0)
    scaling = 1 / (MAX_BID_PRICE * unit.max_power_discharge)

    charge = _storage_reward(unit, market_config, -400, 20.0, next_soc=0.9)
    assert charge["reward"] == pytest.approx(scaling * charge["profit"])
    assert charge["reward"] < 0.0


@pytest.mark.require_learning
def test_soc_value_weight_rejects_invalid_values():
    with pytest.raises(ValueError, match="soc_value_weight"):
        _make_storage(soc_value_weight=-1.0)
