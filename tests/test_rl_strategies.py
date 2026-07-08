# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from assume.common.base import LearningConfig
from assume.common.forecaster import PowerplantForecaster

try:
    from assume.reinforcement_learning import Learning
    from assume.strategies.learning_strategies import (
        EnergyLearningSingleBidStrategy,
        EnergyLearningSingleBidStrategyCongestion,
        EnergyLearningStrategy,
        EnergyLearningStrategyCongestion,
        RenewableEnergyLearningCompatibleStrategy,
        RenewableEnergyLearningSingleBidStrategyCongestion,
        SRMCEnergyLearningStrategy,
        SRMCEnergyLearningStrategyCongestion,
        StorageEnergyLearningStrategyCongestion,
    )

except ImportError:
    EnergyLearningStrategy = None
    EnergyLearningSingleBidStrategy = None
    EnergyLearningStrategyCongestion = None
    EnergyLearningSingleBidStrategyCongestion = None
    StorageEnergyLearningStrategyCongestion = None
    RenewableEnergyLearningCompatibleStrategy = None
    RenewableEnergyLearningSingleBidStrategyCongestion = None
    SRMCEnergyLearningStrategy = None
    SRMCEnergyLearningStrategyCongestion = None

from assume.units import PowerPlant

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)


@pytest.fixture
def power_plant() -> PowerPlant:
    # Create a PowerPlant instance with some example parameters
    index = pd.date_range("2023-06-30 22:00:00", periods=48, freq="h")
    ff = PowerplantForecaster(
        index,
        fuel_prices={"lignite": 10, "co2": 10},
        residual_load={"EOM": 0},
    )
    config = {
        "unit_id": "test_pp",
        "learning_config": LearningConfig(
            algorithm="matd3",
            learning_mode=True,
            training_episodes=3,
        ),
    }
    learning_role = Learning(config["learning_config"], start, end)

    return PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="coal",
        index=ff.index,
        max_power=1000,
        min_power=200,
        efficiency=0.5,
        additional_cost=10,
        bidding_strategies={
            "EOM": EnergyLearningStrategy(learning_role=learning_role, **config)
        },
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=ff,
    )


@pytest.mark.require_learning
@pytest.mark.parametrize(
    "strategy_class, obs_dim, act_dim, unique_obs_dim, actor_architecture, expected_bid_count, expected_volumes",
    [
        (EnergyLearningStrategy, 38, 2, 2, "mlp", 2, [200, 800]),
        (EnergyLearningStrategy, 38, 2, 2, "lstm", 2, [200, 800]),
        (EnergyLearningSingleBidStrategy, 74, 1, 2, "mlp", 1, [1000]),
    ],
)
def test_learning_strategies_parametrized(
    mock_market_config,
    power_plant,
    strategy_class,
    obs_dim,
    act_dim,
    unique_obs_dim,
    actor_architecture,
    expected_bid_count,
    expected_volumes,
):
    product_index = pd.date_range("2023-07-01", periods=1, freq="h")
    mc = mock_market_config
    mc.product_type = "energy_eom"
    product_tuples = [
        (start, start + pd.Timedelta(hours=1), None) for start in product_index
    ]
    # Build LearningConfig dynamically
    config = {
        "unit_id": power_plant.id,
        "learning_config": LearningConfig(
            algorithm="matd3",
            actor_architecture=actor_architecture,
            learning_mode=True,
            training_episodes=3,
        ),
    }

    learning_role = Learning(config["learning_config"], start, end)
    # Override the strategy
    power_plant.bidding_strategies[mc.market_id] = strategy_class(
        learning_role=learning_role, **config
    )
    strategy = power_plant.bidding_strategies[mc.market_id]

    # Check if observation dimension is set accordingly and follows current default structure
    first_observation = strategy.create_observation(
        power_plant,
        mc.market_id,
        product_index[0],
        product_index[0] + pd.Timedelta(hours=1),
    )
    assert len(first_observation) == obs_dim
    assert (
        strategy.unique_obs_dim + strategy.foresight * strategy.num_timeseries_obs_dim
        == obs_dim
    )

    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)

    assert len(bids) == expected_bid_count
    for bid, expected_volume in zip(bids, expected_volumes):
        assert bid["volume"] == expected_volume

    for order in bids:
        order["accepted_price"] = 50
        order["accepted_volume"] = order["volume"]

    strategy.calculate_reward(power_plant, mc, orderbook=bids)

    # Fetch reward, profit, regret from learning_role cache instead of outputs
    # Get the latest timestamp used for reward cache
    learning_role = strategy.learning_role
    reward_cache = learning_role.all_rewards
    profit_cache = learning_role.all_profits
    regret_cache = learning_role.all_regrets

    # Use the last timestamp (should be the one just written)
    last_ts = sorted(reward_cache.keys())[-1]
    unit_id = (
        power_plant.id
        if power_plant.id in reward_cache[last_ts]
        else list(reward_cache[last_ts].keys())[0]
    )

    reward = reward_cache[last_ts][unit_id][0]
    profit = profit_cache[last_ts][unit_id][0]
    regret = regret_cache[last_ts][unit_id][0]
    costs = power_plant.outputs["total_costs"].loc[product_index]

    assert reward == 0.1
    assert profit == 10000.0
    assert regret == 0.0
    assert costs[0] == 40000.0  # Assumes hot_start_cost = 20000 by default


@pytest.mark.require_learning
def test_initial_experience_noise_not_contaminated_by_marginal_cost():
    """Regression: the exploration-noise diagnostic must stay pure.

    During initial-experience collection ``EnergyLearningStrategy.get_actions``
    biases the action towards the marginal cost (``curr_action += marginal_cost``).
    The base class must return the action as a *distinct* tensor from ``noise`` so
    this in-place bias does not leak into the recorded noise. Previously the two
    aliased the same tensor, so the logged noise became ``noise + marginal_cost``
    and looked inflated for scenarios with higher marginal costs (e.g. the
    secondary world in staggered training).
    """
    import torch as th

    config = {
        "unit_id": "test_pp",
        "learning_config": LearningConfig(
            algorithm="matd3",
            learning_mode=True,
            training_episodes=3,
        ),
    }
    learning_role = Learning(config["learning_config"], start, end)
    strategy = EnergyLearningStrategy(learning_role=learning_role, **config)
    strategy.collect_initial_experience_mode = True

    marginal_cost = 0.3
    observation = th.zeros(strategy.obs_dim, dtype=strategy.float_type)
    observation[-1] = marginal_cost

    action, noise = strategy.get_actions(observation)

    # The action is the noise biased by the (scalar) marginal cost; the returned
    # noise must remain the pure exploration component, i.e. they are not aliased.
    assert action.data_ptr() != noise.data_ptr()
    assert th.allclose(
        action - noise,
        th.full_like(noise, marginal_cost),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# SRMC strategy tests
# ---------------------------------------------------------------------------


def _make_srmc_learning_role():
    config = {
        "unit_id": "test_pp",
        "learning_config": LearningConfig(
            algorithm="matd3",
            learning_mode=True,
            training_episodes=3,
        ),
    }
    return Learning(config["learning_config"], start, end), config


def _build_srmc_strategy(strategy_class, learning_role, config):
    """Instantiate an SRMC strategy, supplying congestion kwargs when required."""
    if issubclass(strategy_class, SRMCEnergyLearningStrategyCongestion):
        return strategy_class(
            learning_role=learning_role,
            n_lines=3,
            congestion_foresight=1,
            **config,
        )
    return strategy_class(learning_role=learning_role, **config)


@pytest.mark.require_learning
@pytest.mark.parametrize(
    "strategy_class",
    [SRMCEnergyLearningStrategy, SRMCEnergyLearningStrategyCongestion],
)
@pytest.mark.parametrize(
    "action_value, expected_fraction",
    [(-1.0, 0.0), (0.0, 0.5), (1.0, 1.0)],
)
def test_srmc_bid_price_remap(
    mock_market_config,
    power_plant,
    strategy_class,
    action_value,
    expected_fraction,
):
    """The action must be remapped linearly onto ``[SRMC, max_bid_price]``.

    ``action = -1`` bids exactly at SRMC, ``action = +1`` at the price cap and
    ``action = 0`` at the midpoint. The full available capacity is offered as a
    single bid.
    """
    import torch as th

    product_index = pd.date_range("2023-07-01", periods=1, freq="h")
    mc = mock_market_config
    mc.product_type = "energy_eom"
    product_tuples = [(s, s + pd.Timedelta(hours=1), None) for s in product_index]

    learning_role, config = _make_srmc_learning_role()
    strategy = _build_srmc_strategy(strategy_class, learning_role, config)

    srmc = 25.0
    expected_price = srmc + expected_fraction * (strategy.max_bid_price - srmc)

    with (
        patch.object(
            strategy_class,
            "get_actions",
            return_value=(th.tensor([action_value]), th.tensor(0.0)),
        ),
        patch.object(PowerPlant, "calculate_marginal_cost", return_value=srmc),
    ):
        bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)

    # single bid covering the full available capacity
    assert len(bids) == 1
    assert bids[0]["volume"] == power_plant.max_power
    assert bids[0]["price"] == pytest.approx(expected_price)


@pytest.mark.require_learning
def test_srmc_bid_never_below_marginal_cost(mock_market_config, power_plant):
    """Even the lowest action (-1) must never bid below the unit's SRMC."""
    import torch as th

    product_index = pd.date_range("2023-07-01", periods=1, freq="h")
    mc = mock_market_config
    mc.product_type = "energy_eom"
    product_tuples = [(s, s + pd.Timedelta(hours=1), None) for s in product_index]

    learning_role, config = _make_srmc_learning_role()
    strategy = SRMCEnergyLearningStrategy(learning_role=learning_role, **config)

    srmc = 42.0
    with (
        patch.object(
            SRMCEnergyLearningStrategy,
            "get_actions",
            return_value=(th.tensor([-1.0]), th.tensor(0.0)),
        ),
        patch.object(PowerPlant, "calculate_marginal_cost", return_value=srmc),
    ):
        bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)

    assert bids[0]["price"] == pytest.approx(srmc)


@pytest.mark.require_learning
def test_srmc_get_actions_has_no_marginal_cost_bias():
    """SRMC exploration must not add the marginal-cost bias used by the parent.

    ``EnergyLearningStrategy`` shifts the initial-experience action by the
    marginal cost (``curr_action += marginal_cost``). Because the SRMC strategy
    bakes SRMC into the ``[SRMC, max_bid_price]`` bid remap, that bias would be
    double-counted, so ``SRMCEnergyLearningStrategy.get_actions`` must delegate
    straight to :class:`TorchLearningStrategy` and leave the action unbiased.
    """
    import torch as th

    learning_role, config = _make_srmc_learning_role()
    strategy = SRMCEnergyLearningStrategy(learning_role=learning_role, **config)
    strategy.collect_initial_experience_mode = True

    marginal_cost = 0.3
    observation = th.zeros(strategy.obs_dim, dtype=strategy.float_type)
    observation[-1] = marginal_cost

    action, noise = strategy.get_actions(observation)

    # No marginal-cost bias: the action equals the pure exploration noise.
    assert th.allclose(action, noise, atol=1e-6)


# ---------------------------------------------------------------------------
# Congestion strategy tests
# ---------------------------------------------------------------------------


def _make_learning_role(start, end):
    config = LearningConfig(
        algorithm="matd3",
        learning_mode=True,
        training_episodes=3,
    )
    return Learning(config, start, end)


def _make_congestion_strategy(
    strategy_class, n_lines, congestion_foresight, learning_role
):
    return strategy_class(
        unit_id="test_pp",
        learning_config=LearningConfig(
            algorithm="matd3",
            learning_mode=True,
            training_episodes=3,
        ),
        learning_role=learning_role,
        n_lines=n_lines,
        congestion_foresight=congestion_foresight,
    )


@pytest.mark.require_learning
@pytest.mark.parametrize(
    "strategy_class",
    [
        EnergyLearningStrategyCongestion,
        EnergyLearningSingleBidStrategyCongestion,
        SRMCEnergyLearningStrategyCongestion,
        StorageEnergyLearningStrategyCongestion,
        RenewableEnergyLearningSingleBidStrategyCongestion,
    ],
)
def test_congestion_obs_dim(strategy_class):
    """obs_dim formula: num_timeseries_obs_dim * foresight + n_lines * congestion_foresight + unique_obs_dim."""
    n_lines = 3
    congestion_foresight = 1
    lr = _make_learning_role(start, end)
    strategy = _make_congestion_strategy(
        strategy_class, n_lines, congestion_foresight, lr
    )
    expected = (
        strategy.num_timeseries_obs_dim * strategy.foresight
        + n_lines * congestion_foresight
        + strategy.unique_obs_dim
    )
    assert strategy.obs_dim == expected


@pytest.mark.require_learning
def test_congestion_obs_dim_regression():
    """EnergyLearningStrategy (base) obs_dim must not change."""
    lr = _make_learning_role(start, end)
    base_strategy = EnergyLearningStrategy(
        unit_id="test_pp",
        learning_config=LearningConfig(
            algorithm="matd3",
            learning_mode=True,
            training_episodes=3,
        ),
        learning_role=lr,
    )
    # obs_dim unchanged: 3 * foresight (12) + 2 = 38
    assert base_strategy.obs_dim == (
        base_strategy.num_timeseries_obs_dim * base_strategy.foresight
        + base_strategy.unique_obs_dim
    )


@pytest.mark.require_learning
def test_congestion_zero_grid_fallback(mock_market_config):
    """When congestion_signal_lines is empty, congestion channels are zero-filled."""
    index = pd.date_range("2023-06-30 22:00:00", periods=48, freq="h")
    ff = PowerplantForecaster(
        index,
        fuel_prices={"lignite": 10, "co2": 10},
        residual_load={"EOM": 0},
    )
    # congestion_signal_lines starts as {} by default
    assert ff.congestion_signal_lines == {}

    n_lines = 3
    congestion_foresight = 1
    lr = _make_learning_role(start, end)
    strategy = _make_congestion_strategy(
        EnergyLearningStrategyCongestion, n_lines, congestion_foresight, lr
    )
    pp = PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="coal",
        index=ff.index,
        max_power=1000,
        min_power=200,
        efficiency=0.5,
        additional_cost=10,
        bidding_strategies={"EOM": strategy},
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=ff,
    )

    product_index = pd.date_range("2023-07-01", periods=1, freq="h")
    obs = strategy.create_observation(
        pp,
        mock_market_config.market_id,
        product_index[0],
        product_index[0] + pd.Timedelta(hours=1),
    )
    # obs_dim must equal strategy.obs_dim even with empty grid
    assert len(obs) == strategy.obs_dim

    # The congestion portion of obs (middle n_lines * congestion_foresight values) must be zero
    ts_len = strategy.num_timeseries_obs_dim * strategy.foresight
    cong_len = n_lines * congestion_foresight
    obs_np = obs.cpu().numpy()
    assert (obs_np[ts_len : ts_len + cong_len] == 0).all()


@pytest.mark.require_learning
def test_congestion_line_order_stability():
    """Two agents initialised on the same grid must have identical line channel ordering."""
    from assume.common.fast_pandas import FastSeries

    index = pd.date_range("2023-06-30 22:00:00", periods=48, freq="h")
    ff = PowerplantForecaster(
        index,
        fuel_prices={"lignite": 10, "co2": 10},
        residual_load={"EOM": 0},
    )
    # Inject mock line signals (unsorted insertion order)
    for line_id in ["line_C", "line_A", "line_B"]:
        ff.congestion_signal_lines[line_id] = FastSeries(value=0.0, index=ff.index)

    lr1 = _make_learning_role(start, end)
    lr2 = _make_learning_role(start, end)
    s1 = _make_congestion_strategy(EnergyLearningStrategyCongestion, 3, 1, lr1)
    s2 = _make_congestion_strategy(EnergyLearningStrategyCongestion, 3, 1, lr2)

    pp_kwargs = dict(
        unit_operator="op",
        technology="coal",
        index=ff.index,
        max_power=1000,
        min_power=200,
        efficiency=0.5,
        additional_cost=10,
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=ff,
    )
    pp1 = PowerPlant(id="pp1", bidding_strategies={"EOM": s1}, **pp_kwargs)
    pp2 = PowerPlant(id="pp2", bidding_strategies={"EOM": s2}, **pp_kwargs)

    product_index = pd.date_range("2023-07-01", periods=1, freq="h")
    t_start = product_index[0]
    t_end = t_start + pd.Timedelta(hours=1)

    # Trigger prepare_observations
    s1.prepare_observations(pp1, "EOM")
    s2.prepare_observations(pp2, "EOM")

    assert list(s1.congestion_line_obs.keys()) == list(s2.congestion_line_obs.keys())
    assert list(s1.congestion_line_obs.keys()) == sorted(
        ff.congestion_signal_lines.keys()
    )


# ---------------------------------------------------------------------------
# Activity-mask conflation regression (commit 88ac62d, Tier 2)
# ---------------------------------------------------------------------------


@pytest.mark.require_learning
def test_activity_mask_keys_off_foreign_flag_not_availability(
    mock_market_config,
):
    """The training mask must key off "is foreign", not zero availability.

    The activity mask added in commit 88ac62d originally derived ``active`` from
    ``unit.forecaster.availability.at[start] > 0``. That predicate was meant to
    exclude *foreign* units (the paired scenario's superset, forced to
    availability=0 for the whole horizon) from the shared MATD3 gradient, but it
    also masked a *native* renewable's legitimate zero-availability hours (solar at
    night, wind in a lull) - real, informative states the policy must learn from.

    The fix keys the mask off ``unit.forecaster.is_foreign`` instead. This test
    asserts the corrected behaviour:

    - a native unit (``is_foreign=False``) stays active (1.0) in *both* its
      generating hour and its own zero-availability hour, and
    - a foreign unit (``is_foreign=True``) is masked out (0.0).
    """
    unit_id = "test_renewable"

    index = pd.date_range("2023-06-30 22:00:00", periods=48, freq="h")
    ff = PowerplantForecaster(
        index,
        fuel_prices={"lignite": 10, "co2": 10},
        residual_load={"EOM": 0},
    )

    # A native renewable: available mid-day, genuinely off overnight.
    active_start = pd.Timestamp("2023-07-01 12:00:00")
    off_start = pd.Timestamp("2023-07-01 01:00:00")
    ff.availability.at[off_start] = 0.0
    assert ff.availability.at[active_start] == 1.0
    assert ff.is_foreign is False

    config = LearningConfig(
        algorithm="matd3",
        learning_mode=True,
        training_episodes=3,
        max_bid_price=100,
    )
    lr = Learning(config, start, end)
    strategy = RenewableEnergyLearningCompatibleStrategy(
        unit_id=unit_id,
        learning_config=config,
        learning_role=lr,
    )

    power_plant = PowerPlant(
        id=unit_id,
        unit_operator="test_operator",
        technology="solar",
        index=ff.index,
        max_power=100,
        min_power=0,
        efficiency=1.0,
        additional_cost=0,
        bidding_strategies={"EOM": strategy},
        fuel_type="lignite",
        emission_factor=0.0,
        forecaster=ff,
    )

    def _orderbook(t_start, volume):
        return [
            {
                "start_time": t_start,
                "end_time": t_start + pd.Timedelta(hours=1),
                "only_hours": None,
                "volume": volume,
                "accepted_volume": volume,
                "accepted_price": 50.0,
                "node": "node0",
            }
        ]

    mc = mock_market_config
    mc.market_id = "EOM"
    mc.product_type = "energy"

    with patch.object(PowerPlant, "calculate_marginal_cost", return_value=0.0):
        # Generating hour (availability=1): offered full capacity, dispatched.
        strategy.calculate_reward(
            power_plant, mc, orderbook=_orderbook(active_start, 100.0)
        )
        # Night hour (availability=0): nothing offered, nothing dispatched.
        strategy.calculate_reward(power_plant, mc, orderbook=_orderbook(off_start, 0.0))

    # A native (non-foreign) unit is trainable in both hours - its own
    # zero-availability hour is a genuine, learnable state.
    assert lr.all_active[active_start][unit_id][0] == 1.0
    assert lr.all_active[off_start][unit_id][0] == 1.0

    # A foreign unit (forced off for the whole horizon) is masked out, regardless
    # of which hour we evaluate. (The cache appends, so check the latest entry.)
    ff.is_foreign = True
    with patch.object(PowerPlant, "calculate_marginal_cost", return_value=0.0):
        strategy.calculate_reward(
            power_plant, mc, orderbook=_orderbook(active_start, 100.0)
        )
    assert lr.all_active[active_start][unit_id][-1] == 0.0
