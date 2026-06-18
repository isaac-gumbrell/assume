# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

try:
    from assume.common.base import LearningConfig
    from assume.reinforcement_learning.learning_role import (
        Learning,
        LearningStrategy,
    )
    from assume.reinforcement_learning.neural_network_architecture import (
        Actor,
        CriticTD3,
    )
except ImportError:
    pass

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)


@pytest.mark.require_learning
def test_learning_init():
    config = {
        "foresight": 1,
        "act_dim": 2,
        "unique_obs_dim": 0,
        "learning_config": LearningConfig(
            train_freq="1h",
            algorithm="matd3",
            actor_architecture="mlp",
            learning_mode=True,
            evaluation_mode=False,
            training_episodes=3,
            episodes_collecting_initial_experience=1,
            continue_learning=False,
            trained_policies_save_path=None,
            early_stopping_steps=10,
            early_stopping_threshold=0.05,
        ),
    }

    # test init
    learn = Learning(config["learning_config"], start=start, end=end)
    assert len(learn.rl_strats) == 0

    # we need to add learning strategies first
    learn.rl_strats["test_id"] = LearningStrategy(**config, learning_role=learn)

    # test creating actors
    learn.initialize_policy()
    # now we have an actor for every strategy
    for strategy in learn.rl_strats.values():
        assert isinstance(strategy.actor, Actor)
        assert isinstance(strategy.actor_target, Actor)

    # now we have a critic for every strategy
    for strategy in learn.rl_strats.values():
        assert isinstance(strategy.critics, CriticTD3)
        assert isinstance(strategy.target_critics, CriticTD3)

    ac = learn.rl_algorithm.extract_policy()

    assert ac["target_critics"]["test_id"] == learn.rl_strats["test_id"].target_critics
    assert ac["critics"]["test_id"] == learn.rl_strats["test_id"].critics


@pytest.mark.require_learning
def test_learning_runtime_state_roundtrip(tmp_path):
    class DummyNoise:
        def __init__(self, dt):
            self.dt = dt

    config = {
        "foresight": 1,
        "act_dim": 2,
        "unique_obs_dim": 0,
        "learning_config": LearningConfig(
            train_freq="1h",
            algorithm="matd3",
            actor_architecture="mlp",
            learning_mode=True,
            evaluation_mode=False,
            training_episodes=10,
            episodes_collecting_initial_experience=1,
            continue_learning=False,
            trained_policies_save_path=None,
            early_stopping_steps=10,
            early_stopping_threshold=0.05,
        ),
    }

    learn = Learning(config["learning_config"], start=start, end=end)
    learn.rl_strats["test_id"] = LearningStrategy(**config, learning_role=learn)
    learn.initialize_policy()

    learn.episodes_done = 7
    learn.eval_episodes_done = 2
    learn.max_eval["avg_reward"] = 1.23
    learn.rl_eval["avg_reward"] = [0.5, 1.23]
    learn.avg_rewards = [0.7, 0.8]
    learn.rl_algorithm.n_updates = 42
    learn.rl_algorithm.current_learning_rate = 0.000123
    learn.rl_algorithm.update_learning_rate(
        [
            learn.rl_strats["test_id"].actor.optimizer,
            learn.rl_strats["test_id"].critics.optimizer,
        ],
        learning_rate=learn.rl_algorithm.current_learning_rate,
    )
    learn.rl_strats["test_id"].collect_initial_experience_mode = False
    learn.rl_strats["test_id"].action_noise = DummyNoise(dt=0.321)

    state_path = tmp_path / "learning_state.pt"
    learn.save_runtime_state(str(state_path))

    resumed = Learning(config["learning_config"], start=start, end=end)
    resumed.rl_strats["test_id"] = LearningStrategy(**config, learning_role=resumed)
    resumed.rl_strats["test_id"].collect_initial_experience_mode = True
    resumed.rl_strats["test_id"].action_noise = DummyNoise(dt=0.999)
    resumed.initialize_policy()
    resumed.rl_algorithm.current_learning_rate = 0.009
    resumed.rl_algorithm.update_learning_rate(
        [
            resumed.rl_strats["test_id"].actor.optimizer,
            resumed.rl_strats["test_id"].critics.optimizer,
        ],
        learning_rate=resumed.rl_algorithm.current_learning_rate,
    )
    resumed.load_runtime_state(str(state_path))

    assert resumed.episodes_done == 7
    assert resumed.eval_episodes_done == 2
    assert resumed.max_eval["avg_reward"] == pytest.approx(1.23)
    assert resumed.rl_eval["avg_reward"] == [0.5, 1.23]
    assert resumed.avg_rewards == [0.7, 0.8]
    assert resumed.rl_algorithm.n_updates == 42
    assert resumed.rl_algorithm.current_learning_rate == pytest.approx(0.000123)
    assert resumed.rl_strats["test_id"].actor.optimizer.param_groups[0][
        "lr"
    ] == pytest.approx(0.000123)
    assert resumed.rl_strats["test_id"].critics.optimizer.param_groups[0][
        "lr"
    ] == pytest.approx(0.000123)
    assert resumed.rl_strats["test_id"].collect_initial_experience_mode is False
    assert resumed.rl_strats["test_id"].action_noise.dt == pytest.approx(0.321)


@pytest.mark.require_learning
def test_inter_episodic_data_preserves_n_updates():
    """The global gradient-update counter must survive the inter-episodic
    round-trip. Between episodes ``setup_world`` recreates the rl_algorithm
    (resetting ``n_updates`` to 0); without carrying it through
    ``get_inter_episodic_data``/``load_inter_episodic_data`` the TensorBoard
    gradient-step x-axis would restart at 0 every episode.
    """
    config = {
        "foresight": 1,
        "act_dim": 2,
        "unique_obs_dim": 0,
        "learning_config": LearningConfig(
            train_freq="1h",
            algorithm="matd3",
            actor_architecture="mlp",
            learning_mode=True,
            evaluation_mode=False,
            training_episodes=3,
            episodes_collecting_initial_experience=1,
            continue_learning=False,
            trained_policies_save_path=None,
            early_stopping_steps=10,
            early_stopping_threshold=0.05,
        ),
    }

    learn = Learning(config["learning_config"], start=start, end=end)
    learn.rl_strats["test_id"] = LearningStrategy(**config, learning_role=learn)
    learn.initialize_policy()
    learn.rl_algorithm.n_updates = 137

    inter_episodic_data = learn.get_inter_episodic_data()
    assert inter_episodic_data["n_updates"] == 137

    # Mimic a new episode: fresh learning role / algorithm (n_updates resets to 0).
    next_episode = Learning(config["learning_config"], start=start, end=end)
    next_episode.rl_strats["test_id"] = LearningStrategy(
        **config, learning_role=next_episode
    )
    assert next_episode.rl_algorithm.n_updates == 0

    next_episode.load_inter_episodic_data(inter_episodic_data)
    assert next_episode.rl_algorithm.n_updates == 137


@pytest.mark.require_learning
def test_write_rl_grad_params_uses_global_update_counter():
    """Gradient-step logging must follow TD3.n_updates to avoid collisions in
    staggered mode where update_policy can run more than once per chunk.
    """
    config = {
        "foresight": 1,
        "act_dim": 2,
        "unique_obs_dim": 0,
        "learning_config": LearningConfig(
            train_freq="1h",
            algorithm="matd3",
            actor_architecture="mlp",
            learning_mode=True,
            evaluation_mode=False,
            training_episodes=3,
            episodes_collecting_initial_experience=1,
            continue_learning=False,
            trained_policies_save_path=None,
            early_stopping_steps=10,
            early_stopping_threshold=0.05,
            gradient_steps=2,
        ),
    }

    learn = Learning(config["learning_config"], start=start, end=end)
    learn.db_addr = "db"
    learn._context = MagicMock()

    # Simulate having completed three update_policy calls in the past (2 steps each),
    # and currently logging the 4th call. During this write, n_updates already includes
    # the current call's steps.
    learn.rl_algorithm = SimpleNamespace(n_updates=8)
    learn.update_steps = 999

    unit_params = [
        {
            "unit_1": {
                "actor_loss": 1.0,
                "actor_total_grad_norm": 1.0,
                "actor_max_grad_norm": 1.0,
                "critic_loss": 2.0,
                "critic_total_grad_norm": 2.0,
                "critic_max_grad_norm": 2.0,
            }
        },
        {
            "unit_1": {
                "actor_loss": 1.1,
                "actor_total_grad_norm": 1.1,
                "actor_max_grad_norm": 1.1,
                "critic_loss": 2.1,
                "critic_total_grad_norm": 2.1,
                "critic_max_grad_norm": 2.1,
            }
        },
    ]

    learn.write_rl_grad_params_to_output(
        learning_rate=0.01, unit_params_list=unit_params
    )

    call_kwargs = learn.context.schedule_instant_message.call_args.kwargs
    rows = call_kwargs["content"]["data"]
    assert [row["step"] for row in rows] == [6, 7]


@pytest.fixture
async def learning_role():
    """Fixture that provides a learning role configuration for atomic swap tests."""
    try:
        import torch as th
    except ImportError:
        pytest.skip("torch not available")

    config = {
        "foresight": 1,
        "act_dim": 2,
        "unique_obs_dim": 0,
        "learning_config": LearningConfig(
            train_freq="1h",
            algorithm="matd3",
            actor_architecture="mlp",
            learning_mode=True,
            evaluation_mode=True,  # evaluation mode to skip buffer/policy update
            training_episodes=3,
            episodes_collecting_initial_experience=1,
            continue_learning=False,
            trained_policies_save_path=None,
            early_stopping_steps=10,
            early_stopping_threshold=0.05,
        ),
    }

    learning_role = Learning(config["learning_config"], start=start, end=end)
    learning_role.rl_strats["unit_1"] = LearningStrategy(
        **config, learning_role=learning_role
    )

    # mock function for subsequent processing in store_to_buffer_and_update
    learning_role.write_rl_params_to_output = MagicMock()

    yield learning_role, th


@pytest.mark.require_learning
async def test_atomic_swap_no_data_loss(learning_role):
    """
    Test that the atomic swap in store_to_buffer_and_update does not lose data
    when add_actions_to_cache / add_observation_to_cache / add_reward_to_cache
    are called during the async processing, because the last data point might be in transit.
    """

    learning_role, th = learning_role

    # Add initial data to cache (simulating timestep 1 and 2)
    ts1 = 1000
    ts2 = 2000  # last timestep that has not yet have a reward and hence should be carried over after swap
    ts3 = 3000  # new data that arrives during processing

    obs_ts1 = th.tensor([1.0, 2.0])
    obs_ts2 = th.tensor([3.0, 4.0])
    action_ts1 = th.tensor([0.5])
    action_ts2 = th.tensor([0.6])
    noise_ts1 = th.tensor([0.01])
    noise_ts2 = th.tensor([0.02])
    reward_ts1 = 10.0

    # Add data for ts1 and ts2
    learning_role.add_observation_to_cache("unit_1", ts1, obs_ts1)
    learning_role.add_actions_to_cache("unit_1", ts1, action_ts1, noise_ts1)
    learning_role.add_reward_to_cache(
        "unit_1", ts1, reward_ts1, regret=0.0, profit=10.0
    )

    learning_role.add_observation_to_cache("unit_1", ts2, obs_ts2)
    learning_role.add_actions_to_cache("unit_1", ts2, action_ts2, noise_ts2)
    # ts2 has no reward yet

    # Verify data is in cache before swap
    assert ts1 in learning_role.all_obs
    assert ts2 in learning_role.all_obs
    assert len(learning_role.all_obs) == 2

    # Now call store_to_buffer_and_update (atomic swap happens here)
    # Since we're in evaluation_mode, it won't try to update buffer/policy
    await learning_role.store_to_buffer_and_update()

    # Check that we do not lose the incomplete last timestep ts1
    reward_ts2 = 20  # no reward yet for ts2 to simulate the carry-over logic
    learning_role.add_reward_to_cache("unit_1", ts2, reward_ts2, regret=0, profit=20)

    # After swap: ts1 should be processed, ts2 (last timestamp) should be carried over
    # The cache should now only contain ts2's obs/actions/noises (carried over)
    assert ts2 in learning_role.all_obs, "ts2 should be carried over after swap"
    assert ts1 not in learning_role.all_obs, (
        "ts1 should have been processed and removed"
    )

    # Now simulate new data arriving AFTER the swap (this tests that new dict works)
    obs_ts3 = th.tensor([5.0, 6.0])
    action_ts3 = th.tensor([0.7])
    noise_ts3 = th.tensor([0.03])
    reward_ts3 = 30.0

    learning_role.add_observation_to_cache("unit_1", ts3, obs_ts3)
    learning_role.add_actions_to_cache("unit_1", ts3, action_ts3, noise_ts3)
    learning_role.add_reward_to_cache(
        "unit_1", ts3, reward_ts3, regret=0.0, profit=30.0
    )

    # Verify new data is in cache (not lost due to atomic swap)
    assert ts3 in learning_role.all_obs, "ts3 should be in new cache after swap"
    assert ts3 in learning_role.all_actions, "ts3 actions should be in new cache"
    assert ts3 in learning_role.all_rewards, "ts3 rewards should be in new cache"

    # Verify ts2 carried-over data is still there
    assert ts2 in learning_role.all_obs, "ts2 should still be carried over"
    assert ts2 in learning_role.all_actions, "ts2 actions should still be carried over"
    assert ts2 in learning_role.all_rewards, "ts2 rewards should still be carried over"

    # Verify the actual data content
    assert len(learning_role.all_obs[ts3]["unit_1"]) == 1
    assert th.equal(learning_role.all_obs[ts3]["unit_1"][0], obs_ts3)
    assert len(learning_role.all_actions[ts3]["unit_1"]) == 1
    assert th.equal(learning_role.all_actions[ts3]["unit_1"][0], action_ts3)

    # all timesteps are complete now, so no carry-over should happen in the next swap
    await learning_role.store_to_buffer_and_update()
    assert ts2 not in learning_role.all_obs, (
        "ts2 should have been processed in next swap"
    )
    assert ts3 not in learning_role.all_obs, (
        "ts3 should have been processed in next swap"
    )


@pytest.mark.require_learning
async def test_atomic_swap_carries_over_two_incomplete_timesteps(learning_role):
    """
    Ensure that when two latest timesteps are incomplete (no reward yet),
    both are carried over after store_to_buffer_and_update().
    """
    learning_role, th = learning_role

    ts1, ts2, ts3, ts4 = 1000, 2000, 3000, 4000

    # Complete timesteps
    learning_role.add_observation_to_cache("unit_1", ts1, th.tensor([1.0, 1.1]))
    learning_role.add_actions_to_cache(
        "unit_1", ts1, th.tensor([0.1]), th.tensor([0.01])
    )
    learning_role.add_reward_to_cache("unit_1", ts1, 10.0, regret=0.0, profit=10.0)

    learning_role.add_observation_to_cache("unit_1", ts2, th.tensor([2.0, 2.1]))
    learning_role.add_actions_to_cache(
        "unit_1", ts2, th.tensor([0.2]), th.tensor([0.02])
    )
    learning_role.add_reward_to_cache("unit_1", ts2, 20.0, regret=0.0, profit=20.0)

    # Incomplete timesteps (no reward yet)
    learning_role.add_observation_to_cache("unit_1", ts3, th.tensor([3.0, 3.1]))
    learning_role.add_actions_to_cache(
        "unit_1", ts3, th.tensor([0.3]), th.tensor([0.03])
    )

    learning_role.add_observation_to_cache("unit_1", ts4, th.tensor([4.0, 4.1]))
    learning_role.add_actions_to_cache(
        "unit_1", ts4, th.tensor([0.4]), th.tensor([0.04])
    )

    await learning_role.store_to_buffer_and_update()

    # Both incomplete timesteps should remain in cache
    assert ts3 in learning_role.all_obs, "ts3 should be carried over"
    assert ts4 in learning_role.all_obs, "ts4 should be carried over"
    assert ts3 in learning_role.all_actions, "ts3 actions should be carried over"
    assert ts4 in learning_role.all_actions, "ts4 actions should be carried over"

    # Already complete timesteps should be processed and removed from cache
    assert ts1 not in learning_role.all_obs, "ts1 should have been processed"
    assert ts2 not in learning_role.all_obs, "ts2 should have been processed"

    # Add missing rewards after carry-over and ensure data consistency
    learning_role.add_reward_to_cache("unit_1", ts3, 30.0, regret=0.0, profit=30.0)
    learning_role.add_reward_to_cache("unit_1", ts4, 40.0, regret=0.0, profit=40.0)

    assert ts3 in learning_role.all_obs, (
        "ts3 reward should be accepted after carry-over"
    )
    assert ts4 in learning_role.all_obs, (
        "ts4 reward should be accepted after carry-over"
    )


@pytest.mark.require_learning
async def test_atomic_swap_concurrent_writes(learning_role):
    """
    Test that data written to cache during store_to_buffer_and_update execution
    is not lost. Simulates concurrent-like behavior.
    """
    import asyncio

    learning_role, th = learning_role

    # Add initial data
    ts1 = 1000
    ts2 = 2000
    learning_role.add_observation_to_cache("unit_1", ts1, th.tensor([1.0, 2.0]))
    learning_role.add_actions_to_cache(
        "unit_1", ts1, th.tensor([0.5]), th.tensor([0.01])
    )
    learning_role.add_reward_to_cache("unit_1", ts1, 10.0, regret=0.0, profit=10.0)

    learning_role.add_observation_to_cache("unit_1", ts2, th.tensor([3.0, 4.0]))
    learning_role.add_actions_to_cache(
        "unit_1", ts2, th.tensor([0.6]), th.tensor([0.02])
    )
    learning_role.add_reward_to_cache("unit_1", ts2, 20.0, regret=0.0, profit=20.0)

    # Track data added during "concurrent" operation
    concurrent_data_ts = 4000
    concurrent_obs = th.tensor([9.0, 10.0])

    async def add_data_during_swap():
        """Simulate data being added while swap is processing"""
        await asyncio.sleep(0)  # yield control
        learning_role.add_observation_to_cache(
            "unit_1", concurrent_data_ts, concurrent_obs
        )
        learning_role.add_actions_to_cache(
            "unit_1", concurrent_data_ts, th.tensor([0.9]), th.tensor([0.05])
        )
        learning_role.add_reward_to_cache(
            "unit_1", concurrent_data_ts, 50.0, regret=0.0, profit=50.0
        )

    # Run both concurrently
    await asyncio.gather(
        learning_role.store_to_buffer_and_update(), add_data_during_swap()
    )

    # Verify concurrent data was NOT lost
    assert concurrent_data_ts in learning_role.all_obs, (
        "Concurrent data should be in cache after swap"
    )
    assert concurrent_data_ts in learning_role.all_actions, (
        "Concurrent actions should be in cache"
    )
    assert concurrent_data_ts in learning_role.all_rewards, (
        "Concurrent rewards should be in cache"
    )

    # Verify that the original data added before swap is cleared (since it should have been processed)
    assert ts1 not in learning_role.all_obs, (
        "ts1 should have been processed and moved to buffer"
    )
    assert ts2 not in learning_role.all_obs, (
        "ts2 should have been processed and moved to buffer"
    )

    # Verify data content
    assert len(learning_role.all_obs[concurrent_data_ts]["unit_1"]) == 1
    assert th.equal(
        learning_role.all_obs[concurrent_data_ts]["unit_1"][0], concurrent_obs
    )
