# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest

try:
    import torch as th

    from assume.reinforcement_learning.buffer import ReplayBuffer
except ImportError:
    pass


@pytest.mark.require_learning
def test_replay_buffer_init():
    buffer = ReplayBuffer(
        10,
        obs_dim=2,
        act_dim=3,
        n_rl_units=4,
        device=th.device("cpu"),
        float_type=th.float,
    )

    assert buffer.rewards.shape == (10, 4)
    assert buffer.actions.shape == (10, 4, 3)
    assert buffer.observations.shape == (10, 4, 2)


@pytest.mark.require_learning
def test_replay_buffer_add():
    n_steps = 1
    obs_dim = 2
    act_dim = 3
    n_rl_units = 4
    buffer_size = 10

    buffer = ReplayBuffer(
        buffer_size=buffer_size,
        obs_dim=obs_dim,
        act_dim=act_dim,
        n_rl_units=n_rl_units,
        device=th.device("cpu"),
        float_type=th.float,
    )

    # Simulate cache dicts as in LearningRole
    timestamp = "2024-01-01 00:00:00"
    unit_ids = list(range(1, n_rl_units + 1))
    obs_cache = {timestamp: {u: [np.ones(obs_dim)] for u in unit_ids}}
    actions_cache = {timestamp: {u: [np.ones(act_dim)] for u in unit_ids}}
    rewards_cache = {timestamp: {u: [1.0] for u in unit_ids}}

    # Transform cache dicts to numpy arrays for buffer
    def transform_cache(cache, shape):
        arr = np.zeros((n_steps, n_rl_units, shape))
        for t_idx, t in enumerate([timestamp]):
            for u_idx, u in enumerate(unit_ids):
                arr[t_idx, u_idx] = cache[t][u][0]
        return arr

    obs = transform_cache(obs_cache, obs_dim)
    actions = transform_cache(actions_cache, act_dim)
    # reward shape should be (n_steps, n_rl_units, 1)
    reward = np.zeros((n_steps, n_rl_units, 1))
    for t_idx, t in enumerate([timestamp]):
        for u_idx, u in enumerate(unit_ids):
            reward[t_idx, u_idx, 0] = rewards_cache[t][u][0]

    buffer.add(obs, actions, reward)
    # can't sample with only one entry
    with pytest.raises(Exception):
        sample = buffer.sample(1)
    buffer.add(obs, actions, reward)

    sample = buffer.sample(1)
    observations, actions, next_observations, _, rewards, masks, _ = sample

    assert rewards.shape == (1, 4)
    assert masks.shape == (1, 4)
    assert actions.shape == (1, 4, 3)
    assert next_observations.shape == (1, 4, 2)
    assert observations.shape == (1, 4, 2)

    # now sample twice
    sample = buffer.sample(2)
    observations, actions, next_observations, _, rewards, masks, _ = sample

    assert rewards.shape == (2, 4)
    assert masks.shape == (2, 4)
    assert actions.shape == (2, 4, 3)
    assert next_observations.shape == (2, 4, 2)
    assert observations.shape == (2, 4, 2)


def _make_buffer(buffer_size):
    return ReplayBuffer(
        buffer_size=buffer_size,
        obs_dim=1,
        act_dim=1,
        n_rl_units=1,
        device=th.device("cpu"),
        float_type=th.float,
    )


def _add_steps(buffer, values):
    values = np.asarray(values, dtype=float)
    obs = values.reshape(-1, 1, 1)
    buffer.add(obs, obs.copy(), values.reshape(-1, 1, 1))


@pytest.mark.require_learning
def test_replay_buffer_add_wraps_around():
    buffer = _make_buffer(5)
    _add_steps(buffer, [0, 1, 2])
    # crosses the end of the buffer: 3, 4 at the end, 5, 6 at the start
    _add_steps(buffer, [3, 4, 5, 6])

    assert buffer.full
    assert buffer.pos == 2
    assert buffer.observations[:, 0, 0].tolist() == [5, 6, 2, 3, 4]
    assert buffer.rewards[:, 0].tolist() == [5, 6, 2, 3, 4]


@pytest.mark.require_learning
def test_replay_buffer_add_larger_than_buffer_keeps_newest():
    buffer = _make_buffer(4)
    _add_steps(buffer, [0])
    _add_steps(buffer, list(range(1, 11)))

    assert buffer.full
    # same layout as adding the rows one by one
    reference = _make_buffer(4)
    for value in range(11):
        _add_steps(reference, [value])
    assert buffer.pos == reference.pos
    np.testing.assert_array_equal(buffer.observations, reference.observations)


@pytest.mark.require_learning
def test_replay_buffer_sample_next_is_successor_after_wrap():
    buffer = _make_buffer(5)
    _add_steps(buffer, list(range(8)))  # newest value 7 is at index 2

    for _ in range(20):
        sample = buffer.sample(16)
        obs = sample.observations[:, 0, 0]
        next_obs = sample.next_observations[:, 0, 0]
        # every transition pairs a row with the row added right after it
        assert th.equal(next_obs, obs + 1)
        # the newest row has no successor and is never sampled
        assert not (obs == 7).any()
