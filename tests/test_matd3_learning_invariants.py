# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Invariants the MATD3 learning loop must satisfy, independent of any scenario.

These train on a synthetic replay buffer with a known, constant reward, so the
correct answer is available in closed form. Nothing here depends on a market, a
reward function or a bidding strategy - a failure is a defect in the learning
loop itself.

Two properties are asserted:

1. The critic's prediction must lie inside the achievable discounted return.
   For rewards bounded in ``[r_min, r_max]`` and no terminal state, every
   attainable return lies in ``[r_min / (1 - gamma), r_max / (1 - gamma)]``.
   A prediction outside that interval cannot correspond to any policy.

2. The actor must not saturate. ``softsign`` has derivative ``1 / (1 + |x|)**2``,
   so a policy driven to ``|action| ~ 1`` has a vanishing gradient and can never
   recover, whatever the critic subsequently learns.
"""

from datetime import datetime

import numpy as np
import pytest

from assume.common.base import LearningConfig

try:
    import torch as th

    from assume.common.base import LearningStrategy
    from assume.reinforcement_learning.buffer import ReplayBuffer
    from assume.reinforcement_learning.learning_role import Learning
    from assume.reinforcement_learning.learning_utils import NormalActionNoise
except ImportError:
    th = None

START = datetime(2023, 7, 1)
END = datetime(2023, 7, 2)

N_AGENTS = 2
ACT_DIM = 1
FORESIGHT = 2
NUM_TIMESERIES_OBS_DIM = 2
UNIQUE_OBS_DIM = 2
OBS_DIM = FORESIGHT * NUM_TIMESERIES_OBS_DIM + UNIQUE_OBS_DIM
GAMMA = 0.99
BATCH_SIZE = 64


def _make_learning_role(gamma: float = GAMMA, gradient_steps: int = 1) -> "Learning":
    config = {
        "foresight": FORESIGHT,
        "act_dim": ACT_DIM,
        "unique_obs_dim": UNIQUE_OBS_DIM,
        "num_timeseries_obs_dim": NUM_TIMESERIES_OBS_DIM,
        "obs_dim": OBS_DIM,
        "learning_config": LearningConfig(
            train_freq="1h",
            algorithm="matd3",
            actor_architecture="mlp",
            learning_mode=True,
            evaluation_mode=False,
            training_episodes=1,
            episodes_collecting_initial_experience=0,
            continue_learning=False,
            trained_policies_save_path=None,
            learning_rate=1e-3,
            batch_size=BATCH_SIZE,
            tau=0.005,
            gamma=gamma,
            gradient_steps=gradient_steps,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
        ),
    }
    learn = Learning(config["learning_config"], START, END)
    for i in range(N_AGENTS):
        strategy = LearningStrategy(**config, learning_role=learn)
        # Normally attached by TorchLearningStrategy; the update loop needs both.
        strategy.action_noise = NormalActionNoise(ACT_DIM)
        strategy.unit_id = f"agent_{i}"
        learn.rl_strats[f"agent_{i}"] = strategy
    learn.create_learning_algorithm(config["learning_config"].algorithm)
    learn.initialize_policy()
    # Stands in for the world clock, which drives the noise and learning-rate ramp.
    learn._progress_timestamp = learn.start
    # Normally set when the role is attached to a World.
    learn.db_addr = None
    learn.update_steps = 0
    return learn


def _fill_buffer(learn: "Learning", reward: float, n: int = 512) -> None:
    """A buffer where every transition earns exactly ``reward``."""
    rng = np.random.default_rng(0)
    learn.buffer = ReplayBuffer(
        buffer_size=n + BATCH_SIZE,
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        n_rl_units=N_AGENTS,
        device="cpu",
        float_type=th.float32,
    )
    obs = rng.uniform(-1, 1, size=(n, N_AGENTS, OBS_DIM)).astype(np.float32)
    actions = rng.uniform(-1, 1, size=(n, N_AGENTS, ACT_DIM)).astype(np.float32)
    rewards = np.full((n, N_AGENTS, 1), reward, dtype=np.float32)
    learn.buffer.add(obs, actions, rewards)


def _train(learn: "Learning", updates: int) -> None:
    for _ in range(updates):
        learn.rl_algorithm.update_policy()


def _mean_q(learn: "Learning") -> float:
    """Predicted Q for the first agent, averaged over a fresh batch."""
    sample = learn.buffer.sample(BATCH_SIZE)
    strategies = list(learn.rl_strats.values())
    unique = sample.observations[:, :, -UNIQUE_OBS_DIM:]
    with th.no_grad():
        others = th.cat((unique[:, :0], unique[:, 1:]), dim=1)
        all_states = th.cat(
            (
                sample.observations[:, 0, :].reshape(BATCH_SIZE, -1),
                others.reshape(BATCH_SIZE, -1),
            ),
            dim=1,
        )
        all_actions = sample.actions.view(BATCH_SIZE, -1)
        q = strategies[0].critics.q1_forward(all_states, all_actions)
    return float(q.mean())


@pytest.mark.require_learning
@pytest.mark.parametrize("reward", [1.0, -1.0])
def test_critic_value_stays_within_achievable_return(reward):
    """The critic must not predict a return no policy could ever achieve.

    With a constant reward and no terminal state the only attainable return is
    ``reward / (1 - gamma)``. Anything outside that bound - in particular a
    prediction of the wrong sign - means the target computation is wrong, not
    merely inaccurate.
    """
    learn = _make_learning_role()
    _fill_buffer(learn, reward=reward)
    _train(learn, updates=300)

    q = _mean_q(learn)
    bound = reward / (1 - GAMMA)
    low, high = sorted((0.0, bound))
    # Generous tolerance: this is about sign and order of magnitude, not accuracy.
    margin = 0.5 * abs(bound)
    assert low - margin <= q <= high + margin, (
        f"critic predicts Q={q:.2f} for a constant reward of {reward}; "
        f"the only achievable return is {bound:.2f}"
    )


@pytest.mark.require_learning
def test_actor_does_not_saturate_against_the_action_bound():
    """A saturated actor has a vanishing gradient and can never recover.

    ``softsign`` has derivative ``1 / (1 + |x|)**2``, so an output of 0.999
    implies a pre-activation near 1000 and a gradient of order 1e-6. Once the
    policy reaches the boundary it is frozen there for the rest of training.
    """
    learn = _make_learning_role()
    _fill_buffer(learn, reward=1.0)
    _train(learn, updates=300)

    sample = learn.buffer.sample(BATCH_SIZE)
    strategy = list(learn.rl_strats.values())[0]
    with th.no_grad():
        actions = strategy.actor(sample.observations[:, 0, :])
    saturated = (actions.abs() > 0.99).float().mean().item()

    assert saturated < 0.5, (
        f"{saturated:.0%} of actions are saturated at the action bound; "
        "the actor gradient has vanished and the policy can no longer move"
    )


@pytest.mark.require_learning
def test_actor_gradient_does_not_vanish():
    """The actor must still be able to move after training."""
    learn = _make_learning_role()
    _fill_buffer(learn, reward=1.0)
    _train(learn, updates=300)

    strategy = list(learn.rl_strats.values())[0]
    grad_norm = max(
        float(p.grad.norm()) for p in strategy.actor.parameters() if p.grad is not None
    )
    assert grad_norm > 1e-4, (
        f"actor gradient norm has collapsed to {grad_norm:.2e}; "
        "the policy is frozen regardless of what the critic learns"
    )
