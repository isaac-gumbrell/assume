# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from types import SimpleNamespace
from unittest.mock import MagicMock

from assume.reinforcement_learning.staggered_trainer import (
    StaggeredTrainer,
    _share_learning_state,
)


def test_share_learning_state_does_not_reinit_when_actors_loaded():
    called = {"count": 0}

    def init_policy():
        called["count"] += 1

    # Anchor: already has actor objects on its strategies.
    anchor_strategy = SimpleNamespace(
        actor=object(), actor_target=object(), target_actor=object()
    )
    anchor_role = SimpleNamespace(
        rl_algorithm=SimpleNamespace(initialize_policy=init_policy),
        rl_strats={"u1": anchor_strategy},
        buffer=None,
    )

    # Secondary: has a strategy without actor attributes initially.
    secondary_strategy = SimpleNamespace()
    secondary_role = SimpleNamespace(rl_strats={"u1": secondary_strategy}, buffer=None)

    anchor = SimpleNamespace(learning_role=anchor_role)
    secondary = SimpleNamespace(learning_role=secondary_role)

    # Run the sharing logic.
    _share_learning_state(anchor, secondary)

    # initialize_policy should NOT have been called since anchor already had actors.
    assert called["count"] == 0

    # Secondary strategy should reference anchor's actor objects.
    assert secondary_role.rl_strats["u1"].actor is anchor_role.rl_strats["u1"].actor
    assert (
        secondary_role.rl_strats["u1"].actor_target
        is anchor_role.rl_strats["u1"].actor_target
    )
    assert (
        secondary_role.rl_strats["u1"].target_actor
        is anchor_role.rl_strats["u1"].target_actor
    )


def test_share_learning_state_shares_action_noise():
    # Exploration-noise decay is only applied to the anchor role's strategies
    # during update_policy (the shared algorithm always references the anchor's
    # learning role). If the secondary kept its own action_noise object, its
    # noise would never decay. Sharing the object keeps both worlds on the same
    # decaying schedule.
    anchor_noise = object()
    anchor_strategy = SimpleNamespace(
        actor=object(),
        actor_target=object(),
        target_actor=object(),
        action_noise=anchor_noise,
    )
    anchor_role = SimpleNamespace(
        rl_algorithm=SimpleNamespace(initialize_policy=lambda: None),
        rl_strats={"u1": anchor_strategy},
        buffer=None,
    )

    # Secondary starts with its own, distinct action_noise object.
    secondary_strategy = SimpleNamespace(action_noise=object())
    secondary_role = SimpleNamespace(rl_strats={"u1": secondary_strategy}, buffer=None)

    anchor = SimpleNamespace(learning_role=anchor_role)
    secondary = SimpleNamespace(learning_role=secondary_role)

    _share_learning_state(anchor, secondary)

    # Secondary strategy should now reference the anchor's action_noise object,
    # so a single decay update mutates the noise seen by both worlds.
    assert secondary_role.rl_strats["u1"].action_noise is anchor_noise


def test_pending_policy_update_runs_once_per_chunk():
    """Both worlds flushing in one chunk results in a single policy update."""
    update_policy = MagicMock()
    shared_role = SimpleNamespace(_policy_update_pending=False)
    shared_role.rl_algorithm = SimpleNamespace(
        learning_role=shared_role, update_policy=update_policy
    )
    trainer = object.__new__(StaggeredTrainer)
    trainer.anchor = SimpleNamespace(learning_role=shared_role)

    # nothing flushed: no update
    trainer._run_pending_policy_update()
    update_policy.assert_not_called()

    # both worlds flush in the same chunk and mark the shared update as pending
    shared_role._policy_update_pending = True
    shared_role._policy_update_pending = True
    trainer._run_pending_policy_update()
    trainer._run_pending_policy_update()

    update_policy.assert_called_once()
    assert shared_role._policy_update_pending is False
