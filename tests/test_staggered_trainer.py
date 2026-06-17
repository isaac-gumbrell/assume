# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from types import SimpleNamespace

from assume.reinforcement_learning.staggered_trainer import _share_learning_state


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
