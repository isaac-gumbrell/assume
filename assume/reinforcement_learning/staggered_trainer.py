# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Staggered (paired-scenario) MATD3 trainer — D3 design.

Orchestrates two persistent ``World`` instances that share one MATD3
actor/critic and one replay buffer. Within each training episode, both worlds
advance in *paired-sequential chunks* of ``train_freq`` hours: world A is
advanced over a chunk, then world B over the same chunk, then a single batch of
gradient updates is applied to the shared policy.

See ``.isaac_docs/d3_staggered_training_implementation.md`` for the full design.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from assume.common.exceptions import AssumeException
from assume.common.utils import (
    confirm_learning_save_path,
    datetime2timestamp,
)
from assume.world import World

logger = logging.getLogger(__name__)


def _ensure_persistent_loop(worlds: list[World]) -> asyncio.AbstractEventLoop:
    """Make sure all worlds share a single asyncio loop.

    The ``World`` constructor creates a fresh loop per instance. The staggered
    trainer must drive both worlds from the same loop so the orchestrator can
    interleave their async tasks. We pick world[0]'s loop and force the others
    to share it.
    """
    primary = worlds[0].loop
    for w in worlds[1:]:
        w.loop = primary
    asyncio.set_event_loop(primary)
    return primary


def _share_learning_state(anchor: World, secondary: World) -> None:
    """Wire the secondary world's learning role to use the anchor's policy/buffer.

    After both worlds have been set up via :func:`setup_world`, each owns its
    own :class:`assume.reinforcement_learning.learning_role.Learning` instance
    (one per mango container). For shared MATD3 training we stitch these
    together so that:

    * The replay buffer is shared (transitions from both worlds are pushed into
      the same buffer).
    * The MATD3 ``rl_algorithm`` (and therefore actor / critic networks and
      their optimisers) is the same Python object on both sides.
    * Each world's :class:`LearningStrategy` instances reuse the anchor's
      actor / target actor so the action returned to a unit is identical for a
      given observation regardless of which world is currently stepping.

    Note: each world's LearningRole keeps its own per-timestep observation /
    action / reward caches and its own scheduled flush task. That is fine —
    on flush each role writes into the shared buffer.
    """
    anchor_role = anchor.learning_role
    secondary_role = secondary.learning_role
    if anchor_role is None or secondary_role is None:
        return

    # Initialize the anchor's policy first so actor/critic objects exist.
    if getattr(anchor_role, "rl_algorithm", None) is None:
        return
    anchor_role.rl_algorithm.initialize_policy()

    # Share the algorithm instance — same actor / critic / optimisers.
    secondary_role.rl_algorithm = anchor_role.rl_algorithm

    # Share the (yet-to-be-attached) buffer. The actual buffer is created in
    # :func:`run_learning` flow; until then, both roles will reference whichever
    # is set later via :meth:`load_inter_episodic_data`.
    if anchor_role.buffer is not None:
        secondary_role.buffer = anchor_role.buffer

    # Replace each secondary-world strategy's actor with the anchor's so both
    # worlds query the same network for actions. We mutate the strategy
    # instance in place because units already hold references to it.
    for unit_id, secondary_strategy in secondary_role.rl_strats.items():
        if unit_id not in anchor_role.rl_strats:
            continue
        anchor_strategy = anchor_role.rl_strats[unit_id]
        for attr in ("actor", "actor_target", "target_actor"):
            if hasattr(anchor_strategy, attr):
                setattr(secondary_strategy, attr, getattr(anchor_strategy, attr))


def _chunk_boundaries(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_freq: str,
) -> list[tuple[float, float]]:
    """Return the list of ``(start_ts, end_ts)`` epoch-second pairs that tile
    ``[start, end)`` at ``train_freq`` cadence.

    The simulation horizon is required to be divisible by ``train_freq`` (this
    is enforced by :meth:`Learning.sync_train_freq_with_simulation_horizon`).
    """
    delta = pd.Timedelta(train_freq)
    boundaries: list[tuple[float, float]] = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + delta, end)
        boundaries.append((datetime2timestamp(cursor), datetime2timestamp(nxt)))
        cursor = nxt
    return boundaries


class StaggeredTrainer:
    """Drives MATD3 training across two paired ``World`` instances in lockstep.

    The trainer mirrors the high-level flow of
    :func:`assume.scenario.loader_csv.run_learning` but interleaves two worlds
    at the chunk granularity dictated by ``train_freq``.
    """

    def __init__(
        self,
        worlds: list[World],
        swap_order_per_episode: bool = True,
        verbose: bool = False,
    ) -> None:
        if len(worlds) != 2:
            raise ValueError(
                f"StaggeredTrainer requires exactly 2 worlds, got {len(worlds)}"
            )
        self.worlds = worlds
        self.swap_order_per_episode = swap_order_per_episode
        self.verbose = verbose

        # The first world is the "anchor"; its LearningRole owns shared state.
        self.anchor: World = worlds[0]
        self.secondary: World = worlds[1]

        # Pre-flight: make sure both worlds share an event loop and learning state.
        _ensure_persistent_loop(self.worlds)
        _share_learning_state(self.anchor, self.secondary)

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Run the full staggered training loop (mirrors ``run_learning``)."""
        from assume.reinforcement_learning.buffer import ReplayBuffer
        from assume.scenario.loader_csv import setup_world

        if not self.verbose:
            logger.setLevel(logging.WARNING)

        # CSV export is suppressed during learning, same as run_learning.
        temp_csv_paths = [w.export_csv_path for w in self.worlds]
        for w in self.worlds:
            w.export_csv_path = ""

        # Initialize shared policy once (the anchor owns it).
        self.anchor.learning_role.rl_algorithm.initialize_policy()
        # Re-share so the secondary world picks up the actor refs.
        _share_learning_state(self.anchor, self.secondary)

        learning_config = self.anchor.learning_role.learning_config
        save_path = learning_config.trained_policies_save_path
        continue_learning = learning_config.continue_learning
        confirm_learning_save_path(save_path, continue_learning)

        # Reset tensorboard logs for both worlds' simulation_ids.
        for w in self.worlds:
            tb_path = f"tensorboard/{w.scenario_data['simulation_id']}"
            if os.path.exists(tb_path):
                shutil.rmtree(tb_path, ignore_errors=True)

        # Single shared inter-episodic state (buffer / actors / max_eval).
        inter_episodic_data = {
            "buffer": ReplayBuffer(
                buffer_size=learning_config.replay_buffer_size,
                obs_dim=self.anchor.learning_role.rl_algorithm.obs_dim,
                act_dim=self.anchor.learning_role.rl_algorithm.act_dim,
                n_rl_units=len(self.anchor.learning_role.rl_strats),
                device=self.anchor.learning_role.device,
                float_type=self.anchor.learning_role.float_type,
            ),
            "actors_and_critics": None,
            "max_eval": defaultdict(lambda: -1e9),
            "all_eval": defaultdict(list),
            "avg_all_eval": [],
            "episodes_done": 0,
            "eval_episodes_done": 0,
        }
        for w in self.worlds:
            w.learning_role.load_inter_episodic_data(inter_episodic_data)
        # share buffer reference once it exists
        self.secondary.learning_role.buffer = self.anchor.learning_role.buffer

        validation_interval = self.anchor.learning_role.determine_validation_interval()

        # Sync train_freq with simulation horizon (anchor only — both worlds share
        # the same horizon by design constraint).
        new_train_freq = (
            self.anchor.learning_role.sync_train_freq_with_simulation_horizon()
        )
        if new_train_freq is not None:
            for w in self.worlds:
                w.scenario_data["config"]["learning_config"]["train_freq"] = (
                    new_train_freq
                )
                w.learning_role.learning_config.train_freq = new_train_freq

        eval_episode = 1
        for episode in tqdm(
            range(1, learning_config.training_episodes + 1),
            desc="Staggered Training Episodes",
        ):
            if episode != 1:
                for w in self.worlds:
                    setup_world(world=w, episode=episode)
                _share_learning_state(self.anchor, self.secondary)
                for w in self.worlds:
                    w.learning_role.load_inter_episodic_data(inter_episodic_data)
                self.secondary.learning_role.buffer = self.anchor.learning_role.buffer

            order = self._world_order(episode)
            self._run_episode_chunked(
                order, train_freq=new_train_freq or learning_config.train_freq
            )

            # Aggregate inter-episodic data from anchor (it's the shared owner).
            self.anchor.learning_role.tensor_board_logger.update_tensorboard()
            inter_episodic_data = self.anchor.learning_role.get_inter_episodic_data()
            inter_episodic_data["episodes_done"] = episode

            # Evaluation run
            do_eval = (
                episode % validation_interval == 0
                and episode
                >= learning_config.episodes_collecting_initial_experience
                + validation_interval
            )
            if do_eval:
                for w in self.worlds:
                    w.reset()
                    setup_world(
                        world=w,
                        evaluation_mode=True,
                        episode=episode,
                        eval_episode=eval_episode,
                    )
                _share_learning_state(self.anchor, self.secondary)
                for w in self.worlds:
                    w.learning_role.load_inter_episodic_data(inter_episodic_data)
                self.secondary.learning_role.buffer = self.anchor.learning_role.buffer

                eval_order = self._world_order(episode)
                self._run_episode_chunked(
                    eval_order,
                    train_freq=new_train_freq or learning_config.train_freq,
                )
                self.anchor.learning_role.tensor_board_logger.update_tensorboard()

                per_scenario_avg = {}
                for w in self.worlds:
                    if not w.db_uri:
                        raise AssumeException(
                            "No learning rewards as no database was given"
                        )
                    rewards = w.output_role.get_sum_reward(episode=eval_episode)
                    if len(rewards) == 0:
                        raise AssumeException(
                            "No rewards were collected during evaluation run"
                        )
                    name = w.scenario_data.get(
                        "staggered_scenario_name", w.simulation_id
                    )
                    per_scenario_avg[f"avg_reward_{name}"] = float(np.mean(rewards))

                avg_reward = float(np.mean(list(per_scenario_avg.values())))
                metrics = {"avg_reward": avg_reward, **per_scenario_avg}
                terminate = self.anchor.learning_role.compare_and_save_policies(metrics)
                inter_episodic_data["eval_episodes_done"] = eval_episode
                if terminate:
                    break
                eval_episode += 1

            for w in self.worlds:
                w.reset()

            if (
                episode
                >= learning_config.episodes_collecting_initial_experience
                + validation_interval
            ):
                self.anchor.learning_role.rl_algorithm.save_params(
                    directory=f"{learning_config.trained_policies_save_path}/last_policies"
                )
                lc = self.anchor.learning_role.learning_config
                if getattr(lc, "save_replay_buffer", True):
                    buf = self.anchor.learning_role.buffer
                    if buf is not None:
                        buf_path = getattr(lc, "replay_buffer_save_path", None) or (
                            f"{lc.trained_policies_save_path}/last_policies/replay_buffer.npz"
                        )
                        buf.save(buf_path)

        logger.info("################")
        logger.info("Staggered training finished, starting evaluation run")
        for w, original_csv in zip(self.worlds, temp_csv_paths):
            w.export_csv_path = original_csv
            w.reset()
            w.scenario_data["config"]["learning_config"][
                "trained_policies_load_path"
            ] = f"{learning_config.trained_policies_save_path}/last_policies"
            setup_world(world=w, terminate_learning=True)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _world_order(self, episode: int) -> list[World]:
        """Return the per-episode world advancement order (alternates if enabled)."""
        if self.swap_order_per_episode and episode % 2 == 0:
            return [self.secondary, self.anchor]
        return [self.anchor, self.secondary]

    def _run_episode_chunked(self, order: list[World], train_freq: str) -> None:
        """Advance both worlds chunk-by-chunk through one episode."""
        # Both worlds share start/end; use the first.
        start = order[0].start
        end = order[0].end
        if isinstance(start, datetime):
            start = pd.Timestamp(start)
        if isinstance(end, datetime):
            end = pd.Timestamp(end)
        boundaries = _chunk_boundaries(start, end, train_freq)
        loop = self.anchor.loop

        async def _drive():
            # Open both containers and hold them open across all chunks.
            async with order[0].activate_container(), order[1].activate_container():
                pbar = tqdm(
                    total=len(boundaries),
                    desc=f"Episode chunks ({order[0].simulation_id}/{order[1].simulation_id})",
                    leave=False,
                )
                try:
                    for chunk_start, chunk_end in boundaries:
                        for w in order:
                            await w.async_run_chunk(chunk_start, chunk_end)
                        pbar.update(1)
                finally:
                    pbar.close()

        loop.run_until_complete(_drive())
