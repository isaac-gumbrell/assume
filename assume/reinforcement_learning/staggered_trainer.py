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

    # Avoid unconditionally re-initializing the policy here. If the
    # anchor's strategies already have `actor` objects (for example when
    # policies were loaded by `load_inter_episodic_data`), calling
    # `initialize_policy()` would recreate networks and wipe loaded params.
    needs_init = False
    for strategy in anchor_role.rl_strats.values():
        if not hasattr(strategy, "actor") or getattr(strategy, "actor") is None:
            needs_init = True
            break
    if needs_init:
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

        cfg_save_flag = getattr(learning_config, "save_replay_buffer", True)
        cfg_save_path = getattr(learning_config, "replay_buffer_save_path", None)
        cfg_load_flag = getattr(learning_config, "load_replay_buffer", False)
        cfg_load_path = getattr(learning_config, "replay_buffer_load_path", None)
        cfg_state_save_flag = getattr(learning_config, "save_learning_state", True)
        cfg_state_save_path = getattr(learning_config, "learning_state_save_path", None)
        cfg_state_load_flag = getattr(learning_config, "load_learning_state", False)
        cfg_state_load_path = getattr(learning_config, "learning_state_load_path", None)

        default_buffer_path = f"{save_path}/last_policies/replay_buffer.npz"
        default_state_path = f"{save_path}/last_policies/learning_state.pt"

        def resolve_checkpoint_path(
            load_path: str | None,
            save_path: str | None,
            default_path: str,
        ) -> str:
            """Resolve checkpoint path with precedence: explicit load > explicit save > default."""
            if load_path is not None:
                return load_path
            if save_path is not None:
                return save_path
            return default_path

        resume_mode = continue_learning or cfg_load_flag or cfg_state_load_flag
        if resume_mode:
            logger.info(
                "Resume mode activated (staggered): "
                f"continue_learning={continue_learning}, "
                f"load_replay_buffer={cfg_load_flag}, "
                f"load_learning_state={cfg_state_load_flag}"
            )

        # Reset tensorboard logs for fresh starts only.
        if not resume_mode:
            for w in self.worlds:
                tb_path = f"tensorboard/{w.scenario_data['simulation_id']}"
                if os.path.exists(tb_path):
                    shutil.rmtree(tb_path, ignore_errors=True)

        # Single shared inter-episodic state (buffer / actors / max_eval).
        if cfg_load_flag:
            path_to_load = resolve_checkpoint_path(
                load_path=cfg_load_path,
                save_path=cfg_save_path,
                default_path=default_buffer_path,
            )
            if not os.path.exists(path_to_load):
                raise AssumeException(
                    f"load_replay_buffer is true but no buffer file found at {path_to_load}"
                )
            buffer = ReplayBuffer.load(
                path_to_load,
                device=self.anchor.learning_role.device,
                float_type=self.anchor.learning_role.float_type,
            )
            logger.info(f"Loaded replay buffer from {path_to_load}")
            try:
                for w in self.worlds:
                    w.learning_role.learning_config.episodes_collecting_initial_experience = 0
                    w.scenario_data["config"]["learning_config"][
                        "episodes_collecting_initial_experience"
                    ] = 0
            except Exception:
                logger.warning(
                    "Could not set episodes_collecting_initial_experience to 0 on learning_config"
                )
        else:
            buffer = ReplayBuffer(
                buffer_size=learning_config.replay_buffer_size,
                obs_dim=self.anchor.learning_role.rl_algorithm.obs_dim,
                act_dim=self.anchor.learning_role.rl_algorithm.act_dim,
                n_rl_units=len(self.anchor.learning_role.rl_strats),
                device=self.anchor.learning_role.device,
                float_type=self.anchor.learning_role.float_type,
            )

        inter_episodic_data = {
            "buffer": buffer,
            "actors_and_critics": None,
            "max_eval": defaultdict(lambda: -1e9),
            "all_eval": defaultdict(list),
            "avg_all_eval": [],
            "episodes_done": 0,
            "eval_episodes_done": 0,
        }
        self._load_and_share_state(inter_episodic_data)

        if cfg_state_load_flag:
            state_path_to_load = resolve_checkpoint_path(
                load_path=cfg_state_load_path,
                save_path=cfg_state_save_path,
                default_path=default_state_path,
            )
            if not os.path.exists(state_path_to_load):
                raise AssumeException(
                    "load_learning_state is true but no learning-state file found at "
                    f"{state_path_to_load}"
                )
            self.anchor.learning_role.load_runtime_state(state_path_to_load)
            logger.info(f"Loaded learning runtime state from {state_path_to_load}")

        inter_episodic_data = self.anchor.learning_role.get_inter_episodic_data()
        start_episode = max(int(inter_episodic_data.get("episodes_done", 0)) + 1, 1)
        eval_episode = int(inter_episodic_data.get("eval_episodes_done", 0)) + 1

        # When restarting from episode 1 (no state loaded), existing DB records for
        # these simulation IDs would corrupt the new TB charts. Clear them now so
        # the new run writes into a clean slate.
        if start_episode == 1:
            self._clear_stale_training_db_data()
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

        if start_episode > learning_config.training_episodes:
            logger.info(
                "Training already complete according to loaded runtime state "
                f"(episodes_done={start_episode - 1}). Skipping training loop."
            )

        if start_episode != 1 and start_episode <= learning_config.training_episodes:
            for w in self.worlds:
                setup_world(world=w, episode=start_episode)
            self._load_and_share_state(inter_episodic_data)

        for episode in tqdm(
            range(start_episode, learning_config.training_episodes + 1),
            desc="Staggered Training Episodes",
        ):
            if episode != start_episode:
                for w in self.worlds:
                    setup_world(world=w, episode=episode)
                self._load_and_share_state(inter_episodic_data)

            order = self._world_order(episode)
            self._run_episode_chunked(
                order, train_freq=new_train_freq or learning_config.train_freq
            )

            # Aggregate inter-episodic data from anchor (it's the shared owner).
            for w in self.worlds:
                tb_logger = getattr(
                    getattr(w, "learning_role", None), "tensor_board_logger", None
                )
                if tb_logger is not None:
                    tb_logger.update_tensorboard()
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
                self._load_and_share_state(inter_episodic_data)

                eval_order = self._world_order(episode)
                self._run_episode_chunked(
                    eval_order,
                    train_freq=new_train_freq or learning_config.train_freq,
                )
                for w in self.worlds:
                    tb_logger = getattr(
                        getattr(w, "learning_role", None), "tensor_board_logger", None
                    )
                    if tb_logger is not None:
                        tb_logger.update_tensorboard()

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
                if cfg_save_flag:
                    buf = self.anchor.learning_role.buffer
                    if buf is not None:
                        buf_path = (
                            cfg_save_path
                            if cfg_save_path is not None
                            else default_buffer_path
                        )
                        buf.save(buf_path)
                if cfg_state_save_flag:
                    state_save_path = (
                        cfg_state_save_path
                        if cfg_state_save_path is not None
                        else default_state_path
                    )
                    self.anchor.learning_role.save_runtime_state(state_save_path)

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
    def _load_and_share_state(self, inter_episodic_data: dict) -> None:
        """Load inter-episodic data into both worlds then re-share learning state.

        The correct order is:
        1. Load data into both worlds (``load_inter_episodic_data``).
        2. Re-run ``_share_learning_state`` so that the secondary's strategy
            actor references are updated to whatever the anchor's
            ``initialize_policy`` just installed.
        3. Copy the buffer reference from anchor to secondary.
        4. Sync ``collect_initial_experience_mode`` so secondary mirrors anchor
            (important when ``actors_and_critics`` was ``None`` on the first call).
        """
        for w in self.worlds:
            w.learning_role.load_inter_episodic_data(inter_episodic_data)
        # Re-share AFTER load so secondary gets anchor's up-to-date actor refs.
        _share_learning_state(self.anchor, self.secondary)
        self.secondary.learning_role.buffer = self.anchor.learning_role.buffer
        self._sync_exploration_state()

    def _sync_exploration_state(self) -> None:
        """Copy ``collect_initial_experience_mode`` from anchor strategies to secondary.

        ``load_inter_episodic_data`` on the secondary may fail to turn off
        exploration when the actor ``loaded`` flag is stale (set before the
        latest ``_share_learning_state`` call).  This pass fixes that by
        mirroring the anchor's already-correct exploration state.
        """
        for unit_id, sec_strat in self.secondary.learning_role.rl_strats.items():
            if unit_id not in self.anchor.learning_role.rl_strats:
                continue
            anchor_strat = self.anchor.learning_role.rl_strats[unit_id]
            if not hasattr(anchor_strat, "collect_initial_experience_mode"):
                continue
            sec_strat.collect_initial_experience_mode = (
                anchor_strat.collect_initial_experience_mode
            )

    def _clear_stale_training_db_data(self) -> None:
        """Delete old training-mode rl_params / rl_grad_params rows for this run.

        When ``start_episode == 1`` the episode counters reset to 1.  Any rows
        from a previous run that share the same simulation ID and episode number
        would be mixed with new data when the TB logger reads from the DB, producing
        corrupted charts (e.g. inflated critic-loss at gradient-step 1).

        Only training rows (``evaluation_mode = 0``) are removed; evaluation rows
        are left intact so historical best-policy comparisons still work.
        """
        from sqlalchemy import create_engine, text

        for w in self.worlds:
            if not getattr(w, "db_uri", None):
                continue
            sim_id = w.simulation_id
            try:
                engine = create_engine(w.db_uri)
                with engine.begin() as conn:
                    for tbl in ("rl_params", "rl_grad_params"):
                        try:
                            conn.execute(
                                text(
                                    f"DELETE FROM {tbl} WHERE simulation = :sid"
                                    " AND evaluation_mode = 0"
                                ),
                                {"sid": sim_id},
                            )
                        except Exception:
                            pass  # table may not exist yet on a truly fresh DB
                engine.dispose()
                logger.info(
                    f"Cleared stale training DB records for simulation '{sim_id}'"
                )
            except Exception as exc:
                logger.warning(
                    f"Could not clear stale DB training data for '{sim_id}': {exc}"
                )

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
