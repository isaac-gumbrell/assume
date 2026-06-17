# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unified ASSUME runner.

Dispatches to one of three execution modes based on runner.yaml (or --mode):

  single     — one scenario / study case (equivalent to the old run.py)
  batch      — parallel cases across one or more scenarios
  staggered  — paired-scenario staggered MATD3 training

Configuration lives in runner.yaml. Every YAML field can be overridden
from the CLI; run `python runner.py --help` for the full option list.

Usage:
    python runner.py                                          # uses runner.yaml
    python runner.py --mode single --scenario srmc_bau_2045 --study-case run_seed_e
    python runner.py --mode batch --scenarios profit_bau_2045 profit_invest_2045
    python runner.py --mode batch --workers 4 --dry-run
    python runner.py --mode staggered
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import multiprocessing
import os
import re
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import psutil
import torch as th
import yaml

from assume import World
from assume.common.outputs import DatabaseMaintenance
from assume.scenario.loader_csv import (
    load_config_and_create_forecaster,
    load_scenario_folder,
    run_learning,
    run_staggered_learning,
    setup_world,
)

# Fallback defaults — overridden by runner.yaml at runtime
_DEFAULT_LOG_DIR = "logs"
_DEFAULT_WORKER_START_DELAY = 2
_DEFAULT_THREADS_PER_PROCESS = 1
_DEFAULT_INTEROP_THREADS = 1
_DEFAULT_MONITOR_INTERVAL = 5


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _make_run_id() -> str:
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S") + f"{now.microsecond // 1000:03d}"


def _calc_runtime(start: float, end: float) -> None:
    td = timedelta(seconds=end - start)
    days = td.days
    hours, rem = divmod(td.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Execution time: {td}")
    print(
        f"Execution time: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds"
    )


def format_elapsed(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _parse_elapsed_seconds(label: str) -> float | None:
    """Inverse of :func:`format_elapsed` — parse a compact duration label.

    Accepts tokens like ``"3h 5m"``, ``"12m"``, ``"45s"``, ``"2d 1h"``.
    Returns total seconds, or None if the label cannot be parsed.
    """
    if not label:
        return None
    total = 0.0
    found = False
    for token in label.split():
        m = re.match(r"^(\d+)([dhms])$", token)
        if not m:
            return None
        value = int(m.group(1))
        unit = m.group(2)
        total += value * {"d": 86400, "h": 3600, "m": 60, "s": 1}[unit]
        found = True
    return total if found else None


def _format_eta_compact(seconds: float) -> str:
    """Compact ETA label that drops trailing zero units.

    ``format_elapsed`` always includes seconds (e.g. ``"12m 0s"``); for the
    dashboard we prefer ``"12m"``. Round to the nearest second first.
    """
    seconds = int(round(seconds))
    td = timedelta(seconds=seconds)
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    # Drop seconds unless everything else is zero (keeps sub-minute ETAs).
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def build_task_name(scenario: str, study_case: str) -> str:
    return f"{scenario}__{study_case}"


def get_study_cases_from_config(config_path: str) -> list[str]:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return [k for k in config.keys() if not k.startswith("_")]


def is_staggered_case(inputs_path: str, scenario: str, study_case: str) -> bool:
    """Return True if the case has staggered_training.enabled = true in its learning_config."""
    config_path = os.path.join(inputs_path, scenario, "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    case_cfg = config.get(study_case, {})
    return bool(
        case_cfg.get("learning_config", {})
        .get("staggered_training", {})
        .get("enabled", False)
    )


def _patch_policy_base_scenario(
    scenario_data: dict,
    inputs_path: str,
    policy_base_scenario: str,
) -> None:
    """Re-root trained_policies_load_path to a different scenario folder.

    When policies were trained under a different scenario (e.g. profit_BAU) and
    you want to evaluate them against a new scenario (e.g. profit_INVEST), set
    policy_base_scenario to the folder that holds the trained policies. The
    relative path already stored in learning_config.trained_policies_load_path
    is preserved; only the base scenario folder changes.

    replace_paths() in setup_world skips absolute paths, so by converting to
    an absolute path here we prevent it from being re-rooted to the wrong folder.
    """
    learning_config = scenario_data.get("config", {}).get("learning_config") or {}
    raw_path = learning_config.get("trained_policies_load_path")
    if not raw_path:
        return
    abs_path = os.path.abspath(
        os.path.join(inputs_path, policy_base_scenario, raw_path)
    )
    learning_config["trained_policies_load_path"] = abs_path


def get_staggered_sim_ids(
    inputs_path: str, scenario: str, study_case: str
) -> list[str]:
    """Return the simulation IDs both worlds will write to the DB for a staggered case.

    Mirrors the path resolution in load_staggered_scenario. Each world writes under
    '{scenario_folder_name}_{study_case}' (matching the bug-fixed loader_csv behaviour).
    """
    config_path = os.path.join(inputs_path, scenario, "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    case_cfg = config.get(study_case, {})
    scenario_entries = (
        case_cfg.get("learning_config", {})
        .get("staggered_training", {})
        .get("scenarios", [])
    )
    if not scenario_entries:
        return [f"{scenario}_{study_case}"]
    primary_path = os.path.join(inputs_path, scenario)
    sim_ids = []
    for s in scenario_entries:
        p = Path(s["path"])
        if not p.is_absolute():
            p = (Path(primary_path) / p).resolve()
        sim_ids.append(f"{p.name}_{study_case}")
    return sim_ids


# ---------------------------------------------------------------------------
# tqdm capture — intercepts tqdm's stderr output in each worker process
# ---------------------------------------------------------------------------


class TqdmCapture(io.TextIOBase):
    """
    File-like object that replaces tqdm's output stream. It does three things:

    1. Writes a compact one-line status to a progress file that the monitor
       dashboard reads (updated on every tqdm refresh, throttled to ~1/s).
    2. Logs episode transition milestones to the case logger (e.g. every
       episode start, every 5th episode, and completion).
    3. Writes the full tqdm line to the case log file for post-hoc review.

    ASSUME has two tqdm bars (three for staggered):
      - Outer (run_learning / StaggeredTrainer.run):
        "Training Episodes:  40%|████ | 30/75"
        "Staggered Training Episodes:  40%|████ | 30/75"
      - Inner (world.async_run): "Training Episode 5 2025-03-15 14:00:  8%|█ | 2628000/31276800"
      - Chunk (StaggeredTrainer._run_episode_chunked):
        "Episode chunks (bau/inv):  20%|██ | 4/20"

    For staggered cases the compact status combines episode + chunk progress
    with ETAs, e.g. "Ep 5/75 ep 20% ETA 12m | all 6% ETA 3h".
    """

    _RE_OUTER = re.compile(
        r"(?:Staggered )?Training Episodes.*?(\d+)%\|.*?\|\s*(\d+)/(\d+)"
    )
    _RE_INNER = re.compile(
        r"(Training|Evaluation) Episode (\d+) ([\d-]+ [\d:]+).*?(\d+)%"
    )
    _RE_CHUNKS = re.compile(r"Episode chunks.*?(\d+)%\|.*?\|\s*(\d+)/(\d+)")
    _RE_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

    def __init__(self, logger: logging.Logger, progress_path: str, log_file_path: str):
        super().__init__()
        self._logger = logger
        self._progress_path = progress_path
        self._log_file_path = log_file_path
        self._last_logged_episode = 0
        self._last_mode = None
        self._total_episodes = None
        # Staggered / chunk tracking
        self._is_staggered = False
        self._chunk_current = 0
        self._chunk_total = 0
        self._episode_start_time = None
        self._first_episode_time = None
        # Non-staggered inner-bar tracking (for ETA on sim %)
        self._sim_pct = 0
        self._sim_start_time = None
        os.makedirs(os.path.dirname(progress_path), exist_ok=True)

    def writable(self):
        return True

    def write(self, buf: str) -> int:
        if not buf:
            return 0
        clean = self._RE_ANSI.sub("", buf).strip("\r\n ")
        if not clean:
            return len(buf)
        self._update_progress_file(clean)
        self._log_milestones(clean)
        self._write_to_log_file(clean)
        return len(buf)

    def flush(self):
        pass

    def _update_progress_file(self, line: str):
        try:
            compact = self._extract_compact(line)
            with open(self._progress_path, "w") as f:
                f.write(compact)
        except Exception:
            pass

    def _extract_compact(self, line: str) -> str:
        m = self._RE_OUTER.search(line)
        if m:
            pct, current, total = m.group(1), m.group(2), m.group(3)
            self._total_episodes = int(total)
            self._is_staggered = line.startswith("Staggered ")
            # On outer-bar refresh, (re)start the episode clock if the episode
            # changed. The outer bar fires once per episode in staggered mode.
            cur = int(current)
            if self._episode_start_time is None or cur != self._last_logged_episode:
                self._episode_start_time = time.time()
                if self._first_episode_time is None:
                    self._first_episode_time = self._episode_start_time
            if self._is_staggered:
                # Outer bar alone (between episodes) — show episode-level ETA.
                overall_pct = cur / self._total_episodes if self._total_episodes else 0
                full_eta = self._compute_eta(self._first_episode_time, overall_pct)
                return self._fmt_staggered(
                    cur, self._total_episodes, None, None, full_eta
                )
            return f"Ep {current}/{total} ({pct}%)"
        m = self._RE_INNER.search(line)
        if m:
            mode, ep_num, _sim_date, sim_pct = (
                m.group(1),
                m.group(2),
                m.group(3),
                m.group(4),
            )
            self._sim_pct = int(sim_pct) / 100.0
            if self._sim_start_time is None:
                self._sim_start_time = time.time()
            if mode == "Evaluation":
                eta = self._compute_eta(self._sim_start_time, self._sim_pct)
                eta_s = self._fmt_eta_short(eta)
                return f"Eval {ep_num} sim {sim_pct}% ETA {eta_s}"
            total_str = f"/{self._total_episodes}" if self._total_episodes else ""
            eta = self._compute_eta(self._sim_start_time, self._sim_pct)
            eta_s = self._fmt_eta_short(eta)
            return f"Ep {ep_num}{total_str} sim {sim_pct}% ETA {eta_s}"
        m = self._RE_CHUNKS.search(line)
        if m:
            pct, current, total = m.group(1), m.group(2), m.group(3)
            self._chunk_current = int(current)
            self._chunk_total = int(total)
            if self._episode_start_time is None:
                self._episode_start_time = time.time()
            if self._first_episode_time is None:
                self._first_episode_time = self._episode_start_time
            # Staggered: combine episode + chunk progress with ETAs.
            if self._total_episodes and self._last_logged_episode:
                ep_pct = (
                    self._chunk_current / self._chunk_total
                    if self._chunk_total
                    else 0.0
                )
                overall_pct = (
                    (self._last_logged_episode - 1 + ep_pct) / self._total_episodes
                    if self._total_episodes
                    else 0.0
                )
                ep_eta = self._compute_eta(self._episode_start_time, ep_pct)
                full_eta = self._compute_eta(self._first_episode_time, overall_pct)
                return self._fmt_staggered(
                    self._last_logged_episode,
                    self._total_episodes,
                    ep_pct,
                    ep_eta,
                    full_eta,
                )
            return f"chunks {current}/{total} ({pct}%)"
        if "episode chunks" in line.lower():
            # desc was long enough to truncate the bar — show what we have
            pct_match = re.search(r"(\d+)%", line)
            if pct_match:
                return f"chunks sim {pct_match.group(1)}%"
            return "chunks ..."
        if "evaluation" in line.lower() or "final" in line.lower():
            pct_match = re.search(r"(\d+)%", line)
            if pct_match:
                return f"Final eval sim {pct_match.group(1)}%"
        pct_match = re.search(r"(\d+)%", line)
        if pct_match:
            return f"sim {pct_match.group(1)}%"
        return line[:60]

    @staticmethod
    def _compute_eta(start_time: float | None, pct: float) -> float | None:
        """Extrapolate remaining seconds from elapsed time and fractional progress."""
        if start_time is None or pct is None or pct <= 0.001 or pct >= 0.999:
            return None
        elapsed = time.time() - start_time
        if elapsed < 1.0:
            return None
        return max(0.0, elapsed * (1.0 / pct - 1.0))

    @staticmethod
    def _fmt_eta_short(seconds: float | None) -> str:
        """Compact ETA label: '12m', '3h 5m', '45s', or '…' if unknown."""
        if seconds is None:
            return "…"
        return _format_eta_compact(seconds)

    def _fmt_staggered(
        self,
        ep_cur: int,
        ep_total: int,
        ep_pct: float | None,
        ep_eta: float | None,
        full_eta: float | None,
    ) -> str:
        """Build the compact staggered status line.

        Examples:
            Ep 5/75 ep 20% ETA 12m | all 6% ETA 3h
            Ep 5/75 ep … | all 6% ETA 3h          (chunk bar not yet seen)
        """
        if ep_pct is not None:
            ep_str = f"ep {int(round(ep_pct * 100))}% ETA {self._fmt_eta_short(ep_eta)}"
        else:
            ep_str = "ep …"
        overall = ep_cur / ep_total if ep_total else 0.0
        all_str = f"all {int(round(overall * 100))}% ETA {self._fmt_eta_short(full_eta)}"
        return f"Ep {ep_cur}/{ep_total} {ep_str} | {all_str}"

    def _log_milestones(self, line: str):
        m = self._RE_OUTER.search(line)
        if m:
            current = int(m.group(2))
            total = int(m.group(3))
            self._total_episodes = total
            if current != self._last_logged_episode:
                # New episode: reset chunk + episode clock.
                self._last_logged_episode = current
                self._chunk_current = 0
                self._episode_start_time = time.time()
                if self._first_episode_time is None:
                    self._first_episode_time = self._episode_start_time
                if current == 1 or current == total or current % 5 == 0:
                    self._logger.info(
                        "Training episode %d/%d (%s%%)", current, total, m.group(1)
                    )
            return
        m = self._RE_INNER.search(line)
        if m:
            mode = m.group(1)
            ep_num = int(m.group(2))
            if mode != self._last_mode:
                self._last_mode = mode
                if mode == "Evaluation":
                    self._logger.info("Evaluation episode %d started", ep_num)
                elif ep_num != self._last_logged_episode:
                    self._last_logged_episode = ep_num
                    total_str = (
                        f"/{self._total_episodes}" if self._total_episodes else ""
                    )
                    self._logger.info(
                        "Training episode %d%s started", ep_num, total_str
                    )
            elif mode == "Evaluation" and ep_num != self._last_logged_episode:
                self._last_logged_episode = ep_num
                self._logger.info("Evaluation episode %d started", ep_num)
        if "training finished" in line.lower():
            self._logger.info("Training finished — starting final evaluation")

    def _write_to_log_file(self, line: str):
        try:
            with open(self._log_file_path, "a", encoding="utf-8") as f:
                f.write(f"[tqdm] {line}\n")
        except Exception:
            pass


def patch_tqdm_for_case(
    task_name: str,
    logger: logging.Logger,
    log_dir: str,
    progress_dir: str,
):
    """Monkey-patch tqdm in the current worker process to write to TqdmCapture."""
    import tqdm as tqdm_module

    progress_path = os.path.join(progress_dir, task_name)
    log_file_path = os.path.join(log_dir, f"{task_name}.log")
    capture = TqdmCapture(logger, progress_path, log_file_path)

    _original_init = tqdm_module.tqdm.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs["file"] = capture
        kwargs.setdefault("ncols", 200)
        kwargs.setdefault("mininterval", 1.0)
        _original_init(self, *args, **kwargs)

    tqdm_module.tqdm.__init__ = _patched_init
    return capture


# ---------------------------------------------------------------------------
# Per-case logging setup
# ---------------------------------------------------------------------------


def _timing_log_dir(log_dir: str) -> str:
    return os.path.join(log_dir, "batch_timing")


def setup_case_logger(task_name: str, log_dir: str) -> logging.Logger:
    """
    Create a file-only logger for a single study case.
    No console handler — the live dashboard owns the terminal.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{task_name}.log")

    logger = logging.getLogger(f"assume.batch.{task_name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)

    assume_root = logging.getLogger("assume")
    assume_root.handlers.clear()
    assume_root.addHandler(fh)
    assume_root.setLevel(logging.DEBUG)
    assume_root.propagate = False

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.handlers.clear()
    warnings_logger.addHandler(fh)

    logging.getLogger().handlers.clear()
    return logger


# ---------------------------------------------------------------------------
# Batch worker (runs in a subprocess)
# ---------------------------------------------------------------------------


def run_single_case(
    db_uri: str,
    inputs_path: str,
    scenario: str,
    study_case: str,
    start_delay: float = 0,
    log_dir: str = _DEFAULT_LOG_DIR,
    progress_dir: str = "",
    shared_pids: dict | None = None,
    task_name: str | None = None,
    threads_per_process: int = _DEFAULT_THREADS_PER_PROCESS,
    interop_threads: int = _DEFAULT_INTEROP_THREADS,
    force_no_learning: bool = False,
    policy_base_scenario: str | None = None,
) -> dict:
    """Run a single ASSUME study case in an isolated process."""
    if start_delay > 0:
        time.sleep(start_delay)

    task_name = task_name or build_task_name(scenario, study_case)

    if shared_pids is not None:
        shared_pids[task_name] = os.getpid()

    th.set_num_threads(threads_per_process)
    th.set_num_interop_threads(interop_threads)

    logger = setup_case_logger(task_name, log_dir)

    log_path = os.path.join(log_dir, f"{task_name}.log")
    _log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = _log_file
    sys.stderr = _log_file

    patch_tqdm_for_case(task_name, logger, log_dir, progress_dir)

    import assume.common.utils as _assume_utils

    _assume_utils.interactive_input = lambda prompt, default="y": default

    result = {
        "task_name": task_name,
        "scenario": scenario,
        "study_case": study_case,
        "pid": os.getpid(),
        "status": "unknown",
        "start_time": None,
        "end_time": None,
        "elapsed_seconds": None,
        "error": None,
    }

    start = time.time()
    result["start_time"] = datetime.now().isoformat()

    try:
        logger.info("Starting task %s (PID %d)...", task_name, os.getpid())
        if force_no_learning:
            logger.info(
                "force_no_learning=True — loading scenario without RL, running with fixed policies"
            )
        if policy_base_scenario:
            logger.info(
                "policy_base_scenario=%s — policy load path re-rooted to that folder",
                policy_base_scenario,
            )
        if (
            is_staggered_case(inputs_path, scenario, study_case)
            and not force_no_learning
        ):
            logger.info("Staggered training detected — using run_staggered_learning")
            run_staggered_learning(
                inputs_path=inputs_path,
                scenario=scenario,
                study_case=study_case,
                db_uri=db_uri,
                export_csv_path="",
                log_level="INFO",
                verbose=True,
            )
        else:
            world = World(database_uri=db_uri, export_csv_path="")
            if force_no_learning or policy_base_scenario:
                world.scenario_data = load_config_and_create_forecaster(
                    inputs_path, scenario, study_case
                )
                if policy_base_scenario:
                    _patch_policy_base_scenario(
                        world.scenario_data, inputs_path, policy_base_scenario
                    )
                if force_no_learning:
                    setup_world(world=world, terminate_learning=True)
                else:
                    setup_world(world=world)
                    if world.learning_mode:
                        logger.info("Learning mode engaged — training agents")
                        run_learning(world)
            else:
                load_scenario_folder(
                    world,
                    inputs_path=inputs_path,
                    scenario=scenario,
                    study_case=study_case,
                )
                if world.learning_mode:
                    logger.info("Learning mode engaged — training agents")
                    run_learning(world)
            world.run()
        result["status"] = "completed"
        logger.info("Finished successfully")

    except KeyboardInterrupt:
        result["status"] = "interrupted"
        result["error"] = "KeyboardInterrupt"
        logger.warning("Interrupted by user")

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.exception("Failed with error: %s", e)

    finally:
        end = time.time()
        elapsed = end - start
        result["end_time"] = datetime.now().isoformat()
        result["elapsed_seconds"] = round(elapsed, 2)

        progress_path = os.path.join(progress_dir, task_name)
        try:
            if os.path.exists(progress_path):
                os.remove(progress_path)
        except Exception:
            pass

        td = timedelta(seconds=elapsed)
        days = td.days
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info("Wall-clock time: %sd %sh %sm %ss", days, hours, minutes, seconds)

        try:
            _log_file.flush()
            _log_file.close()
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Live resource monitor
# ---------------------------------------------------------------------------


class ProcessMonitor:
    """
    Reads training progress from per-worker status files and prints a
    refreshing dashboard using ANSI cursor-overwrite (multi-instance safe).
    """

    def __init__(
        self,
        shared_pids: dict,
        cases: list[str],
        run_id: str,
        poll_interval: float = _DEFAULT_MONITOR_INTERVAL,
        progress_dir: str = "",
    ):
        self.poll_interval = poll_interval
        self.progress_dir = progress_dir
        self._shared_pids = shared_pids
        self._cases = cases
        self._run_id = run_id
        self._statuses: dict[str, str] = {c: "waiting" for c in cases}
        self._peak_rss: dict[str, float] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._batch_start = time.time()
        self._system_peak_mem_pct = 0.0
        self._last_line_count = 0

    def mark_done(self, case_name: str, status: str):
        with self._lock:
            self._statuses[case_name] = status

    def start(self):
        os.makedirs(self.progress_dir, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def get_peaks(self) -> dict:
        with self._lock:
            return {
                "per_case": {
                    case: round(self._peak_rss.get(case, 0), 1) for case in self._cases
                },
                "system_peak_mem_pct": round(self._system_peak_mem_pct, 1),
            }

    def _read_progress(self, case_name: str) -> str:
        try:
            path = os.path.join(self.progress_dir, case_name)
            if os.path.exists(path):
                with open(path) as f:
                    return f.read().strip()
        except Exception:
            pass
        return ""

    @staticmethod
    def _parse_eta_seconds(progress_str: str) -> float | None:
        """Extract an ETA in seconds from a compact progress string.

        Looks for the last ``ETA <time>`` token (the full-run ETA in staggered
        strings, the only ETA in non-staggered strings). Returns None if no
        parseable ETA is present.
        """
        if not progress_str:
            return None
        # An ETA token runs from "ETA" up to the next "|" or end of string.
        matches = re.findall(r"ETA\s+([^|]+?)(?=\s*\||\s*$)", progress_str)
        if not matches:
            return None
        return _parse_elapsed_seconds(matches[-1].strip())

    def _loop(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(self.poll_interval)
            if self._stop_event.is_set():
                break
            self._print_dashboard()

    def _print_dashboard(self):
        try:
            current_pids = dict(self._shared_pids)
        except Exception:
            current_pids = {}

        sys_mem = psutil.virtual_memory()
        sys_mem_pct = sys_mem.percent
        sys_mem_used_gb = sys_mem.used / (1024**3)
        sys_mem_total_gb = sys_mem.total / (1024**3)

        with self._lock:
            statuses = dict(self._statuses)
            self._system_peak_mem_pct = max(self._system_peak_mem_pct, sys_mem_pct)

        # Dynamic column widths — handle long scenario names gracefully.
        case_w = max(35, min(max((len(c) for c in self._cases), default=35), 50))
        prog_w = 42
        W = case_w + 55

        rows = []
        for case in self._cases:
            pid = current_pids.get(case)
            status = statuses.get(case, "waiting")
            progress = ""
            rss_mb = 0.0

            if pid and status in ("waiting", "running"):
                with self._lock:
                    if self._statuses.get(case) == "waiting":
                        self._statuses[case] = "running"
                status = "running"
                progress = self._read_progress(case)
                try:
                    proc = psutil.Process(pid)
                    rss_mb = proc.memory_info().rss / (1024 * 1024)
                    with self._lock:
                        self._peak_rss[case] = max(self._peak_rss.get(case, 0), rss_mb)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            rows.append((case, pid, status, rss_mb, progress))

        elapsed = time.time() - self._batch_start
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        # Batch ETA: sum of remaining seconds across running cases.
        batch_eta_seconds: float | None = None
        for _, _, status, _, progress in rows:
            if status == "running":
                eta = self._parse_eta_seconds(progress)
                if eta is not None:
                    batch_eta_seconds = (batch_eta_seconds or 0) + eta
        batch_eta_str = (
            format_elapsed(batch_eta_seconds) if batch_eta_seconds is not None else "—"
        )

        lines = []
        lines.append(f"  {'─' * W}")
        lines.append(
            f"  ASSUME Batch Monitor [{self._run_id}]    "
            f"Elapsed: {elapsed_str}    "
            f"Batch ETA: {batch_eta_str}    "
            f"MEM: {sys_mem_used_gb:.1f} / {sys_mem_total_gb:.1f} GB ({sys_mem_pct:.1f}%)"
        )
        lines.append(f"  {'─' * W}")
        lines.append(
            f"  {'Case':<{case_w}} {'PID':>7}  {'Status':<10} {'RSS MB':>8}  {'Progress':<{prog_w}}"
        )
        lines.append(f"  {'─' * W}")

        for case, pid, status, rss_mb, progress in rows:
            pid_str = f"{pid:>7}" if pid else "    ..."
            # Truncate long case names with an ellipsis.
            if len(case) > case_w:
                case_str = case[: case_w - 1] + "…"
            else:
                case_str = case
            if status == "running":
                status_str, rss_str = "running", f"{rss_mb:7.1f}"
                prog_str = progress[:prog_w] if progress else ""
            elif status in ("completed", "done"):
                status_str, rss_str, prog_str = "done", "      —", "finished"
            elif status == "failed":
                status_str, rss_str, prog_str = "FAILED", "      —", "error"
            elif status == "waiting":
                status_str, rss_str, prog_str = "waiting", "      —", ""
            else:
                status_str, rss_str, prog_str = status, "      —", ""
            lines.append(
                f"  {case_str:<{case_w}} {pid_str}  {status_str:<10} {rss_str}  {prog_str}"
            )

        lines.append(f"  {'─' * W}")
        running = sum(1 for _, _, s, _, _ in rows if s == "running")
        done = sum(1 for _, _, s, _, _ in rows if s in ("completed", "done", "failed"))
        waiting = sum(1 for _, _, s, _, _ in rows if s == "waiting")
        lines.append(
            f"  Running: {running}  |  Waiting: {waiting}  |  "
            f"Done: {done}  |  Total: {len(rows)}"
        )

        total_lines = len(lines)
        buf = io.StringIO()
        if self._last_line_count > 0:
            buf.write(f"\033[{self._last_line_count}A")
        for line in lines:
            buf.write(f"\033[2K\r{line}\n")
        if self._last_line_count > total_lines:
            for _ in range(self._last_line_count - total_lines):
                buf.write("\033[2K\r\n")
        self._last_line_count = total_lines

        try:
            sys.stderr.write(buf.getvalue())
            sys.stderr.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(
    results: list[dict],
    monitor_peaks: dict | None = None,
    log_dir: str = _DEFAULT_LOG_DIR,
):
    print("\n" + "=" * 80)
    print("BATCH RUN SUMMARY")
    print("=" * 80)

    results_sorted = sorted(results, key=lambda r: r.get("elapsed_seconds", 0) or 0)
    print(f"  {'Task':<45} {'Status':<12} {'Wall-clock':>10}  {'Peak RSS':>10}  Notes")
    print("  " + "─" * 96)

    for r in results_sorted:
        case = r.get("task_name") or r.get("study_case", "—")
        status = r.get("status", "unknown")
        elapsed = (
            format_elapsed(r["elapsed_seconds"]) if r.get("elapsed_seconds") else "—"
        )
        notes = r.get("error", "") or ""
        peak_rss = "—"
        if monitor_peaks and case in monitor_peaks.get("per_case", {}):
            mb = monitor_peaks["per_case"][case]
            if mb > 0:
                peak_rss = f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"
        elif r.get("peak_rss_mb"):
            peak_rss = f"{r['peak_rss_mb']:.0f} MB"
        print(f"  {case:<45} {status:<12} {elapsed:>10}  {peak_rss:>10}  {notes}")

    print("  " + "─" * 96)

    completed = [r for r in results if r.get("status") == "completed"]
    failed = [r for r in results if r.get("status") == "failed"]

    if completed:
        fastest = min(completed, key=lambda r: r["elapsed_seconds"])
        slowest = max(completed, key=lambda r: r["elapsed_seconds"])
        print(
            f"\n  Fastest: {fastest.get('task_name') or fastest['study_case']} "
            f"({format_elapsed(fastest['elapsed_seconds'])})"
        )
        print(
            f"  Slowest: {slowest.get('task_name') or slowest['study_case']} "
            f"({format_elapsed(slowest['elapsed_seconds'])})"
        )
        if len(completed) > 1 and fastest["elapsed_seconds"]:
            speedup = slowest["elapsed_seconds"] / fastest["elapsed_seconds"]
            print(f"  Speed ratio (slowest/fastest): {speedup:.2f}x")

    if monitor_peaks:
        print(f"\n  System peak memory: {monitor_peaks['system_peak_mem_pct']}%")

    if failed:
        print(f"\n  {len(failed)} task(s) failed — check {log_dir}/<task_name>.log")

    print(f"\n  Log files: ./{log_dir}/{{task_name}}.log")
    print()


# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------


def run_single_mode(cfg: dict) -> None:
    """Execute a single scenario / study case."""
    scenario = cfg["scenario"]
    study_case = cfg["study_case"]
    db_uri = cfg["db_uri"]
    inputs_path = cfg["inputs_path"]

    print(f"\nSingle run: {scenario} / {study_case}")
    print(f"  inputs_path : {inputs_path}")
    print(f"  db_uri      : {db_uri}\n")

    force_no_learning: bool = cfg.get("force_no_learning", False)
    policy_base_scenario: str | None = cfg.get("policy_base_scenario") or None

    if cfg.get("dry_run"):
        print("  [DRY RUN] — no simulation will be executed")
        return

    start = time.time()
    world = World(database_uri=db_uri, export_csv_path="")
    if force_no_learning or policy_base_scenario:
        if policy_base_scenario:
            print(
                f"policy_base_scenario={policy_base_scenario!r} — policy load path re-rooted to that folder"
            )
        if force_no_learning:
            print(
                "force_no_learning=True — loading scenario without RL, running with fixed policies"
            )
        world.scenario_data = load_config_and_create_forecaster(
            inputs_path, scenario, study_case
        )
        if policy_base_scenario:
            _patch_policy_base_scenario(
                world.scenario_data, inputs_path, policy_base_scenario
            )
        if force_no_learning:
            setup_world(world=world, terminate_learning=True)
        else:
            setup_world(world=world)
            if world.learning_mode:
                print("Learning mode engaged — training agents")
                run_learning(world)
    else:
        load_scenario_folder(
            world, inputs_path=inputs_path, scenario=scenario, study_case=study_case
        )
        if world.learning_mode:
            print("Learning mode engaged — training agents")
            run_learning(world)
    world.run()
    _calc_runtime(start, time.time())
    print(f"Finished: {scenario} / {study_case}")
    print("Tino pai, ē hoa!!!")


def run_batch_mode(cfg: dict) -> None:
    """Execute parallel study cases across one or more scenarios."""
    db_uri = cfg["db_uri"]
    inputs_path = cfg["inputs_path"]
    scenarios_to_run: list[str] = cfg["scenarios"]
    requested_cases: list[str] = cfg.get("cases") or []
    max_workers: int | None = cfg.get("workers")
    log_dir: str = cfg.get("log_dir", _DEFAULT_LOG_DIR)
    monitor_interval: float = cfg.get("monitor_interval", _DEFAULT_MONITOR_INTERVAL)
    dry_run: bool = cfg.get("dry_run", False)
    use_monitor: bool = cfg.get("monitor", True)
    timing_log: str | None = cfg.get("timing_log")
    worker_start_delay: int = cfg.get("worker_start_delay", _DEFAULT_WORKER_START_DELAY)
    threads_per_process: int = cfg.get(
        "threads_per_process", _DEFAULT_THREADS_PER_PROCESS
    )
    interop_threads: int = cfg.get("interop_threads", _DEFAULT_INTEROP_THREADS)
    force_no_learning: bool = cfg.get("force_no_learning", False)
    policy_base_scenario: str | None = cfg.get("policy_base_scenario") or None

    run_id = _make_run_id()

    # Resolve cases per scenario
    scenario_cases: dict[str, list[str]] = {}
    scenario_configs: dict[str, str] = {}

    for scenario_name in scenarios_to_run:
        config_path = os.path.join(inputs_path, scenario_name, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        all_cases = get_study_cases_from_config(config_path)
        scenario_configs[scenario_name] = config_path
        if requested_cases:
            invalid = [c for c in requested_cases if c not in all_cases]
            if invalid:
                raise ValueError(
                    f"Cases not found in {scenario_name}: {invalid}\n"
                    f"Available: {all_cases}"
                )
            scenario_cases[scenario_name] = list(requested_cases)
        else:
            scenario_cases[scenario_name] = list(all_cases)

    tasks_to_run = [
        {
            "scenario": s,
            "study_case": c,
            "task_name": build_task_name(s, c),
        }
        for s in scenarios_to_run
        for c in scenario_cases[s]
    ]

    if not tasks_to_run:
        raise RuntimeError("No tasks found to run.")

    workers = max_workers or len(tasks_to_run)
    progress_dir = os.path.join(log_dir, ".progress", run_id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(progress_dir, exist_ok=True)

    timing_path = timing_log or os.path.join(
        _timing_log_dir(log_dir), f"batch_timing_{run_id}.json"
    )
    os.makedirs(os.path.dirname(timing_path), exist_ok=True)

    print("\nASSUME Batch Runner")
    print(f"  Run ID:    {run_id}")
    print(f"  Inputs:    {inputs_path}")
    print(f"  Scenarios: {', '.join(scenarios_to_run)}")
    print(f"  Database:  {db_uri}")
    print(f"  Workers:   {workers}")
    print(f"  Log dir:   ./{log_dir}/")
    print(f"  Timing:    {timing_path}")
    print(
        f"  Monitor:   {'disabled' if not use_monitor else f'every {monitor_interval}s'}"
    )
    if policy_base_scenario:
        print(f"  Policies:  {policy_base_scenario} (policy_base_scenario override)")
    print(f"  Tasks:     {len(tasks_to_run)} total")
    for s in scenarios_to_run:
        print(
            f"    - {s}: {len(scenario_cases[s])} case(s), config = {scenario_configs[s]}"
        )

    if dry_run:
        print("\n  [DRY RUN] — no tasks will be executed")
        for task in tasks_to_run:
            try:
                task_type = (
                    "staggered"
                    if is_staggered_case(
                        inputs_path, task["scenario"], task["study_case"]
                    )
                    else "regular"
                )
            except Exception:
                task_type = "unknown"
            print(f"    - {task['task_name']}  [{task_type}]")
        return

    print("\nCleaning up old scenario data...")
    maintenance = DatabaseMaintenance(db_uri)
    sim_ids = []
    for task in tasks_to_run:
        if (
            is_staggered_case(inputs_path, task["scenario"], task["study_case"])
            and not force_no_learning
        ):
            sim_ids.extend(
                get_staggered_sim_ids(inputs_path, task["scenario"], task["study_case"])
            )
        else:
            sim_ids.append(f"{task['scenario']}_{task['study_case']}")
    maintenance.delete_simulations(sim_ids)
    print(f"  Cleaned {len(sim_ids)} simulation(s) from database")

    mp_manager = multiprocessing.Manager() if use_monitor else None
    shared_pids = mp_manager.dict() if mp_manager else None
    monitor = (
        ProcessMonitor(
            shared_pids=shared_pids,
            cases=[t["task_name"] for t in tasks_to_run],
            run_id=run_id,
            poll_interval=monitor_interval,
            progress_dir=progress_dir,
        )
        if use_monitor
        else None
    )

    print(f"\nLaunching {len(tasks_to_run)} task(s)...\n")
    if use_monitor:
        print(f"  Tip: tail -f {log_dir}/<task_name>.log to follow a specific task\n")

    batch_start = time.time()
    results: list[dict] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {}
        for i, task in enumerate(tasks_to_run):
            future = executor.submit(
                run_single_case,
                db_uri,
                inputs_path,
                task["scenario"],
                task["study_case"],
                i * worker_start_delay,
                log_dir,
                progress_dir,
                shared_pids,
                task["task_name"],
                threads_per_process,
                interop_threads,
                force_no_learning,
                policy_base_scenario,
            )
            future_to_task[future] = task

        if use_monitor and monitor:
            monitor.start()

        try:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                task_name = task["task_name"]
                try:
                    result = future.result()
                    results.append(result)
                    status = result.get("status", "unknown")
                    if use_monitor and monitor:
                        monitor.mark_done(task_name, status)
                    print(
                        f"\n[{task_name}] {status.upper()} "
                        f"in {format_elapsed(result.get('elapsed_seconds') or 0)}"
                    )
                    if result.get("error"):
                        print(f"  Error: {result['error']}")
                except Exception as e:
                    failed_result = {
                        "task_name": task_name,
                        "scenario": task["scenario"],
                        "study_case": task["study_case"],
                        "status": "failed",
                        "elapsed_seconds": None,
                        "error": str(e),
                    }
                    results.append(failed_result)
                    if use_monitor and monitor:
                        monitor.mark_done(task_name, "failed")
                    print(f"\n[{task_name}] FAILED with unexpected error: {e}")

        except KeyboardInterrupt:
            print("\n\nKeyboardInterrupt received — shutting down workers...")
            executor.shutdown(wait=False, cancel_futures=True)
            raise

        finally:
            if use_monitor and monitor:
                monitor.stop()

    batch_end = time.time()
    batch_elapsed = round(batch_end - batch_start, 2)
    monitor_peaks = monitor.get_peaks() if use_monitor and monitor else None

    timing_payload = {
        "run_id": run_id,
        "inputs_path": inputs_path,
        "scenarios": scenarios_to_run,
        "tasks": tasks_to_run,
        "batch_start": datetime.fromtimestamp(batch_start).isoformat(),
        "batch_end": datetime.fromtimestamp(batch_end).isoformat(),
        "batch_elapsed_seconds": batch_elapsed,
        "max_workers": workers,
        "results": results,
        "monitor_peaks": monitor_peaks,
    }

    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_payload, f, indent=2)

    print_summary(results, monitor_peaks, log_dir=log_dir)
    print(f"Saved timing JSON: {timing_path}")

    try:
        for child in Path(progress_dir).iterdir():
            if child.is_file():
                child.unlink(missing_ok=True)
        Path(progress_dir).rmdir()
        parent = Path(progress_dir).parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except Exception:
        pass

    if mp_manager:
        mp_manager.shutdown()

    failed_count = sum(1 for r in results if r.get("status") == "failed")
    if failed_count:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Config loading and CLI
# ---------------------------------------------------------------------------


def load_config(args: argparse.Namespace) -> dict:
    """Load runner.yaml and apply CLI overrides on top."""
    config_file = args.config or "runner.yaml"
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # CLI overrides — only applied when explicitly provided
    if args.mode is not None:
        cfg["mode"] = args.mode
    if args.inputs_path is not None:
        cfg["inputs_path"] = args.inputs_path
    if args.db_uri is not None:
        cfg["db_uri"] = args.db_uri
    if args.scenario is not None:
        cfg["scenario"] = args.scenario
    if args.study_case is not None:
        cfg["study_case"] = args.study_case
    if args.scenarios is not None:
        cfg["scenarios"] = args.scenarios
    if args.cases is not None:
        cfg["cases"] = args.cases
    if args.workers is not None:
        cfg["workers"] = args.workers
    if args.log_dir is not None:
        cfg["log_dir"] = args.log_dir
    if args.monitor_interval is not None:
        cfg["monitor_interval"] = args.monitor_interval
    if args.timing_log is not None:
        cfg["timing_log"] = args.timing_log
    if args.dry_run:
        cfg["dry_run"] = True
    if args.no_monitor:
        cfg["monitor"] = False
    if args.force_no_learning:
        cfg["force_no_learning"] = True
    if args.policy_base_scenario is not None:
        cfg["policy_base_scenario"] = args.policy_base_scenario

    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified ASSUME runner — dispatches to single or batch mode (staggered cases auto-detected)",
        epilog=(
            "Examples:\n"
            "  python runner.py\n"
            "  python runner.py --mode single --scenario srmc_bau_2045 --study-case run_seed_e\n"
            "  python runner.py --mode batch --scenarios profit_bau_2045 profit_bau_2045_staggered\n"
            "  python runner.py --mode batch --workers 4 --dry-run\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML (default: runner.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default=None,
        help="Execution mode — overrides runner.yaml",
    )
    parser.add_argument("--inputs-path", default=None, help="Path to inputs folder")
    parser.add_argument("--db-uri", default=None, help="Database URI")

    # single mode
    parser.add_argument(
        "--scenario",
        default=None,
        help="Scenario folder name (single mode)",
    )
    parser.add_argument(
        "--study-case",
        dest="study_case",
        default=None,
        help="Study case name (single mode)",
    )

    # batch mode
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="One or more scenario folders (batch mode)",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Specific study cases to run (batch mode; default: all)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Max parallel workers (batch mode)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="List tasks without executing (batch mode)",
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        default=False,
        help="Disable the live resource dashboard (batch mode)",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=None,
        help="Dashboard refresh interval in seconds (batch mode)",
    )
    parser.add_argument(
        "--timing-log",
        default=None,
        help="Path to save JSON timing results (batch mode)",
    )

    # shared
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for log files",
    )
    parser.add_argument(
        "--force-no-learning",
        action="store_true",
        default=False,
        help="Disable RL training for all cases — run with fixed/SRMC policies",
    )
    parser.add_argument(
        "--policy-base-scenario",
        default=None,
        help=(
            "Re-root trained_policies_load_path to this scenario folder. "
            "Use when policies were trained under a different scenario "
            "(e.g. profit_BAU) and you want to evaluate them against another "
            "(e.g. profit_INVEST). The relative path in each case's config is "
            "preserved; only the base folder changes."
        ),
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args)

    mode = cfg.get("mode")
    if not mode:
        raise ValueError("'mode' must be set in runner.yaml or via --mode")

    if mode == "single":
        for key in ("scenario", "study_case", "db_uri", "inputs_path"):
            if not cfg.get(key):
                raise ValueError(f"'{key}' is required for mode: single")
        run_single_mode(cfg)

    elif mode == "batch":
        if not cfg.get("scenarios"):
            raise ValueError("'scenarios' is required for mode: batch")
        if not cfg.get("db_uri"):
            raise ValueError("'db_uri' is required for mode: batch")
        run_batch_mode(cfg)

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose from: single, batch")


if __name__ == "__main__":
    main()
