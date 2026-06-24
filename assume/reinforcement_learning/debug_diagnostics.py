# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Opt-in per-step diagnostics for debugging RL training divergence.

This module is a *debugging aid* for investigating learning instabilities
(e.g. exploding reward/regret, diverging critic loss, growing gradients). It is
completely inert unless the ``ASSUME_RL_DEBUG`` environment variable is set, so
production and CI runs are unaffected and pay zero overhead.

Usage
-----
Set the environment variable to a directory (or ``1`` to use ``./rl_debug``)::

    # Windows PowerShell
    $env:ASSUME_RL_DEBUG = "rl_debug/staggered"
    python run.py

It writes one CSV per "table" into that directory. Each ``log_row`` call appends
a row; the header is written from the first row's keys, so every row for a given
table must carry the same set of keys.

Tables emitted by the current hooks:

``rewards``
    One row per unit per market period: ``episode, start, unit_id, reward,
    regret, profit, active``. Use this to confirm reward magnitude (the "10x"
    symptom) and watch regret blow up over training.

``critic_updates``
    One row per gradient step per agent: ``n_updates, agent, active_count,
    target_q_absmean, current_q_absmean, critic_loss``. Use this to see whether
    ``target_q`` (and therefore the critic loss) is drifting upward.

``actor_updates``
    One row per (delayed) gradient step per agent: ``n_updates, agent,
    active_count, actor_loss``.

``step_masks``
    One row per gradient step: ``n_updates, n_agents, n_fully_inactive,
    inactive_action_absmean, active_action_absmean``. ``n_fully_inactive`` and
    ``inactive_action_absmean`` quantify foreign (forced-off) units whose stale
    actions still feed the centralized critic input — the suspected staggered
    contamination leak.
"""

import csv
import os
import threading

_ENV_VAR = "ASSUME_RL_DEBUG"
_FALSEY = {"", "0", "false", "False", "no", "off"}

_lock = threading.Lock()
_instance = None
_initialised = False


class _Diagnostics:
    """Lazily-opened, append-mode CSV multiplexer keyed by table name."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self._files: dict[str, object] = {}
        self._writers: dict[str, csv.DictWriter] = {}
        self._write_lock = threading.Lock()

    def log(self, table: str, row: dict) -> None:
        with self._write_lock:
            writer = self._writers.get(table)
            if writer is None:
                path = os.path.join(self.out_dir, f"{table}.csv")
                is_new = not os.path.exists(path) or os.path.getsize(path) == 0
                handle = open(path, "a", newline="", encoding="utf-8")
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                if is_new:
                    writer.writeheader()
                self._files[table] = handle
                self._writers[table] = writer
            writer.writerow(row)
            self._files[table].flush()


def _resolve_out_dir(value: str) -> str:
    if value in {"1", "true", "True", "yes", "on"}:
        return os.path.join(os.getcwd(), "rl_debug")
    return value


def get_diagnostics():
    """Return the active diagnostics sink, or ``None`` when disabled.

    The environment variable is read exactly once per process. Use
    :func:`configure` to point subsequent logging at a different directory (e.g.
    when running several experiments in one process).
    """
    global _instance, _initialised
    if not _initialised:
        with _lock:
            if not _initialised:
                value = os.environ.get(_ENV_VAR)
                if value is not None and value not in _FALSEY:
                    _instance = _Diagnostics(_resolve_out_dir(value))
                _initialised = True
    return _instance


def configure(out_dir: str | None) -> None:
    """Explicitly (re)point diagnostics at ``out_dir`` (or disable with ``None``).

    Overrides the environment variable. Useful for harnesses that run multiple
    experiments in a single process and want each in its own directory.
    """
    global _instance, _initialised
    with _lock:
        _instance = _Diagnostics(out_dir) if out_dir else None
        _initialised = True


def is_enabled() -> bool:
    """Cheap guard so hot paths can skip building row dicts when disabled."""
    return get_diagnostics() is not None


def log_row(table: str, **row) -> None:
    """Append ``row`` to ``table``'s CSV if diagnostics are enabled, else no-op."""
    sink = get_diagnostics()
    if sink is not None:
        sink.log(table, row)
