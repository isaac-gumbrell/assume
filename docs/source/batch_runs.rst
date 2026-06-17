.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

Batch runs on remote servers
============================

The ``runner.py`` script at the repository root executes one or more ASSUME
study cases. In ``--mode batch`` it runs cases in parallel via a
``ProcessPoolExecutor`` and prints a live dashboard to stderr showing per-case
progress, memory usage, and estimated time to completion.

This page covers how to keep a batch run alive across SSH disconnects on a
locked-down remote server (e.g. RHEL on AWS) where you cannot install a
process supervisor.

.. contents::
   :local:
   :depth: 2

Why not just run it in the foreground?
--------------------------------------

A long batch (especially reinforcement-learning cases) can run for hours or
days. If your SSH connection drops, the shell receives ``SIGHUP`` and the
whole batch — including every worker subprocess — is killed. The two
workarounds below detach the run from your terminal so it survives a
disconnect.

Option 1: tmux (recommended)
----------------------------

``tmux`` is a terminal multiplexer that keeps a shell session alive after you
disconnect. It is present on almost every RHEL install and is rarely on a
locked-down package blocklist.

Start the batch in a detached session::

    tmux new -s assume 'python runner.py --mode batch 2>&1'

Reconnect from any later SSH session::

    tmux attach -t assume

From inside the session:

* **Detach** (leave it running): ``Ctrl-b`` then ``d``.
* **Cancel the whole batch**: ``Ctrl-C`` (this stops every worker at once —
  there is no per-job cancellation in this mode).
* **Scroll back**: ``Ctrl-b`` then ``[``, then use PageUp/PageDown; ``q`` to
  exit scroll mode.

To list sessions: ``tmux ls``. To kill a session outright:
``tmux kill-session -t assume``.

Option 2: nohup (no interactivity)
----------------------------------

If ``tmux`` is unavailable, ``nohup`` detaches the process from the terminal
and redirects output to a file. You can watch the file from anywhere but you
cannot interact with the run::

    nohup python runner.py --mode batch --no-monitor > run.log 2>&1 &
    disown

Follow the log from any session::

    tail -f run.log

Because the live dashboard uses ANSI cursor-overwrite codes that look messy in
a plain log file, pass ``--no-monitor`` and instead tail the per-case logs::

    tail -f logs/<scenario>__<study_case>.log

To cancel the batch: ``pkill -f runner.py`` (kills every worker).

Option 3: tmux + tee (live dashboard + log file)
------------------------------------------------

If you want both the interactive dashboard *and* a persistent log file::

    tmux new -s assume 'python runner.py --mode batch 2>&1 | tee run.log'

Attach for the live view; ``tail -f run.log`` for read-only monitoring from
elsewhere. ``Ctrl-C`` inside the session cancels the batch.

Reading the live dashboard
--------------------------

The batch monitor refreshes every few seconds and looks like this::

  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ ASSUME Batch Monitor [20260617_120000123]   Elapsed: 0:05:12   Batch ETA: 3h 2m  │
  │                                                                                  │
  │ Case                                PID    Status     RSS MB   Progress          │
  │ example_01a__tiny                  12345   running     450.2   Ep 3/10 sim 45%…  │
  │ example_01a__base                  12346   waiting        —                      │
  │ …                                                                                 │
  │ Running: 1  |  Waiting: 1  |  Done: 0  |  Total: 2                                   │
  └──────────────────────────────────────────────────────────────────────────────────┘

Columns:

* **Case** — ``<scenario>__<study_case>``. Long names are truncated with ``…``.
* **PID** — worker process ID (useful for ``top`` / ``htop``).
* **Status** — ``waiting``, ``running``, ``done``, ``FAILED``.
* **RSS MB** — resident memory of the worker process.
* **Progress** — compact tqdm status. For RL cases this includes an ETA, e.g.
  ``Ep 3/10 sim 45% ETA 12m``. For staggered cases it shows both episode and
  overall progress: ``Ep 5/75 ep 20% ETA 12m | all 6% ETA 3h``.

The header line shows the total **Elapsed** time since the batch started and a
**Batch ETA** which sums the remaining ETAs of all running cases. Queued cases
do not contribute a numeric ETA (they show as ``waiting``).

Per-case log files
------------------

Every worker writes a full log to ``logs/<scenario>__<study_case>.log``. This
includes every tqdm line (with ETAs) and any tracebacks. Tail it for detail
the dashboard omits::

    tail -f logs/example_01a__tiny.log

A JSON timing summary is written to ``logs/batch_timing/batch_timing_<run_id>.json``
when the batch finishes.

Configuration
-------------

All batch behaviour is controlled by ``runner.yaml`` and can be overridden on
the CLI. See ``python runner.py --help`` for the full option list. Key fields:

* ``workers`` — max parallel workers (default: one per task).
* ``monitor_interval`` — dashboard refresh seconds (default: 5).
* ``monitor`` — set ``false`` (or pass ``--no-monitor``) to disable the
  dashboard entirely.
* ``log_dir`` — where per-case logs and timing JSON are written.
* ``threads_per_process`` / ``interop_threads`` — PyTorch thread caps per
  worker (default: 1 each) to avoid thread explosion in parallel runs.

Surviving a server reboot
-------------------------

Neither ``tmux`` nor ``nohup`` survives a server reboot. If your AWS instance
may reboot, wrap the launch in a ``systemd`` user service so the batch
restarts automatically. A minimal unit (``~/.config/systemd/user/assume-batch.service``)::

    [Unit]
    Description=ASSUME batch runner
    After=network.target

    [Service]
    Type=simple
    WorkingDirectory=%h/Code/assume
    ExecStart=%h/Code/assume/.venv/bin/python runner.py --mode batch
    Restart=on-failure
    RestartSec=10

    [Install]
    WantedBy=default.target

Enable and start::

    systemctl --user daemon-reload
    systemctl --user enable --now assume-batch.service

View output via ``journalctl --user -u assume-batch -f``. Note that with
``systemd`` the live ANSI dashboard goes to the journal; use ``journalctl`` or
the per-case log files to follow progress.
