# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Debug harness for the staggered RL training divergence.

Runs a short repro in two modes — staggered (``short_CF`` paired with
``short_UC``) and single-world (``short_CF`` alone) — each with per-step
diagnostics enabled (see ``assume.reinforcement_learning.debug_diagnostics``),
then prints a side-by-side summary so you can localise the divergence.

The bisecting logic:

* If the single-world run also shows the 10x reward / climbing critic loss, the
  bug is in the **reward change** (runs in both modes).
* If single-world is clean and only staggered diverges, the bug is in the
  **staggered masking / centralized-critic** path (foreign-unit contamination).

Usage (from the repo root, with the venv active)::

    # Run both experiments and compare
    python tools/debug_rl.py bisect

    # Or run one experiment on its own
    python tools/debug_rl.py run --study e        --debug-dir rl_debug/staggered
    python tools/debug_rl.py run --study e_single --debug-dir rl_debug/single

    # Compare already-collected CSV directories
    python tools/debug_rl.py compare rl_debug/staggered rl_debug/single

Each run writes CSVs (``rewards.csv``, ``critic_updates.csv``,
``actor_updates.csv``, ``step_masks.csv``) into its ``--debug-dir``.
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUTS = os.path.join(REPO_ROOT, "inputs")
DEFAULT_SCENARIO = "short_CF"
DEFAULT_DB = "sqlite:///local_db/assume_debug.db"


def _clear_debug_dir(debug_dir):
    """Remove stale diagnostic CSVs so a re-run starts clean (the sink appends).

    Only deletes the diagnostic ``*.csv`` files this harness produces; it never
    removes the directory itself or any other files, so pointing ``--debug-dir``
    at a populated folder cannot destroy unrelated data.
    """
    if not os.path.isdir(debug_dir):
        os.makedirs(debug_dir, exist_ok=True)
        return
    for name in (
        "rewards",
        "critic_updates",
        "actor_updates",
        "step_masks",
        "critic_grads",
    ):
        path = os.path.join(debug_dir, f"{name}.csv")
        if os.path.exists(path):
            os.remove(path)
            print(f"[debug_rl] cleared stale {path}")


def _run_worker(inputs_path, scenario, study_case, debug_dir, db_uri):
    """In-process worker: train one scenario with diagnostics enabled."""
    debug_dir = os.path.abspath(debug_dir)
    _clear_debug_dir(debug_dir)
    os.environ["ASSUME_RL_DEBUG"] = debug_dir

    # Imported here so ASSUME_RL_DEBUG is set before anything reads it.
    import yaml

    from assume import World
    from assume.scenario.loader_csv import (
        load_scenario_folder,
        run_learning,
        run_staggered_learning,
    )

    config_yaml = os.path.join(inputs_path, scenario, "config.yaml")
    with open(config_yaml, encoding="utf-8") as f:
        study_cfg = yaml.safe_load(f).get(study_case, {})
    staggered = (study_cfg.get("learning_config") or {}).get("staggered_training") or {}

    if staggered.get("enabled"):
        print(f"[debug_rl] staggered run: {scenario}/{study_case} -> {debug_dir}")
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
        print(f"[debug_rl] single-world run: {scenario}/{study_case} -> {debug_dir}")
        world = World(database_uri=db_uri, export_csv_path="")
        load_scenario_folder(
            world,
            inputs_path=inputs_path,
            scenario=scenario,
            study_case=study_case,
        )
        if world.learning_mode:
            run_learning(world)
        world.run()


def _spawn_run(inputs_path, scenario, study_case, debug_dir, db_uri):
    """Run a worker in a fresh subprocess so the two experiments stay isolated."""
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "run",
        "--inputs",
        inputs_path,
        "--scenario",
        scenario,
        "--study",
        study_case,
        "--debug-dir",
        debug_dir,
        "--db",
        db_uri,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _summarise(debug_dir, label):
    import pandas as pd

    print(f"\n================ {label}  ({debug_dir}) ================")

    rewards_path = os.path.join(debug_dir, "rewards.csv")
    if os.path.exists(rewards_path):
        df = pd.read_csv(rewards_path)
        per_ep = df.groupby("episode").agg(
            reward_mean=("reward", "mean"),
            reward_absmax=("reward", lambda s: s.abs().max()),
            regret_mean=("regret", "mean"),
            regret_max=("regret", "max"),
            active_frac=("active", "mean"),
        )
        print("\nrewards by episode:")
        print(per_ep.to_string())
    else:
        print(f"  (no rewards.csv at {rewards_path})")

    critic_path = os.path.join(debug_dir, "critic_updates.csv")
    if os.path.exists(critic_path):
        df = pd.read_csv(critic_path).sort_values("n_updates")
        n = len(df)
        head = df.head(max(1, n // 10))
        tail = df.tail(max(1, n // 10))
        print("\ncritic loss / target-Q drift (first vs last 10% of updates):")
        print(
            f"  critic_loss:      {head['critic_loss'].mean():.4g}"
            f"  ->  {tail['critic_loss'].mean():.4g}"
        )
        print(
            f"  target_q_absmean: {head['target_q_absmean'].mean():.4g}"
            f"  ->  {tail['target_q_absmean'].mean():.4g}"
        )
        print(
            f"  current_q_absmean:{head['current_q_absmean'].mean():.4g}"
            f"  ->  {tail['current_q_absmean'].mean():.4g}"
        )

    masks_path = os.path.join(debug_dir, "step_masks.csv")
    if os.path.exists(masks_path):
        df = pd.read_csv(masks_path)
        print("\nforeign-unit contamination (centralized critic input):")
        print(
            f"  mean fully-inactive agents/step: {df['n_fully_inactive'].mean():.2f}"
            f"  of {int(df['n_agents'].max())}"
        )
        print(
            f"  mean |action| inactive agents:   {df['inactive_action_absmean'].mean():.4g}"
        )
        print(
            f"  mean |action| active agents:     {df['active_action_absmean'].mean():.4g}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="train one scenario with diagnostics")
    p_run.add_argument("--inputs", default=DEFAULT_INPUTS)
    p_run.add_argument("--scenario", default=DEFAULT_SCENARIO)
    p_run.add_argument("--study", required=True)
    p_run.add_argument("--debug-dir", required=True)
    p_run.add_argument("--db", default=DEFAULT_DB)

    p_bis = sub.add_parser("bisect", help="run staggered + single-world and compare")
    p_bis.add_argument("--inputs", default=DEFAULT_INPUTS)
    p_bis.add_argument("--scenario", default=DEFAULT_SCENARIO)
    p_bis.add_argument("--staggered-study", default="e")
    p_bis.add_argument("--single-study", default="e_single")
    p_bis.add_argument("--out", default=os.path.join(REPO_ROOT, "rl_debug"))
    p_bis.add_argument("--db", default=DEFAULT_DB)

    p_cmp = sub.add_parser("compare", help="summarise two debug dirs")
    p_cmp.add_argument("staggered_dir")
    p_cmp.add_argument("single_dir")

    args = parser.parse_args()

    if args.cmd == "run":
        _run_worker(args.inputs, args.scenario, args.study, args.debug_dir, args.db)
    elif args.cmd == "bisect":
        staggered_dir = os.path.join(args.out, "staggered")
        single_dir = os.path.join(args.out, "single")
        _spawn_run(
            args.inputs, args.scenario, args.staggered_study, staggered_dir, args.db
        )
        _spawn_run(args.inputs, args.scenario, args.single_study, single_dir, args.db)
        _summarise(staggered_dir, "STAGGERED")
        _summarise(single_dir, "SINGLE-WORLD")
    elif args.cmd == "compare":
        _summarise(args.staggered_dir, "STAGGERED")
        _summarise(args.single_dir, "SINGLE-WORLD")


if __name__ == "__main__":
    main()
