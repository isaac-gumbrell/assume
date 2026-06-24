# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Ad-hoc analysis: compare episode-1 diagnostics between staggered and single runs.

Usage: python tools/analyse_ep1.py
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl_debug"
)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def load(run):
    d = os.path.join(ROOT, run)
    out = {}
    for t in (
        "rewards",
        "critic_updates",
        "actor_updates",
        "step_masks",
        "critic_grads",
    ):
        p = os.path.join(d, f"{t}.csv")
        out[t] = pd.read_csv(p) if os.path.exists(p) else None
    return out


def summarise(run, data):
    print(f"\n{'=' * 70}\n{run.upper()}\n{'=' * 70}")

    rw = data["rewards"]
    if rw is not None:
        print(f"\n[rewards] episodes present: {sorted(rw['episode'].unique())}")
        ep1 = rw[rw["episode"] == 1]
        print(
            f"[rewards] episode==1 rows: {len(ep1)}, units: {ep1['unit_id'].nunique()}"
        )
        if len(ep1):
            print("\nepisode==1 reward/regret/profit stats:")
            print(
                ep1[["reward", "regret", "profit", "active"]]
                .describe(percentiles=[0.5, 0.99])
                .to_string()
            )
            print(f"\n[rewards] active fraction (ep1): {ep1['active'].mean():.4f}")
            print(f"[rewards] |reward| max (ep1): {ep1['reward'].abs().max():.4g}")
            print(f"[rewards] regret max (ep1): {ep1['regret'].max():.4g}")
            # top regret contributors
            top = (
                ep1.groupby("unit_id")["regret"]
                .max()
                .sort_values(ascending=False)
                .head(8)
            )
            print("\ntop-8 units by max regret (ep1):")
            print(top.to_string())

    cu = data["critic_updates"]
    if cu is not None and len(cu):
        # episode-1 updates: from the first n_updates up to where rewards ep1 ends is
        # unknown here, so report the EARLY portion (first 6 = one flush of gradient_steps)
        # plus overall trend.
        cu = cu.sort_values("n_updates")
        first_step = cu["n_updates"].min()
        ep1_like = cu[cu["n_updates"] <= first_step + 5]  # first gradient flush
        print(
            f"\n[critic] n_updates range: {cu['n_updates'].min()}..{cu['n_updates'].max()}"
        )
        print("\ncritic_loss / target_q_absmean — first flush (<= first+5):")
        print(
            ep1_like.groupby("n_updates")[
                ["critic_loss", "target_q_absmean", "current_q_absmean"]
            ]
            .mean()
            .to_string()
        )
        n = len(cu)
        head = cu.head(max(1, n // 10))
        tail = cu.tail(max(1, n // 10))
        print("\ncritic trend (first 10% -> last 10% of all updates):")
        print(
            f"  critic_loss:       {head['critic_loss'].mean():.4g} -> {tail['critic_loss'].mean():.4g}"
        )
        print(
            f"  target_q_absmean:  {head['target_q_absmean'].mean():.4g} -> {tail['target_q_absmean'].mean():.4g}"
        )
        print(
            f"  current_q_absmean: {head['current_q_absmean'].mean():.4g} -> {tail['current_q_absmean'].mean():.4g}"
        )
        print(f"  active_count mean: {cu['active_count'].mean():.2f}")
        print(
            f"  active_count min/median/max: {cu['active_count'].min():.0f} / "
            f"{cu['active_count'].median():.0f} / {cu['active_count'].max():.0f}"
        )

    sm = data["step_masks"]
    if sm is not None and len(sm):
        print("\n[step_masks] foreign contamination:")
        print(
            f"  n_agents: {int(sm['n_agents'].max())},  "
            f"mean n_fully_inactive/step: {sm['n_fully_inactive'].mean():.2f}"
        )
        print(
            f"  inactive_action_absmean (mean): {sm['inactive_action_absmean'].mean():.4g}"
        )
        print(
            f"  active_action_absmean (mean):   {sm['active_action_absmean'].mean():.4g}"
        )
        if "obs_absmax_foreign" in sm.columns:
            print(
                f"  obs_absmax_foreign (max): {sm['obs_absmax_foreign'].max():.4g},  "
                f"obs_absmax_native (max): {sm['obs_absmax_native'].max():.4g}"
            )
        if "batch_obs_or_action_naninf" in sm.columns:
            print(
                f"  steps with NaN/Inf in obs/action batch: "
                f"{int((sm['batch_obs_or_action_naninf'] > 0).sum())}"
            )

    cg = data.get("critic_grads")
    if cg is not None and len(cg):
        cg = cg.sort_values("n_updates")
        print("\n[critic_grads] pre-clip gradient norms:")
        first = cg[cg["n_updates"] <= cg["n_updates"].min() + 2]
        print(
            first.groupby("n_updates")[
                ["pre_clip_grad_norm", "pre_clip_max_param_grad_norm"]
            ]
            .agg(["mean", "max"])
            .to_string()
        )
        print(
            f"  clip_threshold: {cg['clip_threshold'].iloc[0]:.4g}  "
            f"(pre-clip norms far above this confirm a huge-gradient explosion)"
        )
        print(
            f"  steps with NaN/Inf in critic grads: "
            f"{int((cg['grad_has_naninf'] > 0).sum())} / {len(cg)}"
        )


def main():
    runs = {r: load(r) for r in ("staggered", "single")}
    for r, d in runs.items():
        summarise(r, d)

    # Direct episode-1 comparison
    print(f"\n{'#' * 70}\nEPISODE==1 SIDE BY SIDE\n{'#' * 70}")
    rows = []
    for r, d in runs.items():
        rw = d["rewards"]
        if rw is None:
            continue
        ep1 = rw[rw["episode"] == 1]
        if not len(ep1):
            continue
        rows.append(
            {
                "run": r,
                "n_rows": len(ep1),
                "units": ep1["unit_id"].nunique(),
                "reward_mean": ep1["reward"].mean(),
                "reward_absmax": ep1["reward"].abs().max(),
                "regret_mean": ep1["regret"].mean(),
                "regret_max": ep1["regret"].max(),
                "profit_absmax": ep1["profit"].abs().max(),
                "active_frac": ep1["active"].mean(),
            }
        )
    if rows:
        cmp = pd.DataFrame(rows).set_index("run")
        print("\n" + cmp.to_string())
        if {"staggered", "single"} <= set(cmp.index):
            print("\nratios (staggered / single):")
            for c in ["reward_absmax", "regret_max", "reward_mean", "regret_mean"]:
                s, sg = cmp.loc["single", c], cmp.loc["staggered", c]
                ratio = (
                    sg / s if s not in (0, np.nan) and abs(s) > 1e-12 else float("nan")
                )
                print(f"  {c}: {ratio:.3f}")


if __name__ == "__main__":
    main()
