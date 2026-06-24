# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Inspect a saved replay buffer for extreme actions/observations.

Localises the source of the staggered-training critic divergence by reporting
per-agent and per-feature magnitude statistics for the stored transitions, with
special attention to whether masked (forced-off / foreign) transitions carry
non-trivial actions that still feed the centralised critic input.

Usage:
    .venv\\Scripts\\python.exe tools\\inspect_buffer.py <path-to-replay_buffer.npz>
"""

from __future__ import annotations

import sys

import numpy as np


def main(path: str) -> None:
    data = np.load(path)
    obs = data["observations"]
    acts = data["actions"]
    rews = data["rewards"]
    masks = data["masks"] if "masks" in data.files else None
    pos = int(data["pos"][0]) if "pos" in data.files else obs.shape[0]
    full = bool(data["full"][0]) if "full" in data.files else False

    n_valid = obs.shape[0] if full else pos
    print(f"file: {path}")
    print(f"shapes: obs={obs.shape} acts={acts.shape} rews={rews.shape}")
    print(f"pos={pos} full={full} -> valid rows: {n_valid}")
    if masks is not None:
        print(f"masks shape: {masks.shape}")
    print()

    o = obs[:n_valid]
    a = acts[:n_valid]
    r = rews[:n_valid]
    m = masks[:n_valid] if masks is not None else None

    # ---- global sanity ----
    print("=== GLOBAL ===")
    print(f"obs    min/max: {np.nanmin(o):.4g} / {np.nanmax(o):.4g}")
    print(f"action min/max: {np.nanmin(a):.4g} / {np.nanmax(a):.4g}")
    print(f"reward min/max: {np.nanmin(r):.4g} / {np.nanmax(r):.4g}")
    print(f"obs    NaN/Inf: {np.isnan(o).sum()} / {np.isinf(o).sum()}")
    print(f"action NaN/Inf: {np.isnan(a).sum()} / {np.isinf(a).sum()}")
    print(f"reward NaN/Inf: {np.isnan(r).sum()} / {np.isinf(r).sum()}")
    print()

    # ---- per-agent action magnitude ----
    # a: (N, n_agents, act_dim)
    act_absmax_per_agent = np.abs(a).max(axis=(0, 2))
    obs_absmax_per_agent = np.abs(o).max(axis=(0, 2))
    order = np.argsort(act_absmax_per_agent)[::-1]
    print("=== TOP-15 AGENTS BY ACTION ABS-MAX ===")
    print(f"{'agent_idx':>9} {'act_absmax':>12} {'obs_absmax':>12}", end="")
    if m is not None:
        print(f" {'mask_mean':>10} {'mask_min':>9}", end="")
    print()
    for idx in order[:15]:
        line = f"{idx:>9} {act_absmax_per_agent[idx]:>12.4g} {obs_absmax_per_agent[idx]:>12.4g}"
        if m is not None:
            mm = m[:, idx]
            line += f" {mm.mean():>10.4f} {mm.min():>9.0f}"
        print(line)
    print()

    # ---- masked vs unmasked action magnitude ----
    if m is not None:
        # broadcast mask over act_dim
        mb = np.broadcast_to(m[:, :, None], a.shape)
        active = mb > 0.5
        inactive = ~active
        if inactive.any():
            print("=== MASKED (forced-off) TRANSITIONS ===")
            print(f"count inactive action entries: {inactive.sum()}")
            print(f"inactive action abs-max:  {np.abs(a[inactive]).max():.4g}")
            print(f"inactive action abs-mean: {np.abs(a[inactive]).mean():.4g}")
            print(f"active   action abs-max:  {np.abs(a[active]).max():.4g}")
            print(f"active   action abs-mean: {np.abs(a[active]).mean():.4g}")
            # which agents are ever masked?
            ever_masked = np.where(m.min(axis=0) < 0.5)[0]
            print(f"agents ever masked (idx): {ever_masked.tolist()}")
            for idx in ever_masked:
                col = a[:, idx, :]
                mm = m[:, idx]
                off = mm < 0.5
                print(
                    f"  agent {idx}: masked_steps={off.sum()} "
                    f"action_absmax_when_off={np.abs(col[off]).max():.4g} "
                    f"action_absmax_when_on="
                    f"{(np.abs(col[~off]).max() if (~off).any() else 0):.4g}"
                )
        else:
            print("=== no masked transitions present ===")
    print()

    # ---- per-observation-feature extremes ----
    print("=== TOP-10 (agent, obs_feature) BY ABS VALUE ===")
    flat = np.abs(o).reshape(n_valid, -1)
    feat_absmax = flat.max(axis=0)
    obs_dim = o.shape[2]
    top = np.argsort(feat_absmax)[::-1][:10]
    for f in top:
        ag, feat = divmod(f, obs_dim)
        print(f"  agent={ag:>3} obs_feature={feat:>3} abs_max={feat_absmax[f]:.4g}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
