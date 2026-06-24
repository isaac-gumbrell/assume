# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Offline reproduction of the staggered MATD3 critic divergence.

Loads a saved replay buffer and replays the *exact* centralized-critic update
from ``TD3.update_policy`` with freshly initialised critics, logging the
per-step Q magnitude. This isolates whether the explosion is intrinsic to the
critic regression on the stored (clean) transitions, independent of the live
simulation, the actors, and the reward computation.

Bisect knobs:
    --n-agents N   : restrict to the first N agents (92 = staggered superset,
                     89 = single-world native set) to test the agent-count
                     hypothesis.
    --no-mask      : ignore masks (treat every transition as active).
    --clip / --no-clip : toggle gradient-norm clipping.
    --steps K      : number of gradient steps to replay.

Usage:
    .venv\\Scripts\\python.exe tools\\repro_divergence.py <buffer.npz> [opts]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch as th

from assume.reinforcement_learning.neural_network_architecture import CriticTD3

GAMMA = 0.99
LR = 0.00075
GRAD_CLIP = 1.0
UNIQUE_OBS_DIM = 2
BATCH = 192
SEED = 42


def build_centralized(states, i, unique_obs_dim, n_agents):
    """Replicate update_policy's per-agent centralized state construction."""
    b = states.shape[0]
    own = states[:, i, :].reshape(b, -1)
    uniq = states[:, :, states.shape[2] - unique_obs_dim :].reshape(b, n_agents, -1)
    other = th.cat((uniq[:, :i], uniq[:, i + 1 :]), dim=1).reshape(b, -1)
    return th.cat((own, other), dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("buffer")
    ap.add_argument("--n-agents", type=int, default=None)
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--unique-obs-dim", type=int, default=UNIQUE_OBS_DIM)
    args = ap.parse_args()

    th.manual_seed(SEED)
    np.random.seed(SEED)

    data = np.load(args.buffer)
    obs = data["observations"]
    acts = data["actions"]
    rews = data["rewards"]
    masks = data["masks"] if "masks" in data.files else np.ones(rews.shape, np.float32)
    pos = int(data["pos"][0]) if "pos" in data.files else obs.shape[0]
    full = bool(data["full"][0]) if "full" in data.files else False
    n_valid = obs.shape[0] if full else pos

    n_agents_total = obs.shape[1]
    n = args.n_agents or n_agents_total
    obs_dim = obs.shape[2]
    act_dim = acts.shape[2]
    uod = args.unique_obs_dim

    print(
        f"buffer={args.buffer} valid={n_valid} n_agents_total={n_agents_total} "
        f"-> using n={n} obs_dim={obs_dim} act_dim={act_dim} unique_obs_dim={uod}"
    )
    print(
        f"opts: mask={'OFF' if args.no_mask else 'ON'} "
        f"clip={'OFF' if args.no_clip else f'{GRAD_CLIP}'} steps={args.steps}\n"
    )

    ft = th.float32
    o = th.tensor(obs[:n_valid, :n, :], dtype=ft)
    a = th.tensor(acts[:n_valid, :n, :], dtype=ft)
    r = th.tensor(rews[:n_valid, :n], dtype=ft)
    m = th.tensor(masks[:n_valid, :n], dtype=ft)
    if args.no_mask:
        m = th.ones_like(m)

    # next-step index proxy (clamped within valid range)
    nxt = th.clamp(th.arange(n_valid) + 1, max=n_valid - 1)
    o_next = o[nxt]
    a_next = a[nxt]

    # Build fresh critics + targets (Xavier init, identical to create_critics).
    critics, targets, opts = [], [], []
    for _ in range(n):
        c = CriticTD3(
            n_agents=n,
            obs_dim=obs_dim,
            act_dim=act_dim,
            float_type=ft,
            unique_obs_dim=uod,
        )
        t = CriticTD3(
            n_agents=n,
            obs_dim=obs_dim,
            act_dim=act_dim,
            float_type=ft,
            unique_obs_dim=uod,
        )
        t.load_state_dict(c.state_dict())
        t.train(mode=False)
        critics.append(c)
        targets.append(t)
        opts.append(th.optim.AdamW(c.parameters(), lr=LR))

    rng = np.random.default_rng(SEED)

    print(
        f"{'step':>4} {'agent0_curQ':>14} {'maxQ_all':>14} "
        f"{'maxLoss':>12} {'preClipNorm0':>13}"
    )
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, n_valid, size=BATCH)
        bi = th.as_tensor(idx)
        states = o[bi]
        actions = a[bi]
        next_states = o_next[bi]
        next_actions_full = a_next[bi].reshape(BATCH, -1)
        all_actions = actions.reshape(BATCH, -1)
        rewards = r[bi]
        mb = m[bi]

        # zero grads
        for opt in opts:
            opt.zero_grad(set_to_none=True)

        total = 0.0
        per_agent_q = []
        per_agent_loss = []
        for i in range(n):
            all_states = build_centralized(states, i, uod, n)
            all_next = build_centralized(next_states, i, uod, n)
            with th.no_grad():
                tq = th.cat(targets[i](all_next, next_actions_full), dim=1)
                tq, _ = th.min(tq, dim=1, keepdim=True)
                target_Q = rewards[:, i].unsqueeze(1) + GAMMA * tq
            cur = critics[i](all_states, all_actions)
            mask_i = mb[:, i].unsqueeze(1)
            mask_norm = mask_i.sum().clamp(min=1.0)
            closs = sum((mask_i * (cq - target_Q) ** 2).sum() / mask_norm for cq in cur)
            total = total + closs
            per_agent_q.append(float(cur[0].detach().abs().mean()))
            per_agent_loss.append(float(closs.detach()))

        total.backward()

        pre0 = None
        for i in range(n):
            params = list(critics[i].parameters())
            max_norm = float("inf") if args.no_clip else GRAD_CLIP
            tn = th.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
            if i == 0:
                pre0 = float(tn)
            opts[i].step()

        print(
            f"{step:>4} {per_agent_q[0]:>14.4g} {max(per_agent_q):>14.4g} "
            f"{max(per_agent_loss):>12.4g} {pre0:>13.4g}"
        )


if __name__ == "__main__":
    main()
