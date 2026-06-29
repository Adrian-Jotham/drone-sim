# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Evaluate a SHAC policy (checkpoint from train_shac.py) on a CLEAN, fixed protocol —
no curriculum ramp, no mid-rollout resampling, so the score reflects the policy alone.

Two diagnostics:
  * HOLD  : spawn AT the target (upright hover) and measure whether the policy keeps it
            there. Tests stabilisation.
  * REACH : spawn at the origin, target 0.5-1.5 m away; measure final distance / success.
            Tests navigation.

Usage:
  python eval_shac.py --model shac_drone.pt --num_envs 256 --steps 300
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

import diff_drone_env as E
from diff_drone_env import OMEGA_HOVER, TURN_DIR, OBS_DIM
from train_shac import Actor


def load_actor(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    actor = Actor(OBS_DIM, 4).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    mean = ckpt["rms_mean"].to(device); var = ckpt["rms_var"].to(device)
    return actor, mean, var


@torch.no_grad()
def rollout(env, actor, mean, var, q, qd, om, steps):
    """Deterministic rollout; returns per-step mean distance trace + final dist tensor."""
    obs = env.obs_of(q, qd)
    trace = []
    for t in range(steps):
        env.new_window()                       # detached each step → reuse slot 0
        on = ((obs - mean) / torch.sqrt(var + 1e-5)).clamp(-5.0, 5.0)
        a = actor(on, deterministic=True)
        q, qd, om, obs, rew = env.step(q, qd, om, a)
        env.update_prev_dist()
        d = torch.from_numpy(env.dist.numpy())
        trace.append(d.mean().item())
    dist = torch.from_numpy(env.dist.numpy())
    return np.array(trace), dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="shac_drone.pt")
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = args.device
    env = E.DiffDroneEnv(num_envs=args.num_envs, device=dev)
    actor, mean, var = load_actor(args.model, dev)
    rng = np.random.default_rng(args.seed)
    N = args.num_envs

    def report(name, trace, dist):
        d = dist.numpy()
        blew = ~np.isfinite(d)
        ok = ~blew
        print(f"\n  [{name}]  steps={args.steps}")
        print(f"    blew up   : {blew.mean()*100:5.1f}%   (diverged to NaN/inf)")
        if ok.any():
            ds = d[ok]
            print(f"    survivors : mean dist {ds.mean():.3f} m   success {(ds < 0.15).mean()*100:.1f}%"
                  f"   within0.3 {(ds < 0.30).mean()*100:.1f}%   (of {ok.sum()} alive)")
        else:
            print("    survivors : none")

    # ── HOLD: spawn at target, upright hover ─────────────────────────────
    ang = rng.uniform(0, 2*np.pi, N); rad = rng.uniform(0.5, 1.5, N); alt = rng.uniform(0.3, 1.2, N)
    tgt = np.stack([rad*np.cos(ang), rad*np.sin(ang), alt], 1).astype(np.float32)
    q = np.zeros((N, 7), np.float32); q[:, :3] = tgt; q[:, 6] = 1.0
    qd = np.zeros((N, 6), np.float32)
    om = (TURN_DIR[None, :]*OMEGA_HOVER).repeat(N, 0).astype(np.float32)
    env.set_targets(tgt); env.a_prev.zero_()
    env.prev_dist.assign(np.zeros(N, np.float32))
    tq = torch.tensor(q, device=dev); tqd = torch.tensor(qd, device=dev); tom = torch.tensor(om, device=dev)
    tr, dist = rollout(env, actor, mean, var, tq, tqd, tom, args.steps)
    report("HOLD  (spawn on target)", tr, dist)

    # ── REACH: spawn at origin-ish, target 0.5-1.5 m away ────────────────
    spawn = np.zeros((N, 3), np.float32); spawn[:, 2] = 0.5
    q = np.zeros((N, 7), np.float32); q[:, :3] = spawn; q[:, 6] = 1.0
    qd = np.zeros((N, 6), np.float32)
    env.set_targets(tgt); env.a_prev.zero_()
    env.prev_dist.assign(np.linalg.norm(spawn - tgt, axis=1).astype(np.float32))
    tq = torch.tensor(q, device=dev); tqd = torch.tensor(qd, device=dev); tom = torch.tensor(om, device=dev)
    tr, dist = rollout(env, actor, mean, var, tq, tqd, tom, args.steps)
    report("REACH (origin -> waypoint)", tr, dist)


if __name__ == "__main__":
    main()
