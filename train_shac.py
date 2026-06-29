# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
SHAC trainer — Short-Horizon Actor-Critic (Xu et al., ICLR 2022, arXiv:2204.07137)
on the differentiable Featherstone drone env.

This revision targets the navigation-convergence failure with five changes:
  1. SHORT navigation — spawn offset capped (env.MAX_OFFSET≈0.6 m) so a trip fits the
     8-step BPTT window + critic bootstrap.
  2. DECOUPLED curriculum — tilt ramps early (stabilise first), navigation offset later
     (env.sample_spawn).
  3. NO tanh squashing — its saturation made the policy bang-bang and non-generalising;
     the actor outputs a LINEAR action (the env clamps for physics), kept small by an
     action cost + a pre-activation L2 (mu) penalty.
  4. SUCCESS-GATED curriculum — c advances only when survivors actually master the level,
     so the policy always trains on an achievable (dense-reward) task.
  5. STRONGER critic — deeper net + a target network for stable TD(λ) bootstrapping, which
     carries the ~100-step navigation credit the short window cannot.
"""

from __future__ import annotations

import argparse
from collections import deque
import numpy as np
import torch
import torch.nn as nn

import diff_drone_env as E
from diff_drone_env import sample_spawn, OMEGA_HOVER, TURN_DIR

SUCC = 0.15      # success distance threshold [m]


def mlp(i, o, h):
    layers, d = [], i
    for w in h:
        layers += [nn.Linear(d, w), nn.ELU()]
        d = w
    layers += [nn.Linear(d, o)]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Linear-output policy (no tanh). The env clamps the action for the physics; an action
    cost + mu penalty keep it small, avoiding the tanh-saturation bang-bang failure."""
    def __init__(self, obs_dim, act_dim, std=0.05):
        super().__init__()
        self.mu = mlp(obs_dim, act_dim, (256, 256))
        self.register_buffer("std", torch.full((act_dim,), float(std)))

    def set_std(self, s):
        self.std.fill_(float(s))

    def forward(self, obs, deterministic=False):
        mu = self.mu(obs)
        if deterministic:
            return mu
        return mu + self.std * torch.randn_like(mu)


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.v = mlp(obs_dim, 1, (256, 256, 256))   # deeper

    def forward(self, obs):
        return self.v(obs).squeeze(-1)


class RMS:
    def __init__(self, dim, device):
        self.mean = torch.zeros(dim, device=device); self.var = torch.ones(dim, device=device)
        self.count = 1e-4

    def update(self, x):
        bm = x.mean(0); bv = x.var(0, unbiased=False); bn = x.shape[0]
        d = bm - self.mean; tot = self.count + bn
        self.mean += d * bn / tot
        self.var = (self.var*self.count + bv*bn + d*d*self.count*bn/tot) / tot
        self.count = tot

    def norm(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-5)).clamp(-5.0, 5.0)


class ValueRMS:
    """Scalar running mean/std for value-target normalisation (stabilises the critic when
    the value horizon — and hence the return scale — is long)."""
    def __init__(self, device):
        self.mean = torch.zeros((), device=device); self.var = torch.ones((), device=device)
        self.count = 1e-4

    def update(self, x):
        bm = x.mean(); bv = x.var(unbiased=False); bn = x.numel()
        d = bm - self.mean; tot = self.count + bn
        self.mean += d * bn / tot
        self.var = (self.var*self.count + bv*bn + d*d*self.count*bn/tot) / tot
        self.count = tot

    def norm(self, x):   return (x - self.mean) / torch.sqrt(self.var + 1e-5)
    def denorm(self, x): return x * torch.sqrt(self.var + 1e-5) + self.mean


def compute_done(env, q, dev):
    dist = torch.from_numpy(env.dist.numpy()).to(dev)
    return (q[:, 2] < 0.05) | (dist > 2.5) | (q[:, 6].abs() < 0.5)


def resample(env, q, qd, om, mask, c, seed):
    n = int(mask.sum().item())
    if n == 0:
        return q, qd, om
    idx = mask.nonzero(as_tuple=True)[0].cpu().numpy()
    rng = np.random.default_rng(seed)
    tgt, nq, nqd, nom = sample_spawn(c, n, rng)
    dev = q.device
    q = q.clone();  q[mask]  = torch.tensor(nq,  device=dev)
    qd = qd.clone(); qd[mask] = torch.tensor(nqd, device=dev)
    om = om.clone(); om[mask] = torch.tensor(nom, device=dev)
    tg = env.target.numpy().copy(); tg[idx] = tgt; env.target.assign(tg)
    pd = env.prev_dist.numpy().copy(); pd[idx] = np.linalg.norm(nq[:, :3] - tgt, axis=1); env.prev_dist.assign(pd)
    ap = env.a_prev.numpy().copy().reshape(-1, 4); ap[idx] = 0.0; env.a_prev.assign(ap.reshape(-1))
    return q, qd, om


def train(args):
    dev = args.device
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    env = E.DiffDroneEnv(num_envs=args.num_envs, device=dev)
    obs_dim, act_dim = env.num_obs, env.num_act
    actor = Actor(obs_dim, act_dim).to(dev)
    critic = Critic(obs_dim).to(dev)
    critic_t = Critic(obs_dim).to(dev); critic_t.load_state_dict(critic.state_dict())
    a_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    rms = RMS(obs_dim, dev)
    vrms = ValueRMS(dev)
    gamma, lam, h, L = args.gamma, args.lam, args.horizon, args.critic_horizon

    q, qd, om = env.reset(curriculum=0.0, seed=args.seed)
    obs = env.obs_of(q, qd)
    cur = 0.0; succ_ema = 0.0; best = -1e9

    for it in range(args.iters):
        frac = min(it / max(args.std_anneal, 1), 1.0)
        actor.set_std(args.std_start + frac * (args.std_end - args.std_start))
        warm = min(it / max(args.critic_warmup, 1), 1.0)
        q, qd, om = q.detach(), qd.detach(), om.detach(); obs = obs.detach()

        # ── ACTOR rollout: h steps WITH gradient (BPTT through the sim) ─────
        actor_loss = torch.zeros(args.num_envs, device=dev)
        mu_reg = torch.zeros((), device=dev)
        gt = 1.0; obs_tr, rew_tr, done_tr = [], [], []
        env.new_window()
        o = obs
        for t in range(h):
            on = rms.norm(o); obs_tr.append(o.detach())
            mu = actor.mu(on)
            a = mu + actor.std * torch.randn_like(mu)
            mu_reg = mu_reg + (mu * mu).mean()
            q, qd, om, o2, rew = env.step(q, qd, om, a)
            env.update_prev_dist()
            actor_loss = actor_loss - gt * rew; gt *= gamma
            rew_tr.append(rew.detach()); done_tr.append(compute_done(env, q, dev))
            o = o2
        # bootstrap with the value-normalised critic (denorm to real return scale)
        term_v = vrms.denorm(critic(rms.norm(o)))
        loss = (actor_loss - warm * gt * term_v).mean() / h + args.mu_reg * mu_reg / h
        a_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        a_opt.step()

        # ── clean success on survivors, success-gated curriculum ───────────
        with torch.no_grad():
            done = compute_done(env, q, dev)
            dist = torch.from_numpy(env.dist.numpy()).to(dev)
            surv = ~done
            if surv.any():
                succ_ema = 0.9*succ_ema + 0.1*(dist[surv] < SUCC).float().mean().item()
        if succ_ema > args.gate and cur < 1.0:
            cur = min(cur + args.c_step, 1.0)
        q, qd, om = resample(env, q, qd, om, done, cur, args.seed + it)
        o = env.obs_of(q.detach(), qd.detach())

        # ── CRITIC rollout: continue L MORE steps (no grad) so the value
        #     target spans a LONG horizon → V captures the slow drift→crash that
        #     the 8-step actor window cannot see. State carries forward (no waste).
        with torch.no_grad():
            for t in range(L):
                env.new_window()                    # no-grad → reuse slot 0
                on = rms.norm(o); obs_tr.append(o)
                a = actor.mu(on) + actor.std * torch.randn_like(on[:, :act_dim])
                q, qd, om, o2, rew = env.step(q, qd, om, a)
                env.update_prev_dist()
                d = compute_done(env, q, dev)
                rew_tr.append(rew); done_tr.append(d)
                q, qd, om = resample(env, q, qd, om, d, cur, args.seed + 7919*it + t)
                o = env.obs_of(q, qd) if bool(d.any()) else o2
            final_obs = o

        # ── TD(λ) value targets over the full h+L trajectory ───────────────
        with torch.no_grad():
            Lt = len(rew_tr)
            vals = [vrms.denorm(critic_t(rms.norm(ob))) for ob in obs_tr]
            vals.append(vrms.denorm(critic_t(rms.norm(final_obs))))
            targets = [None]*Lt; gae = vals[Lt]
            for t in reversed(range(Lt)):
                nonterm = (~done_tr[t]).float()
                gae = rew_tr[t] + gamma*nonterm*((1-lam)*vals[t+1] + lam*gae)
                targets[t] = gae
            T = torch.stack(targets); O = torch.stack(obs_tr)
        vrms.update(T.reshape(-1))
        On = rms.norm(O).reshape(-1, obs_dim); Tn = vrms.norm(T).reshape(-1)
        for _ in range(args.critic_epochs):
            idx = torch.randperm(On.shape[0], device=dev)[:args.critic_batch]
            c_opt.zero_grad(set_to_none=True)
            cl = ((critic(On[idx]) - Tn[idx])**2).mean()
            cl.backward(); c_opt.step()
        with torch.no_grad():
            for p, pt in zip(critic.parameters(), critic_t.parameters()):
                pt.mul_(1 - args.tau).add_(args.tau * p)
        rms.update(O.reshape(-1, obs_dim))
        obs = o                                      # carry state to next iteration
        rew_buf = rew_tr                             # for logging

        if succ_ema > best and it > 50:
            best = succ_ema
            if args.save:
                torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                            "rms_mean": rms.mean, "rms_var": rms.var, "cur": cur}, args.save)
        if it % args.log_every == 0:
            with torch.no_grad():
                d = env.dist.numpy()
                print(f"it {it:5d} | c={cur:.2f} | succ_ema={succ_ema:.2f} "
                      f"| ep_rew~{torch.stack(rew_buf).sum(0).mean():.2f} | mean_dist={np.nanmean(d):.3f} "
                      f"| done={int(done.sum())}")
    if args.save:
        print("best succ_ema:", round(best, 3), "-> saved", args.save)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--critic_horizon", type=int, default=56,
                   help="extra no-grad steps for critic value targets (long-horizon V)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--actor_lr", type=float, default=1e-4)
    p.add_argument("--critic_lr", type=float, default=1e-3)
    p.add_argument("--critic_epochs", type=int, default=16)
    p.add_argument("--critic_batch", type=int, default=4096)
    p.add_argument("--tau", type=float, default=0.01, help="target-critic polyak")
    p.add_argument("--critic_warmup", type=int, default=40)
    p.add_argument("--std_start", type=float, default=0.08)
    p.add_argument("--std_end", type=float, default=0.03)
    p.add_argument("--std_anneal", type=int, default=600)
    p.add_argument("--mu_reg", type=float, default=0.02)
    p.add_argument("--gate", type=float, default=0.6, help="success EMA to advance curriculum")
    p.add_argument("--c_step", type=float, default=0.01, help="curriculum increment when gated")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save", default="shac_drone.pt")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
