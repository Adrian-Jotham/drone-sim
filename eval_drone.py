# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Drone evaluation — SKRL PPO/SAC, MLP/GRU policies (CLAUDE.md T9).

Runs the same multi-waypoint rollout as before, but against the batched Newton/MuJoCo
env (one world for eval) and a SKRL checkpoint. Each episode visits
``waypoints_per_ep`` random waypoints in sequence WITHOUT resetting between them; a slot
ends on success (dist < 0.15 m), step-budget expiry, or a crash. Waypoints are drawn
from the training distribution (radius ∈ [0.5,1.5] m, altitude ∈ [0.3,1.2] m).

Usage:
    python eval_drone.py --model runs/ppo_mlp_s0/checkpoints/best_agent.pt --algo ppo --policy mlp
    python eval_drone.py --model <ckpt.pt> --algo sac --policy gru --num_episodes 20
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import numpy as np
import torch

from drone_gym_env import CF_HOVER_RPM, CF_MAX_RPM, FPS
from drone_env_batched import RAD_TO_RPM, SUCCESS_R, SIM_DT, GROUND_Z, BODIES_PER_WORLD
import train_drone as T

DEFAULT_STEPS_PER_WP = 200
DEFAULT_WAYPOINTS    = 4


def _random_waypoints(rng: np.random.Generator, n: int) -> list[np.ndarray]:
    angles = rng.uniform(0.0, 2.0 * np.pi, n)
    radii  = rng.uniform(0.5, 1.5, n)
    alts   = rng.uniform(0.3, 1.2, n)
    return [np.array([r*np.cos(a), r*np.sin(a), z], dtype=np.float32)
            for a, r, z in zip(angles, radii, alts)]


class Policy:
    """Deterministic SKRL policy inference (handles MLP + GRU hidden state)."""

    def __init__(self, agent, device):
        self.agent = agent
        self.device = device
        self.is_rnn = bool(getattr(agent, "_rnn", False))
        self.rnn = None
        self.reset_rnn()

    def reset_rnn(self):
        if self.is_rnn:
            self.rnn = [s.clone() for s in self.agent._rnn_initial_states["policy"]]

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        proc = self.agent._observation_preprocessor(obs)
        inputs = {"observations": proc}
        if self.is_rnn:
            inputs["rnn"] = self.rnn
        action, outputs = self.agent.policy.act(inputs, role="policy")
        if self.is_rnn:
            self.rnn = outputs.get("rnn", self.rnn)
        # Gaussian policies (PPO/SAC) expose the deterministic mean; the TD3
        # deterministic actor returns the greedy action directly.
        return outputs.get("mean_actions", action)


def run_evaluation(policy, env, num_episodes, waypoints_per_ep, steps_per_wp, seed,
                   realtime=False):
    rng = np.random.default_rng(seed)
    ep_rewards, ep_lengths, ep_wpts, ep_all = [], [], [], []
    all_dists, all_succ, all_rpm = [], [], []

    for ep in range(num_episodes):
        waypoints = _random_waypoints(rng, waypoints_per_ep)
        env.place(pos=(0.0, 0.0, 0.5))
        policy.reset_rnn()
        env.set_target(waypoints[0])
        obs = env.obs_t

        ep_reward = 0.0; ep_len = 0; wpts_reached = 0
        wpt_results = []; crashed = False; ep_rpms = []

        for wpt_idx, target in enumerate(waypoints):
            env.set_target(target)
            obs = env.obs_t
            slot_steps = 0; slot_dist = float("inf"); reached = False

            while slot_steps < steps_per_wp:
                t0 = time.perf_counter()
                action = policy.act(obs)
                obs, reward, term, trunc, info = env.step(action, auto_reset=False)
                if realtime:
                    dt = SIM_DT - (time.perf_counter() - t0)
                    if dt > 0:
                        time.sleep(dt)
                ep_reward += float(reward[0]); ep_len += 1; slot_steps += 1
                slot_dist = float(info["dist"][0])
                ep_rpms.append(float(np.abs(env.motor_omega.numpy()).mean() * RAD_TO_RPM))
                if slot_dist < SUCCESS_R:
                    reached = True; wpts_reached += 1; break
                # The training termination box (|p_err| > 0.6 m) is NOT a crash here:
                # waypoints are 0.5-1.5 m away, so the drone is legitimately outside the
                # box while flying toward one. Only a genuine ground impact counts.
                z0 = float(env.state_0.body_q.numpy()[0][2])   # airframe z, world 0
                if z0 < GROUND_Z:
                    crashed = True; break

            wpt_results.append((slot_dist, reached))
            all_dists.append(slot_dist); all_succ.append(float(reached))
            if crashed:
                for _ in range(wpt_idx + 1, waypoints_per_ep):
                    wpt_results.append((float("inf"), False))
                    all_dists.append(float("inf")); all_succ.append(0.0)
                break

        ep_rewards.append(ep_reward); ep_lengths.append(ep_len)
        ep_wpts.append(wpts_reached)
        all_ok = wpts_reached == waypoints_per_ep
        ep_all.append(float(all_ok))
        mean_rpm = float(np.mean(ep_rpms)) if ep_rpms else float("nan")
        all_rpm.append(mean_rpm)

        wpt_str = "  ".join(f"wp{i+1}({'OK' if ok else 'x'},{d:.2f}m)"
                            for i, (d, ok) in enumerate(wpt_results))
        tag = "ALL" if all_ok else f"{wpts_reached}/{waypoints_per_ep}"
        print(f"  ep {ep+1:>3}/{num_episodes} | rew={ep_reward:8.2f} | len={ep_len:>4} "
              f"| rpm~{mean_rpm:.0f} | {wpt_str} | [{tag}]")

    finite = [d for d in all_dists if d < 1e9]
    rpms   = [r for r in all_rpm if not np.isnan(r)]
    return {
        "mean_reward": float(np.mean(ep_rewards)), "std_reward": float(np.std(ep_rewards)),
        "mean_length": float(np.mean(ep_lengths)),
        "mean_wpts_reached": float(np.mean(ep_wpts)),
        "all_success_rate": float(np.mean(ep_all)),
        "wpt_success_rate": float(np.mean(all_succ)),
        "mean_wpt_dist": float(np.mean(finite)) if finite else float("inf"),
        "mean_motor_rpm": float(np.mean(rpms)) if rpms else float("nan"),
        "num_episodes": num_episodes, "waypoints_per_ep": waypoints_per_ep,
    }


def print_summary(stats, algo, policy, steps_per_wp):
    n = stats["waypoints_per_ep"]; mrpm = stats["mean_motor_rpm"]
    print("\n" + "-" * 66)
    print(f"  Evaluation summary  [{algo.upper()}/{policy.upper()}]")
    print("-" * 66)
    print(f"  Episodes            : {stats['num_episodes']}")
    print(f"  Waypoints / ep      : {n}  (random, r=0.5-1.5 m, z=0.3-1.2 m)")
    print(f"  Step budget / wp    : {steps_per_wp} steps = {steps_per_wp/FPS:.1f} s")
    print(f"  Mean reward         : {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
    print(f"  Mean ep length      : {stats['mean_length']:.1f} steps")
    print(f"  Mean wpts reached   : {stats['mean_wpts_reached']:.2f} / {n}")
    print(f"  Per-waypoint succ   : {stats['wpt_success_rate']*100:.1f}%  (dist < {SUCCESS_R} m)")
    print(f"  All-waypoints succ  : {stats['all_success_rate']*100:.1f}%")
    print(f"  Mean waypoint dist  : {stats['mean_wpt_dist']:.4f} m")
    print(f"  Mean motor RPM      : {mrpm:.0f} RPM  ({mrpm/CF_HOVER_RPM*100:.1f}% of hover)")
    print("-" * 66)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to SKRL checkpoint (.pt)")
    p.add_argument("--algo", choices=["ppo", "sac", "td3"], default="ppo")
    p.add_argument("--policy", choices=["mlp", "gru"], default="mlp")
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--waypoints_per_ep", type=int, default=DEFAULT_WAYPOINTS)
    p.add_argument("--steps_per_wp", type=int, default=DEFAULT_STEPS_PER_WP)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--render", action="store_true",
                   help="open a real-time OpenGL window to watch the rollout")
    p.add_argument("--realtime", action="store_true",
                   help="pace the rollout to wall-clock 100 Hz (use with --render)")
    args = p.parse_args()

    # Build the same env+agent as training (1 world, eval), then load weights.
    build_args = SimpleNamespace(
        algo=args.algo, policy=args.policy, seed=args.seed, num_envs=1,
        rollouts=8, buffer_size=2048, curriculum_steps=1, entropy_decay_steps=1,
        logdir="/tmp/eval_runs", device=args.device, no_graph=True,
        render=args.render, render_worlds=1, render_every=1,
    )
    wrapped, agent = T.build(build_args)
    agent.init()
    agent.load(args.model)
    agent.enable_training_mode(False, apply_to_models=True)
    env = wrapped.unwrapped

    # Curriculum at full so eval spawn/physics match the trained regime.
    env.set_curriculum(1.0)
    policy = Policy(agent, args.device)

    wpts = min(max(args.waypoints_per_ep, 1), 8)
    print(f"\nLoaded {args.algo.upper()}/{args.policy.upper()} from '{args.model}'")
    print(f"Running {args.num_episodes} episodes x {wpts} waypoints "
          f"({args.steps_per_wp} steps/wp, seed={args.seed})\n")

    stats = run_evaluation(policy, env, args.num_episodes, wpts, args.steps_per_wp,
                           args.seed, realtime=args.realtime)
    print_summary(stats, args.algo, args.policy, args.steps_per_wp)
    if env._viewer is not None:
        env._viewer.close()


if __name__ == "__main__":
    main()
