# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Drone Evaluation — PPO | SAC | TD3, multi-waypoint with random targets
#
# Each episode visits `waypoints_per_ep` randomly generated waypoints in
# sequence WITHOUT resetting the drone between them.  A waypoint slot ends
# when dist < 0.15 m (success), the step budget expires, or the drone
# crashes.  Waypoints are drawn uniformly from the training distribution:
#   radius ∈ [0.5, 1.5] m,  altitude ∈ [0.3, 1.2] m,  angle ∈ [0, 2π)
#
# Usage:
#   python eval_drone.py --model sac_drone_final --algo sac
#   python eval_drone.py --model td3_drone_final --algo td3
#   python eval_drone.py --model ppo_drone_final --algo ppo
#   python eval_drone.py --model sac_drone_final --algo sac --num_episodes 20
#   python eval_drone.py --model sac_drone_final --algo sac --seed 7
###########################################################################

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import numpy as np

import newton.examples
from drone_gym_env import DroneEnv

DEFAULT_STEPS_PER_WP = 150   # 1.5 s at 100 Hz
DEFAULT_WAYPOINTS    = 4


# ── Random waypoint generation ────────────────────────────────────────────

def _random_waypoints(rng: np.random.Generator, n: int) -> list[np.ndarray]:
    """Generate `n` random waypoints drawn from the training distribution."""
    angles = rng.uniform(0.0, 2.0 * np.pi, n)
    radii  = rng.uniform(0.5, 1.5, n)
    alts   = rng.uniform(0.3, 1.2, n)
    return [
        np.array([r * np.cos(a), r * np.sin(a), z], dtype=np.float32)
        for a, r, z in zip(angles, radii, alts)
    ]


# ── Evaluation loop ───────────────────────────────────────────────────────

def run_evaluation(
    model,
    env: DroneEnv,
    num_episodes: int,
    waypoints_per_ep: int   = DEFAULT_WAYPOINTS,
    steps_per_wp: int       = DEFAULT_STEPS_PER_WP,
    deterministic: bool     = True,
    seed: int               = 42,
) -> dict:
    """Run `num_episodes` multi-waypoint episodes and return aggregate stats."""
    rng = np.random.default_rng(seed)

    ep_rewards, ep_lengths, ep_wpts_reached, ep_all_success = [], [], [], []
    all_wpt_dists, all_wpt_success = [], []

    for ep in range(num_episodes):
        waypoints = _random_waypoints(rng, waypoints_per_ep)

        obs, _ = env.reset()
        env.set_target(waypoints[0])
        obs = env.get_obs()   # refresh obs with new target

        ep_reward   = 0.0
        ep_len      = 0
        wpts_reached = 0
        wpt_results: list[tuple[float, bool]] = []
        crashed     = False

        for wpt_idx, target in enumerate(waypoints):
            if wpt_idx > 0:
                env.set_target(target)
                obs = env.get_obs()

            slot_steps = 0
            slot_dist  = float("inf")
            reached    = False

            while slot_steps < steps_per_wp:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward  += reward
                ep_len     += 1
                slot_steps += 1
                slot_dist   = info["dist"]

                if slot_dist < 0.15:
                    reached = True
                    wpts_reached += 1
                    break

                if terminated:
                    crashed = True
                    break

            wpt_results.append((slot_dist, reached))
            all_wpt_dists.append(slot_dist if slot_dist < 1e9 else float("inf"))
            all_wpt_success.append(float(reached))

            if crashed:
                for _ in range(wpt_idx + 1, waypoints_per_ep):
                    wpt_results.append((float("inf"), False))
                    all_wpt_dists.append(float("inf"))
                    all_wpt_success.append(0.0)
                break

        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_len)
        ep_wpts_reached.append(wpts_reached)
        all_ok = wpts_reached == waypoints_per_ep
        ep_all_success.append(float(all_ok))

        wpt_str = "  ".join(
            f"wp{i+1}({'✓' if ok else '✗'},{d:.2f}m)"
            for i, (d, ok) in enumerate(wpt_results)
        )
        tag = "ALL✓" if all_ok else (f"{wpts_reached}/{waypoints_per_ep}")
        print(f"  ep {ep+1:>3}/{num_episodes} | rew={ep_reward:8.2f} | "
              f"len={ep_len:>4} | {wpt_str} | [{tag}]")

    finite_dists = [d for d in all_wpt_dists if d < 1e9]
    return {
        "mean_reward":       float(np.mean(ep_rewards)),
        "std_reward":        float(np.std(ep_rewards)),
        "mean_length":       float(np.mean(ep_lengths)),
        "mean_wpts_reached": float(np.mean(ep_wpts_reached)),
        "all_success_rate":  float(np.mean(ep_all_success)),
        "wpt_success_rate":  float(np.mean(all_wpt_success)),
        "mean_wpt_dist":     float(np.mean(finite_dists)) if finite_dists else float("inf"),
        "num_episodes":      num_episodes,
        "waypoints_per_ep":  waypoints_per_ep,
    }


def print_summary(stats: dict, algo: str) -> None:
    n = stats["waypoints_per_ep"]
    print()
    print("─" * 64)
    print(f"  Evaluation summary  [{algo.upper()}]")
    print("─" * 64)
    print(f"  Episodes            : {stats['num_episodes']}")
    print(f"  Waypoints / ep      : {n}  (random, r=0.5–1.5 m, z=0.3–1.2 m)")
    print(f"  Mean reward         : {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Mean ep length      : {stats['mean_length']:.1f} steps")
    print(f"  Mean wpts reached   : {stats['mean_wpts_reached']:.2f} / {n}")
    print(f"  Per-waypoint succ   : {stats['wpt_success_rate']*100:.1f}%  (dist < 0.15 m)")
    print(f"  All-waypoints succ  : {stats['all_success_rate']*100:.1f}%  (all wpts hit)")
    print(f"  Mean waypoint dist  : {stats['mean_wpt_dist']:.4f} m")
    print("─" * 64)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = newton.examples.create_parser()
    parser.add_argument("--model", type=str, default="sac_drone_final",
                        help="Path to saved model (without .zip).")
    parser.add_argument("--algo",  type=str, default="sac",
                        choices=["ppo", "sac", "td3"],
                        help="Algorithm used to train the model.")
    parser.add_argument("--num_episodes",    type=int, default=10)
    parser.add_argument("--waypoints_per_ep", type=int, default=DEFAULT_WAYPOINTS,
                        help="Random waypoints per episode (default 4).")
    parser.add_argument("--steps_per_wp",    type=int, default=DEFAULT_STEPS_PER_WP,
                        help=f"Step budget per waypoint (default {DEFAULT_STEPS_PER_WP}).")
    parser.add_argument("--seed",            type=int, default=42,
                        help="RNG seed for reproducible waypoint generation.")
    parser.add_argument("--stochastic",      action="store_true",
                        help="Use stochastic (non-deterministic) policy.")

    viewer, args = newton.examples.init(parser)

    model_path = args.model
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"Model not found: '{model_path}'.  "
            "Train first with train_drone.py or pass --model <path>."
        )

    algo = args.algo.lower()
    print(f"\nLoading {algo.upper()} model from '{model_path}' …")

    eval_env = DroneEnv(render_mode="human", viewer=viewer, random_targets=False)

    if algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(model_path, env=eval_env)
    elif algo == "sac":
        from sbx import SAC
        model = SAC.load(model_path, env=eval_env)
    else:
        from sbx import TD3
        model = TD3.load(model_path, env=eval_env)

    wpts = min(max(args.waypoints_per_ep, 1), 8)
    print(f"Running {args.num_episodes} episodes × {wpts} random waypoints "
          f"({args.steps_per_wp} steps/wp, seed={args.seed}) …\n")

    stats = run_evaluation(
        model=model,
        env=eval_env,
        num_episodes=args.num_episodes,
        waypoints_per_ep=wpts,
        steps_per_wp=args.steps_per_wp,
        deterministic=not args.stochastic,
        seed=args.seed,
    )

    print_summary(stats, algo)
    eval_env.close()


if __name__ == "__main__":
    main()
