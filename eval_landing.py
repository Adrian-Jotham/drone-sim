# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Drone Landing Evaluation — PPO
#
# Each episode spawns the drone at a random aerial position and places the
# landing platform at a random (x, y) on the ground.  The episode runs
# until the drone lands successfully, crashes, or times out.
#
# Usage:
#   python eval_landing.py
#   python eval_landing.py --model ppo_landing_final
#   python eval_landing.py --num_episodes 20 --seed 7
#   python eval_landing.py --headless --num_episodes 100   # fast batch eval
###########################################################################

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import numpy as np

import newton.examples
from stable_baselines3 import PPO
from drone_landing_env import DroneLandingEnv, LAND_DIST, LAND_SPEED, MAX_EPISODE_STEPS


# ── Evaluation loop ───────────────────────────────────────────────────────

def run_evaluation(
    model,
    env: DroneLandingEnv,
    num_episodes: int,
    deterministic: bool = True,
    seed: int = 42,
) -> dict:
    """Run `num_episodes` landing attempts and return aggregate stats."""
    ep_rewards, ep_lengths = [], []
    ep_landed, ep_dists, ep_speeds = [], [], []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)

        ep_reward = 0.0
        ep_len    = 0
        landed    = False

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len    += 1

            if terminated or truncated:
                landed = info.get("terminal_landed", False)
                dist   = info.get("terminal_dist",   info["dist"])
                speed  = info.get("terminal_speed",  info["speed"])
                break

        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_len)
        ep_landed.append(float(landed))
        ep_dists.append(dist)
        ep_speeds.append(speed)

        status = "LANDED ✓" if landed else "failed ✗"
        print(
            f"  ep {ep+1:>3}/{num_episodes} | rew={ep_reward:8.2f} | "
            f"len={ep_len:>4} | dist={dist:.3f}m | speed={speed:.3f}m/s | {status}"
        )

    return {
        "num_episodes":  num_episodes,
        "success_rate":  float(np.mean(ep_landed)),
        "mean_reward":   float(np.mean(ep_rewards)),
        "std_reward":    float(np.std(ep_rewards)),
        "mean_length":   float(np.mean(ep_lengths)),
        "mean_dist":     float(np.mean(ep_dists)),
        "mean_speed":    float(np.mean(ep_speeds)),
        "best_dist":     float(np.min(ep_dists)),
    }


def print_summary(stats: dict) -> None:
    print()
    print("─" * 60)
    print("  Evaluation summary  [PPO Landing]")
    print("─" * 60)
    print(f"  Episodes            : {stats['num_episodes']}")
    print(f"  Success rate        : {stats['success_rate']*100:.1f}%")
    print(f"    (threshold: dist < {LAND_DIST} m,  speed < {LAND_SPEED} m/s)")
    print(f"  Mean reward         : {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Mean ep length      : {stats['mean_length']:.1f} / {MAX_EPISODE_STEPS} steps")
    print(f"  Mean final dist     : {stats['mean_dist']:.4f} m")
    print(f"  Mean final speed    : {stats['mean_speed']:.4f} m/s")
    print(f"  Best final dist     : {stats['best_dist']:.4f} m")
    print("─" * 60)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = newton.examples.create_parser()
    parser.add_argument("--model",        type=str, default="ppo_landing_final",
                        help="Path to saved model (without .zip).")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--stochastic",   action="store_true",
                        help="Use stochastic policy instead of deterministic.")

    viewer, args = newton.examples.init(parser)

    model_path = args.model
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"Model not found: '{model_path}'.  "
            "Train first with train_landing.py or pass --model <path>."
        )

    print(f"\nLoading PPO model from '{model_path}' …")
    eval_env = DroneLandingEnv(render_mode="human", viewer=viewer)
    model    = PPO.load(model_path, env=eval_env)

    print(
        f"Running {args.num_episodes} episodes  "
        f"(seed={args.seed}, deterministic={not args.stochastic}) …\n"
    )

    stats = run_evaluation(
        model=model,
        env=eval_env,
        num_episodes=args.num_episodes,
        deterministic=not args.stochastic,
        seed=args.seed,
    )

    print_summary(stats)
    eval_env.close()


if __name__ == "__main__":
    main()
