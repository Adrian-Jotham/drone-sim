# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Quadrotor Hover Evaluation — PPO | SAC | TD3
#
# Two evaluation modes:
#
#   waypoint (default)
#     Visits `waypoints_per_ep` random waypoints per episode without
#     resetting the drone.  Waypoints: radius 0.5–1.5 m, alt 0.3–1.2 m.
#
#   disturbance
#     Drone holds a fixed hover point under increasing OU wind.
#     Reports mean/max position error and % time within 0.15 m.
#     The position trail and wind indicator are rendered in the viewer.
#
# Usage:
#   python eval_hover.py --model sac_hover_final --algo sac
#   python eval_hover.py --model sac_hover_final --algo sac --mode disturbance --wind_scale 0.8
#   python eval_hover.py --model ppo_hover_final --algo ppo --num_episodes 20
###########################################################################

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import numpy as np

import newton.examples
from disturbance.quadrotor_hover_env import DroneEnv, WIND_MAX

DEFAULT_STEPS_PER_WP  = 150    # 1.5 s per waypoint
DEFAULT_WAYPOINTS     = 4
DISTURBANCE_STEPS     = 1000   # 10 s hold per episode


# ── Waypoint helpers ──────────────────────────────────────────────────────

def _random_waypoints(rng: np.random.Generator, n: int) -> list[np.ndarray]:
    angles = rng.uniform(0.0, 2.0 * np.pi, n)
    radii  = rng.uniform(0.5, 1.5, n)
    alts   = rng.uniform(0.3, 1.2, n)
    return [
        np.array([r * np.cos(a), r * np.sin(a), z], dtype=np.float32)
        for a, r, z in zip(angles, radii, alts)
    ]


# ── Waypoint evaluation ───────────────────────────────────────────────────

def run_waypoint_eval(
    model,
    env: DroneEnv,
    num_episodes: int,
    waypoints_per_ep: int = DEFAULT_WAYPOINTS,
    steps_per_wp: int     = DEFAULT_STEPS_PER_WP,
    deterministic: bool   = True,
    seed: int             = 42,
) -> dict:
    rng = np.random.default_rng(seed)

    ep_rewards, ep_lengths, ep_wpts_reached, ep_all_success = [], [], [], []
    all_wpt_dists, all_wpt_success = [], []

    for ep in range(num_episodes):
        waypoints = _random_waypoints(rng, waypoints_per_ep)

        obs, _ = env.reset()
        env.set_target(waypoints[0])
        obs = env.get_obs()

        ep_reward   = 0.0
        ep_len      = 0
        wpts_reached = 0
        wpt_results: list[tuple[float, bool]] = []
        crashed = False

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
        tag = "ALL✓" if all_ok else f"{wpts_reached}/{waypoints_per_ep}"
        print(f"  ep {ep+1:>3}/{num_episodes} | rew={ep_reward:8.2f} | "
              f"len={ep_len:>4} | {wpt_str} | [{tag}]")

    finite_dists = [d for d in all_wpt_dists if d < 1e9]
    return {
        "mode":              "waypoint",
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


# ── Disturbance rejection evaluation ─────────────────────────────────────

def run_disturbance_eval(
    model,
    env: DroneEnv,
    num_episodes: int,
    steps_per_ep: int   = DISTURBANCE_STEPS,
    deterministic: bool = True,
    seed: int           = 42,
) -> dict:
    """
    Hold a fixed target for `steps_per_ep` steps under OU wind.
    Measures steady-state position error and robustness metrics.
    """
    rng = np.random.default_rng(seed)

    all_mean_dists, all_max_dists, all_time_on_target = [], [], []
    all_rewards, all_wind_mags = [], []

    for ep in range(num_episodes):
        obs, _ = env.reset()

        dists     = []
        wind_mags = []
        ep_reward = 0.0

        for step in range(steps_per_ep):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            dists.append(info["dist"])
            wind_mags.append(info.get("wind_magnitude", 0.0))

            if terminated:
                # Fill rest with crash distance
                dists.extend([info["dist"]] * (steps_per_ep - step - 1))
                wind_mags.extend([0.0] * (steps_per_ep - step - 1))
                break

        mean_d  = float(np.mean(dists))
        max_d   = float(np.max(dists))
        on_tgt  = float(np.mean([d < 0.15 for d in dists]))
        mean_w  = float(np.mean(wind_mags))

        all_mean_dists.append(mean_d)
        all_max_dists.append(max_d)
        all_time_on_target.append(on_tgt)
        all_rewards.append(ep_reward)
        all_wind_mags.append(mean_w)

        print(f"  ep {ep+1:>3}/{num_episodes} | rew={ep_reward:8.2f} | "
              f"mean_dist={mean_d:.3f} m | max_dist={max_d:.3f} m | "
              f"on_target={on_tgt*100:.1f}% | wind={mean_w:.2f} N")

    return {
        "mode":              "disturbance",
        "wind_scale":        env.wind_scale,
        "mean_reward":       float(np.mean(all_rewards)),
        "mean_pos_error":    float(np.mean(all_mean_dists)),
        "std_pos_error":     float(np.std(all_mean_dists)),
        "mean_max_error":    float(np.mean(all_max_dists)),
        "mean_on_target":    float(np.mean(all_time_on_target)),
        "mean_wind_N":       float(np.mean(all_wind_mags)),
        "num_episodes":      num_episodes,
        "steps_per_ep":      steps_per_ep,
    }


# ── Summary printers ──────────────────────────────────────────────────────

def print_summary(stats: dict, algo: str) -> None:
    print()
    print("─" * 64)
    print(f"  Evaluation summary  [{algo.upper()}]  mode={stats['mode']}")
    print("─" * 64)

    if stats["mode"] == "waypoint":
        n = stats["waypoints_per_ep"]
        print(f"  Episodes            : {stats['num_episodes']}")
        print(f"  Waypoints / ep      : {n}  (r=0.5–1.5 m, z=0.3–1.2 m)")
        print(f"  Mean reward         : {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Mean ep length      : {stats['mean_length']:.1f} steps")
        print(f"  Mean wpts reached   : {stats['mean_wpts_reached']:.2f} / {n}")
        print(f"  Per-waypoint succ   : {stats['wpt_success_rate']*100:.1f}%  (dist < 0.15 m)")
        print(f"  All-waypoints succ  : {stats['all_success_rate']*100:.1f}%")
        print(f"  Mean waypoint dist  : {stats['mean_wpt_dist']:.4f} m")

    else:  # disturbance
        ws = stats["wind_scale"]
        print(f"  Episodes            : {stats['num_episodes']}  "
              f"× {stats['steps_per_ep']} steps ({stats['steps_per_ep']/100:.0f} s)")
        print(f"  Wind scale          : {ws:.2f}  (max {ws*WIND_MAX:.2f} N)")
        print(f"  Mean reward         : {stats['mean_reward']:.2f}")
        print(f"  Mean pos error      : {stats['mean_pos_error']:.4f} ± "
              f"{stats['std_pos_error']:.4f} m")
        print(f"  Mean max error      : {stats['mean_max_error']:.4f} m")
        print(f"  On-target (< 0.15m) : {stats['mean_on_target']*100:.1f}%")
        print(f"  Mean wind force     : {stats['mean_wind_N']:.2f} N")

    print("─" * 64)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = newton.examples.create_parser()
    parser.add_argument("--model", type=str, default="sac_hover_final")
    parser.add_argument("--algo",  type=str, default="sac",
                        choices=["ppo", "sac", "td3"])
    parser.add_argument("--mode",  type=str, default="waypoint",
                        choices=["waypoint", "disturbance"],
                        help="Evaluation mode.")
    parser.add_argument("--num_episodes",     type=int,   default=10)
    parser.add_argument("--waypoints_per_ep", type=int,   default=DEFAULT_WAYPOINTS)
    parser.add_argument("--steps_per_wp",     type=int,   default=DEFAULT_STEPS_PER_WP)
    parser.add_argument("--disturbance_steps",type=int,   default=DISTURBANCE_STEPS,
                        help="Steps per episode in disturbance mode.")
    parser.add_argument("--wind_scale",       type=float, default=0.7,
                        help=f"Wind strength for disturbance mode (0–1, max {WIND_MAX} N).")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--stochastic",       action="store_true")

    viewer, args = newton.examples.init(parser)

    model_path = args.model
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"Model not found: '{model_path}'.  "
            "Train first with train_hover.py or pass --model <path>."
        )

    algo = args.algo.lower()
    print(f"\nLoading {algo.upper()} model from '{model_path}' …")

    wind = args.wind_scale if args.mode == "disturbance" else 0.0
    eval_env = DroneEnv(
        render_mode="human",
        viewer=viewer,
        random_targets=(args.mode == "waypoint"),
        wind_scale=wind,
    )

    if algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(model_path, env=eval_env)
    elif algo == "sac":
        from sbx import SAC
        model = SAC.load(model_path, env=eval_env)
    else:
        from sbx import TD3
        model = TD3.load(model_path, env=eval_env)

    deterministic = not args.stochastic

    if args.mode == "waypoint":
        wpts = min(max(args.waypoints_per_ep, 1), 8)
        print(f"Running {args.num_episodes} ep × {wpts} waypoints "
              f"({args.steps_per_wp} steps/wp, seed={args.seed}) …\n")
        stats = run_waypoint_eval(
            model=model, env=eval_env,
            num_episodes=args.num_episodes,
            waypoints_per_ep=wpts,
            steps_per_wp=args.steps_per_wp,
            deterministic=deterministic,
            seed=args.seed,
        )
    else:
        print(f"Running {args.num_episodes} disturbance episodes "
              f"({args.disturbance_steps} steps, wind_scale={wind:.2f}) …\n")
        stats = run_disturbance_eval(
            model=model, env=eval_env,
            num_episodes=args.num_episodes,
            steps_per_ep=args.disturbance_steps,
            deterministic=deterministic,
            seed=args.seed,
        )

    print_summary(stats, algo)
    eval_env.close()


if __name__ == "__main__":
    main()
