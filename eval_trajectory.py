###########################################################################
# Drone Trajectory Evaluation — continuous waypoint chaining
#
# Unlike eval_drone.py there is NO per-waypoint step budget.
# The drone flies until it reaches each waypoint (dist < 0.15 m), then the
# next target is immediately assigned — no physics reset.
# An episode ends when:
#   • the drone crashes / leaves the arena  (terminated)
#   • it has reached `max_waypoints` waypoints  (success stop)
#   • it has taken more than `max_steps` total steps  (safety truncation)
#
# Trajectory shapes  (--traj):
#   random    — uniform sample each step (training distribution)
#   square    — 4-corner loop, side ≈ 1.13 m, z = 0.7 m
#   lissajous — figure-8  x=sin(t)  y=0.7·sin(2t),  7 waypoints / cycle
#
# Usage:
#   python eval_trajectory.py --model ppo_drone_final --algo ppo
#   python eval_trajectory.py --model ppo_drone_final --algo ppo --traj square
#   python eval_trajectory.py --model ppo_drone_final --algo ppo --traj lissajous
#   python eval_trajectory.py --model ppo_drone_final --algo ppo --max_waypoints 30
###########################################################################

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import numpy as np

import newton.examples
from drone_gym_env import DroneEnv, CF_HOVER_RPM, CF_MAX_RPM, FPS

DEFAULT_MAX_WAYPOINTS = 20
DEFAULT_MAX_STEPS     = 8000   # safety valve: ~80 s per episode
DEFAULT_TRAJ          = "random"


# ── Trajectory generators ─────────────────────────────────────────────────

def _make_waypoints(
    traj: str,
    n: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Return a list of `n` waypoints for the requested trajectory shape."""

    if traj == "random":
        angles = rng.uniform(0.0, 2.0 * np.pi, n)
        radii  = rng.uniform(0.5, 1.5, n)
        alts   = rng.uniform(0.3, 1.2, n)
        return [
            np.array([r * np.cos(a), r * np.sin(a), z], dtype=np.float32)
            for a, r, z in zip(angles, radii, alts)
        ]

    if traj == "square":
        # Four corners at (±0.8, ±0.8, 0.7) — distance from origin ≈ 1.13 m
        h = 0.8
        z = 0.7
        corners = [
            np.array([ h,  h, z], dtype=np.float32),
            np.array([-h,  h, z], dtype=np.float32),
            np.array([-h, -h, z], dtype=np.float32),
            np.array([ h, -h, z], dtype=np.float32),
        ]
        return [corners[i % len(corners)] for i in range(n)]

    if traj == "lissajous":
        # Figure-8: x = A·sin(t),  y = B·sin(2t)
        # 7 evenly-spaced sample points at t = π/4, π/2, …, 7π/4 — skips t=0
        # so the cycle avoids the duplicate center point at t=0 and t=π.
        A, B, z = 1.0, 0.7, 0.7
        ts = np.arange(1, 8) * (np.pi / 4)   # [π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4]
        pts = [
            np.array([A * np.sin(t), B * np.sin(2 * t), z], dtype=np.float32)
            for t in ts
        ]
        return [pts[i % len(pts)] for i in range(n)]

    raise ValueError(f"Unknown --traj '{traj}'. Choose: random | square | lissajous")


def _traj_description(traj: str) -> str:
    if traj == "random":
        return "random  (r=0.5–1.5 m, z=0.3–1.2 m, uniform)"
    if traj == "square":
        return "square  (4-corner loop, half=0.8 m, z=0.7 m)"
    if traj == "lissajous":
        return "lissajous  figure-8  x=sin(t), y=0.7·sin(2t), 7 pts/cycle, z=0.7 m"
    return traj


# ── Evaluation loop ───────────────────────────────────────────────────────

def run_evaluation(
    model,
    env: DroneEnv,
    num_episodes:  int  = 5,
    max_waypoints: int  = DEFAULT_MAX_WAYPOINTS,
    max_steps:     int  = DEFAULT_MAX_STEPS,
    traj:          str  = DEFAULT_TRAJ,
    deterministic: bool = True,
    seed:          int  = 42,
) -> dict:
    rng = np.random.default_rng(seed)

    ep_wpts_reached: list[int]   = []
    ep_total_steps:  list[int]   = []
    ep_crashed:      list[bool]  = []
    ep_mean_rpms:    list[float] = []

    all_leg_steps:   list[int]   = []   # steps per successful leg
    all_leg_dists:   list[float] = []   # wp-to-wp 3-D distance

    for ep in range(num_episodes):
        waypoints = _make_waypoints(traj, max_waypoints, rng)

        obs, _ = env.reset()
        env.set_target(waypoints[0])
        obs = env.get_obs()

        wpts_reached = 0
        total_steps  = 0
        crashed      = False
        ep_rpms:     list[float] = []

        leg_steps   = 0
        prev_wp     = np.zeros(3, dtype=np.float32)   # drone starts at origin
        symbols     = []                               # ✓ / ✗ per leg

        while wpts_reached < max_waypoints and total_steps < max_steps:
            target = waypoints[wpts_reached]

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, _, info = env.step(action)

            total_steps += 1
            leg_steps   += 1

            rpms = info.get("motor_rpms")
            if rpms is not None:
                ep_rpms.append(float(np.mean(rpms)))

            if info["dist"] < 0.15:
                leg_dist = float(np.linalg.norm(target - prev_wp))
                all_leg_steps.append(leg_steps)
                all_leg_dists.append(leg_dist)
                symbols.append("✓")

                wpts_reached += 1
                prev_wp   = target.copy()
                leg_steps = 0

                if wpts_reached < max_waypoints:
                    env.set_target(waypoints[wpts_reached])
                    obs = env.get_obs()

            elif terminated:
                symbols.append("✗")
                crashed = True
                break

        ep_wpts_reached.append(wpts_reached)
        ep_total_steps.append(total_steps)
        ep_crashed.append(crashed)
        ep_mean_rpms.append(float(np.mean(ep_rpms)) if ep_rpms else float("nan"))

        # ── Per-episode print ─────────────────────────────────────────────
        flight_s  = total_steps / FPS
        mean_s_wp = (total_steps / wpts_reached / FPS) if wpts_reached > 0 else float("nan")
        rpm_str   = f"{ep_mean_rpms[-1]:.0f}" if not np.isnan(ep_mean_rpms[-1]) else "n/a"
        crash_tag = "CRASH" if crashed else "ok   "
        sym_str   = "".join(symbols)
        leg_avg   = (np.mean(all_leg_dists[-wpts_reached:])
                     if wpts_reached > 0 else float("nan"))
        leg_str   = f"  leg_avg={leg_avg:.2f}m" if not np.isnan(leg_avg) else ""

        print(f"  ep {ep+1:>2}/{num_episodes} | "
              f"reached={wpts_reached:>2}/{max_waypoints} | "
              f"{crash_tag} | "
              f"flight={flight_s:5.1f}s | "
              f"mean={mean_s_wp:.2f}s/wp | "
              f"rpm≈{rpm_str}")
        print(f"    {sym_str}{leg_str}")

    # ── Aggregate ─────────────────────────────────────────────────────────
    finite_rpms = [r for r in ep_mean_rpms if not np.isnan(r)]
    return {
        "traj":               traj,
        "num_episodes":       num_episodes,
        "max_waypoints":      max_waypoints,
        "mean_wpts_reached":  float(np.mean(ep_wpts_reached)),
        "std_wpts_reached":   float(np.std(ep_wpts_reached)),
        "crash_rate":         float(np.mean(ep_crashed)),
        "mean_flight_s":      float(np.mean(ep_total_steps)) / FPS,
        "mean_s_per_wp":      float(np.mean(all_leg_steps)) / FPS if all_leg_steps else float("nan"),
        "std_s_per_wp":       float(np.std(all_leg_steps))  / FPS if all_leg_steps else float("nan"),
        "mean_leg_dist":      float(np.mean(all_leg_dists))       if all_leg_dists else float("nan"),
        "mean_motor_rpm":     float(np.mean(finite_rpms))         if finite_rpms   else float("nan"),
    }


def print_summary(stats: dict, algo: str) -> None:
    mrpm      = stats["mean_motor_rpm"]
    rpm_str   = f"{mrpm:.0f} RPM" if not np.isnan(mrpm) else "n/a"
    hover_pct = (mrpm / CF_HOVER_RPM * 100) if not np.isnan(mrpm) else float("nan")
    hp_str    = f"  ({hover_pct:.1f}% of hover RPM {CF_HOVER_RPM:.0f})" if not np.isnan(hover_pct) else ""
    spw       = stats["mean_s_per_wp"]
    spw_str   = f"{spw:.2f} ± {stats['std_s_per_wp']:.2f} s" if not np.isnan(spw) else "n/a"
    ld        = stats["mean_leg_dist"]
    ld_str    = f"{ld:.3f} m" if not np.isnan(ld) else "n/a"

    print()
    print("─" * 66)
    print(f"  Trajectory eval  [{algo.upper()}]  — continuous waypoint chaining")
    print("─" * 66)
    print(f"  Trajectory            : {_traj_description(stats['traj'])}")
    print(f"  Episodes              : {stats['num_episodes']}")
    print(f"  Waypoint cap / ep     : {stats['max_waypoints']}")
    print(f"  No step budget        : ends on crash or waypoint cap")
    print(f"  Mean wpts reached     : {stats['mean_wpts_reached']:.1f} ± {stats['std_wpts_reached']:.1f} / {stats['max_waypoints']}")
    print(f"  Crash rate            : {stats['crash_rate']*100:.0f}%")
    print(f"  Mean flight time      : {stats['mean_flight_s']:.1f} s / episode")
    print(f"  Mean time per wp      : {spw_str}")
    print(f"  Mean leg distance     : {ld_str}  (wp-to-wp, not from origin)")
    print(f"  Mean motor RPM        : {rpm_str}{hp_str}")
    print(f"  Max motor RPM         : {CF_MAX_RPM:.0f} RPM")
    print("─" * 66)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = newton.examples.create_parser()
    parser.add_argument("--model", type=str, default="ppo_drone_final",
                        help="Path to saved model (without .zip).")
    parser.add_argument("--algo",  type=str, default="ppo",
                        choices=["ppo", "sac", "td3"])
    parser.add_argument("--traj",  type=str, default=DEFAULT_TRAJ,
                        choices=["random", "square", "lissajous"],
                        help="Trajectory shape: random | square | lissajous (figure-8).")
    parser.add_argument("--num_episodes",  type=int, default=5)
    parser.add_argument("--max_waypoints", type=int, default=DEFAULT_MAX_WAYPOINTS,
                        help=f"Waypoints to attempt per episode (default {DEFAULT_MAX_WAYPOINTS}).")
    parser.add_argument("--max_steps",     type=int, default=DEFAULT_MAX_STEPS,
                        help=f"Safety step cap per episode (default {DEFAULT_MAX_STEPS} ≈ {DEFAULT_MAX_STEPS//FPS}s).")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--stochastic",    action="store_true")

    viewer, args = newton.examples.init(parser)

    model_path = args.model
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"Model not found: '{model_path}'.  Pass --model <path>."
        )

    algo = args.algo.lower()
    print(f"\nLoading {algo.upper()} model from '{model_path}' …")
    print(f"Trajectory : {_traj_description(args.traj)}")
    print(f"CF hover ≈ {CF_HOVER_RPM:.0f} RPM  |  max {CF_MAX_RPM:.0f} RPM\n")

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

    print(f"Running {args.num_episodes} ep × up to {args.max_waypoints} wps "
          f"(no time limit per wp, seed={args.seed}) …\n")

    stats = run_evaluation(
        model=model,
        env=eval_env,
        num_episodes=args.num_episodes,
        max_waypoints=args.max_waypoints,
        max_steps=args.max_steps,
        traj=args.traj,
        deterministic=not args.stochastic,
        seed=args.seed,
    )

    print_summary(stats, algo)
    eval_env.close()


if __name__ == "__main__":
    main()
