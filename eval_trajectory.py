###########################################################################
# Drone Trajectory Evaluation — continuous waypoint chaining
#
# An episode ends when:
#   • the drone crashes / leaves the arena  (CRASH)
#   • it reaches `max_waypoints` waypoints  (SUCCESS)
#   • it exceeds `max_steps` total steps    (TIMEOUT)
#
# Trajectory shapes  (--traj):
#   random    — uniform sample, r=0.5–1.5 m, z=0.3–1.2 m  (training dist.)
#   square    — 4-corner loop, half=0.8 m, z=0.7 m
#   lissajous — figure-8  x=sin(t)  y=0.7·sin(2t),  7 pts / cycle
#
# Usage:
#   python eval_trajectory.py --model ppo_drone_final_s1 --algo ppo
#   python eval_trajectory.py --model ppo_drone_final_s1 --algo ppo --traj square
#   python eval_trajectory.py --model ppo_drone_final_s1 --algo ppo --traj lissajous
#   python eval_trajectory.py --model ppo_drone_final_s1 --algo ppo \
#       --max_waypoints 1 --num_episodes 50   # clean per-attempt success rate
###########################################################################

import os
import time
import numpy as np
import torch as th

import newton.examples
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy, Actor as TD3Actor
from stable_baselines3.sac.policies import SACPolicy, Actor as SACActor
from drone_gym_env import DroneEnv, CF_HOVER_RPM, CF_MAX_RPM, FPS


# ── AAC policy stubs (mirrors train_drone.py) ─────────────────────────────
# Required to load checkpoints saved with the asymmetric actor-critic setup.

_ACTOR_OBS_DIM = 22

class _ActorSliceExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space) -> None:
        super().__init__(observation_space, features_dim=_ACTOR_OBS_DIM)
    def forward(self, obs: th.Tensor) -> th.Tensor:
        return obs[..., :_ACTOR_OBS_DIM]

class AACTd3Policy(TD3Policy):
    def make_actor(self, features_extractor=None):
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, _ActorSliceExtractor(self.observation_space)
        )
        return TD3Actor(**actor_kwargs).to(self.device)

class AACSacPolicy(SACPolicy):
    def make_actor(self, features_extractor=None):
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, _ActorSliceExtractor(self.observation_space)
        )
        return SACActor(**actor_kwargs).to(self.device)


# ── Model loader with obs-space mismatch handling ─────────────────────────

def _load_model(algo: str, path: str, env: DroneEnv):
    """Load a saved model and return (model, obs_dim).

    obs_dim is the number of obs features the model expects. When the
    environment now returns more features than the model was trained on
    (e.g. old 22-D checkpoint with new 26-D env), obs_dim < env obs dim
    and the caller must slice obs[:obs_dim] before calling model.predict().
    Returns obs_dim=None when the model matches the current env.
    """
    from stable_baselines3 import PPO, SAC, TD3

    cls        = {"ppo": PPO, "sac": SAC, "td3": TD3}[algo]
    aac_policy = {"ppo": None, "sac": AACSacPolicy, "td3": AACTd3Policy}[algo]

    # ── Try normal load first (obs spaces already match) ──────────────────
    try:
        model = cls.load(path, env=env)
        return model, None
    except Exception:
        pass

    # ── Obs-space mismatch: load without env, then detect the gap ─────────
    for custom_objects in [None, {"policy_class": aac_policy}]:
        try:
            model = cls.load(path, env=None, custom_objects=custom_objects)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f"Could not load model from '{path}' for algo '{algo}'.")

    model_obs_dim = model.observation_space.shape[0]
    env_obs_dim   = env.observation_space.shape[0]
    obs_dim       = model_obs_dim if model_obs_dim != env_obs_dim else None
    return model, obs_dim

_STEP_DT = 1.0 / FPS   # wall-clock budget per step for real-time playback

DEFAULT_MAX_WAYPOINTS = 20
DEFAULT_MAX_STEPS     = 8000   # ~80 s per episode
DEFAULT_TRAJ          = "random"

# Distance bands for per-bin success rate (random trajectory only)
_DIST_BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 3.0)]


# ── Trajectory generators ─────────────────────────────────────────────────

def _make_waypoints(traj: str, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    if traj == "random":
        angles = rng.uniform(0.0, 2.0 * np.pi, n)
        radii  = rng.uniform(0.5, 1.5, n)
        alts   = rng.uniform(0.3, 1.2, n)
        return [
            np.array([r * np.cos(a), r * np.sin(a), z], dtype=np.float32)
            for a, r, z in zip(angles, radii, alts)
        ]

    if traj == "square":
        h, z = 0.8, 0.7
        corners = [
            np.array([ h,  h, z], dtype=np.float32),
            np.array([-h,  h, z], dtype=np.float32),
            np.array([-h, -h, z], dtype=np.float32),
            np.array([ h, -h, z], dtype=np.float32),
        ]
        return [corners[i % len(corners)] for i in range(n)]

    if traj == "lissajous":
        A, B, z = 1.0, 0.7, 0.7
        ts  = np.arange(1, 8) * (np.pi / 4)
        pts = [np.array([A * np.sin(t), B * np.sin(2 * t), z], dtype=np.float32) for t in ts]
        return [pts[i % len(pts)] for i in range(n)]

    raise ValueError(f"Unknown --traj '{traj}'. Choose: random | square | lissajous")


def _traj_description(traj: str) -> str:
    return {
        "random":    "random    r=0.5–1.5 m, z=0.3–1.2 m, uniform",
        "square":    "square    4-corner loop, half=0.8 m, z=0.7 m",
        "lissajous": "lissajous figure-8  x=sin(t), y=0.7·sin(2t), 7 pts/cycle, z=0.7 m",
    }.get(traj, traj)


# ── Evaluation loop ───────────────────────────────────────────────────────

def run_evaluation(
    model,
    env:           DroneEnv,
    num_episodes:  int       = 5,
    max_waypoints: int       = DEFAULT_MAX_WAYPOINTS,
    max_steps:     int       = DEFAULT_MAX_STEPS,
    traj:          str       = DEFAULT_TRAJ,
    deterministic: bool      = True,
    seed:          int       = 42,
    realtime:      bool      = True,
    obs_dim:       int|None  = None,
) -> dict:
    rng = np.random.default_rng(seed)

    # Episode-level accumulators
    ep_wpts_reached: list[int]   = []
    ep_total_steps:  list[int]   = []
    ep_outcomes:     list[str]   = []   # "crash" | "timeout" | "success"
    ep_mean_rpms:    list[float] = []

    # Successful-leg accumulators
    ok_leg_steps:   list[int]   = []   # steps to complete a leg
    ok_leg_dists:   list[float] = []   # wp-to-wp distance (m)
    ok_arrival_spd: list[float] = []   # speed at moment of arrival (m/s)

    # Failed-leg accumulators (crash or timeout on that leg)
    fail_min_dist:  list[float] = []   # closest distance reached before failing
    fail_leg_dists: list[float] = []   # intended leg distance

    for ep in range(num_episodes):
        waypoints = _make_waypoints(traj, max_waypoints, rng)

        # Set target BEFORE reset so the spawn position is within curriculum
        # distance of the first waypoint — matches training exactly.
        env._target = waypoints[0].copy()
        obs, _ = env.reset()
        env.set_target(waypoints[0])
        obs = env.get_obs()

        wpts_reached = 0
        total_steps  = 0
        outcome      = "timeout"
        ep_rpms: list[float] = []

        leg_steps    = 0
        leg_min_dist = float("inf")
        prev_wp      = np.zeros(3, dtype=np.float32)
        symbols: list[str] = []

        while wpts_reached < max_waypoints and total_steps < max_steps:
            t0 = time.perf_counter()
            obs_input = obs[:obs_dim] if obs_dim is not None else obs
            action, _ = model.predict(obs_input, deterministic=deterministic)
            obs, _, terminated, _, info = env.step(action)
            if realtime:
                remaining = _STEP_DT - (time.perf_counter() - t0)
                if remaining > 0:
                    time.sleep(remaining)

            total_steps  += 1
            leg_steps    += 1

            rpms = info.get("motor_rpms")
            if rpms is not None:
                ep_rpms.append(float(np.mean(rpms)))

            dist = float(info.get("dist", float("inf")))
            leg_min_dist = min(leg_min_dist, dist)

            if dist < 0.15:
                leg_dist     = float(np.linalg.norm(waypoints[wpts_reached] - prev_wp))
                arrival_spd  = float(np.linalg.norm(obs[12:15]))   # v in obs layout

                ok_leg_steps.append(leg_steps)
                ok_leg_dists.append(leg_dist)
                ok_arrival_spd.append(arrival_spd)
                symbols.append("✓")

                prev_wp      = waypoints[wpts_reached].copy()
                wpts_reached += 1
                leg_steps    = 0
                leg_min_dist = float("inf")

                if wpts_reached < max_waypoints:
                    env.set_target(waypoints[wpts_reached])
                    obs = env.get_obs()
                else:
                    outcome = "success"

            elif terminated:
                leg_dist = float(np.linalg.norm(waypoints[wpts_reached] - prev_wp))
                fail_min_dist.append(leg_min_dist)
                fail_leg_dists.append(leg_dist)
                symbols.append("✗")
                outcome = "crash"
                break

        # Record final incomplete leg for timeout
        if outcome == "timeout" and leg_min_dist < float("inf"):
            leg_dist = float(np.linalg.norm(waypoints[wpts_reached] - prev_wp))
            fail_min_dist.append(leg_min_dist)
            fail_leg_dists.append(leg_dist)
            symbols.append("…")

        ep_wpts_reached.append(wpts_reached)
        ep_total_steps.append(total_steps)
        ep_outcomes.append(outcome)
        ep_mean_rpms.append(float(np.mean(ep_rpms)) if ep_rpms else float("nan"))

        # ── Per-episode print ─────────────────────────────────────────────
        flight_s    = total_steps / FPS
        mean_s_leg  = (float(np.mean(ok_leg_steps[-wpts_reached:])) / FPS
                       if wpts_reached > 0 else float("nan"))
        rpm_str     = f"{ep_mean_rpms[-1]:.0f}" if not np.isnan(ep_mean_rpms[-1]) else "n/a"
        tag         = {"crash": "CRASH  ", "timeout": "TIMEOUT", "success": "SUCCESS"}[outcome]
        mean_str    = f"  {mean_s_leg:.1f}s/wp" if not np.isnan(mean_s_leg) else ""

        print(f"  ep {ep+1:>2}/{num_episodes} | reached={wpts_reached:>2}/{max_waypoints} | "
              f"{tag} | flight={flight_s:5.1f}s{mean_str} | rpm≈{rpm_str}")
        print(f"    {''.join(symbols)}")

    # ── Aggregate stats ───────────────────────────────────────────────────
    total_attempts   = len(ok_leg_dists) + len(fail_leg_dists)
    attempt_sr       = len(ok_leg_dists) / total_attempts if total_attempts > 0 else float("nan")

    finite_rpms      = [r for r in ep_mean_rpms if not np.isnan(r)]
    mean_rpm         = float(np.mean(finite_rpms))      if finite_rpms      else float("nan")
    mean_rpm_dev     = float(np.mean(np.abs(np.array(finite_rpms) - CF_HOVER_RPM))) \
                       if finite_rpms else float("nan")

    # Distance-binned success rate (random trajectory only)
    bin_stats: dict = {}
    if traj == "random":
        all_pairs = ([(d, True)  for d in ok_leg_dists] +
                     [(d, False) for d in fail_leg_dists])
        for lo, hi in _DIST_BINS:
            band = [(d, s) for d, s in all_pairs if lo <= d < hi]
            if band:
                bin_stats[f"{lo:.1f}–{hi:.1f}m"] = {
                    "n":            len(band),
                    "success_rate": sum(s for _, s in band) / len(band),
                }

    return {
        "traj":               traj,
        "num_episodes":       num_episodes,
        "max_waypoints":      max_waypoints,
        # Episode-level
        "crash_rate":         float(np.mean([o == "crash"   for o in ep_outcomes])),
        "timeout_rate":       float(np.mean([o == "timeout" for o in ep_outcomes])),
        "success_rate":       float(np.mean([o == "success" for o in ep_outcomes])),
        "mean_wpts_reached":  float(np.mean(ep_wpts_reached)),
        "std_wpts_reached":   float(np.std(ep_wpts_reached)),
        "mean_flight_s":      float(np.mean(ep_total_steps)) / FPS,
        # Per-attempt (leg-level)
        "attempt_success_rate": attempt_sr,
        "total_attempts":       total_attempts,
        # Successful legs
        "mean_s_per_wp":      float(np.mean(ok_leg_steps)) / FPS if ok_leg_steps else float("nan"),
        "std_s_per_wp":       float(np.std(ok_leg_steps))  / FPS if ok_leg_steps else float("nan"),
        "mean_leg_dist":      float(np.mean(ok_leg_dists))        if ok_leg_dists else float("nan"),
        "mean_arrival_speed": float(np.mean(ok_arrival_spd))      if ok_arrival_spd else float("nan"),
        # Failed legs
        "mean_fail_min_dist": float(np.mean(fail_min_dist))       if fail_min_dist else float("nan"),
        # Motor
        "mean_motor_rpm":     mean_rpm,
        "mean_rpm_dev":       mean_rpm_dev,
        # Bins
        "bin_stats":          bin_stats,
    }


def print_summary(stats: dict, algo: str) -> None:
    nan = float("nan")

    def _f(v, fmt, suffix=""):
        return f"{v:{fmt}}{suffix}" if not np.isnan(v) else "n/a"

    mrpm      = stats["mean_motor_rpm"]
    hover_pct = (mrpm / CF_HOVER_RPM * 100) if not np.isnan(mrpm) else nan

    print()
    print("═" * 68)
    print(f"  EVALUATION REPORT  [{algo.upper()}]  —  {_traj_description(stats['traj'])}")
    print("═" * 68)
    print(f"  Episodes run          : {stats['num_episodes']}  "
          f"(waypoint cap = {stats['max_waypoints']} / ep)")

    print()
    print("  ── Episode outcomes ────────────────────────────────────────────")
    print(f"  Crash rate            : {stats['crash_rate']*100:5.1f}%")
    print(f"  Timeout rate          : {stats['timeout_rate']*100:5.1f}%")
    print(f"  Full-success rate     : {stats['success_rate']*100:5.1f}%  "
          f"(all {stats['max_waypoints']} wps reached)")
    print(f"  Mean wps / episode    : {_f(stats['mean_wpts_reached'], '.1f')} "
          f"± {_f(stats['std_wpts_reached'], '.1f')}  "
          f"(cap {stats['max_waypoints']})")
    print(f"  Mean flight time      : {_f(stats['mean_flight_s'], '.1f', ' s / ep')}")

    print()
    print("  ── Per-waypoint-attempt (leg-level) ────────────────────────────")
    print(f"  Leg success rate      : {_f(stats['attempt_success_rate']*100, '.1f', '%')}  "
          f"({stats['total_attempts']} total attempts)")
    print(f"  Mean time to reach wp : {_f(stats['mean_s_per_wp'], '.2f')} "
          f"± {_f(stats['std_s_per_wp'], '.2f')} s  (successful legs only)")
    print(f"  Mean leg distance     : {_f(stats['mean_leg_dist'], '.3f', ' m')}  "
          f"(wp-to-wp)")
    print(f"  Mean arrival speed    : {_f(stats['mean_arrival_speed'], '.3f', ' m/s')}  "
          f"(speed when dist < 0.15 m)")
    print(f"  Mean closest (failed) : {_f(stats['mean_fail_min_dist'], '.3f', ' m')}  "
          f"(best dist before crash/timeout on failed legs)")

    print()
    print("  ── Motor / Thrust ──────────────────────────────────────────────")
    print(f"  Mean motor RPM        : {_f(mrpm, '.0f', ' RPM')}  "
          f"({_f(hover_pct, '.1f', '% of hover')})")
    print(f"  Hover RPM             : {CF_HOVER_RPM:.0f} RPM  (required for level flight)")
    print(f"  Mean |RPM − hover|    : {_f(stats['mean_rpm_dev'], '.0f', ' RPM')}  "
          f"(0 = perfect hover thrust)")
    print(f"  Max RPM (hardware)    : {CF_MAX_RPM:.0f} RPM")

    if stats["bin_stats"]:
        print()
        print("  ── Leg success rate by distance (random traj) ──────────────────")
        for band, bs in stats["bin_stats"].items():
            bar = "█" * int(bs["success_rate"] * 20) + "░" * (20 - int(bs["success_rate"] * 20))
            print(f"  {band:>10}  [{bar}]  {bs['success_rate']*100:5.1f}%  (n={bs['n']})")

    print("═" * 68)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = newton.examples.create_parser()
    parser.add_argument("--model", type=str, default="ppo_drone_final_s1",
                        help="Path to saved model (without .zip).")
    parser.add_argument("--algo",  type=str, default="ppo",
                        choices=["ppo", "sac", "td3"])
    parser.add_argument("--traj",  type=str, default=DEFAULT_TRAJ,
                        choices=["random", "square", "lissajous"])
    parser.add_argument("--num_episodes",  type=int, default=10)
    parser.add_argument("--max_waypoints", type=int, default=DEFAULT_MAX_WAYPOINTS,
                        help=f"Waypoints per episode (default {DEFAULT_MAX_WAYPOINTS}). "
                             f"Use 1 + many episodes for a clean per-attempt success rate.")
    parser.add_argument("--max_steps",     type=int, default=DEFAULT_MAX_STEPS,
                        help=f"Step cap per episode (default {DEFAULT_MAX_STEPS} ≈ {DEFAULT_MAX_STEPS//FPS} s).")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--stochastic",    action="store_true")
    parser.add_argument("--no_realtime",   action="store_true",
                        help="Run as fast as possible (default: throttle to real-world 100 Hz).")
    parser.add_argument("--clean_start",   action="store_true",
                        help="Spawn upright with zero velocity (curriculum=0). "
                             "Tests pure navigation without spawn-state recovery.")

    viewer, args = newton.examples.init(parser)

    if not os.path.exists(args.model) and not os.path.exists(args.model + ".zip"):
        raise FileNotFoundError(f"Model not found: '{args.model}'.  Pass --model <path>.")

    algo = args.algo.lower()
    print(f"\nLoading {algo.upper()} model from '{args.model}' …")
    print(f"Trajectory : {_traj_description(args.traj)}")
    print(f"CF hover ≈ {CF_HOVER_RPM:.0f} RPM  |  max {CF_MAX_RPM:.0f} RPM\n")

    eval_env = DroneEnv(render_mode="human", viewer=viewer, random_targets=False)
    eval_env.curriculum = 0.0 if args.clean_start else 1.0

    model, obs_dim = _load_model(algo, args.model, eval_env)
    if obs_dim is not None:
        print(f"[compat] model expects {obs_dim}-D obs, env returns "
              f"{eval_env.observation_space.shape[0]}-D — slicing obs automatically.\n")

    spawn_mode = "clean (curriculum=0.0)" if args.clean_start else "randomised (curriculum=1.0)"
    print(f"Running {args.num_episodes} ep × up to {args.max_waypoints} wps  "
          f"(spawn={spawn_mode}, seed={args.seed}) …\n")

    stats = run_evaluation(
        model=model,
        env=eval_env,
        num_episodes=args.num_episodes,
        max_waypoints=args.max_waypoints,
        max_steps=args.max_steps,
        traj=args.traj,
        deterministic=not args.stochastic,
        seed=args.seed,
        realtime=not args.no_realtime,
        obs_dim=obs_dim,
    )

    print_summary(stats, algo)
    eval_env.close()


if __name__ == "__main__":
    main()
