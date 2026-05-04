# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Unified Drone Training — PPO | SAC | TD3
#
# All algorithms log to ./drone_logs/<algo>/ so a single TensorBoard
# session can compare convergence across all three:
#   tensorboard --logdir drone_logs
#
# Usage:
#   python train_drone.py --algo ppo
#   python train_drone.py --algo sac
#   python train_drone.py --algo td3
#   python train_drone.py --algo sac --headless           # no OpenGL (newton flag)
#   python train_drone.py --algo sac --obs_noise         # sensor noise
#   python train_drone.py --algo sac --total_timesteps 3000000
#
# PPO  → stable_baselines3 (on-policy, parallel rollouts)
# SAC  → sbx (off-policy, JAX-accelerated, entropy regularisation)
# TD3  → sbx (off-policy, JAX-accelerated, delayed policy update)
###########################################################################

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import math
from collections import deque

import numpy as np
import warp as wp

import newton
import newton.examples
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from drone_gym_env import DroneEnv, MAX_EPISODE_STEPS, FPS, SIM_DT, DRONE_SIZE


# ── Curriculum callback ───────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """Linearly ramps env.curriculum from 0 → 1 over the first half of training.

    With curriculum=0 (early): small penalty weights let the drone survive
    even when far from the target.  As it learns to fly, weights increase,
    demanding tighter position and velocity control.
    """

    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self._total = total_timesteps

    def _on_step(self) -> bool:
        t = min(self.num_timesteps / (self._total * 0.5), 1.0)
        for env in self.training_env.envs:
            env.curriculum = t
        return True


# ── Render callback ───────────────────────────────────────────────────────

class RenderCallback(BaseCallback):
    """Renders all parallel training envs in a grid every N steps."""

    GRID_SPACING = 4.0

    def __init__(self, viewer, train_vec_env, render_freq: int = 5_000, verbose: int = 0):
        super().__init__(verbose)
        self._viewer     = viewer
        self._train_env  = train_vec_env
        self.render_freq = render_freq
        self._last_render = 0
        self._render_t    = 0.0
        n = train_vec_env.num_envs
        self._grid_cols = max(1, math.ceil(math.sqrt(n)))

    def _on_step(self) -> bool:
        if self._viewer is None:
            return True
        if self.num_timesteps - self._last_render >= self.render_freq:
            self._last_render = self.num_timesteps
            self._render_grid()
        return True

    def _grid_offset(self, i: int) -> np.ndarray:
        row, col = divmod(i, self._grid_cols)
        return np.array([col * self.GRID_SPACING, row * self.GRID_SPACING, 0.0], dtype=np.float32)

    def _dist_color(self, dist: float) -> wp.vec3:
        t = float(np.clip(dist / 2.0, 0.0, 1.0))
        return wp.vec3(t, 1.0 - t * 0.8, 0.0)

    def _render_grid(self) -> None:
        envs = self._train_env.envs
        drone_tfs, target_tfs, drone_colors, target_colors = [], [], [], []

        for i, env in enumerate(envs):
            offset = self._grid_offset(i)
            q_np   = env._state.body_q.numpy()[0]
            pos    = q_np[:3].astype(np.float32)
            quat   = q_np[3:].astype(np.float32)
            drone_tfs.append(wp.transform(
                wp.vec3(*(pos + offset).tolist()),
                wp.quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            ))
            drone_colors.append(self._dist_color(env.last_dist))
            target_tfs.append(wp.transform(
                wp.vec3(*(env._target + offset).tolist()), wp.quat_identity(),
            ))
            target_colors.append(wp.vec3(1.0, 0.2, 0.0))

        self._render_t += SIM_DT
        try:
            self._viewer.begin_frame(self._render_t)
            self._viewer.log_shapes(
                "/train/drones", newton.GeoType.BOX,
                (DRONE_SIZE, DRONE_SIZE, DRONE_SIZE * 0.08),
                wp.array(drone_tfs,    dtype=wp.transform),
                wp.array(drone_colors, dtype=wp.vec3),
            )
            self._viewer.log_shapes(
                "/train/targets", newton.GeoType.SPHERE, (0.07,),
                wp.array(target_tfs,    dtype=wp.transform),
                wp.array(target_colors, dtype=wp.vec3),
            )
            self._viewer.end_frame()
        except Exception:
            pass

        if self.verbose >= 1:
            dists = [e.last_dist for e in envs]
            print(f"[render @ {self.num_timesteps:>8,}]  "
                  f"mean_dist={np.mean(dists):.3f}  min_dist={np.min(dists):.3f}")


# ── Metrics callback ──────────────────────────────────────────────────────

class MetricsCallback(BaseCallback):
    """Logs per-episode metrics and reward components to TensorBoard."""

    _RC_KEYS = ("pos_c", "orient_c", "vel_c", "ang_c", "act_c", "survival")

    def __init__(self, log_freq: int = 1_000, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq    = log_freq
        self._last_log   = 0
        self._dists      = deque(maxlen=window)
        self._uprights   = deque(maxlen=window)
        self._ep_lens    = deque(maxlen=window)
        self._ep_rewards = deque(maxlen=window)
        self._successes  = deque(maxlen=window)
        self._rc: dict[str, deque] = {k: deque(maxlen=window) for k in self._RC_KEYS}

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if done and "terminal_dist" in info:
                d = info["terminal_dist"]
                self._dists.append(d)
                self._uprights.append(info["terminal_upright"])
                self._ep_lens.append(info["terminal_ep_len"])
                self._ep_rewards.append(info["terminal_reward"])
                self._successes.append(float(d < 0.15))
            for k in self._RC_KEYS:
                v = info.get("reward_components", {}).get(k)
                if v is not None:
                    self._rc[k].append(v)

        if self.num_timesteps - self._last_log >= self.log_freq and self._dists:
            self._last_log = self.num_timesteps
            self._flush()
        return True

    def _flush(self) -> None:
        rec = self.logger.record
        rec("metrics/terminal_dist",    np.mean(self._dists))
        rec("metrics/success_rate",     np.mean(self._successes))
        rec("metrics/terminal_upright", np.mean(self._uprights))
        rec("metrics/ep_length",        np.mean(self._ep_lens))
        rec("metrics/ep_reward",        np.mean(self._ep_rewards))
        for k, buf in self._rc.items():
            if buf:
                rec(f"reward_components/{k}", np.mean(buf))


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = newton.examples.create_parser()
    parser.add_argument("--algo",            type=str,   default="sac",
                        choices=["ppo", "sac", "td3"],
                        help="RL algorithm to train.")
    parser.add_argument("--num_envs",        type=int,   default=16)
    parser.add_argument("--total_timesteps", type=int,   default=1_500_000)
    parser.add_argument("--checkpoint_freq", type=int,   default=50_000)
    parser.add_argument("--render_freq",     type=int,   default=5_000)
    parser.add_argument("--checkpoint_dir",  type=str,   default="checkpoints")
    parser.add_argument("--learning_rate",   type=float, default=3e-4)
    parser.add_argument("--gamma",           type=float, default=0.99)
    parser.add_argument("--obs_noise",       action="store_true",
                        help="Add sensor noise to observations (robustness).")
    parser.add_argument("--no_curriculum",   action="store_true",
                        help="Disable reward curriculum (fixed target weights).")

    viewer, args = newton.examples.init(parser)
    # viewer is already None when --headless or --viewer null is passed by newton

    algo = args.algo.lower()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Training environments ─────────────────────────────────────────────
    def _make_env():
        def _init():
            return DroneEnv(
                render_mode=None, viewer=None,
                random_targets=True,
                obs_noise=args.obs_noise,
                curriculum=0.0,
            )
        return _init

    train_env = DummyVecEnv([_make_env() for _ in range(args.num_envs)])

    # ── Model ─────────────────────────────────────────────────────────────
    tb_log = "./drone_logs/"

    if algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO(
            "MlpPolicy", train_env,
            verbose=1,
            learning_rate=args.learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=tb_log,
        )
    elif algo == "sac":
        from sbx import SAC
        model = SAC(
            "MlpPolicy", train_env,
            verbose=1,
            learning_rate=args.learning_rate,
            buffer_size=500_000,
            batch_size=256,
            learning_starts=5_000,
            gamma=args.gamma,
            tau=0.005,
            ent_coef="auto",
            target_entropy="auto",
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=tb_log,
        )
    else:  # td3
        from sbx import TD3
        model = TD3(
            "MlpPolicy", train_env,
            verbose=1,
            learning_rate=args.learning_rate,
            buffer_size=500_000,
            batch_size=256,
            learning_starts=5_000,
            gamma=args.gamma,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=tb_log,
        )

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.num_envs, 1),
            save_path=args.checkpoint_dir,
            name_prefix=f"{algo}_drone",
            verbose=1,
        ),
        RenderCallback(
            viewer=viewer,
            train_vec_env=train_env,
            render_freq=args.render_freq,
            verbose=1,
        ),
        MetricsCallback(log_freq=1_000, window=100),
    ]
    if not args.no_curriculum:
        callbacks.append(CurriculumCallback(args.total_timesteps))

    print(
        f"\n  algo={algo.upper()}  steps={args.total_timesteps:,}  "
        f"envs={args.num_envs}  viewer={'off' if viewer is None else 'on'}  "
        f"obs_noise={args.obs_noise}  curriculum={not args.no_curriculum}\n"
        f"  checkpoints → {args.checkpoint_dir}/\n"
        f"  TensorBoard → tensorboard --logdir drone_logs\n"
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name=algo,   # creates drone_logs/sac_1/, drone_logs/ppo_1/, etc.
        progress_bar=True,
    )

    save_path = f"{algo}_drone_final"
    model.save(save_path)
    print(f"\nSaved → {save_path}.zip")
    train_env.close()


if __name__ == "__main__":
    main()
