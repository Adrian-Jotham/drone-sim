# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Drone Landing Training — PPO
#
# Trains a policy to descend and land softly on a static platform that is
# placed at a random (x, y) position each episode.
# The drone also spawns at a random position in the air.
#
# Usage:
#   python train_landing.py
#   python train_landing.py --headless
#   python train_landing.py --num_envs 64 --total_timesteps 3000000
#
# Monitor:
#   tensorboard --logdir landing_logs
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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from drone_landing_env import (
    DroneLandingEnv,
    MAX_EPISODE_STEPS,
    FPS,
    SIM_DT,
    DRONE_SIZE,
    LAND_DIST,
    LAND_SPEED,
)


# ── Metrics callback ──────────────────────────────────────────────────────

class LandingMetricsCallback(BaseCallback):
    """Logs per-episode landing metrics to TensorBoard."""

    _RC_KEYS = ("pos_xy", "pos_z", "att", "vel_xy", "vel_z", "ang", "smooth", "sym", "descent", "land", "crash")

    def __init__(self, log_freq: int = 1_000, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq    = log_freq
        self._last_log   = 0
        self._dists      = deque(maxlen=window)
        self._speeds     = deque(maxlen=window)
        self._uprights   = deque(maxlen=window)
        self._ep_lens    = deque(maxlen=window)
        self._ep_rewards = deque(maxlen=window)
        self._landings   = deque(maxlen=window)   # 1 = success, 0 = fail
        self._rc: dict[str, deque] = {k: deque(maxlen=window) for k in self._RC_KEYS}

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if done and "terminal_dist" in info:
                self._dists.append(info["terminal_dist"])
                self._speeds.append(info["terminal_speed"])
                self._uprights.append(info["terminal_upright"])
                self._ep_lens.append(info["terminal_ep_len"])
                self._ep_rewards.append(info["terminal_reward"])
                self._landings.append(float(info["terminal_landed"]))
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
        rec("landing/success_rate",     np.mean(self._landings))
        rec("landing/terminal_dist",    np.mean(self._dists))
        rec("landing/terminal_speed",   np.mean(self._speeds))
        rec("landing/terminal_upright", np.mean(self._uprights))
        rec("landing/ep_length",        np.mean(self._ep_lens))
        rec("landing/ep_reward",        np.mean(self._ep_rewards))
        for k, buf in self._rc.items():
            if buf:
                rec(f"reward_components/{k}", np.mean(buf))

        if self.verbose >= 1:
            print(
                f"[{self.num_timesteps:>8,}]  "
                f"success={np.mean(self._landings):.2%}  "
                f"dist={np.mean(self._dists):.3f}m  "
                f"speed={np.mean(self._speeds):.3f}m/s"
            )


# ── Render callback ───────────────────────────────────────────────────────

class LandingRenderCallback(BaseCallback):
    """Renders all parallel training envs in a grid every N steps."""

    GRID_SPACING = 5.0

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

    def _render_grid(self) -> None:
        envs = self._train_env.envs
        drone_tfs, plat_tfs = [], []
        drone_colors, plat_colors = [], []

        for i, env in enumerate(envs):
            offset = self._grid_offset(i)

            q_np = env._state.body_q.numpy()[0]
            pos  = q_np[:3].astype(np.float32)
            quat = q_np[3:].astype(np.float32)
            drone_tfs.append(wp.transform(
                wp.vec3(*(pos + offset).tolist()),
                wp.quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            ))
            # colour: green when close, red when far
            t = float(np.clip(env.last_dist / 2.0, 0.0, 1.0))
            drone_colors.append(wp.vec3(t, 1.0 - t * 0.8, 0.0))

            px, py, pz = (env._platform + offset).tolist()
            plat_tfs.append(wp.transform(wp.vec3(px, py, pz), wp.quat_identity()))
            plat_colors.append(wp.vec3(0.2, 0.8, 0.2))

        self._render_t += SIM_DT
        try:
            self._viewer.begin_frame(self._render_t)
            self._viewer.log_shapes(
                "/train/drones", newton.GeoType.BOX,
                (DRONE_SIZE, DRONE_SIZE, DRONE_SIZE * 0.08),
                wp.array(drone_tfs,   dtype=wp.transform),
                wp.array(drone_colors, dtype=wp.vec3),
            )
            self._viewer.log_shapes(
                "/train/platforms", newton.GeoType.BOX, (0.40, 0.40, 0.02),
                wp.array(plat_tfs,   dtype=wp.transform),
                wp.array(plat_colors, dtype=wp.vec3),
            )
            self._viewer.end_frame()
        except Exception:
            pass


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = newton.examples.create_parser()
    parser.add_argument("--num_envs",        type=int,   default=64)
    parser.add_argument("--total_timesteps", type=int,   default=2_000_000)
    parser.add_argument("--checkpoint_freq", type=int,   default=50_000)
    parser.add_argument("--render_freq",     type=int,   default=5_000)
    parser.add_argument("--checkpoint_dir",  type=str,   default="checkpoints_landing")
    parser.add_argument("--learning_rate",   type=float, default=3e-4)
    parser.add_argument("--obs_noise",       action="store_true")

    viewer, args = newton.examples.init(parser)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Environments ──────────────────────────────────────────────────────
    def _make_env():
        def _init():
            return DroneLandingEnv(
                render_mode=None,
                viewer=None,
                obs_noise=args.obs_noise,
            )
        return _init

    train_env = DummyVecEnv([_make_env() for _ in range(args.num_envs)])

    # ── PPO model ─────────────────────────────────────────────────────────
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,       # small entropy bonus to keep exploration alive
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log="./landing_logs/",
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.num_envs, 1),
            save_path=args.checkpoint_dir,
            name_prefix="ppo_landing",
            verbose=1,
        ),
        LandingRenderCallback(
            viewer=viewer,
            train_vec_env=train_env,
            render_freq=args.render_freq,
            verbose=1,
        ),
        LandingMetricsCallback(log_freq=1_000, window=100, verbose=1),
    ]

    print(
        f"\n  PPO Landing  steps={args.total_timesteps:,}  "
        f"envs={args.num_envs}  viewer={'off' if viewer is None else 'on'}  "
        f"obs_noise={args.obs_noise}\n"
        f"  Success threshold: dist<{LAND_DIST}m, speed<{LAND_SPEED}m/s\n"
        f"  checkpoints → {args.checkpoint_dir}/\n"
        f"  TensorBoard → tensorboard --logdir landing_logs\n"
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name="ppo",
        progress_bar=True,
    )

    save_path = "ppo_landing_final"
    model.save(save_path)
    print(f"\nSaved → {save_path}.zip")
    train_env.close()


if __name__ == "__main__":
    main()
