# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Drone Landing Environment

The drone spawns at a random aerial position.
A static landing platform is placed at a random (x, y) on the ground.
Goal: descend and land softly on the platform.

Observation layout (22-D): identical to DroneEnv
  [0:3]   p_err  — position error = pos − platform_center (xyzw quat)
  [3:12]  R_flat — 3×3 rotation matrix, row-major
  [12:15] v      — linear velocity, world frame
  [15:18] w      — angular velocity, body frame
  [18:22] a_prev — last action

Action (4-D, ∈ [-1, 1]): hover-centred motor setpoints.

Reward per step (shaped exponential kernels, no survival bias):
  r = 1.0·exp(−3‖p_xy‖²) + 2.0·exp(−5·Δz²) + 1.5·cos_tilt
    − 0.3·‖v_xy‖² − 0.5·max(|vz|−0.3, 0)²  − 0.2·‖ω‖²
    − 0.1·‖Δaction‖² − 0.05·|rotor_imbalance|
    + 100.0 on first soft landing   − 100.0 on crash

Landing (all must hold simultaneously):
  dist_xy < 0.15 m,  dist_z < 0.10 m,  speed_xy < 0.20 m/s,
  speed_z  < 0.30 m/s,  cos_tilt > 0.95

Episode ends on:
  - Successful landing (terminated=True, +100 reward)
  - Crash: inverted (cos_tilt < 0) | ground | out of arena (dist_xy > 5 m)
  - Timeout: MAX_EPISODE_STEPS steps
"""

import numpy as np
import gymnasium
from gymnasium import spaces

import warp as wp
import newton
import newton.solvers

from disturbance.quadrotor_hover_env import (
    Propeller,
    _apply_prop_forces,
    _make_prop,
    _quat_to_rotmat,
    FPS,
    SIM_DT,
    DRONE_SIZE,
    MOTOR_TAU,
    MOTOR_ALPHA,
    HOVER_FRAC,
    THRUST_RANGE,
)

# ── Constants ─────────────────────────────────────────────────────────────

MAX_EPISODE_STEPS = 600    # 6 s at 100 Hz

N_ACTION_HIST = 1
OBS_DIM       = 3 + 9 + 3 + 3 + N_ACTION_HIST * 4   # 22

# Landing pad
PLAT_Z        = 0.10   # logical surface height (m); acts as target z
PLAT_XY_RANGE = 1.5    # platform x,y ∈ uniform [-1.5, 1.5]

# Landing success thresholds (all must hold simultaneously)
LAND_DIST_XY  = 0.20   # horizontal distance from pad center (m)
LAND_DIST_Z   = 0.10   # vertical distance from pad surface (m)
LAND_SPEED_XY = 0.20   # horizontal speed (m/s)
LAND_SPEED_Z  = 0.30   # vertical speed — allows controlled descent (m/s)
LAND_TILT     = 0.95   # cos_tilt minimum (~18° max lean)

# Kept for external use (eval scripts)
LAND_DIST  = LAND_DIST_XY
LAND_SPEED = LAND_SPEED_XY

# Drone spawn envelope (relative to platform xy)
SPAWN_XY_OFFSET = 1.0   # lateral offset from platform (m)
SPAWN_Z_LOW     = 0.8   # minimum spawn altitude (m)
SPAWN_Z_HIGH    = 2.5   # maximum spawn altitude (m)


# ── Reward function ───────────────────────────────────────────────────────

def _compute_reward(
    pos:         np.ndarray,   # world position          (3,)  float32
    quat:        np.ndarray,   # orientation [qx,qy,qz,qw] (4,)
    v:           np.ndarray,   # linear velocity, world  (3,)
    w:           np.ndarray,   # angular velocity, body  (3,)
    action:      np.ndarray,   # current motor commands  (4,)
    prev_action: np.ndarray,   # previous motor commands (4,)
    platform:    np.ndarray,   # pad center [x,y,z]      (3,)
) -> tuple[float, dict, bool, bool]:
    """Returns (total_reward, components, landed, crashed)."""
    p_err    = pos - platform
    dist_xy  = float(np.linalg.norm(p_err[:2]))
    dist_z   = float(abs(p_err[2]))
    speed_xy = float(np.linalg.norm(v[:2]))
    speed_z  = float(abs(v[2]))

    # Exponential position kernels — sharp peak at goal, dense gradient everywhere
    r_pos_xy = float(np.exp(-3.0 * dist_xy ** 2))
    r_pos_z  = float(np.exp(-5.0 * dist_z  ** 2))

    # Attitude: project drone's body-z axis onto world-z
    # _quat_to_rotmat returns 9-D row-major; column 2 is the body z-axis in world
    body_z   = _quat_to_rotmat(quat).reshape(3, 3)[:, 2]
    cos_tilt = float(np.clip(body_z[2], -1.0, 1.0))   # body_z · [0,0,1]
    r_att    = cos_tilt                                 # ∈ [-1, 1]; 1 = perfectly level

    # Velocity — allow slow descent, penalise everything else
    r_vel_xy = -(speed_xy ** 2)
    r_vel_z  = -(max(speed_z - LAND_SPEED_Z, 0.0) ** 2)   # free below threshold

    # Angular velocity (spinning = bad for touchdown)
    r_ang = -float(np.dot(w, w))

    # Action smoothness — penalise rapid RPM changes (jerk)
    delta    = action - prev_action
    r_smooth = -float(np.dot(delta, delta))

    # Rotor symmetry: opposing diagonal pairs should be balanced
    r_sym = -abs(float((action[0] + action[2]) - (action[1] + action[3])))

    # Z-pull — when horizontally close, dense reward for being near pad altitude.
    # Position-based so it can't be gamed by holding vel_z = 0.
    # Activates inside 0.40m radius, peaks when directly over pad at pad altitude.
    if dist_xy < 0.40:
        xy_weight = (0.40 - dist_xy) / 0.40
        r_descent = 3.0 * xy_weight * float(np.exp(-8.0 * dist_z ** 2))
    else:
        r_descent = 0.0

    # Sparse landing bonus — all conditions must hold simultaneously
    landed = bool(
        dist_xy  < LAND_DIST_XY  and
        dist_z   < LAND_DIST_Z   and
        speed_xy < LAND_SPEED_XY and
        speed_z  < LAND_SPEED_Z  and
        cos_tilt > LAND_TILT
    )
    r_land = 100.0 if landed else 0.0

    # Sparse crash penalty
    crashed = bool(
        cos_tilt < 0.0               or   # fully inverted
        dist_xy  > 5.0               or   # out of arena
        pos[2]   < platform[2] - 0.05     # clipped through platform
    )
    r_crash = -100.0 if crashed else 0.0

    r_total = (
        1.00 * r_pos_xy +
        3.00 * r_pos_z  +
        1.50 * r_att    +
        0.30 * r_vel_xy +
        0.50 * r_vel_z  +
        0.20 * r_ang    +
        0.10 * r_smooth +
        0.05 * r_sym    +
        r_descent       +
        r_land          +
        r_crash
    )

    components = dict(
        pos_xy=r_pos_xy, pos_z=r_pos_z, att=r_att,
        vel_xy=r_vel_xy, vel_z=r_vel_z, ang=r_ang,
        smooth=r_smooth, sym=r_sym,
        descent=r_descent, land=r_land, crash=r_crash,
    )
    return float(r_total), components, landed, crashed


# ── Gymnasium Environment ─────────────────────────────────────────────────

class DroneLandingEnv(gymnasium.Env):
    """Newton quadrotor trained to land on a static platform at a random location."""

    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(
        self,
        render_mode:  str | None = None,
        viewer                   = None,
        obs_noise:    bool       = False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self._viewer     = viewer
        self.obs_noise   = obs_noise

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32,
        )

        self._build_sim()

        self._step_count  = 0
        self._ep_reward   = 0.0
        self._render_t    = 0.0
        self._platform    = np.array([0.0, 0.0, PLAT_Z], dtype=np.float32)
        self._landed      = False
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._motor_fracs = np.full(4, HOVER_FRAC, dtype=np.float32)

        # Readable by callbacks
        self.last_dist    = 1.0
        self.last_upright = 1.0
        self.last_speed   = 0.0

    # ── Build simulation ──────────────────────────────────────────────────

    def _build_sim(self) -> None:
        s = DRONE_SIZE
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.05
        builder.add_ground_plane()

        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.5), wp.quat_identity()),
            label="drone",
        )
        density = 1750.0
        for hx, hy, hz in [(s * 0.05, s, s * 0.05), (s, s * 0.05, s * 0.05)]:
            builder.add_shape_box(
                body, hx=hx, hy=hy, hz=hz,
                cfg=newton.ModelBuilder.ShapeConfig(density=density),
            )

        self._props = wp.array([
            _make_prop(body, wp.vec3( 0.0,  s, 0.0), turning_direction=-1.0),
            _make_prop(body, wp.vec3( 0.0, -s, 0.0), turning_direction= 1.0),
            _make_prop(body, wp.vec3( s,  0.0, 0.0), turning_direction= 1.0),
            _make_prop(body, wp.vec3(-s,  0.0, 0.0), turning_direction=-1.0),
        ], dtype=Propeller)

        self._model           = builder.finalize(requires_grad=False)
        self._solver          = newton.solvers.SolverSemiImplicit(self._model)
        self._state           = self._model.state()
        self._state1          = self._model.state()
        self._motor_fracs_gpu = wp.zeros(4, dtype=float)

        if self._viewer is not None:
            self._viewer.set_model(self._model)

    # ── Observation ───────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        body_q  = self._state.body_q.numpy()[0].astype(np.float32)
        body_qd = self._state.body_qd.numpy()[0].astype(np.float32)

        pos  = body_q[:3]
        quat = body_q[3:]
        v    = body_qd[3:]
        w    = body_qd[:3]

        p_err  = pos - self._platform
        R_flat = _quat_to_rotmat(quat)

        obs = np.concatenate([p_err, R_flat, v, w, self._prev_action])

        if self.obs_noise:
            rng = self.np_random
            obs[0:3]   += rng.normal(0, 0.01, 3).astype(np.float32)
            obs[12:15] += rng.normal(0, 0.01, 3).astype(np.float32)
            obs[15:18] += rng.normal(0, 0.05, 3).astype(np.float32)

        return obs.astype(np.float32)

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_count  = 0
        self._ep_reward   = 0.0
        self._render_t    = 0.0
        self._landed      = False
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._motor_fracs = np.full(4, HOVER_FRAC, dtype=np.float32)

        # Randomise platform position
        px = float(self.np_random.uniform(-PLAT_XY_RANGE, PLAT_XY_RANGE))
        py = float(self.np_random.uniform(-PLAT_XY_RANGE, PLAT_XY_RANGE))
        self._platform = np.array([px, py, PLAT_Z], dtype=np.float32)

        # Randomise drone start: near platform laterally, random altitude
        dx = float(self.np_random.uniform(-SPAWN_XY_OFFSET, SPAWN_XY_OFFSET))
        dy = float(self.np_random.uniform(-SPAWN_XY_OFFSET, SPAWN_XY_OFFSET))
        dz = float(self.np_random.uniform(SPAWN_Z_LOW, SPAWN_Z_HIGH))

        init_xform = wp.transform(
            wp.vec3(px + dx, py + dy, dz),
            wp.quat_identity(),
        )
        self._state.body_q.assign([init_xform])
        self._state.body_qd.assign([np.zeros(6, dtype=np.float32)])

        obs = self._get_obs()
        self.last_dist    = float(np.linalg.norm(obs[0:3]))
        self.last_upright = 1.0
        self.last_speed   = 0.0
        return obs, {}

    # ── Step ──────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        # Motor low-pass filter
        setpoint = np.clip(
            HOVER_FRAC + np.clip(action, -1.0, 1.0) * THRUST_RANGE,
            0.05, 1.0,
        ).astype(np.float32)
        self._motor_fracs = (
            (1.0 - MOTOR_ALPHA) * self._motor_fracs + MOTOR_ALPHA * setpoint
        ).astype(np.float32)

        # Physics step
        self._state.clear_forces()
        self._motor_fracs_gpu.assign(self._motor_fracs)
        wp.launch(
            _apply_prop_forces, dim=4,
            inputs =(self._props, self._motor_fracs_gpu,
                     self._state.body_q, self._model.body_com),
            outputs=(self._state.body_f,),
        )
        self._solver.step(self._state, self._state1, None, None, SIM_DT)
        self._state, self._state1 = self._state1, self._state

        prev_action       = self._prev_action          # capture before overwrite
        self._prev_action = action.astype(np.float32)

        obs     = self._get_obs()
        body_q  = self._state.body_q.numpy()[0].astype(np.float32)
        body_qd = self._state.body_qd.numpy()[0].astype(np.float32)
        pos     = body_q[:3]
        quat    = body_q[3:]   # [qx, qy, qz, qw]
        v       = body_qd[3:]
        w       = body_qd[:3]
        z       = float(pos[2])

        # ── Reward ────────────────────────────────────────────────────────
        reward, components, just_landed, crashed = _compute_reward(
            pos, quat, v, w, action, prev_action, self._platform,
        )

        if just_landed and not self._landed:
            self._landed = True

        p_err    = pos - self._platform
        dist_xy  = float(np.linalg.norm(p_err[:2]))
        dist     = float(np.linalg.norm(p_err))
        speed    = float(np.linalg.norm(v))
        cos_tilt = float(components["att"])

        # ── Termination ───────────────────────────────────────────────────
        terminated = bool(
            self._landed or   # successful soft landing
            crashed      or   # inverted / out of arena / clipped through pad
            z < 0.05          # ground contact outside the pad
        )

        self._step_count += 1
        truncated         = self._step_count >= MAX_EPISODE_STEPS
        self._ep_reward  += reward
        self.last_dist    = dist
        self.last_upright = cos_tilt
        self.last_speed   = speed

        info: dict = {
            "dist":     dist,
            "dist_xy":  dist_xy,
            "speed":    speed,
            "upright":  cos_tilt,
            "z":        z,
            "landed":   self._landed,
            "reward_components": components,
        }
        if terminated or truncated:
            info["terminal_dist"]    = dist
            info["terminal_speed"]   = speed
            info["terminal_upright"] = cos_tilt
            info["terminal_ep_len"]  = self._step_count
            info["terminal_reward"]  = self._ep_reward
            info["terminal_landed"]  = self._landed

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ── Render ────────────────────────────────────────────────────────────

    def render(self) -> None:
        if self._viewer is None:
            return
        self._render_t += SIM_DT
        self._viewer.begin_frame(self._render_t)
        self._viewer.log_state(self._state)

        px, py, pz = self._platform.tolist()
        pad_size = 0.40

        # Landing pad visualised as a flat box
        self._viewer.log_shapes(
            "/platform",
            newton.GeoType.BOX, (pad_size, pad_size, 0.02),
            wp.array(
                [wp.transform(wp.vec3(px, py, pz - 0.02), wp.quat_identity())],
                dtype=wp.transform,
            ),
            wp.array([wp.vec3(0.2, 0.8, 0.2)], dtype=wp.vec3),   # green pad
        )
        # Bullseye marker
        self._viewer.log_shapes(
            "/platform_center",
            newton.GeoType.SPHERE, (0.05,),
            wp.array(
                [wp.transform(wp.vec3(px, py, pz), wp.quat_identity())],
                dtype=wp.transform,
            ),
            wp.array([wp.vec3(1.0, 0.1, 0.1)], dtype=wp.vec3),   # red dot
        )
        self._viewer.end_frame()

    def close(self) -> None:
        pass
