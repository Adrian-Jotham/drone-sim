# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Quadrotor Hover / Disturbance-Rejection Environment
Based on "Learning to Fly in Seconds" (Eschmann et al., RAL 2024)

Physical model: generic ~400-class quadrotor (≈560 g, 400 mm diagonal,
9-inch props, ~6400 RPM max).  The Crazyflie USD mesh is loaded for
visualisation only — the simulated body is the two cross-shaped boxes below.

Observation layout (22-D):
  [0:3]   p_err  — position error = pos − target
  [3:12]  R_flat — 3×3 rotation matrix, row-major
  [12:15] v      — linear velocity, world frame
  [15:18] w      — angular velocity, body frame
  [18:22] a_prev — last action (action history N_H=1)

Action (4-D, ∈ [-1, 1]):
  Normalised motor setpoints, hover-centred (0 → stable hover).
  First-order low-pass motor dynamics applied internally (τ = 0.15 s).

Reward (paper Eq. 1):
  r = −C_rp‖p_err‖² − C_rq(1−qw²) − C_rv‖v‖² − C_rω‖ω‖² − C_ra‖a‖² + C_rs
  Weights ramp from conservative to strict via env.curriculum ∈ [0, 1].

Disturbance injection (wind_scale > 0):
  An Ornstein-Uhlenbeck wind force is applied to the body each step.
  The wind is NOT observable — the policy must infer it from state drift.
  wind_scale = 0  → no disturbance (standard hover training)
  wind_scale = 1  → full-strength OU wind (max ≈ 55 % of hover thrust)
"""

from collections import deque

import numpy as np
import gymnasium
from gymnasium import spaces

import warp as wp
import newton
import newton.solvers


# ── Crazyflie mesh helpers (cosmetic visualisation only) ──────────────────

_DRONE_MESH_NAME = "/quadrotor/body"


def _load_drone_mesh(arm_length: float):
    """Load a mesh for visualisation, scaled to arm_length.  Returns None on failure."""
    try:
        import newton.examples
        from pxr import Gf, Usd, UsdGeom
    except ImportError:
        return None

    usd_path = newton.examples.get_asset("crazyflie.usd")
    stage = Usd.Stage.Open(usd_path)

    all_verts, all_faces = [], []
    vert_offset = 0

    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        if "propeller" in str(prim.GetPath()).lower():
            continue

        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        fvc = mesh.GetFaceVertexCountsAttr().Get()
        fvi = mesh.GetFaceVertexIndicesAttr().Get()
        if pts is None or fvc is None or fvi is None:
            continue

        world_mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        verts = np.array(
            [[*world_mat.Transform(Gf.Vec3d(*p))] for p in pts], dtype=np.float32
        )

        tris, idx = [], 0
        for cnt in fvc:
            for k in range(1, cnt - 1):
                tris.append([fvi[idx], fvi[idx + k], fvi[idx + k + 1]])
            idx += cnt

        all_verts.append(verts)
        all_faces.append(np.array(tris, dtype=np.int32) + vert_offset)
        vert_offset += len(verts)

    if not all_verts:
        return None

    verts = np.concatenate(all_verts)
    faces = np.concatenate(all_faces)

    # Y-up → Z-up
    verts = np.stack([verts[:, 0], -verts[:, 2], verts[:, 1]], axis=1)

    # Scale XY extent to arm_length
    xy_ext = np.abs(verts[:, :2]).max()
    verts *= arm_length / xy_ext

    # Centre vertically
    verts[:, 2] -= (verts[:, 2].max() + verts[:, 2].min()) * 0.5

    norms = np.zeros_like(verts)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    np.add.at(norms, faces[:, 0], fn)
    np.add.at(norms, faces[:, 1], fn)
    np.add.at(norms, faces[:, 2], fn)
    nlen = np.linalg.norm(norms, axis=1, keepdims=True)
    norms /= np.where(nlen > 1e-8, nlen, 1.0)

    return (
        wp.array(verts, dtype=wp.vec3),
        wp.array(faces.flatten(), dtype=wp.int32),
        wp.array(norms.astype(np.float32), dtype=wp.vec3),
    )


# Keep old name as alias so drone_landing_env and train scripts still work
_CRAZYFLIE_MESH_NAME = _DRONE_MESH_NAME
_load_crazyflie_mesh = _load_drone_mesh


# ── Simulation constants ──────────────────────────────────────────────────

FPS               = 100          # Hz
SIM_DT            = 1.0 / FPS
# Arm length (centre → motor), metres.  Matches ~400-class (F450-style) frame.
DRONE_SIZE        = 0.20
MAX_EPISODE_STEPS = 500          # 5 s at 100 Hz

# First-order motor low-pass: τ ≈ 0.15 s (matches real ESC + motor lag)
MOTOR_TAU   = 0.15
MOTOR_ALPHA = SIM_DT / MOTOR_TAU

# Hover-centred action map: action=0 → hover, ±1 → hover ± THRUST_RANGE
HOVER_FRAC   = 0.33
THRUST_RANGE = 0.33

N_ACTION_HIST = 1
OBS_DIM       = 3 + 9 + 3 + 3 + N_ACTION_HIST * 4   # 22

# Training waypoints
TARGETS = [
    np.array([ 1.0,  0.0, 0.5], dtype=np.float32),
    np.array([ 0.0,  1.0, 0.5], dtype=np.float32),
    np.array([-1.0,  0.0, 0.5], dtype=np.float32),
    np.array([ 0.0, -1.0, 0.5], dtype=np.float32),
]

# ── Reward weight curriculum ──────────────────────────────────────────────

_C_RP_INIT, _C_RP_TGT = 0.05, 1.00
_C_RV_INIT, _C_RV_TGT = 0.01, 0.30
_C_RW_INIT, _C_RW_TGT = 0.001, 0.05
_C_RA_INIT, _C_RA_TGT = 0.005, 0.02

_C_RQ = 0.10
_C_RS = 0.50

# ── Wind / disturbance constants ──────────────────────────────────────────

# Ornstein-Uhlenbeck process: dw = -θ·w·dt + σ·√dt·ξ
WIND_THETA  = 0.5    # mean-reversion rate (s⁻¹), correlation time ≈ 2 s
WIND_SIGMA  = 2.0    # noise diffusion (N · s^{-½})
WIND_MAX    = 3.0    # magnitude clamp (N) — ≈ 55 % of hover thrust
WIND_Z_FRAC = 0.25   # vertical component fraction (wind is mostly horizontal)

# ── Position trail ────────────────────────────────────────────────────────

TRAIL_LEN = 120   # 1.2 s of history at 100 Hz


# ── Propeller physics ─────────────────────────────────────────────────────

@wp.struct
class Propeller:
    body:              int
    pos:               wp.vec3
    dir:               wp.vec3
    max_thrust:        float
    max_torque:        float
    turning_direction: float


@wp.kernel
def _apply_prop_forces(
    props:       wp.array[Propeller],
    motor_fracs: wp.array[float],
    body_q:      wp.array[wp.transform],
    body_com:    wp.array[wp.vec3],
    body_f:      wp.array[wp.spatial_vector],
):
    tid  = wp.tid()
    prop = props[tid]
    frac = motor_fracs[tid]
    tf   = body_q[prop.body]
    d    = wp.transform_vector(tf, prop.dir)
    force  = d * prop.max_thrust * frac
    torque = d * prop.max_torque * frac * prop.turning_direction
    arm    = wp.transform_point(tf, prop.pos) - wp.transform_point(tf, body_com[prop.body])
    torque += wp.cross(arm, force)
    torque *= 0.8
    wp.atomic_add(body_f, prop.body, wp.spatial_vector(force, torque))


@wp.kernel
def _apply_wind_force(
    wind:   wp.vec3,
    body_f: wp.array[wp.spatial_vector],
):
    """Add a pure world-frame force (no torque) to body 0."""
    wp.atomic_add(body_f, 0, wp.spatial_vector(wind, wp.vec3(0.0, 0.0, 0.0)))


def _make_prop(
    body: int,
    pos: wp.vec3,
    turning_direction: float = 1.0,
    # Coefficients from momentum theory; calibrated for ~400-class / 9-inch props
    thrust: float  = 0.109919,
    power:  float  = 0.040164,
    diam:   float  = 0.2286,      # 9-inch propeller diameter (m)
    max_rpm: float = 6396.667,    # motor no-load speed at rated voltage
) -> Propeller:
    rho    = 1.225                # air density, sea level (kg/m³)
    rps    = max_rpm / 60.0
    rps_sq = rps ** 2
    p = Propeller()
    p.body              = body
    p.pos               = pos
    p.dir               = wp.vec3(0.0, 0.0, 1.0)
    p.max_thrust        = thrust * rho * rps_sq * diam ** 4
    p.max_torque        = power  * rho * rps_sq * diam ** 5 / wp.TAU
    p.turning_direction = turning_direction
    return p


# ── Rotation-matrix helper ────────────────────────────────────────────────

def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """[qx, qy, qz, qw] → 9-D row-major rotation matrix (float32)."""
    qx, qy, qz, qw = q.astype(np.float64)
    return np.array([
        1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw),
            2*(qx*qy + qz*qw),   1 - 2*(qx*qx + qz*qz),   2*(qy*qz - qx*qw),
            2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy),
    ], dtype=np.float32)


# ── Gymnasium Environment ─────────────────────────────────────────────────

class DroneEnv(gymnasium.Env):
    """
    Quadrotor hover / disturbance-rejection environment.

    wind_scale = 0  → standard waypoint hover (no wind)
    wind_scale > 0  → OU wind disturbance injected each step; the policy
                      must learn to reject it from state drift alone.
    """

    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(
        self,
        render_mode:    str | None = None,
        viewer         = None,
        random_targets: bool  = True,
        obs_noise:      bool  = False,
        curriculum:     float = 0.0,
        wind_scale:     float = 0.0,
    ):
        super().__init__()
        self.render_mode     = render_mode
        self._viewer         = viewer
        self._random_targets = random_targets
        self.obs_noise       = obs_noise
        self.curriculum      = float(np.clip(curriculum, 0.0, 1.0))
        self.wind_scale      = float(np.clip(wind_scale, 0.0, 1.0))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32,
        )

        self._has_drone_mesh = False
        self._build_sim()

        self._step_count  = 0
        self._ep_reward   = 0.0
        self._render_t    = 0.0
        self._target      = TARGETS[0].copy()
        self._arrived     = False
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._motor_fracs = np.full(4, HOVER_FRAC, dtype=np.float32)

        # OU wind state (world frame, N)
        self._wind        = np.zeros(3, dtype=np.float32)

        # Position trail for visualisation
        self._pos_trail: deque = deque(maxlen=TRAIL_LEN)

        self.last_dist    = 1.0
        self.last_upright = 1.0

    # ── Build simulation ──────────────────────────────────────────────────

    def _build_sim(self) -> None:
        s = DRONE_SIZE
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.05
        builder.add_ground_plane()

        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            label="drone",
        )
        # Cross-shaped frame: two thin boxes at right angles
        density = 1750.0   # kg/m³ (carbon fibre composite)
        for hx, hy, hz in [(s*0.05, s, s*0.05), (s, s*0.05, s*0.05)]:
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
            self._setup_drone_mesh()

    def _setup_drone_mesh(self) -> None:
        mesh = _load_drone_mesh(DRONE_SIZE)
        if mesh is None:
            self._has_drone_mesh = False
            return
        points, indices, normals = mesh
        self._viewer.log_mesh(_DRONE_MESH_NAME, points, indices, normals=normals)
        self._has_drone_mesh = True

    # ── Observation ───────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        body_q  = self._state.body_q.numpy()[0].astype(np.float32)
        body_qd = self._state.body_qd.numpy()[0].astype(np.float32)

        pos  = body_q[:3]
        quat = body_q[3:]
        v    = body_qd[3:]
        w    = body_qd[:3]

        p_err  = pos - self._target
        R_flat = _quat_to_rotmat(quat)

        obs = np.concatenate([p_err, R_flat, v, w, self._prev_action])

        if self.obs_noise:
            rng = self.np_random
            obs[0:3]   += rng.normal(0, 0.01,  3).astype(np.float32)
            obs[12:15] += rng.normal(0, 0.01,  3).astype(np.float32)
            obs[15:18] += rng.normal(0, 0.05,  3).astype(np.float32)

        return obs.astype(np.float32)

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_count  = 0
        self._ep_reward   = 0.0
        self._render_t    = 0.0
        self._arrived     = False
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._motor_fracs = np.full(4, HOVER_FRAC, dtype=np.float32)
        self._wind        = np.zeros(3, dtype=np.float32)
        self._pos_trail.clear()

        if self._random_targets:
            self._target = TARGETS[self.np_random.integers(len(TARGETS))].copy()
        else:
            self._target = TARGETS[0].copy()

        pos_range = 0.1 + 0.4 * self.curriculum
        vel_range = 0.3 * self.curriculum

        xy    = self.np_random.uniform(-pos_range, pos_range, 2).astype(np.float32)
        dz    = float(self.np_random.uniform(-pos_range * 0.5, pos_range * 0.5))
        init_z = max(self._target[2] + dz, 0.15)

        self._state.body_q.assign([wp.transform(
            wp.vec3(float(self._target[0] + xy[0]),
                    float(self._target[1] + xy[1]),
                    init_z),
            wp.quat_identity(),
        )])
        v0 = self.np_random.uniform(-vel_range, vel_range, 6).astype(np.float32)
        self._state.body_qd.assign([v0])

        obs = self._get_obs()
        self.last_dist    = float(np.linalg.norm(obs[0:3]))
        self.last_upright = 1.0
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

        # Propeller forces
        self._state.clear_forces()
        self._motor_fracs_gpu.assign(self._motor_fracs)
        wp.launch(
            _apply_prop_forces, dim=4,
            inputs =(self._props, self._motor_fracs_gpu,
                     self._state.body_q, self._model.body_com),
            outputs=(self._state.body_f,),
        )

        # OU wind disturbance (not observable by policy — learned from state drift)
        if self.wind_scale > 0.0:
            noise = self.np_random.standard_normal(3).astype(np.float32)
            noise[2] *= WIND_Z_FRAC
            self._wind = (
                self._wind * (1.0 - WIND_THETA * SIM_DT)
                + WIND_SIGMA * np.sqrt(SIM_DT) * noise
            )
            mag = float(np.linalg.norm(self._wind))
            if mag > WIND_MAX:
                self._wind *= WIND_MAX / mag
            wind_scaled = (self._wind * self.wind_scale).astype(np.float32)
            wp.launch(
                _apply_wind_force, dim=1,
                inputs =[wp.vec3(*wind_scaled.tolist())],
                outputs=[self._state.body_f],
            )

        self._solver.step(self._state, self._state1, None, None, SIM_DT)
        self._state, self._state1 = self._state1, self._state

        self._prev_action = action.astype(np.float32)

        obs    = self._get_obs()
        p_err  = obs[0:3]
        quat   = self._state.body_q.numpy()[0][3:].astype(np.float32)
        body_qd = self._state.body_qd.numpy()[0].astype(np.float32)
        v      = body_qd[3:]
        w      = body_qd[:3]
        z      = float(self._state.body_q.numpy()[0][2])

        dist = float(np.linalg.norm(p_err))
        qw   = float(quat[3])
        R22  = float(1.0 - 2.0 * (quat[0]**2 + quat[1]**2))

        # Reward (curriculum-weighted, paper Eq. 1)
        c = float(self.curriculum)
        C_rp = _C_RP_INIT + c * (_C_RP_TGT - _C_RP_INIT)
        C_rv = _C_RV_INIT + c * (_C_RV_TGT - _C_RV_INIT)
        C_rw = _C_RW_INIT + c * (_C_RW_TGT - _C_RW_INIT)
        C_ra = _C_RA_INIT + c * (_C_RA_TGT - _C_RA_INIT)

        pos_c    = -C_rp  * float(np.dot(p_err, p_err))
        orient_c = -_C_RQ * float(1.0 - qw**2)
        vel_c    = -C_rv  * float(np.dot(v, v))
        ang_c    = -C_rw  * float(np.dot(w, w))
        act_c    = -C_ra  * float(np.dot(action, action))
        survival =  _C_RS

        crash = -2.0 if z < 0.05 else 0.0

        arrival = 0.0
        if dist < 0.1 and not self._arrived:
            arrival = 1.0
            self._arrived = True

        reward = pos_c + orient_c + vel_c + ang_c + act_c + survival + crash + arrival

        terminated = bool(
            z    < 0.05  or
            z    > 6.0   or
            R22  < -0.5  or
            dist > 4.0
        )

        self._step_count += 1
        truncated         = self._step_count >= MAX_EPISODE_STEPS
        self._ep_reward  += reward
        self.last_dist    = dist
        self.last_upright = R22

        info: dict = {
            "dist":    dist,
            "upright": R22,
            "z":       z,
            "wind_magnitude": float(np.linalg.norm(self._wind)) * self.wind_scale,
            "reward_components": {
                "pos_c":    pos_c,
                "orient_c": orient_c,
                "vel_c":    vel_c,
                "ang_c":    ang_c,
                "act_c":    act_c,
                "survival": survival,
            },
        }
        if terminated or truncated:
            info["terminal_dist"]    = dist
            info["terminal_upright"] = R22
            info["terminal_ep_len"]  = self._step_count
            info["terminal_reward"]  = self._ep_reward

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ── Helpers ───────────────────────────────────────────────────────────

    def set_target(self, target: np.ndarray) -> None:
        self._target  = np.asarray(target, dtype=np.float32).copy()
        self._arrived = False

    def get_obs(self) -> np.ndarray:
        return self._get_obs()

    # ── Render ────────────────────────────────────────────────────────────

    def render(self) -> None:
        if self._viewer is None:
            return

        self._render_t += SIM_DT
        self._viewer.begin_frame(self._render_t)
        self._viewer.log_state(self._state)

        q_np = self._state.body_q.numpy()[0]
        pos  = q_np[:3].astype(np.float32)
        quat = q_np[3:].astype(np.float32)

        # ── Drone mesh / fallback box ─────────────────────────────────────
        drone_tf = wp.array(
            [wp.transform(
                wp.vec3(*pos.tolist()),
                wp.quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            )],
            dtype=wp.transform,
        )
        if self._has_drone_mesh:
            self._viewer.log_instances(
                "/drone/body", _DRONE_MESH_NAME, drone_tf,
                scales=None,
                colors=wp.array([wp.vec3(0.2, 0.2, 0.25)], dtype=wp.vec3),
                materials=None,
            )

        # ── Target marker ─────────────────────────────────────────────────
        self._viewer.log_shapes(
            "/target",
            newton.GeoType.SPHERE, (0.05,),
            wp.array(
                [wp.transform(wp.vec3(*self._target.tolist()), wp.quat_identity())],
                dtype=wp.transform,
            ),
            wp.array([wp.vec3(1.0, 0.2, 0.0)], dtype=wp.vec3),
        )

        # ── Position trail ────────────────────────────────────────────────
        self._pos_trail.append(pos.copy())
        n = len(self._pos_trail)
        if n > 1:
            trail_list = list(self._pos_trail)
            trail_tfs    = []
            trail_colors = []
            for i, p in enumerate(trail_list):
                t = i / (n - 1)   # 0 = oldest, 1 = newest
                trail_tfs.append(
                    wp.transform(wp.vec3(*p.tolist()), wp.quat_identity())
                )
                # Fade: dark blue (old) → bright cyan (recent)
                trail_colors.append(wp.vec3(0.0, 0.5 * t, 0.4 + 0.6 * t))
            self._viewer.log_shapes(
                "/drone/trail",
                newton.GeoType.SPHERE, (0.015,),
                wp.array(trail_tfs,    dtype=wp.transform),
                wp.array(trail_colors, dtype=wp.vec3),
            )

        # ── Wind force indicator ──────────────────────────────────────────
        if self.wind_scale > 0.0:
            wind_mag = float(np.linalg.norm(self._wind)) * self.wind_scale
            if wind_mag > 0.05:
                wind_dir = (self._wind * self.wind_scale) / wind_mag
                # Sphere placed 0.6 m upwind (direction the wind pushes from),
                # sized and coloured by intensity
                intensity = min(wind_mag / WIND_MAX, 1.0)
                indicator_pos = pos + wind_dir * 0.6
                self._viewer.log_shapes(
                    "/wind_indicator",
                    newton.GeoType.SPHERE, (0.03 + 0.05 * intensity,),
                    wp.array(
                        [wp.transform(wp.vec3(*indicator_pos.tolist()), wp.quat_identity())],
                        dtype=wp.transform,
                    ),
                    # Orange (weak) → red (strong)
                    wp.array(
                        [wp.vec3(0.9, 0.5 * (1.0 - intensity), 0.05)],
                        dtype=wp.vec3,
                    ),
                )

        self._viewer.end_frame()

    def close(self) -> None:
        pass
