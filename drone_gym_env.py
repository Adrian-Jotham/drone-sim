# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Drone Gymnasium Environment
Based on "Learning to Fly in Seconds" (Eschmann et al., RAL 2024)

Observation layout (22-D):
  [0:3]   p_err  — position error = pos − target  (policy always aims for origin)
  [3:12]  R_flat — 3×3 rotation matrix, row-major  (avoids quaternion double-coverage)
  [12:15] v      — linear velocity, world frame
  [15:18] w      — angular velocity, body frame
  [18:22] a_prev — last action (action history N_H=1)

Action (4-D, ∈ [-1, 1]):
  Normalised RPM setpoints, hover-centred (0 → stable hover).
  Maps to actual motor RPMs: n_sp = CF_HOVER_RPM + action × CF_RPM_RANGE
  First-order low-pass motor dynamics applied internally (τ = 0.15 s).
  Physics: F = CF_KT × n², Q = CF_KD × n²  (Level 5.1 — direct RPM control)

Reward (paper Eq. 1):
  r = −C_rp‖p_err‖² − C_rq(1−qw²) − C_rv‖v‖² − C_rω‖ω‖² − C_ra‖Δa‖² + C_rs
  Weights ramp from conservative to strict via env.curriculum ∈ [0, 1].

Crazyflie 2.x physical parameters (Förster 2015 system ID + Bitcraze docs):
  Mass:  27 g,  Arm: 32.5 mm
  Ixx = Iyy = 1.657e-5 kg·m²,  Izz = 2.9e-5 kg·m²
  KT = 3.16e-10 N/RPM²,  KD = 7.94e-12 N·m/RPM²
  Max RPM: 21 702,  Hover RPM: ≈ 14 476
"""

import numpy as np
import gymnasium
from gymnasium import spaces

import warp as wp
import newton
import newton.solvers


# ── Crazyflie mesh helpers ────────────────────────────────────────────────

_CRAZYFLIE_MESH_NAME = "/crazyflie/body"


def _load_crazyflie_mesh(arm_length: float):
    """Extract body geometry from crazyflie.usd, transform Y-up→Z-up, scale to arm_length.

    Returns (points, indices, normals) as warp arrays, or None on failure.
    """
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

    # Y-up → Z-up: (x, y, z) → (x, -z, y)
    verts = np.stack([verts[:, 0], -verts[:, 2], verts[:, 1]], axis=1)

    # Scale so XY extent matches arm_length
    xy_ext = np.abs(verts[:, :2]).max()
    verts *= arm_length / xy_ext

    # Center vertically around Z=0
    verts[:, 2] -= (verts[:, 2].max() + verts[:, 2].min()) * 0.5

    # Per-vertex normals via area-weighted face normal accumulation
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


# ── Simulation constants ──────────────────────────────────────────────────

FPS               = 100          # Hz — matches paper's simulator frequency
SIM_DT            = 1.0 / FPS
MAX_EPISODE_STEPS = 800          # 8 s — matches eval budget (4 wp × 200 steps)

# ── Crazyflie 2.x physical parameters ────────────────────────────────────
# Sources: Förster 2015 (ETH system ID), Bitcraze documentation, Eschmann 2024

CF_MASS  = 0.027       # kg  — total vehicle mass (27 g)
CF_ARM   = 0.0325      # m   — centre-to-motor distance (32.5 mm)

# Inertia tensor (diagonal, body frame), Förster 2015:
CF_IXX   = 1.657e-5    # kg·m²
CF_IYY   = 1.657e-5    # kg·m²
CF_IZZ   = 2.900e-5    # kg·m²

# Blade-element thrust / torque constants (F = KT·n², Q = KD·n², n in RPM):
# Derived from: max thrust ≈ 15.2 g per motor at 21 702 RPM
CF_KT    = 3.16e-10    # N / RPM²
CF_KD    = 7.94e-12    # N·m / RPM²

# Motor speed limits
CF_MAX_RPM = 21702.0   # RPM — full-throttle
CF_MIN_RPM = 1000.0    # RPM — idle (ESC minimum; prevents motor cut-off)

# Hover RPM: 4·KT·n² = m·g  →  n = sqrt(m·g / 4·KT)
CF_HOVER_RPM = float(np.sqrt(CF_MASS * 9.81 / (4.0 * CF_KT)))  # ≈ 14 476 RPM

# Action → RPM:  n_sp = CF_HOVER_RPM + action·CF_RPM_RANGE
# Symmetric range so action=0 ↔ hover, ±1 ↔ min/max RPM
CF_RPM_RANGE = CF_MAX_RPM - CF_HOVER_RPM  # ≈ 7 226 RPM

# Motor first-order LPF (paper §IV: τ ≈ 0.15 s for Crazyflie)
MOTOR_TAU   = 0.15
MOTOR_ALPHA = SIM_DT / MOTOR_TAU   # ≈ 0.067 per step

# Observation / action sizes
N_ACTION_HIST = 1
OBS_DIM       = 3 + 9 + 3 + 3 + N_ACTION_HIST * 4   # 22

# Fixed waypoints used when random_targets=False (eval / render)
TARGETS = [
    np.array([ 1.0,  0.0, 0.5], dtype=np.float32),
    np.array([ 0.0,  1.0, 0.5], dtype=np.float32),
    np.array([-1.0,  0.0, 0.5], dtype=np.float32),
    np.array([ 0.0, -1.0, 0.5], dtype=np.float32),
]

def _sample_random_target(rng: np.random.Generator) -> np.ndarray:
    """Sample a random target from the same distribution as the evaluator.

    radius ∈ [0.5, 1.5] m,  altitude ∈ [0.3, 1.2] m,  angle ∈ [0, 2π)
    Matching eval_drone.py so the training distribution covers the eval range.
    """
    angle  = rng.uniform(0.0, 2.0 * np.pi)
    radius = rng.uniform(0.5, 1.5)
    alt    = rng.uniform(0.3, 1.2)
    return np.array([radius * np.cos(angle), radius * np.sin(angle), alt], dtype=np.float32)

# ── Reward weight curriculum ──────────────────────────────────────────────
# env.curriculum ∈ [0,1]: 0 = init (easy), 1 = target (hard).

_C_RP_INIT, _C_RP_TGT = 0.05, 1.00   # position error ‖p_err‖²
_C_RV_INIT, _C_RV_TGT = 0.01, 0.30   # linear velocity ‖v‖²
_C_RW_INIT, _C_RW_TGT = 0.001, 0.05  # angular velocity ‖ω‖²
_C_RA_INIT, _C_RA_TGT = 0.005, 0.02  # action-change regularisation ‖Δa‖²

_C_RQ = 0.10   # orientation cost  (1 − qw²)  — fixed
_C_RS = 0.50   # survival bonus per step       — fixed

# Approach / hover shaping constants
_APPROACH_COEF      = 1.0   # potential-shaping weight (was 2.0 — reduced to curb overshoot)
_APPROACH_GATE      = 0.25  # m — suppress approach reward inside this radius to stop oscillation
_HOVER_BONUS        = 0.15  # per-step bonus for settling at target (halved from 0.30)
_HOVER_SPEED_GATE   = 0.5   # m/s — must be slow to earn hover bonus; stops rush-to-target


# ── Propeller physics (Level 5.1 — direct RPM) ───────────────────────────

@wp.struct
class Propeller:
    body:              int
    pos:               wp.vec3
    dir:               wp.vec3
    kt:                float        # N / RPM² thrust coefficient
    kd:                float        # N·m / RPM² drag-torque coefficient
    turning_direction: float        # +1 CCW, −1 CW (determines reaction torque sign)


@wp.kernel
def _apply_prop_forces(
    props:      wp.array[Propeller],
    motor_rpms: wp.array[float],         # filtered motor speed [RPM] per rotor
    body_q:     wp.array[wp.transform],
    body_com:   wp.array[wp.vec3],
    body_f:     wp.array[wp.spatial_vector],
):
    tid  = wp.tid()
    prop = props[tid]
    rpm  = motor_rpms[tid]
    n2   = rpm * rpm                     # RPM² for quadratic thrust/torque model

    tf     = body_q[prop.body]
    d      = wp.transform_vector(tf, prop.dir)

    thrust = d * (prop.kt * n2)
    torque = d * (prop.kd * n2 * prop.turning_direction)
    arm    = wp.transform_point(tf, prop.pos) - wp.transform_point(tf, body_com[prop.body])
    torque = torque + wp.cross(arm, thrust)

    wp.atomic_add(body_f, prop.body, wp.spatial_vector(thrust, torque))


def _make_prop(body: int, pos: wp.vec3, turning_direction: float = 1.0) -> Propeller:
    p                   = Propeller()
    p.body              = body
    p.pos               = pos
    p.dir               = wp.vec3(0.0, 0.0, 1.0)
    p.kt                = CF_KT
    p.kd                = CF_KD
    p.turning_direction = turning_direction
    return p


# ── Rotation-matrix helper ────────────────────────────────────────────────

def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """[qx, qy, qz, qw] → 9-D row-major rotation matrix (float32)."""
    qx, qy, qz, qw = q.astype(np.float64)
    return np.array([
        1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw),
            2*(qx*qy + qz*qw),   1 - 2*(qx*qx + qz*qz),   2*(qy*qz - qx*qw),
            2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),   1 - 2*(qx*qx + qy*qy),
    ], dtype=np.float32)


# ── Gymnasium Environment ─────────────────────────────────────────────────

class DroneEnv(gymnasium.Env):
    """Newton quadrotor — Crazyflie 2.x physics, Level-5.1 RPM control (Eschmann 2024)."""

    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(
        self,
        render_mode:    str | None = None,
        viewer         = None,
        random_targets: bool  = True,
        obs_noise:      bool  = False,
        curriculum:     float = 0.0,
    ):
        super().__init__()
        self.render_mode     = render_mode
        self._viewer         = viewer
        self._random_targets = random_targets
        self.obs_noise       = obs_noise
        self.curriculum      = float(np.clip(curriculum, 0.0, 1.0))

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
        self._motor_rpms  = np.full(4, CF_HOVER_RPM, dtype=np.float32)

        # Public: readable by training callbacks
        self.last_dist    = 1.0
        self.last_upright = 1.0

    # ── Build simulation ──────────────────────────────────────────────────

    def _build_sim(self) -> None:
        al = CF_ARM   # arm length for propeller positions

        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.05
        builder.add_ground_plane()

        # Body: explicit Crazyflie mass + inertia tensor (Förster 2015).
        # Thin cross-arm geometry is collision-only (density=0).
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            mass=CF_MASS,
            inertia=wp.mat33(
                CF_IXX, 0.0,    0.0,
                0.0,    CF_IYY, 0.0,
                0.0,    0.0,    CF_IZZ,
            ),
            label="drone",
        )
        # Collision geometry: cross arms scaled to real CF proportions (density=0
        # so they contribute no additional mass/inertia beyond what's set above).
        for hx, hy, hz in [(al * 0.05, al, al * 0.05), (al, al * 0.05, al * 0.05)]:
            builder.add_shape_box(
                body, hx=hx, hy=hy, hz=hz,
                cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
            )

        self._props = wp.array([
            _make_prop(body, wp.vec3( 0.0,  al, 0.0), turning_direction=-1.0),
            _make_prop(body, wp.vec3( 0.0, -al, 0.0), turning_direction= 1.0),
            _make_prop(body, wp.vec3( al,  0.0, 0.0), turning_direction= 1.0),
            _make_prop(body, wp.vec3(-al,  0.0, 0.0), turning_direction=-1.0),
        ], dtype=Propeller)

        self._model           = builder.finalize(requires_grad=False)
        self._solver          = newton.solvers.SolverSemiImplicit(self._model)
        self._state           = self._model.state()
        self._state1          = self._model.state()
        self._motor_rpms_gpu  = wp.zeros(4, dtype=float)

        if self._viewer is not None:
            self._viewer.set_model(self._model)
            self._setup_drone_mesh()

    def _setup_drone_mesh(self) -> None:
        mesh = _load_crazyflie_mesh(CF_ARM)
        if mesh is None:
            self._has_drone_mesh = False
            return
        points, indices, normals = mesh
        self._viewer.log_mesh(_CRAZYFLIE_MESH_NAME, points, indices, normals=normals)
        self._has_drone_mesh = True

    # ── Observation ───────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        body_q  = self._state.body_q.numpy()[0].astype(np.float32)
        body_qd = self._state.body_qd.numpy()[0].astype(np.float32)

        pos  = body_q[:3]
        quat = body_q[3:]     # [qx, qy, qz, qw]
        v    = body_qd[3:]    # linear velocity  (world frame)
        w    = body_qd[:3]    # angular velocity (body frame)

        p_err  = pos - self._target
        R_flat = _quat_to_rotmat(quat)

        obs = np.concatenate([p_err, R_flat, v, w, self._prev_action])

        if self.obs_noise:
            rng = self.np_random
            obs[0:3]   += rng.normal(0, 0.01,  3).astype(np.float32)
            obs[12:15] += rng.normal(0, 0.01,  3).astype(np.float32)
            obs[15:18] += rng.normal(0, 0.05,  3).astype(np.float32)

        return obs.astype(np.float32)

    # ── Gymnasium reset ───────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_count  = 0
        self._ep_reward   = 0.0
        self._render_t    = 0.0
        self._arrived     = False
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._motor_rpms  = np.full(4, CF_HOVER_RPM, dtype=np.float32)

        if self._random_targets:
            self._target = _sample_random_target(self.np_random)
        else:
            self._target = TARGETS[0].copy()

        # Spawn range grows with curriculum. Cap at 0.7 m so that the
        # survival bonus (+0.5/step) always exceeds the position penalty
        # (-C_rp × dist²) at spawn, keeping episode returns positive and
        # giving PPO a learnable gradient from the first update.
        pos_range = 0.1 + 0.6 * self.curriculum   # 0.1 m → 0.7 m
        vel_range = 0.5 * self.curriculum           # 0 → 0.5 m/s

        xy  = self.np_random.uniform(-pos_range, pos_range, 2).astype(np.float32)
        dz  = float(self.np_random.uniform(-pos_range * 0.5, pos_range * 0.5))
        init_z = max(self._target[2] + dz, 0.15)

        init_pos = wp.transform(
            wp.vec3(float(self._target[0] + xy[0]),
                    float(self._target[1] + xy[1]),
                    init_z),
            wp.quat_identity(),
        )
        self._state.body_q.assign([init_pos])

        v0 = self.np_random.uniform(-vel_range, vel_range, 6).astype(np.float32)
        self._state.body_qd.assign([v0])

        obs = self._get_obs()
        self.last_dist    = float(np.linalg.norm(obs[0:3]))
        self.last_upright = 1.0
        return obs, {}

    # ── Gymnasium step ────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        # Action ∈ [-1, 1] → RPM setpoint (hover-centred, Level 5.1)
        rpm_sp = np.clip(
            CF_HOVER_RPM + np.clip(action, -1.0, 1.0) * CF_RPM_RANGE,
            CF_MIN_RPM, CF_MAX_RPM,
        ).astype(np.float32)

        # First-order LPF on motor RPMs (τ = 0.15 s, paper §IV)
        self._motor_rpms = (
            (1.0 - MOTOR_ALPHA) * self._motor_rpms + MOTOR_ALPHA * rpm_sp
        ).astype(np.float32)

        # Physics step
        self._state.clear_forces()
        self._motor_rpms_gpu.assign(self._motor_rpms)
        wp.launch(
            _apply_prop_forces, dim=4,
            inputs =(self._props, self._motor_rpms_gpu,
                     self._state.body_q, self._model.body_com),
            outputs=(self._state.body_f,),
        )
        self._solver.step(self._state, self._state1, None, None, SIM_DT)
        self._state, self._state1 = self._state1, self._state

        prev_action       = self._prev_action.copy()
        self._prev_action = action.astype(np.float32)

        obs     = self._get_obs()
        p_err   = obs[0:3]
        quat    = self._state.body_q.numpy()[0][3:].astype(np.float32)
        body_qd = self._state.body_qd.numpy()[0].astype(np.float32)
        v       = body_qd[3:]
        w       = body_qd[:3]
        z       = float(self._state.body_q.numpy()[0][2])

        dist = float(np.linalg.norm(p_err))
        qw   = float(quat[3])
        R22  = float(1.0 - 2.0 * (quat[0]**2 + quat[1]**2))  # drone_up · world_z

        # ── Reward (paper Eq. 1) ──────────────────────────────────────────
        c    = float(self.curriculum)
        C_rp = _C_RP_INIT + c * (_C_RP_TGT - _C_RP_INIT)
        C_rv = _C_RV_INIT + c * (_C_RV_TGT - _C_RV_INIT)
        C_rw = _C_RW_INIT + c * (_C_RW_TGT - _C_RW_INIT)
        C_ra = _C_RA_INIT + c * (_C_RA_TGT - _C_RA_INIT)

        delta_a  = action - prev_action

        pos_c    = -C_rp  * float(np.dot(p_err,    p_err))
        orient_c = -_C_RQ * float(1.0 - qw**2)
        vel_c    = -C_rv  * float(np.dot(v,         v))
        ang_c    = -C_rw  * float(np.dot(w,         w))
        act_c    = -C_ra  * float(np.dot(delta_a,   delta_a))
        survival =  _C_RS

        crash = -2.0 if z < 0.05 else 0.0

        arrival = 0.0
        if dist < 0.1 and not self._arrived:
            arrival = 1.0
            self._arrived = True

        # Potential-based shaping: only active far from target so the drone
        # doesn't oscillate trying to generate approach reward near the waypoint.
        approach = (
            _APPROACH_COEF * (self.last_dist - dist)
            if dist > _APPROACH_GATE else 0.0
        )

        # Dense bonus for settling at the target.
        # Gated on speed so the policy must decelerate before earning it —
        # a pure distance gate rewards rushing through the target at high speed.
        speed = float(np.linalg.norm(v))
        hover_bonus = _HOVER_BONUS if (dist < 0.15 and speed < _HOVER_SPEED_GATE) else 0.0

        reward = pos_c + orient_c + vel_c + ang_c + act_c + survival + crash + arrival + approach + hover_bonus

        # ── Termination ───────────────────────────────────────────────────
        terminated = bool(
            z    < 0.05  or   # ground impact
            z    > 6.0   or   # escaped upward
            R22  < -0.5  or   # severely inverted
            dist > 4.0        # out of arena
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
            "motor_rpms": self._motor_rpms.tolist(),
            "reward_components": {
                "pos_c":      pos_c,
                "orient_c":   orient_c,
                "vel_c":      vel_c,
                "ang_c":      ang_c,
                "act_c":      act_c,
                "survival":   survival,
                "approach":   approach,
                "hover_bonus": hover_bonus,
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
        """Switch waypoint mid-episode; call get_obs() to refresh observation."""
        self._target  = np.asarray(target, dtype=np.float32).copy()
        self._arrived = False

    def get_obs(self) -> np.ndarray:
        """Re-read current simulator state as observation (useful after set_target)."""
        return self._get_obs()

    def render(self) -> None:
        if self._viewer is None:
            return
        self._render_t += SIM_DT
        self._viewer.begin_frame(self._render_t)
        self._viewer.log_state(self._state)
        if self._has_drone_mesh:
            q_np = self._state.body_q.numpy()[0]
            quat = q_np[3:]
            drone_tf = wp.array(
                [wp.transform(
                    wp.vec3(*q_np[:3].tolist()),
                    wp.quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
                )],
                dtype=wp.transform,
            )
            self._viewer.log_instances(
                "/drone/body", _CRAZYFLIE_MESH_NAME, drone_tf,
                scales=None,
                colors=wp.array([wp.vec3(0.2, 0.2, 0.25)], dtype=wp.vec3),
                materials=None,
            )
        self._viewer.log_shapes(
            "/target",
            newton.GeoType.SPHERE, (0.05,),
            wp.array(
                [wp.transform(wp.vec3(*self._target.tolist()), wp.quat_identity())],
                dtype=wp.transform,
            ),
            wp.array([wp.vec3(1.0, 0.2, 0.0)], dtype=wp.vec3),
        )
        self._viewer.end_frame()

    def close(self) -> None:
        pass
