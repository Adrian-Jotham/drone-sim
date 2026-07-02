# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Drone Gymnasium Environment
Based on "Learning to Fly in Seconds" (Eschmann et al., RAL 2024).

Training is a position-controller primitive: each episode samples one random
target and one random initial state (wide spawn covering the eval waypoint
range, full SO(3) attitude ≤ 90°, random vel/RPM). Multi-waypoint navigation
emerges by composing this primitive at eval time.

Observation layout (22-D):
  [0:3]   p_err  — position error = pos − target
  [3:12]  R_flat — 3×3 rotation matrix, row-major
  [12:15] v      — linear velocity, world frame
  [15:18] w      — angular velocity, body frame
  [18:22] a_prev — last action (N_H=1; paper uses 32 but the extra 124 dims
                   drown out the 18-D state signal under PPO; smaller helps).

Action (4-D, ∈ [-1, 1]):
  Normalised RPM setpoints, hover-centred (0 → stable hover).
  Maps to actual motor RPMs: n_sp = CF_HOVER_RPM + action × CF_RPM_RANGE
  First-order low-pass motor dynamics applied internally (τ = 0.15 s).
  Physics: F = CF_KT × n², Q = CF_KD × n²  (Level 5.1 — direct RPM control)

Reward (paper Eq. 1 + Table 2 weights, action-baseline form):
  r = −C_rp‖p_err‖² − C_rq(1−qw²) − C_rv‖v‖² − C_rω‖ω‖² − C_ra‖a − C_rab‖² + C_rs
  Weights linearly ramp init → target via env.curriculum ∈ [0, 1].
  C_rab = 0 here because action is already hover-centred (paper uses 0.334 for
  their [-1,1] → [0, MAX_RPM] mapping; equivalent hover point in our parameterisation).

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
MAX_EPISODE_STEPS = 500          # 5 s per single-target episode (paper Table 6)

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
# Note: paper uses N_H=32; empirically N_H=1 trains better with our PPO setup
# (146-D obs with 128 action-history dims swamps the 18-D state signal).
N_ACTION_HIST = 1
OBS_DIM = 3 + 9 + 3 + 3 + N_ACTION_HIST * 4 + 4 + 3 + 3  # 32 (22 state + 4 ωm + 3 fr + 3 τr)

# Training target: fixed at centre (paper trains "fly to origin from anywhere").
# The policy only observes p_err = pos − target, so any fixed point is equivalent
# to the paper's origin. At deployment: p_err = pos − actual_target (same trick).
TRAIN_TARGET = np.array([0.0, 0.0, 0.5], dtype=np.float32)

# Fixed waypoints used when random_targets=False in EVAL / render mode
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


def _sample_random_orientation_capped(rng: np.random.Generator, max_tilt: float) -> np.ndarray:
    """Uniform SO(3) restricted to body-z tilted at most `max_tilt` rad from world-z.

    Paper Table 3 (at max_tilt = π/2) plus a curriculum-tunable cap for bootstrap.
    Direct construction (no rejection): sample body-z on spherical cap × uniform yaw.
    Returns [qx, qy, qz, qw].
    """
    # Body-z direction uniformly on spherical cap around (0,0,1) of half-angle max_tilt.
    # cos(theta) uniform in [cos(max_tilt), 1] gives area-uniform sampling on the cap.
    cos_theta = float(rng.uniform(np.cos(max_tilt), 1.0))
    phi       = float(rng.uniform(0.0, 2.0 * np.pi))

    if cos_theta > 0.9999:
        q_tilt = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        # Axis = world_z × body_z (normalized): rotates world-z to the sampled direction.
        axis = np.array([-np.sin(phi), np.cos(phi), 0.0])
        half = np.arccos(cos_theta) * 0.5
        s, c = np.sin(half), np.cos(half)
        q_tilt = np.array([axis[0] * s, axis[1] * s, 0.0, c], dtype=np.float64)

    # Uniform yaw around world-z; compose as q = q_tilt * q_yaw.
    half_y = float(rng.uniform(0.0, 2.0 * np.pi)) * 0.5
    q_yaw  = np.array([0.0, 0.0, np.sin(half_y), np.cos(half_y)], dtype=np.float64)

    x1, y1, z1, w1 = q_tilt
    x2, y2, z2, w2 = q_yaw
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float32)

# ── Reward weight curriculum ──────────────────────────────────────────────
# Paper Table 2: linear interpolation from C_init to C_target via env.curriculum ∈ [0, 1].

_C_RP_INIT, _C_RP_TGT = 2.5, 20.0    # position error ‖p_err‖²        (paper: 2.5 → 20)
_C_RV_INIT, _C_RV_TGT = 0.005, 0.5   # linear velocity ‖v‖²            (paper: 0.005 → 0.5)
_C_RA_INIT, _C_RA_TGT = 0.005, 0.5   # action magnitude ‖a − C_rab‖²   (paper: 0.005 → 0.5)

_C_RQ  = 2.5   # orientation cost (1 − qw²)               — fixed (paper)
_C_RS  = 2.0   # survival bonus per step                  — fixed (paper)
_C_RW  = 0.0   # angular velocity ‖ω‖²                    — fixed at 0 (paper)
_C_RAB = 0.0   # action baseline: 0 for hover-centred a    (paper 0.334 in their RPM-direct mapping)


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
        disturbances:   bool  = False,
        curriculum:     float = 0.0,
    ):
        super().__init__()
        self.render_mode     = render_mode
        self._viewer         = viewer
        self._random_targets = random_targets
        self.obs_noise       = obs_noise
        self.disturbances    = disturbances
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
        self._action_hist = np.zeros((N_ACTION_HIST, 4), dtype=np.float32)
        self._motor_rpms  = np.full(4, CF_HOVER_RPM, dtype=np.float32)
        # Episode-level disturbances (sampled in reset, constant per episode)
        self._force_dist  = np.zeros(3, dtype=np.float32)
        self._torque_dist = np.zeros(3, dtype=np.float32)

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

        obs = np.concatenate([p_err, R_flat, v, w, self._action_hist.flatten()])

        if self.obs_noise:
            rng = self.np_random
            obs[0:3]   += rng.normal(0, 0.01, 3).astype(np.float32)   # position  σ=1 cm
            obs[3:12]  += rng.normal(0, 0.01, 9).astype(np.float32)   # rotation  σ≈0.01 rad (IMU)
            obs[12:15] += rng.normal(0, 0.01, 3).astype(np.float32)   # lin. vel  σ=1 cm/s
            obs[15:18] += rng.normal(0, 0.01, 3).astype(np.float32)   # ang. vel  σ=0.01 rad/s (gyro)

        obs = obs.astype(np.float32)
        # Guard against physics blowup: NaN/inf obs would corrupt the whole batch.
        np.nan_to_num(obs, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs

    # ── Gymnasium reset ───────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_count  = 0
        self._ep_reward   = 0.0
        self._render_t    = 0.0
        self._action_hist = np.zeros((N_ACTION_HIST, 4), dtype=np.float32)

        rng = self.np_random
        if self._random_targets:
            self._target = _sample_random_target(rng)
        else:
            # Training mode (paper): fixed centre target, drone spawns at random
            # absolute position around it — equivalent to paper's "fly to origin"
            self._target = TRAIN_TARGET.copy()

        # ── Curriculum-gated spawn extremes ──────────────────────────────
        # env.curriculum ∈ [0, 1] interpolates from an easy bootstrap distribution
        # (c=0: small perturbation around target, mild tilt, near-hover RPM) to
        # full paper Table 3 + widened-for-navigation spawn (c=1).
        #
        # Without this gating, PPO faces the hardest spawn from step 0 while the
        # reward weights are still at their gentlest — death-spirals immediately
        # because terminating the episode is cheaper than enduring the position
        # penalty from a 1.5 m random spawn.
        c = self.curriculum
        pos_range = 0.15 + (1.5  - 0.15) * c            # ±0.15 m → ±1.5 m on each axis
        max_tilt  = (15.0 + (90.0 - 15.0) * c) * (np.pi / 180.0)  # ±15° → ±90°
        vel_range = 0.10 + (1.0  - 0.10) * c            # ±0.1 m/s   → ±1.0 m/s
        ang_range = 0.10 + (1.0  - 0.10) * c            # ±0.1 rad/s → ±1.0 rad/s
        rpm_low   = (1.0 - c) * (CF_HOVER_RPM * 0.95)                                # 0.95·hover  → 0
        rpm_high  = (1.0 - c) * (CF_HOVER_RPM * 1.05) + c * (CF_MAX_RPM / 2.0)        # 1.05·hover → MAX/2

        # 10 % guidance branch (paper Table 3) — always on, curriculum-independent.
        # Spawn at target with identity attitude; vel/RPM still sampled below.
        # Acts as a permanent supply of "hold-at-target" data so the policy
        # doesn't forget the stabilisation sub-skill as spawn widens.
        if rng.uniform() < 0.10:
            init_x, init_y, init_z = float(self._target[0]), float(self._target[1]), float(self._target[2])
            init_quat = wp.quat_identity()
        else:
            dx, dy, dz = rng.uniform(-pos_range, pos_range, 3).astype(np.float32)
            init_x = float(self._target[0] + dx)
            init_y = float(self._target[1] + dy)
            init_z = float(np.clip(self._target[2] + dz, 0.15, 1.5))
            q = _sample_random_orientation_capped(rng, max_tilt)
            init_quat = wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

        self._state.body_q.assign([wp.transform(
            wp.vec3(init_x, init_y, init_z), init_quat,
        )])

        lin_v = rng.uniform(-vel_range, vel_range, 3).astype(np.float32)
        ang_v = rng.uniform(-ang_range, ang_range, 3).astype(np.float32)
        self._state.body_qd.assign([np.concatenate([ang_v, lin_v])])

        self._motor_rpms = np.clip(
            rng.uniform(rpm_low, rpm_high, 4).astype(np.float32),
            CF_MIN_RPM, CF_MAX_RPM,
        )

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
        if self.disturbances:
            wp.launch(
                _add_disturbance_force, dim=1,
                inputs=(
                    self._state.body_f,
                    wp.vec3(float(self._force_dist[0]),
                            float(self._force_dist[1]),
                            float(self._force_dist[2])),
                    wp.vec3(float(self._torque_dist[0]),
                            float(self._torque_dist[1]),
                            float(self._torque_dist[2])),
                ),
            )
        self._solver.step(self._state, self._state1, None, None, SIM_DT)
        self._state, self._state1 = self._state1, self._state

        # Push current action onto the FIFO history buffer (most recent at end).
        self._action_hist = np.roll(self._action_hist, -1, axis=0)
        self._action_hist[-1] = action.astype(np.float32)

        obs       = self._get_obs()
        p_err     = obs[0:3]
        body_q_np = self._state.body_q.numpy()[0].astype(np.float32)
        pos       = body_q_np[:3]
        quat      = body_q_np[3:]
        body_qd   = self._state.body_qd.numpy()[0].astype(np.float32)
        v         = body_qd[3:]
        w         = body_qd[:3]
        z         = float(pos[2])

        dist = float(np.linalg.norm(p_err))
        qw   = float(quat[3])
        R22  = float(1.0 - 2.0 * (quat[0]**2 + quat[1]**2))  # drone_up · world_z

        # ── Reward (paper Eq. 1 + Table 2) ────────────────────────────────
        c    = float(self.curriculum)
        C_rp = _C_RP_INIT + c * (_C_RP_TGT - _C_RP_INIT)
        C_rv = _C_RV_INIT + c * (_C_RV_TGT - _C_RV_INIT)
        C_ra = _C_RA_INIT + c * (_C_RA_TGT - _C_RA_INIT)

        act_dev  = action - _C_RAB

        pos_c    = -C_rp  * float(np.dot(p_err,    p_err))
        orient_c = -_C_RQ * float(1.0 - qw**2)
        vel_c    = -C_rv  * float(np.dot(v,         v))
        ang_c    = -_C_RW * float(np.dot(w,         w))
        act_c    = -C_ra  * float(np.dot(act_dev,   act_dev))
        survival =  _C_RS

        reward = pos_c + orient_c + vel_c + ang_c + act_c + survival

        # ── Termination ───────────────────────────────────────────────────
        speed = float(np.linalg.norm(v))
        terminated = bool(
            z     < 0.05              or   # ground impact
            z     > 6.0               or   # escaped upward
            R22   < -0.5              or   # severely inverted
            dist  > _TERMINATION_DIST or   # left the tight survival box
            speed > 20.0              or   # physics blowup guard
            not np.isfinite(z)             # NaN/inf state — reset immediately
        )

        crash_c = _C_CRASH if terminated else 0.0
        reward += crash_c

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
                "pos_c":    pos_c,
                "orient_c": orient_c,
                "vel_c":    vel_c,
                "ang_c":    ang_c,
                "act_c":    act_c,
                "survival": survival,
            },
        }
        if terminated or truncated:
            info["terminal_dist"]     = dist
            info["terminal_upright"]  = R22
            info["terminal_ep_len"]   = self._step_count
            info["terminal_reward"]   = self._ep_reward
            info["terminal_survived"] = truncated

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ── Helpers ───────────────────────────────────────────────────────────

    def set_target(self, target: np.ndarray) -> None:
        """Switch waypoint mid-episode; call get_obs() to refresh observation."""
        self._target   = np.asarray(target, dtype=np.float32).copy()
        body_q         = self._state.body_q.numpy()[0]
        self.last_dist = float(np.linalg.norm(body_q[:3].astype(np.float32) - self._target))

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
