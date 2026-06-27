# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Batched, GPU-parallel quadrotor position-controller environment (Newton + MuJoCo).

Migration of ``drone_gym_env.py`` to the architecture mandated by CLAUDE.md:

  * One drone *template* (FREE airframe + 4 physical REVOLUTE rotor joints with
    ``armature`` / ``effort_limit`` / ``friction``) replicated into N worlds via
    ``builder.replicate`` → a single ``model`` / ``solver`` stepped together.
  * ``SolverMuJoCo`` (``integrator="implicitfast"``) operating on the articulation.
  * All per-step I/O stays on the GPU (warp arrays / torch views) — no ``.numpy()``
    syncs in the hot loop. ``ArticulationView`` is used for batched root I/O.
  * A batched aerodynamic Warp kernel converts rotor ω → axial thrust + drag torque
    written into ``state.body_f`` (world-frame wrench at COM, consumed by MuJoCo).
    Thrust is applied *at each rotor body*, so MuJoCo produces the roll/pitch moments
    automatically and the rotor drag torque transmits through the REVOLUTE joints to
    give the yaw reaction (and gyroscopic coupling) for free.
  * The CUDA-graph substep loop is captured once and replayed each control step.

Preserved verbatim from the single-env version (CLAUDE.md §6): the 22-D observation
layout, the hover-centred [-1,1] 4-D action, the paper reward (Eq. 1 + Table 2), the
curriculum schedule, the Crazyflie physical parameters, 100 Hz control, the spawn /
target distribution, and termination/truncation rules.

Deviations from CLAUDE.md (flagged per §9):
  * ``Control`` exposes ``joint_target_vel`` (not ``joint_target_qd`` as the doc states);
    velocity targets are written there.
  * "Option A" motor dynamics realized as a software first-order LPF (τ = 0.15 s, the
    paper value) on the rotor velocity *setpoint* feeding a stiff velocity actuator
    (``target_kd``), with real spin inertia (``armature``) retained for gyroscopic
    coupling. A pure proportional velocity actuator cannot reproduce both the 0.15 s lag
    and zero steady-state tracking error under aero drag, so the lag is kept where the
    paper defines it and the actuator provides tracking fidelity.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp
import newton
import newton.solvers
from newton._src.utils.selection import ArticulationView

from drone_gym_env import (
    FPS, SIM_DT, MAX_EPISODE_STEPS, OBS_DIM,
    CF_MASS, CF_ARM, CF_IXX, CF_IYY, CF_IZZ, CF_KT, CF_KD,
    CF_MAX_RPM, CF_MIN_RPM, CF_HOVER_RPM, CF_RPM_RANGE, MOTOR_TAU,
)

# ── Reward (paper "Learning to Fly in Seconds", Eschmann 2024, §IV-D) ──────
#
#   r = −C_rp‖p‖²  − C_rq(1−q_w²)  − C_rv‖v‖²  − C_rω‖ω‖²  − C_ra‖a−a_rab‖²  + C_rs
#
# The paper uses "a negative squared cost with an additive constant incentivizing
# survival to mitigate the 'learning to terminate' problem". For that mitigation to
# actually hold, the per-step reward must stay roughly positive across the spawn
# envelope — otherwise crashing early (ending the negative stream) is optimal and the
# policy learns to dive into the ground (the exact failure mode this fixes). Two needs:
#
#   1. The position cost is **clipped** (rl-tools `position_clip`) so a far spawn can't
#      produce an unbounded penalty that dwarfs the survival bonus.
#   2. The survival constant C_rs dominates the clipped cost budget.
#
# The exact constants live in the paper's (unavailable) supplementary material; the
# values below implement the paper's *formula and design principle* with rl-tools-scale
# magnitudes that train reliably (survival-dominant, all six terms active, ω now penalised
# so the drone cannot spin freely). Weights ramp init→target over the curriculum.
P_ERR_CLIP  = 2.0                     # m — clip ‖p_err‖ at the termination radius only:
P_ERR_CLIP2 = P_ERR_CLIP * P_ERR_CLIP  #     keeps a navigation gradient everywhere INSIDE
                                       #     the valid box, just bounds the cost at the edge.

# Position weight ramps low→high over the curriculum. Low early so survival dominates and
# the drone bootstraps flying/approaching; high late for tight precision once it already
# hovers near the target (so the large weight rarely produces a large cost). C_rp·dist²max
# at the boundary stays comparable to survival early on, avoiding "learning to terminate".
_C_RP_INIT, _C_RP_TGT = 1.0, 4.0      # position  ‖p‖² (clipped at the boundary)
_C_RV_INIT, _C_RV_TGT = 0.05, 0.30    # linear velocity ‖v‖²  — brakes overshoot/fly-away
_C_RW_INIT, _C_RW_TGT = 0.02, 0.15    # angular velocity ‖ω‖²  (was 0 → caused free spin)
_C_RA_INIT, _C_RA_TGT = 0.0, 0.20     # action ‖a − a_rab‖²
_C_RQ  = 1.0                          # orientation (1 − q_w²), fixed
# Survival no longer *dominates*: with stable (low-variance) control the drone can actually
# fly to the target to escape the position cost, so a moderate survival bonus avoids the
# "learning to terminate" crash without removing the pressure to navigate.
_C_RS  = 2.5
_C_RAB = 0.0                          # action baseline (hover action = 0 in hover-centred a)
_C_APPROACH = 0.0                     # approach shaping disabled — destabilised PPO in
                                      # practice (it exploited the dense signal / dove).

# ── Derived rotor constants (rad/s domain) ────────────────────────────────
RPM_TO_RAD = 2.0 * np.pi / 60.0
RAD_TO_RPM = 60.0 / (2.0 * np.pi)

OMEGA_HOVER = CF_HOVER_RPM * RPM_TO_RAD          # ≈ 1516 rad/s
OMEGA_RANGE = CF_RPM_RANGE * RPM_TO_RAD
OMEGA_MAX   = CF_MAX_RPM   * RPM_TO_RAD
OMEGA_MIN   = CF_MIN_RPM   * RPM_TO_RAD

KT_SI = CF_KT * RAD_TO_RPM * RAD_TO_RPM          # N    per (rad/s)²
KD_SI = CF_KD * RAD_TO_RPM * RAD_TO_RPM          # N·m  per (rad/s)²

MOTOR_ALPHA = SIM_DT / MOTOR_TAU

# Rotor layout (CLAUDE.md §4.1): 0:+y CCW, 1:-y CW, 2:+x CW, 3:-x CCW
ROTOR_POS = np.array([
    [0.0,  CF_ARM, 0.0],
    [0.0, -CF_ARM, 0.0],
    [ CF_ARM, 0.0, 0.0],
    [-CF_ARM, 0.0, 0.0],
], dtype=np.float32)
TURN_DIR = np.array([+1.0, -1.0, -1.0, +1.0], dtype=np.float32)  # +1 CCW, -1 CW

BODIES_PER_WORLD = 5
DOFS_PER_WORLD   = 10
COORDS_PER_WORLD = 11
ROTOR_MASS       = 0.0015
AIRFRAME_MASS    = CF_MASS - 4.0 * ROTOR_MASS

TARGET_KD    = 0.05
EFFORT_LIMIT = 0.10
FRICTION     = 0.0
ARMATURE     = 1e-7

SIM_SUBSTEPS = 4
SUBSTEP_DT   = SIM_DT / SIM_SUBSTEPS

GROUND_Z   = 0.05
MAX_DIST   = 2.0
SUCCESS_R  = 0.15


# ── Warp kernels ──────────────────────────────────────────────────────────

@wp.kernel
def aero_kernel(
    body_q:    wp.array(dtype=wp.transform),
    joint_qd:  wp.array(dtype=wp.float32),
    kt:        wp.float32,
    kd:        wp.float32,
    body_f:    wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()                  # one per (world, rotor)
    w   = tid // 4
    r   = tid % 4
    af  = w * 5
    rb  = af + 1 + r
    dof = w * 10 + 6 + r

    omega = joint_qd[dof]
    n2    = omega * omega

    q_af  = wp.transform_get_rotation(body_q[af])
    zaxis = wp.quat_rotate(q_af, wp.vec3(0.0, 0.0, 1.0))

    thrust = zaxis * (kt * n2)
    drag   = zaxis * (-(kd * n2 * wp.sign(omega)))   # opposes spin → yaw reaction
    body_f[rb] = wp.spatial_vector(thrust, drag)


@wp.kernel
def actuation_kernel(
    action:        wp.array(dtype=wp.float32),   # [N*4] in [-1,1]
    turn_dir:      wp.array(dtype=wp.float32),   # [4]
    motor_omega:   wp.array(dtype=wp.float32),   # [N*4] LPF state (signed), in/out
    omega_hover:   wp.float32,
    omega_range:   wp.float32,
    omega_min:     wp.float32,
    omega_max:     wp.float32,
    alpha:         wp.float32,
    joint_target:  wp.array(dtype=wp.float32),   # [N*10] out
):
    tid = wp.tid()
    w   = tid // 4
    r   = tid % 4
    a   = wp.clamp(action[tid], -1.0, 1.0)
    mag = wp.clamp(omega_hover + a * omega_range, omega_min, omega_max)
    sp  = turn_dir[r] * mag
    new = (1.0 - alpha) * motor_omega[tid] + alpha * sp
    motor_omega[tid] = new
    joint_target[w * 10 + 6 + r] = new


@wp.kernel
def obs_kernel(
    body_q:    wp.array(dtype=wp.transform),
    joint_qd:  wp.array(dtype=wp.float32),
    target:    wp.array(dtype=wp.vec3),
    a_prev:    wp.array(dtype=wp.float32),
    obs:       wp.array(dtype=wp.float32),
):
    w   = wp.tid()
    af  = w * 5
    tf  = body_q[af]
    p   = wp.transform_get_translation(tf)
    q   = wp.transform_get_rotation(tf)
    R   = wp.quat_to_matrix(q)

    base = w * 22
    tgt  = target[w]
    obs[base + 0] = p[0] - tgt[0]
    obs[base + 1] = p[1] - tgt[1]
    obs[base + 2] = p[2] - tgt[2]
    obs[base + 3] = R[0, 0]; obs[base + 4] = R[0, 1]; obs[base + 5] = R[0, 2]
    obs[base + 6] = R[1, 0]; obs[base + 7] = R[1, 1]; obs[base + 8] = R[1, 2]
    obs[base + 9] = R[2, 0]; obs[base +10] = R[2, 1]; obs[base +11] = R[2, 2]
    d = w * 10
    obs[base +12] = joint_qd[d + 0]
    obs[base +13] = joint_qd[d + 1]
    obs[base +14] = joint_qd[d + 2]
    wx = joint_qd[d + 3]; wy = joint_qd[d + 4]; wz = joint_qd[d + 5]
    obs[base +15] = R[0, 0]*wx + R[1, 0]*wy + R[2, 0]*wz   # Rᵀ·w_world (body frame)
    obs[base +16] = R[0, 1]*wx + R[1, 1]*wy + R[2, 1]*wz
    obs[base +17] = R[0, 2]*wx + R[1, 2]*wy + R[2, 2]*wz
    obs[base +18] = a_prev[w*4 + 0]
    obs[base +19] = a_prev[w*4 + 1]
    obs[base +20] = a_prev[w*4 + 2]
    obs[base +21] = a_prev[w*4 + 3]


@wp.kernel
def reward_done_kernel(
    obs:        wp.array(dtype=wp.float32),
    body_q:     wp.array(dtype=wp.transform),
    action:     wp.array(dtype=wp.float32),
    step_count: wp.array(dtype=wp.int32),
    c:          wp.float32,
    crp_i: wp.float32, crp_t: wp.float32,
    crv_i: wp.float32, crv_t: wp.float32,
    cra_i: wp.float32, cra_t: wp.float32,
    crw_i: wp.float32, crw_t: wp.float32,
    crq: wp.float32, crs: wp.float32, crab: wp.float32,
    p_clip2:    wp.float32,
    c_app:      wp.float32,
    prev_dist:  wp.array(dtype=wp.float32),
    max_steps:  wp.int32,
    ground_z:   wp.float32,
    max_dist:   wp.float32,
    reward:     wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.float32),
    truncated:  wp.array(dtype=wp.float32),
    dist_out:   wp.array(dtype=wp.float32),
):
    w = wp.tid()
    b = w * 22
    px = obs[b+0]; py = obs[b+1]; pz = obs[b+2]
    tr = obs[b+3] + obs[b+7] + obs[b+11]
    qw2 = wp.clamp((tr + 1.0) * 0.25, 0.0, 1.0)
    vx = obs[b+12]; vy = obs[b+13]; vz = obs[b+14]
    wx = obs[b+15]; wy = obs[b+16]; wz = obs[b+17]

    crp = crp_i + c * (crp_t - crp_i)
    crv = crv_i + c * (crv_t - crv_i)
    cra = cra_i + c * (cra_t - cra_i)
    crw = crw_i + c * (crw_t - crw_i)

    p2 = px*px + py*py + pz*pz
    p2c = wp.min(p2, p_clip2)              # clipped position cost (bounds far-spawn penalty)
    v2 = vx*vx + vy*vy + vz*vz
    w2 = wx*wx + wy*wy + wz*wz
    a0 = action[w*4+0]-crab; a1 = action[w*4+1]-crab
    a2 = action[w*4+2]-crab; a3 = action[w*4+3]-crab
    asum = a0*a0 + a1*a1 + a2*a2 + a3*a3

    dist = wp.sqrt(p2)
    # Potential-based approach shaping (policy-invariant; Ng et al. 1999): reward each
    # metre of distance reduced toward the target. Gives PPO a dense navigation gradient
    # at every distance — the squared penalty alone is nearly flat far from the goal, so
    # a survival-dominant reward leaves the drone content to hover anywhere. Φ(s)=−c_app·d.
    approach = c_app * (prev_dist[w] - dist)
    prev_dist[w] = dist

    reward[w] = (-crp*p2c - crq*(1.0 - qw2) - crv*v2 - crw*w2 - cra*asum + crs
                 + approach)

    dist_out[w] = dist

    z = wp.transform_get_translation(body_q[w*5])[2]
    sc = step_count[w] + 1
    step_count[w] = sc
    term = 0.0
    if z < ground_z or dist > max_dist:
        term = 1.0
    terminated[w] = term
    trunc = 0.0
    if sc >= max_steps:
        trunc = 1.0
    truncated[w] = trunc


@wp.func
def _rand_uniform(state: wp.uint32, lo: wp.float32, hi: wp.float32):
    return lo + (hi - lo) * wp.randf(state)


@wp.kernel
def reset_kernel(
    reset_mask: wp.array(dtype=wp.float32),    # [N] 1=reset
    seed:       wp.int32,
    c:          wp.float32,
    omega_hover: wp.float32,
    omega_max:   wp.float32,
    turn_dir:   wp.array(dtype=wp.float32),    # [4]
    joint_q:    wp.array(dtype=wp.float32),    # [N*11] in/out
    joint_qd:   wp.array(dtype=wp.float32),    # [N*10] in/out
    motor_omega: wp.array(dtype=wp.float32),   # [N*4] out
    a_prev:     wp.array(dtype=wp.float32),    # [N*4] out
    step_count: wp.array(dtype=wp.int32),      # [N] out
    target:     wp.array(dtype=wp.vec3),       # [N] out
    prev_dist:  wp.array(dtype=wp.float32),    # [N] out (approach-shaping baseline)
):
    w = wp.tid()
    if reset_mask[w] == 0.0:
        return

    rng = wp.rand_init(seed, w)

    # ── Random target (matches eval distribution) ────────────────────────
    angle = _rand_uniform(rng, 0.0, 6.2831853)
    radius = _rand_uniform(rng, 0.5, 1.5)
    alt    = _rand_uniform(rng, 0.3, 1.2)
    tx = radius * wp.cos(angle)
    ty = radius * wp.sin(angle)
    tz = alt
    target[w] = wp.vec3(tx, ty, tz)

    # ── Curriculum spawn extremes ────────────────────────────────────────
    pos_range = 0.15 + (1.5 - 0.15) * c
    max_tilt  = (0.2617994 + (1.5707963 - 0.2617994) * c)        # 15°→90° in rad
    vel_range = 1.0 * c      # zero linear velocity at c=0 → true hover start
    ang_range = 1.0 * c      # zero body rate at c=0
    # Rotor speed init: exactly hover at c=0 (the drone starts in thrust equilibrium),
    # widening to [0, MAX/2] as the curriculum ramps.
    rpm_lo    = omega_hover - c * omega_hover
    rpm_hi    = omega_hover + c * (omega_max * 0.5 - omega_hover)

    coord = w * 11
    qx = 0.0; qy = 0.0; qz = 0.0; qw = 1.0

    guide = wp.randf(rng)
    if guide < 0.10:
        # 10% guidance branch: spawn at target, identity attitude.
        joint_q[coord + 0] = tx
        joint_q[coord + 1] = ty
        joint_q[coord + 2] = tz
    else:
        dx = _rand_uniform(rng, -pos_range, pos_range)
        dy = _rand_uniform(rng, -pos_range, pos_range)
        dz = _rand_uniform(rng, -pos_range, pos_range)
        joint_q[coord + 0] = tx + dx
        joint_q[coord + 1] = ty + dy
        joint_q[coord + 2] = wp.clamp(tz + dz, 0.15, 1.5)
        # capped-tilt orientation: body-z on spherical cap × uniform yaw
        cos_t = _rand_uniform(rng, wp.cos(max_tilt), 1.0)
        phi   = _rand_uniform(rng, 0.0, 6.2831853)
        half  = wp.acos(wp.clamp(cos_t, -1.0, 1.0)) * 0.5
        s = wp.sin(half); cc = wp.cos(half)
        ax = -wp.sin(phi) * s
        ay =  wp.cos(phi) * s
        # q_tilt = (ax, ay, 0, cc)
        hy = _rand_uniform(rng, 0.0, 6.2831853) * 0.5
        yz = wp.sin(hy); yw = wp.cos(hy)
        # q = q_tilt * q_yaw  (q_yaw = (0,0,yz,yw))
        qx = cc*0.0 + ax*yw + ay*yz - 0.0*0.0
        qy = cc*0.0 - ax*yz + ay*yw + 0.0*0.0
        qz = cc*yz + ax*0.0 - ay*0.0 + 0.0*yw
        qw = cc*yw - ax*0.0 - ay*0.0 - 0.0*yz

    joint_q[coord + 3] = qx
    joint_q[coord + 4] = qy
    joint_q[coord + 5] = qz
    joint_q[coord + 6] = qw
    joint_q[coord + 7] = 0.0     # rotor angles
    joint_q[coord + 8] = 0.0
    joint_q[coord + 9] = 0.0
    joint_q[coord +10] = 0.0

    dof = w * 10
    joint_qd[dof + 0] = _rand_uniform(rng, -vel_range, vel_range)
    joint_qd[dof + 1] = _rand_uniform(rng, -vel_range, vel_range)
    joint_qd[dof + 2] = _rand_uniform(rng, -vel_range, vel_range)
    joint_qd[dof + 3] = _rand_uniform(rng, -ang_range, ang_range)
    joint_qd[dof + 4] = _rand_uniform(rng, -ang_range, ang_range)
    joint_qd[dof + 5] = _rand_uniform(rng, -ang_range, ang_range)
    for i in range(4):
        spd = wp.clamp(_rand_uniform(rng, rpm_lo, rpm_hi), 0.0, omega_max)
        sgn = turn_dir[i] * spd
        joint_qd[dof + 6 + i] = sgn
        motor_omega[w*4 + i]  = sgn
        a_prev[w*4 + i] = 0.0

    step_count[w] = 0
    # Approach-shaping baseline = spawn distance to target (no spurious shaping on step 0).
    ex = joint_q[coord + 0] - tx
    ey = joint_q[coord + 1] - ty
    ez = joint_q[coord + 2] - tz
    prev_dist[w] = wp.sqrt(ex*ex + ey*ey + ez*ez)


@wp.kernel
def copy_action_kernel(
    action: wp.array(dtype=wp.float32),
    a_prev: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    a_prev[tid] = action[tid]


# ── Batched environment ───────────────────────────────────────────────────

class BatchedDroneEnv:
    """N-world Crazyflie position controller on SolverMuJoCo (CLAUDE.md §4)."""

    def __init__(self, num_envs: int = 4096, device: str = "cuda",
                 curriculum: float = 0.0, seed: int = 0, capture_graph: bool = True,
                 viewer=None, render_worlds: int = 1, render_every: int = 2,
                 render_spacing: float = 4.0):
        self.num_envs   = int(num_envs)
        self.device     = device
        self.curriculum = float(np.clip(curriculum, 0.0, 1.0))
        self.max_episode_steps = MAX_EPISODE_STEPS
        self._rng_ctr   = int(seed) * 1_000_003 + 1
        self._capture   = capture_graph

        self.num_obs = OBS_DIM
        self.num_act = 4

        # Real-time OpenGL viewer (optional)
        self._viewer       = viewer
        self._render_every = max(int(render_every), 1)
        self._render_t     = 0.0
        self._render_tick  = 0
        self._n_render     = min(int(render_worlds), self.num_envs)
        self._render_spacing = float(render_spacing)

        self._build()
        self._alloc()
        if self._viewer is not None:
            self._setup_viewer()
        self._graph = None
        if self._capture:
            self._capture_graph()
        # Prime everything with a full reset.
        self.reset()

    # ── Model construction (T1) ──────────────────────────────────────────
    def _build(self) -> None:
        template = self._make_template()
        main = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(main)
        main.replicate(template, world_count=self.num_envs)
        self.model = main.finalize()

        self.solver = newton.solvers.SolverMuJoCo(
            self.model, integrator="implicitfast", solver="newton", cone="elliptic",
        )
        self.view = ArticulationView(self.model, pattern="drone*", verbose=False)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

    def _make_template(self) -> newton.ModelBuilder:
        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.add_ground_plane()

        # Airframe inertia = CF tensor minus the rotors' parallel-axis contribution,
        # so the composite vehicle inertia stays ≈ the Crazyflie tensor (CLAUDE §4.1).
        ixx_pa = 0.0; iyy_pa = 0.0; izz_pa = 0.0
        for (rx, ry, rz) in ROTOR_POS:
            ixx_pa += ROTOR_MASS * (ry*ry + rz*rz)
            iyy_pa += ROTOR_MASS * (rx*rx + rz*rz)
            izz_pa += ROTOR_MASS * (rx*rx + ry*ry)
        ixx = max(CF_IXX - ixx_pa, 1e-6)
        iyy = max(CF_IYY - iyy_pa, 1e-6)
        izz = max(CF_IZZ - izz_pa, 1e-6)

        af = b.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            mass=AIRFRAME_MASS,
            inertia=wp.mat33(ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz),
            label="drone",
        )
        al = CF_ARM
        for hx, hy, hz in [(al*0.05, al, al*0.05), (al, al*0.05, al*0.05)]:
            b.add_shape_box(af, hx=hx, hy=hy, hz=hz,
                            cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        jf = b.add_joint_free(child=af, label="drone_free")

        rotor_I = wp.mat33(1e-8, 0, 0, 0, 1e-8, 0, 0, 0, ARMATURE)
        joints = [jf]
        for i, (rx, ry, rz) in enumerate(ROTOR_POS):
            r = b.add_link(
                xform=wp.transform(wp.vec3(float(rx), float(ry), 0.5), wp.quat_identity()),
                mass=ROTOR_MASS, inertia=rotor_I, label=f"rotor{i}",
            )
            j = b.add_joint_revolute(
                parent=af, child=r, axis=(0.0, 0.0, 1.0),
                armature=ARMATURE, effort_limit=EFFORT_LIMIT, friction=FRICTION,
                target_kd=TARGET_KD, actuator_mode=newton.JointTargetMode.VELOCITY,
                label=f"rotorj{i}",
            )
            joints.append(j)
        b.add_articulation(joints, label="drone")
        return b

    # ── Buffers ──────────────────────────────────────────────────────────
    def _alloc(self) -> None:
        N = self.num_envs
        d = self.device
        self.turn_dir    = wp.array(TURN_DIR, dtype=wp.float32, device=d)
        self.motor_omega = wp.zeros(N*4, dtype=wp.float32, device=d)
        self.action_wp   = wp.zeros(N*4, dtype=wp.float32, device=d)
        self.a_prev      = wp.zeros(N*4, dtype=wp.float32, device=d)
        self.obs_wp      = wp.zeros(N*OBS_DIM, dtype=wp.float32, device=d)
        self.reward_wp   = wp.zeros(N, dtype=wp.float32, device=d)
        self.term_wp     = wp.zeros(N, dtype=wp.float32, device=d)
        self.trunc_wp    = wp.zeros(N, dtype=wp.float32, device=d)
        self.dist_wp     = wp.zeros(N, dtype=wp.float32, device=d)
        self.step_count  = wp.zeros(N, dtype=wp.int32, device=d)
        self.target      = wp.zeros(N, dtype=wp.vec3, device=d)
        self.reset_mask  = wp.zeros(N, dtype=wp.float32, device=d)
        self.prev_dist   = wp.zeros(N, dtype=wp.float32, device=d)

        # torch views (zero-copy) for the RL interface
        self.obs_t    = wp.to_torch(self.obs_wp).view(N, OBS_DIM)
        self.reward_t = wp.to_torch(self.reward_wp)
        self.term_t   = wp.to_torch(self.term_wp)
        self.trunc_t  = wp.to_torch(self.trunc_wp)
        self.dist_t   = wp.to_torch(self.dist_wp)

    # ── CUDA-graph substep loop (T6) ─────────────────────────────────────
    def _substeps(self) -> None:
        for _ in range(SIM_SUBSTEPS):
            self.state_0.clear_forces()
            wp.launch(aero_kernel, dim=self.num_envs*4,
                      inputs=[self.state_0.body_q, self.state_0.joint_qd, KT_SI, KD_SI],
                      outputs=[self.state_0.body_f], device=self.device)
            self.solver.step(self.state_0, self.state_1, self.control, None, SUBSTEP_DT)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _capture_graph(self) -> None:
        # Warm up (compile kernels) before capture.
        self._substeps()
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(self.device) as capture:
            self._substeps()
        self._graph = capture.graph

    # ── Forward kinematics helper ────────────────────────────────────────
    def _fk(self) -> None:
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    # ── Obs assembly (T4) ────────────────────────────────────────────────
    def _compute_obs(self) -> None:
        wp.launch(obs_kernel, dim=self.num_envs,
                  inputs=[self.state_0.body_q, self.state_0.joint_qd, self.target, self.a_prev],
                  outputs=[self.obs_wp], device=self.device)

    # ── Masked curriculum reset (T5) ─────────────────────────────────────
    def _do_reset(self, mask_wp) -> None:
        self._rng_ctr += 1
        wp.launch(reset_kernel, dim=self.num_envs,
                  inputs=[mask_wp, self._rng_ctr, self.curriculum,
                          OMEGA_HOVER, OMEGA_MAX, self.turn_dir],
                  outputs=[self.state_0.joint_q, self.state_0.joint_qd,
                           self.motor_omega, self.a_prev, self.step_count, self.target,
                           self.prev_dist],
                  device=self.device)
        self._fk()

    def reset(self):
        wp.launch(_fill_ones, dim=self.num_envs, inputs=[], outputs=[self.reset_mask],
                  device=self.device)
        self._do_reset(self.reset_mask)
        self._compute_obs()
        return self.obs_t, {}

    # ── Real-time OpenGL viewer (CUDA/GL interop) ────────────────────────
    def _setup_viewer(self) -> None:
        """Attach the GL viewer to a subset of worlds (display-only offsets)."""
        v = self._viewer
        v.set_model(self.model)
        ids = list(range(self._n_render))
        v.set_visible_worlds(ids)
        if self._n_render > 1:
            v.set_world_offsets((self._render_spacing, self._render_spacing, 0.0))
            try:
                from newton.utils import compute_world_offsets
                off = compute_world_offsets(
                    self._n_render,
                    (self._render_spacing, self._render_spacing, 0.0),
                    self.model.up_axis,
                )
                self._render_offsets = np.asarray(
                    off.numpy() if hasattr(off, "numpy") else off, dtype=np.float32
                ).reshape(-1, 3)[:self._n_render]
            except Exception:
                self._render_offsets = np.zeros((self._n_render, 3), dtype=np.float32)
        else:
            self._render_offsets = np.zeros((1, 3), dtype=np.float32)
        self._tgt_color = wp.array(
            [wp.vec3(1.0, 0.25, 0.0)] * self._n_render, dtype=wp.vec3, device=self.device)

    def render(self) -> None:
        """Draw one frame: drone state + target markers for the visible worlds."""
        v = self._viewer
        if v is None:
            return
        if not v.is_running():          # window closed → stop rendering, keep running
            self._viewer = None
            return
        self._render_t += SIM_DT
        v.begin_frame(self._render_t)
        v.log_state(self.state_0)
        tgt = self.target.numpy()[:self._n_render] + self._render_offsets
        xf = wp.array(
            [wp.transform(wp.vec3(float(p[0]), float(p[1]), float(p[2])), wp.quat_identity())
             for p in tgt], dtype=wp.transform, device=self.device)
        v.log_shapes("/targets", newton.GeoType.SPHERE, 0.05, xf, colors=self._tgt_color)
        v.end_frame()

    def _maybe_render(self) -> None:
        if self._viewer is None:
            return
        self._render_tick += 1
        if self._render_tick % self._render_every == 0:
            self.render()

    # ── Eval helpers (single-/few-world rollout; CPU sync OK off the hot loop) ──
    def set_target(self, target) -> None:
        """Set the goal position(s); ``target`` broadcastable to [num_envs, 3]."""
        t = np.asarray(target, dtype=np.float32).reshape(-1, 3)
        if t.shape[0] == 1 and self.num_envs > 1:
            t = np.repeat(t, self.num_envs, axis=0)
        self.target.assign(t)
        self._compute_obs()

    def place(self, pos=(0.0, 0.0, 0.5), quat=(0.0, 0.0, 0.0, 1.0)) -> None:
        """Place every world at a fixed hover pose (eval initialisation)."""
        jq  = self.state_0.joint_q.numpy()
        jqd = self.state_0.joint_qd.numpy()
        mo  = self.motor_omega.numpy()
        for w in range(self.num_envs):
            c = w * COORDS_PER_WORLD
            jq[c:c+3] = pos
            jq[c+3:c+7] = quat
            jq[c+7:c+11] = 0.0
            d = w * DOFS_PER_WORLD
            jqd[d:d+DOFS_PER_WORLD] = 0.0
            for i in range(4):
                jqd[d+6+i] = TURN_DIR[i] * OMEGA_HOVER
                mo[w*4+i]  = TURN_DIR[i] * OMEGA_HOVER
        self.state_0.joint_q.assign(jq)
        self.state_0.joint_qd.assign(jqd)
        self.motor_omega.assign(mo)
        self.step_count.zero_()
        self.a_prev.zero_()
        self._fk()
        self._compute_obs()

    # ── Step ─────────────────────────────────────────────────────────────
    def step(self, actions: torch.Tensor, auto_reset: bool = True):
        a = actions.detach().to(self.device, torch.float32).contiguous().view(-1)
        wp.copy(self.action_wp, wp.from_torch(a, dtype=wp.float32))

        # Motor LPF + write velocity targets (T3).
        wp.launch(actuation_kernel, dim=self.num_envs*4,
                  inputs=[self.action_wp, self.turn_dir, self.motor_omega,
                          OMEGA_HOVER, OMEGA_RANGE, OMEGA_MIN, OMEGA_MAX, MOTOR_ALPHA],
                  outputs=[self.control.joint_target_vel], device=self.device)

        # Physics (replay captured substep loop, or run eagerly).
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._substeps()

        # Obs / reward / termination on the post-step state (T4).
        self._compute_obs()
        wp.launch(reward_done_kernel, dim=self.num_envs,
                  inputs=[self.obs_wp, self.state_0.body_q, self.action_wp, self.step_count,
                          self.curriculum,
                          _C_RP_INIT, _C_RP_TGT, _C_RV_INIT, _C_RV_TGT, _C_RA_INIT, _C_RA_TGT,
                          _C_RW_INIT, _C_RW_TGT, _C_RQ, _C_RS, _C_RAB,
                          P_ERR_CLIP2, _C_APPROACH, self.prev_dist,
                          self.max_episode_steps, GROUND_Z, MAX_DIST],
                  outputs=[self.reward_wp, self.term_wp, self.trunc_wp, self.dist_wp],
                  device=self.device)

        # a_prev = action (history depth 1), then auto-reset done envs.
        wp.launch(copy_action_kernel, dim=self.num_envs*4,
                  inputs=[self.action_wp], outputs=[self.a_prev], device=self.device)

        if auto_reset:
            wp.launch(_or_mask, dim=self.num_envs,
                      inputs=[self.term_wp, self.trunc_wp], outputs=[self.reset_mask],
                      device=self.device)
            self._do_reset(self.reset_mask)
            self._compute_obs()

        self._maybe_render()

        info = {"dist": self.dist_t, "success": (self.dist_t < SUCCESS_R).float()}
        return self.obs_t, self.reward_t, self.term_t, self.trunc_t, info

    def set_curriculum(self, c: float) -> None:
        self.curriculum = float(np.clip(c, 0.0, 1.0))


@wp.kernel
def _fill_ones(out: wp.array(dtype=wp.float32)):
    out[wp.tid()] = 1.0


@wp.kernel
def _or_mask(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32),
             out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    v = 0.0
    if a[tid] > 0.5 or b[tid] > 0.5:
        v = 1.0
    out[tid] = v
