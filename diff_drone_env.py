# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Differentiable batched quadrotor env on Newton's SolverFeatherstone, for SHAC
(Short-Horizon Actor-Critic, Xu et al., ICLR 2022, arXiv:2204.07137).

SHAC backpropagates the policy gradient *through* the simulator, so the whole
forward path (action → motor LPF → aero wrench → rigid-body dynamics → obs → reward)
must be differentiable. We therefore use:

  * **SolverFeatherstone** — Newton's reduced-coordinate articulation solver, which
    supports Warp autodiff (requires_grad, custom adjoints) — unlike SolverMuJoCo.
  * A **single FREE-joint rigid body** (no stiff rotor DOFs) with a smooth force-kernel
    aero model: 4 rotor thrusts produce a body-frame wrench (collective thrust +
    roll/pitch from arm geometry + yaw from drag), applied via ``state.body_f``. This
    keeps the dynamics smooth → clean gradients (stiff rotor joints would wreck them).
  * Hover-centred [-1,1] action and the 22-D observation, preserved from the RL envs.

The class exposes a PyTorch-differentiable ``step``: state and actions are torch
tensors; ``DiffStep`` (a torch.autograd.Function) runs one control step under a Warp
``Tape`` and bridges gradients back to PyTorch, so a SHAC rollout chains steps in the
torch graph and ``loss.backward()`` flows into the policy weights.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp
import newton
import newton.solvers

from drone_gym_env import (
    SIM_DT, OBS_DIM, CF_MASS, CF_ARM, CF_IXX, CF_IYY, CF_IZZ, CF_KT, CF_KD,
    CF_MAX_RPM, CF_MIN_RPM, CF_HOVER_RPM, CF_RPM_RANGE, MOTOR_TAU,
)

RPM_TO_RAD = 2.0 * np.pi / 60.0
RAD_TO_RPM = 60.0 / (2.0 * np.pi)
OMEGA_HOVER = CF_HOVER_RPM * RPM_TO_RAD
OMEGA_RANGE = CF_RPM_RANGE * RPM_TO_RAD
OMEGA_MAX   = CF_MAX_RPM * RPM_TO_RAD
OMEGA_MIN   = CF_MIN_RPM * RPM_TO_RAD
KT_SI = CF_KT * RAD_TO_RPM * RAD_TO_RPM
KD_SI = CF_KD * RAD_TO_RPM * RAD_TO_RPM
MOTOR_ALPHA = SIM_DT / MOTOR_TAU

ROTOR_POS = np.array([[0.0, CF_ARM, 0.0], [0.0, -CF_ARM, 0.0],
                      [CF_ARM, 0.0, 0.0], [-CF_ARM, 0.0, 0.0]], dtype=np.float32)
TURN_DIR = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32)

SIM_SUBSTEPS = 4
SUBSTEP_DT   = SIM_DT / SIM_SUBSTEPS

# Reward (redesigned for a WELL-POSED objective whose unique optimum is "on target,
# upright, still"). SHAC follows exact gradients, so dense smooth shaping is ideal:
#   r = −w_pos‖p‖²              dense quadratic pull (gradient everywhere, strongest far)
#       + w_prox·exp(−‖p‖²/σ²)  sharp positive bonus, MAXIMAL on target → rewards precise
#                               arrival (the thing every method failed at)
#       + w_up·R22              keep upright (R22 = body-z·world-z ∈ [−1,1])
#       − w_vel‖v‖²             settle (no orbiting)
#       − w_spin‖ω‖²            damp rotation
#       − w_act‖a‖²             small, smooth, hover-centred actions
# No survival constant and no clip: the only way to raise reward is to get ON target and
# stay there, so there is no comfortable off-target hover basin to collapse into.
_W_POS, _W_PROX, _PROX_SIG2 = 1.0, 3.0, 0.09     # σ = 0.3 m
_W_UP, _W_VEL, _W_SPIN, _W_ACT = 1.0, 0.15, 0.02, 0.02


# ── Differentiable kernels ────────────────────────────────────────────────

@wp.kernel
def motor_lpf_kernel(
    action: wp.array(dtype=wp.float32),       # [N*4]
    omega_in: wp.array(dtype=wp.float32),     # [N*4]
    turn_dir: wp.array(dtype=wp.float32),     # [4]
    omega_out: wp.array(dtype=wp.float32),    # [N*4] out (signed rad/s)
):
    tid = wp.tid()
    r = tid % 4
    a = wp.clamp(action[tid], -1.0, 1.0)
    mag = wp.clamp(OMEGA_HOVER + a * OMEGA_RANGE, OMEGA_MIN, OMEGA_MAX)
    sp = turn_dir[r] * mag
    omega_out[tid] = (1.0 - MOTOR_ALPHA) * omega_in[tid] + MOTOR_ALPHA * sp


@wp.kernel
def aero_kernel(
    joint_q: wp.array(dtype=wp.float32),      # [N*7] free joint (pos3, quat4)
    omega: wp.array(dtype=wp.float32),        # [N*4] signed rotor speed
    arm: wp.array(dtype=wp.vec3),             # [4] rotor positions (body frame)
    joint_f: wp.array(dtype=wp.float32),      # [N*6] out: free-joint world wrench
):
    # NOTE: forces enter Featherstone differentiably through control.joint_f (the free
    # joint's 6-D generalized wrench, world frame) — state.body_f is NOT differentiable.
    w = wp.tid()
    q = wp.quat(joint_q[w*7+3], joint_q[w*7+4], joint_q[w*7+5], joint_q[w*7+6])
    fz = float(0.0)
    mx = float(0.0); my = float(0.0); mz = float(0.0)
    for i in range(4):
        om = omega[w*4+i]
        n2 = om * om
        thr = KT_SI * n2
        drg = KD_SI * n2 * wp.sign(om)
        fz += thr
        p = arm[i]
        mx += p[1] * thr        # (p × T·ẑ)_x =  p_y·T
        my += -p[0] * thr       # (p × T·ẑ)_y = -p_x·T
        mz += -drg              # reaction torque about +z
    f_world = wp.quat_rotate(q, wp.vec3(0.0, 0.0, fz))
    m_world = wp.quat_rotate(q, wp.vec3(mx, my, mz))
    joint_f[w*6+0] = f_world[0]; joint_f[w*6+1] = f_world[1]; joint_f[w*6+2] = f_world[2]
    joint_f[w*6+3] = m_world[0]; joint_f[w*6+4] = m_world[1]; joint_f[w*6+5] = m_world[2]


@wp.kernel
def obs_kernel(
    joint_q: wp.array(dtype=wp.float32),      # [N*7]
    joint_qd: wp.array(dtype=wp.float32),     # [N*6]
    target: wp.array(dtype=wp.vec3),          # [N]
    a_prev: wp.array(dtype=wp.float32),       # [N*4]
    obs: wp.array(dtype=wp.float32),          # [N*22] out
):
    w = wp.tid()
    base = w * 22
    px = joint_q[w*7+0]; py = joint_q[w*7+1]; pz = joint_q[w*7+2]
    q = wp.quat(joint_q[w*7+3], joint_q[w*7+4], joint_q[w*7+5], joint_q[w*7+6])
    R = wp.quat_to_matrix(q)
    t = target[w]
    obs[base+0] = px - t[0]; obs[base+1] = py - t[1]; obs[base+2] = pz - t[2]
    obs[base+3] = R[0,0]; obs[base+4] = R[0,1]; obs[base+5] = R[0,2]
    obs[base+6] = R[1,0]; obs[base+7] = R[1,1]; obs[base+8] = R[1,2]
    obs[base+9] = R[2,0]; obs[base+10] = R[2,1]; obs[base+11] = R[2,2]
    # free-joint qd: [0:3] linear (world), [3:6] angular (world) — verified for Newton
    d = w * 6
    vx = joint_qd[d+0]; vy = joint_qd[d+1]; vz = joint_qd[d+2]
    obs[base+12] = vx; obs[base+13] = vy; obs[base+14] = vz
    wx = joint_qd[d+3]; wy = joint_qd[d+4]; wz = joint_qd[d+5]
    obs[base+15] = R[0,0]*wx + R[1,0]*wy + R[2,0]*wz       # Rᵀω → body frame
    obs[base+16] = R[0,1]*wx + R[1,1]*wy + R[2,1]*wz
    obs[base+17] = R[0,2]*wx + R[1,2]*wy + R[2,2]*wz
    obs[base+18] = a_prev[w*4+0]; obs[base+19] = a_prev[w*4+1]
    obs[base+20] = a_prev[w*4+2]; obs[base+21] = a_prev[w*4+3]


@wp.kernel
def reward_kernel(
    obs: wp.array(dtype=wp.float32),          # [N*22]
    action: wp.array(dtype=wp.float32),       # [N*4]
    w_pos: wp.float32, w_prox: wp.float32, prox_sig2: wp.float32,
    w_up: wp.float32, w_vel: wp.float32, w_spin: wp.float32, w_act: wp.float32,
    reward: wp.array(dtype=wp.float32),       # [N] out
    dist_out: wp.array(dtype=wp.float32),     # [N] out
):
    w = wp.tid()
    b = w * 22
    px = obs[b+0]; py = obs[b+1]; pz = obs[b+2]
    R22 = obs[b+11]                            # body-z · world-z (upright measure)
    vx = obs[b+12]; vy = obs[b+13]; vz = obs[b+14]
    wx = obs[b+15]; wy = obs[b+16]; wz = obs[b+17]
    p2 = px*px + py*py + pz*pz
    v2 = vx*vx + vy*vy + vz*vz
    w2 = wx*wx + wy*wy + wz*wz
    a2 = (action[w*4+0]*action[w*4+0] + action[w*4+1]*action[w*4+1]
          + action[w*4+2]*action[w*4+2] + action[w*4+3]*action[w*4+3])
    dist_out[w] = wp.sqrt(p2 + 1.0e-8)
    reward[w] = (-w_pos*p2 + w_prox*wp.exp(-p2/prox_sig2) + w_up*R22
                 - w_vel*v2 - w_spin*w2 - w_act*a2)


@wp.kernel
def copy_f32(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
    t = wp.tid()
    dst[t] = src[t]


# ── Curriculum spawn (shared by env.reset and the trainer's resample) ─────
MAX_OFFSET = 0.6     # m — short navigation distance that fits the BPTT window + critic

def sample_spawn(c, n, rng):
    """Decoupled curriculum: tilt ramps EARLY (stabilise first), navigation offset ramps
    LATER and short. Returns (target[n,3], q[n,7], qd[n,6], om[n,4]) as float32 numpy."""
    c = float(np.clip(c, 0.0, 1.0))
    tilt_c = min(c / 0.5, 1.0)            # full attitude challenge by c=0.5
    off_c  = max((c - 0.3) / 0.7, 0.0)    # navigation only starts after c=0.3
    ang = rng.uniform(0, 2*np.pi, n); rad = rng.uniform(0.5, 1.5, n); alt = rng.uniform(0.3, 1.2, n)
    tgt = np.stack([rad*np.cos(ang), rad*np.sin(ang), alt], 1).astype(np.float32)
    pr = MAX_OFFSET * off_c
    spawn = tgt + rng.uniform(-pr, pr, (n, 3)).astype(np.float32)
    spawn[:, 2] = np.clip(spawn[:, 2], 0.15, 1.5)
    mt = (90.0 * tilt_c) * np.pi/180.0
    cos_t = rng.uniform(np.cos(mt), 1.0, n); phi = rng.uniform(0, 2*np.pi, n)
    half = np.arccos(np.clip(cos_t, -1, 1))*0.5; s = np.sin(half); cc = np.cos(half)
    ax = -np.sin(phi)*s; ay = np.cos(phi)*s
    hy = rng.uniform(0, 2*np.pi, n)*0.5; yz = np.sin(hy); yw = np.cos(hy)
    q = np.zeros((n, 7), np.float32); q[:, :3] = spawn
    q[:, 3] = ax*yw + ay*yz; q[:, 4] = -ax*yz + ay*yw; q[:, 5] = cc*yz; q[:, 6] = cc*yw
    qd = (rng.uniform(-1, 1, (n, 6)) * 0.5 * c).astype(np.float32)
    om = (TURN_DIR[None, :]*OMEGA_HOVER).repeat(n, 0).astype(np.float32)
    return tgt, q, qd, om


# ── Differentiable environment ────────────────────────────────────────────

class DiffDroneEnv:
    def __init__(self, num_envs: int, device: str = "cuda"):
        self.num_envs = int(num_envs)
        self.device = device
        self.num_obs = OBS_DIM
        self.num_act = 4
        self._build()
        self._alloc()

    def _build(self):
        b = newton.ModelBuilder()
        # single free-floating drone body with the Crazyflie inertia
        body = b.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            mass=CF_MASS,
            inertia=wp.mat33(CF_IXX, 0, 0, 0, CF_IYY, 0, 0, 0, CF_IZZ),
            label="drone",
        )
        jf = b.add_joint_free(child=body, label="drone_free")
        b.add_articulation([jf], label="drone")
        # replicate to N worlds
        main = newton.ModelBuilder()
        main.replicate(b, world_count=self.num_envs)
        self.model = main.finalize(requires_grad=True)
        self.solver = newton.solvers.SolverFeatherstone(self.model)
        # state/control buffers live in the per-step slot pool (see _alloc)

    def _alloc(self):
        N, d = self.num_envs, self.device
        self.arm = wp.array(ROTOR_POS, dtype=wp.vec3, device=d)
        self.turn_dir = wp.array(TURN_DIR, dtype=wp.float32, device=d)
        # per-env (not per-step) buffers — enter the reward linearly / aren't differentiated
        self.a_prev  = wp.zeros(N*4, dtype=wp.float32, device=d)
        self.target  = wp.zeros(N, dtype=wp.vec3, device=d)
        self.prev_dist = wp.zeros(N, dtype=wp.float32, device=d)
        self.dist    = wp.zeros(N, dtype=wp.float32, device=d)
        # Per-step slot pool: each step in a SHAC window uses a distinct slot so ALL of its
        # forward intermediates survive until that step's backward (correct multi-step BPTT;
        # sharing buffers would corrupt gradients once ω drifts from hover across steps).
        self.MAX_H = 64
        self.slots = [self._make_slot() for _ in range(self.MAX_H)]
        self._slot = 0
        # non-differentiable helpers for obs_of()
        self.qh = wp.zeros(N*7, dtype=wp.float32, device=d)
        self.qdh = wp.zeros(N*6, dtype=wp.float32, device=d)
        self.obsh = wp.zeros(N*OBS_DIM, dtype=wp.float32, device=d)

    def _make_slot(self):
        N, d = self.num_envs, self.device
        g = dict(dtype=wp.float32, device=d, requires_grad=True)
        return dict(
            q_in=wp.zeros(N*7, **g), qd_in=wp.zeros(N*6, **g), om_in=wp.zeros(N*4, **g),
            act_in=wp.zeros(N*4, **g), om_out=wp.zeros(N*4, **g),
            obs_out=wp.zeros(N*OBS_DIM, **g), rew_out=wp.zeros(N, **g),
            q_out=wp.zeros(N*7, **g), qd_out=wp.zeros(N*6, **g),
            states=[self.model.state(requires_grad=True) for _ in range(SIM_SUBSTEPS + 1)],
            control=self.model.control(requires_grad=True),
        )

    def new_window(self):
        """Reset the slot counter at the start of each SHAC short-horizon window."""
        self._slot = 0

    # ── one differentiable control step on slot `s` (recorded on a tape) ──
    def _forward(self, s):
        N = self.num_envs
        s0 = s["states"][0]
        wp.copy(s0.joint_q, s["q_in"])
        wp.copy(s0.joint_qd, s["qd_in"])
        newton.eval_fk(self.model, s0.joint_q, s0.joint_qd, s0)
        wp.launch(motor_lpf_kernel, dim=N*4, inputs=[s["act_in"], s["om_in"], self.turn_dir],
                  outputs=[s["om_out"]], device=self.device)
        wp.launch(aero_kernel, dim=N, inputs=[s0.joint_q, s["om_out"], self.arm],
                  outputs=[s["control"].joint_f], device=self.device)
        for t in range(SIM_SUBSTEPS):
            self.solver.step(s["states"][t], s["states"][t+1], s["control"], None, SUBSTEP_DT)
        sN = s["states"][SIM_SUBSTEPS]
        wp.launch(obs_kernel, dim=N, inputs=[sN.joint_q, sN.joint_qd, self.target, self.a_prev],
                  outputs=[s["obs_out"]], device=self.device)
        wp.launch(reward_kernel, dim=N,
                  inputs=[s["obs_out"], s["act_in"],
                          _W_POS, _W_PROX, _PROX_SIG2, _W_UP, _W_VEL, _W_SPIN, _W_ACT],
                  outputs=[s["rew_out"], self.dist], device=self.device)
        # outputs copied AFTER obs/reward so their adjoints don't clobber joint_q/qd grads
        wp.launch(copy_f32, dim=N*7, inputs=[sN.joint_q], outputs=[s["q_out"]], device=self.device)
        wp.launch(copy_f32, dim=N*6, inputs=[sN.joint_qd], outputs=[s["qd_out"]], device=self.device)

    def _copy_in(self, dst_wp, src_torch):
        wp.launch(copy_f32, dim=dst_wp.shape[0],
                  inputs=[wp.from_torch(src_torch.contiguous().view(-1), dtype=wp.float32)],
                  outputs=[dst_wp], device=self.device)

    # ── public differentiable step ───────────────────────────────────────
    def step(self, q, qd, om, action):
        """One control step. All args/returns are torch tensors; differentiable
        w.r.t. q[N,7], qd[N,6], om[N,4], action[N,4]. Uses the next slot in the window."""
        idx = self._slot
        self._slot += 1
        return _DiffStep.apply(self, idx, q, qd, om, action)

    # ── reset (non-differentiable bookkeeping) ───────────────────────────
    def reset(self, curriculum: float = 1.0, seed: int = 0):
        N = self.num_envs
        rng = np.random.default_rng(seed)
        tgt, q, qd, om = sample_spawn(curriculum, N, rng)
        self.target.assign(tgt); self.a_prev.zero_()
        self.prev_dist.assign(np.linalg.norm(q[:, :3] - tgt, axis=1).astype(np.float32))
        d = self.device
        return (torch.tensor(q, device=d), torch.tensor(qd, device=d), torch.tensor(om, device=d))

    def obs_of(self, q, qd):
        """Observation from state torch tensors (for the actor at the window start)."""
        N = self.num_envs
        self._copy_in(self.qh, q); self._copy_in(self.qdh, qd)
        wp.launch(obs_kernel, dim=N, inputs=[self.qh, self.qdh, self.target, self.a_prev],
                  outputs=[self.obsh], device=self.device)
        return wp.to_torch(self.obsh).view(N, OBS_DIM).clone()

    def set_targets(self, tgt_np):
        self.target.assign(np.asarray(tgt_np, np.float32).reshape(-1, 3))

    def update_prev_dist(self):
        wp.copy(self.prev_dist, self.dist)


class _DiffStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, env, idx, q, qd, om, action):
        s = env.slots[idx]
        env._copy_in(s["q_in"], q);  env._copy_in(s["qd_in"], qd)
        env._copy_in(s["om_in"], om); env._copy_in(s["act_in"], action)
        for a in (s["q_in"], s["qd_in"], s["om_in"], s["act_in"],
                  s["q_out"], s["qd_out"], s["om_out"], s["obs_out"], s["rew_out"]):
            a.grad.zero_()
        tape = wp.Tape()
        with tape:
            env._forward(s)
        ctx.env = env; ctx.tape = tape; ctx.s = s
        N = env.num_envs
        return (wp.to_torch(s["q_out"]).view(N, 7).clone(),
                wp.to_torch(s["qd_out"]).view(N, 6).clone(),
                wp.to_torch(s["om_out"]).view(N, 4).clone(),
                wp.to_torch(s["obs_out"]).view(N, OBS_DIM).clone(),
                wp.to_torch(s["rew_out"]).view(N).clone())

    @staticmethod
    def backward(ctx, gq, gqd, gom, gobs, grew):
        env, tape, s = ctx.env, ctx.tape, ctx.s
        grads = {
            s["q_out"]:   wp.from_torch(gq.contiguous().view(-1),   dtype=wp.float32),
            s["qd_out"]:  wp.from_torch(gqd.contiguous().view(-1),  dtype=wp.float32),
            s["om_out"]:  wp.from_torch(gom.contiguous().view(-1),  dtype=wp.float32),
            s["obs_out"]: wp.from_torch(gobs.contiguous().view(-1), dtype=wp.float32),
            s["rew_out"]: wp.from_torch(grew.contiguous().view(-1), dtype=wp.float32),
        }
        tape.backward(grads=grads)
        N = env.num_envs
        out = (None, None,    # env, idx
               wp.to_torch(s["q_in"].grad).view(N, 7).clone(),
               wp.to_torch(s["qd_in"].grad).view(N, 6).clone(),
               wp.to_torch(s["om_in"].grad).view(N, 4).clone(),
               wp.to_torch(s["act_in"].grad).view(N, 4).clone())
        tape.zero()
        return out
