# CLAUDE.md — Drone Position Controller: Newton/MuJoCo + SKRL Migration

> Context for the Claude Code agent working on this repo.
> **Scope of this document: the position controller only** (`drone_gym_env.py`,
> `train_drone.py`, `eval_drone.py`). The landing (`landing/`) and hover/disturbance
> (`disturbance/`) envs share the same anti-patterns but are **out of scope** — port them
> later using the same recipe.

---

## 1. Goal

Migrate the quadrotor **position controller** from its current
single-body, sequential, CPU-bound form to a **GPU-parallel** Newton pipeline that follows
the DeepWiki architecture recommendation:

- **`SolverMuJoCo`** (not `SolverSemiImplicit`)
- **Generalized coordinates**: drone modeled as a `FREE` joint + **4 physical `REVOLUTE`
  rotor joints** with `armature` / `effort_limit` / `friction` (the **motor-level sim2real**
  path — chosen deliberately over the simpler force-kernel approach)
- **Massively parallel**: one model, N worlds via `builder.replicate`, `ArticulationView`
  for all I/O, CUDA-graph capture of the substep loop
- **RL via SKRL** (GPU-native): **PPO and SAC**, each with **MLP and GRU** policies

**Preserve** (this is non-negotiable — see §6): the paper reward (Eschmann et al., "Learning
to Fly in Seconds", RAL 2024), the curriculum schedule, the 22-D observation layout, the
hover-centred `[-1,1]` action space, the Crazyflie physical parameters, 100 Hz control, and
the algorithm hyperparameters. What changes is **how** the 4-D action drives the physics
(rotor DOFs instead of an LPF→force kernel) and **how many** envs run in parallel.

---

## 2. Repo map (in-scope files)

| File | Role | Disposition |
|---|---|---|
| `drone_gym_env.py` | Single-env Gym wrapper + Newton sim + propeller kernel | **Rewrite** into a batched N-world env |
| `train_drone.py` | SB3 PPO/SAC/TD3 training, `DummyVecEnv` | **Replace** with SKRL training (PPO+SAC, MLP+GRU) |
| `eval_drone.py` | Multi-waypoint eval rollout | **Adapt** to the new env/policy I/O |
| `LearningtoFlyinSeconds.pdf`, `Paper.md` | Reference paper | Keep (source of truth for reward/curriculum) |
| `physicssummmary.txt` | Honest physics audit | Keep (notes missing drag/ground-effect) |

Everything else (`drone_logs/`, `checkpoints*/`, `ppo_*.zip`, `sac_*.zip`, `td3_*.zip`,
`*.md` result logs) is **historical SB3 output** — not used by the new pipeline. The `.zip`
checkpoints are SB3 format and are **not loadable** by SKRL; treat them as archival.

---

## 3. Current architecture (what exists, and why it's wrong for Newton)

All three envs follow the identical pattern; here it is for `drone_gym_env.py`:

- **Single free rigid body**, no joint → *maximal coordinates* (`add_body`, line ~355).
- **`SolverSemiImplicit`** (line 381).
- **Propeller aerodynamics** applied by a custom Warp kernel `_apply_prop_forces`
  (lines 254–275) that writes `thrust + torque` into `state.body_f`. This physics is
  **correct** (F = K_T·n², Q = K_D·n², motor LPF τ = 0.15 s) and is the part worth keeping
  conceptually.
- **Per-step GPU→CPU sync**: `self._state.body_q.numpy()[0]` on every `step()` and `_get_obs()`
  (lines ~402, ~520). A single-body read, every step.
- **`gymnasium.Env`** — one environment per instance.

Training (`train_drone.py`):

- **`DummyVecEnv([_make_env(i) for i in range(num_envs)])`** (line 418): N **independent**
  single-body Newton models stepped **one at a time in a Python loop**, each syncing to CPU
  every step. **This serialization — not the solver — is the dominant cost.** It is the exact
  opposite of Newton's one-model-N-worlds design.
- **Stable-Baselines3** runs PPO/SAC/TD3; **GRU exists only for PPO** via
  `sb3_contrib.RecurrentPPO` (and it is actually LSTM, not GRU).
- ⚠️ **`train_drone.py` currently contains 26 unresolved git merge-conflict markers**
  (`<<<<<<< HEAD`, `=======`, `>>>>>>> ...`). **It will not run.** Resolving these is the
  first task (§7, T0) — but since we are replacing this file wholesale, the practical move is
  to write the new SKRL trainer fresh and delete the conflicted one.

### Gap summary

| Axis | Current | Target |
|---|---|---|
| Solver | `SolverSemiImplicit` | `SolverMuJoCo`, `integrator="implicitfast"` |
| Coordinates | maximal (free body, no joint) | generalized (`FREE` + 4 `REVOLUTE` rotors) |
| Motor model | first-order LPF (τ=0.15 s) + force kernel | physical rotor inertia (`armature`) + actuator |
| Parallelism | `DummyVecEnv`, N sequential Python envs | one model, N worlds (`builder.replicate`) |
| Obs/action transfer | per-step `.numpy()` sync | `ArticulationView`, GPU-resident tensors |
| GPU graph | none | CUDA-graph capture of substep loop |
| RL framework | SB3 (CPU-bound rollout) | SKRL (GPU-native) |
| Algorithms | PPO/SAC/TD3, GRU=PPO-only | PPO + SAC, both MLP + GRU |

---

## 4. Target architecture

### 4.1 Model: FREE base + 4 physical rotor joints

```
World (parent = -1)
 └── FREE joint ─────────────► drone body  (Crazyflie airframe)
       ├── REVOLUTE (axis +z, armature, effort_limit, friction) ─► rotor 0  (+y arm, CCW)
       ├── REVOLUTE (axis +z, armature, effort_limit, friction) ─► rotor 1  (−y arm,  CW)
       ├── REVOLUTE (axis +z, armature, effort_limit, friction) ─► rotor 2  (+x arm,  CW)
       └── REVOLUTE (axis +z, armature, effort_limit, friction) ─► rotor 3  (−x arm, CCW)
```

**Hard requirements** (verified against Newton source):

- `SolverMuJoCo` operates **only on articulations** — it cannot step a free maximal-coordinate
  body. The `FREE` joint is mandatory.
- Call **`SolverMuJoCo.register_custom_attributes(builder)` BEFORE adding any bodies/joints.**
  `armature` / `effort_limit` / `friction` / actuator modes are MuJoCo custom attributes; the
  solver will not work without this.
- After writing initial `joint_q` (post-`finalize` and on every reset), call
  **`newton.eval_fk(model, state.joint_q, state.joint_qd, state)`** to propagate generalized →
  maximal coords (`body_q`), which collision/rendering read.

**`add_joint_revolute` signature** (verified, `newton/_src/sim/builder.py:4263`):
```python
builder.add_joint_revolute(
    parent, child, *,
    axis=(0,0,1),
    armature=...,          # rotor reflected/added inertia about spin axis [kg·m²]
    effort_limit=...,      # max motor torque [N·m] — models saturation (sim2real)
    friction=...,          # bearing friction [N·m] — models shaft drag (sim2real)
    velocity_limit=...,    # optional max rotor speed
    target_kd=...,         # velocity-actuator gain (see 4.2 Option A)
    actuator_mode=...,     # newton.JointTargetMode.{VELOCITY|EFFORT} (see 4.2)
    label="rotor{i}",
)
```

**Mass / inertia consistency (CRITICAL — easy to get wrong):**
Total vehicle mass must remain **27 g**. Adding 4 rotor bodies with mass means the airframe body
mass must be reduced: `m_body + 4·m_rotor = 0.027 kg`. Crazyflie motor+prop ≈ 1–2 g each.
Likewise the composite inertia (airframe own-inertia + rotor parallel-axis contributions) should
still ≈ the CF tensor (Ixx=Iyy=1.657e-5, Izz=2.9e-5). Simplest safe approach: keep rotor masses
small, set `armature` to the reflected rotor inertia (CF rotor spin inertia is tiny, ~1e-7 kg·m² —
**do not** copy the PDF's `1e-5` example value, that was for a larger drone), and **verify with a
hover test** (§8).

### 4.2 Rotor actuation + aerodynamics

**MuJoCo simulates rotor spin, NOT propeller aerodynamics.** Two pieces:

**(a) Actuation — how the policy drives the rotors.** Pick one:

- **Option A — VELOCITY actuator (RECOMMENDED; preserves the hover-centred action semantics).**
  `actuator_mode=JointTargetMode.VELOCITY`, tune `target_kd`, set `effort_limit` to cap torque.
  Map action exactly like today, but onto the rotor DOF target speed:
  `ω_target = ω_hover + action · ω_range` (hover-centred, action=0 → hover). Write via
  `control.joint_target_qd` `[N×4]`. The LPF is **replaced** by `armature` + `target_kd` dynamics —
  tune these to reproduce the documented τ ≈ 0.15 s spin-up lag.
  `ω_hover = CF_HOVER_RPM · 2π/60 ≈ 1516 rad/s`.

- **Option B — EFFORT/torque (PDF's literal default).** `actuator_mode=JointTargetMode.EFFORT`,
  policy outputs motor torque via `control.joint_f` `[N×4]`. More "raw", but hover-centring is
  awkward (hover torque depends on K_D·ω²) and it diverges more from the preserved action space.

> **Default to Option A.** It keeps the `[-1,1]` hover-centred action (matching the preserved
> config) while delivering the physical motor dynamics that motivated this path. Confirm
> empirically; switch to B only if velocity tracking is unsatisfactory.

**(b) Aerodynamic kernel (still required).** A batched Warp kernel, run each substep:
- Read rotor angular velocities `ω_i` `[N×4]` from `state` (via `ArticulationView.get_dof_velocities`
  or directly from `joint_qd`).
- **Unit conversion bug-trap:** `ω` is rad/s; the thrust constants are per-RPM. Either convert
  `n = ω·60/(2π)` then `thrust = K_T·n²`, or precompute `K_T' = K_T·(60/2π)²` and use
  `thrust = K_T'·ω²`. Same for `K_D`.
- Apply to **each rotor body**: axial thrust force (`+z` in rotor frame) + aerodynamic drag torque
  `Q = K_D·n²·(−sign ω)` about the spin axis, via `state.body_f` (indexed by rotor body).

**Why physical rotors simplify the kernel:** because thrust is applied at the rotor bodies'
arm offsets, **MuJoCo produces the roll/pitch moments automatically** — the old kernel's manual
`cross(arm, thrust)` (line 273) is gone. The rotor drag torque transmits through the `REVOLUTE`
joint to give the yaw reaction. Spinning rotor bodies also give **gyroscopic coupling for free**
(the force-kernel approach missed this). The aero kernel reduces to: per rotor, axial thrust +
axial drag torque, both in the rotor frame.

**`body_f` IS consumed by `SolverMuJoCo` (verified):** `apply_mjc_body_f_kernel`
(`newton/_src/solvers/mujoco/kernels.py`) reads `body_f[0:3]` as **force** and `body_f[3:6]` as
**torque** and writes MuJoCo `xfrc_applied` (world-frame wrench at COM). Express the kernel's
output accordingly.

### 4.3 Parallelism (the real throughput win)

- **One template, N worlds**: `builder.replicate(template, world_count=N)` → single `model`,
  single `solver`, double-buffered `state_0/state_1`. **No `DummyVecEnv`. No per-step `.numpy()`.**
- **`ArticulationView(model, pattern="drone*")`** for **all** batched I/O (see §5 API).
- **CUDA-graph capture** of the substep loop with `wp.ScopedCapture`; replay each control step.
  - Write the motor-command buffer (`control.joint_target_qd` or `joint_f`) **before**
    `wp.capture_launch(graph)` — the graph references the buffer, writes are visible on replay.
  - **`SolverMuJoCo` does its own contacts**: pass `contacts=None` to `solver.step(...)`; do **not**
    call `model.collide()` separately (unless `use_mujoco_contacts=False`).
  - **Reset & graph validity:** reset done envs by writing **in-place** into the existing `state_0`
    (masked) and calling `eval_fk` — the graph stays valid. **Reallocating** `state_0/1`/`control`
    invalidates the graph; rebuild it if you do.
- **Control rate vs substeps:** keep **100 Hz** control (paper). Sub-step the solver for stability
  (`sim_substeps`, e.g. 4–8); `sim_dt = (1/100)/sim_substeps`. `implicitfast` is stable under the
  extreme-tilt spawns, so fewer substeps may suffice — tune.

### 4.4 RL: SKRL (PPO + SAC, MLP + GRU)

**Why SKRL (verified):** it ships dedicated RNN agents — `PPO_RNN` **and** `SAC_RNN` (plus
DDPG/TD3/A2C RNN) — so the full **(PPO+SAC) × (MLP+GRU)** matrix comes from one library. RL Games
cannot do recurrent SAC cleanly. SKRL consumes GPU-resident tensors, preserving Newton's
parallelism. (Precedent: a published quadrotor PPO-vs-SAC study built on SKRL at 4096 envs.)

- **Agents:** MLP → `skrl.agents.torch.ppo.PPO`, `skrl.agents.torch.sac.SAC`.
  GRU → `PPO_RNN`, `SAC_RNN`. **Drop TD3** (per decision).
- **Models:** Gaussian policy + deterministic/value critic for PPO; Gaussian policy + twin
  Q-critics for SAC. Head `net_arch=[256, 256]`, `Tanh` (matches current). GRU variants prepend a
  GRU layer feeding the `[256,256]` head; implement `get_specification()` for hidden-state sizes
  per SKRL's RNN model contract.
- **Env wrapper:** a thin SKRL-compatible **vectorized** wrapper around the batched Newton model
  exposing GPU tensors — `reset() → obs[N,22]`, `step(actions[N,4]) → (obs[N,22], reward[N],
  terminated[N], truncated[N], info)`. Match SKRL's wrapped-env interface (it supports
  Isaac-Gym-style and Gymnasium-vectorized contracts). Keep `device="cuda"` end-to-end.

---

## 5. Verified API quick-reference (`ArticulationView`)

From `newton/_src/utils/selection.py` (source-checked). `pattern="drone*"` selects by label.

| Need | Call |
|---|---|
| Drone pose (pos+quat) → `p_err`, `R_flat` | `get_root_transforms(state)` → `[N,1,7]` |
| Drone twist → `v`, `w` (obs) | `get_root_velocities(state)` → `[N,1,6]` |
| Rotor speeds `ω` (aero kernel input) | `get_dof_velocities(state)` → `[N,1,4]` |
| Read motor torques | `get_dof_forces(control)` |
| Write motor torques (EFFORT mode) | `set_dof_forces(control, ...)` |
| Reset: drone pose | `set_root_transforms(state, ...)` (writes `joint_q` for the FREE root) |
| Reset: drone velocity | `set_root_velocities(state, ...)` |
| Reset: rotor speeds | `set_dof_velocities(state, ...)` |
| Generic attr get/set | `get_attribute / set_attribute` |

`JointTargetMode` (`newton/_src/sim/enums.py:213`): `NONE=0`, `POSITION=1`, `VELOCITY=2`,
`POSITION_VELOCITY=3`, `EFFORT=4`. Velocity actuators track `joint_target_qd`; `EFFORT` expects
force via `joint_f`.

---

## 6. Preserved configuration (DO NOT silently change)

These are ported from per-env NumPy into **batched GPU kernels** but must remain numerically
identical.

**Observation (22-D)** — `OBS_DIM = 3 + 9 + 3 + 3 + N_ACTION_HIST*4`, `N_ACTION_HIST=1`:
`[p_err(3), R_flat(9, row-major), v(3, world), w(3, body), a_prev(4)]`.
*Optional (clearly mark if added): append normalized rotor ω (4 dims → 26-D) — now physically
meaningful state. Default keep 22-D for config continuity.*

**Action (4-D, `[-1,1]`, hover-centred):** action=0 → hover. Maps to rotor target speed
(Option A) instead of LPF'd RPM setpoint. **Action space shape and hover-centring are preserved.**

**Reward** (paper Eq. 1 + Table 2; `drone_gym_env.py:232–239, 533–547`):
```
r = −C_rp‖p_err‖² − C_rq(1−q_w²) − C_rv‖v‖² − C_rω‖ω‖² − C_ra‖a − C_rab‖² + C_rs
```
Weights, curriculum-ramped via `c ∈ [0,1]`: `C_rp 2.5→20`, `C_rv 0.005→0.5`, `C_ra 0.005→0.5`;
fixed `C_rq=2.5`, `C_rs=2.0` (survival), `C_rω=0`, `C_rab=0`.

**Curriculum** (`reset`, lines 447–453) — linear `c` over `curriculum_steps` (5M), ramps **both**
reward weights and spawn extremes: `pos_range ±0.15→±1.5 m`, `max_tilt ±15°→±90°`,
`vel_range ±0.1→±1.0`, `ang_range ±0.1→±1.0`, RPM init band. Keep the 10% "spawn-at-target"
guidance branch (line 459). Implement as a per-step scalar broadcast into the batched reset/reward.

**Termination:** `z < 0.05` (ground) or `dist > 2.0`. **Truncation:** `MAX_EPISODE_STEPS = 500`
(5 s @ 100 Hz).

**Physical params (Crazyflie 2.x):** mass 27 g; arm 32.5 mm; Ixx=Iyy=1.657e-5, Izz=2.9e-5;
K_T=3.16e-10 N/RPM², K_D=7.94e-12 N·m/RPM²; max RPM 21702; hover RPM ≈ 14476; CW/CCW per the 4
rotor positions; 100 Hz.

**Spawn/eval target distribution** (keep aligned, `eval_drone.py:11`):
radius ∈ [0.5,1.5] m, altitude ∈ [0.3,1.2] m, angle ∈ [0,2π).

**Algorithm hyperparameters** — preserve via SKRL cfg keys:

| | SB3 (current) | SKRL cfg |
|---|---|---|
| PPO rollout | `n_steps=2048` per env | `rollouts` (steps/env) — set so `rollouts×N` matches intended batch |
| PPO epochs | `n_epochs=10` | `learning_epochs=10` |
| PPO minibatch | `batch_size` | `mini_batches` (so minibatch size ≈ preserved) |
| PPO clip | `clip_range=0.3` | `ratio_clip=0.3` |
| PPO GAE | `gae_lambda=0.95` | `lambda=0.95` |
| PPO entropy | `ent_coef=0.02` (→0.0005 sched) | `entropy_loss_scale=0.02` (replicate the decay) |
| PPO value coef | `vf_coef=0.3` | `value_loss_scale=0.3` |
| PPO grad clip | `max_grad_norm=0.5` | `grad_norm_clip=0.5` |
| SAC buffer | `buffer_size=500_000` | `memory_size` |
| SAC batch | `batch_size=256` | `batch_size=256` |
| SAC target | `tau=0.005` | `polyak=0.005` |
| SAC entropy | `ent_coef` auto / 0.005 | `learn_entropy=True` (auto) |
| both | `gamma=0.99`, `lr=3e-4` | `discount_factor=0.99`, `lr=3e-4` |

> **PPO rollout note:** SB3 `n_steps` is per-env; with N envs the batch is `n_steps×N`. In SKRL set
> `rollouts` (steps/env) and `mini_batches` to preserve the *effective* minibatch size, not the
> literal numbers.

**The one intended change:** `num_envs`. The entire point of the migration is to scale it up
(default **4096**, tunable to GPU memory). "Keep the config" means hyperparameters/reward/curriculum
— **not** the env count.

---

## 7. Migration task list (suggested order)

- **T0 — Clean slate.** Remove `train_drone.py`'s merge conflicts by writing the new SKRL trainer
  fresh; delete/retire the old file. Don't try to salvage the conflicted SB3 version.
- **T1 — Batched model builder.** New module: `register_custom_attributes` → build one drone
  template (`FREE` + 4 `REVOLUTE` rotors, armature/effort_limit/friction, mass/inertia decomposed to
  conserve 27 g) → `builder.replicate(world_count=N)` → `finalize()` →
  `SolverMuJoCo(model, integrator="implicitfast", solver="newton", cone="elliptic")`. Initial
  `eval_fk`.
- **T2 — Aerodynamic kernel (batched).** Read rotor ω `[N×4]`, apply axial thrust + drag torque to
  rotor bodies via `body_f` (correct rad/s→RPM conversion; world-frame wrench at COM).
- **T3 — Actuation.** Option A velocity actuators: action `[N,4]` → `joint_target_qd`
  (hover-centred), tune `target_kd`+`armature` to ≈0.15 s lag.
- **T4 — Batched obs + reward + termination kernels.** Assemble 22-D obs from
  `get_root_transforms`/`get_root_velocities` + action buffer; port reward (Eq.1) and
  termination/truncation to GPU; broadcast curriculum scalar.
- **T5 — Masked batched reset.** Curriculum-driven random pose/vel/rotor-speed via
  `set_root_transforms`/`set_root_velocities`/`set_dof_velocities`, masked in-place, then `eval_fk`.
- **T6 — CUDA-graph substep loop.** Capture once; replay per control step; write command buffer
  before replay; `contacts=None`.
- **T7 — SKRL env wrapper.** Vectorized GPU-tensor env matching SKRL's wrapped-env API.
- **T8 — SKRL trainer.** PPO + SAC; MLP + GRU models; map hyperparameters (§6); curriculum +
  entropy-decay callbacks; TensorBoard logging mirroring current metrics
  (`success_rate` = terminal `dist<0.15`, `terminal_dist`, `ep_reward`, reward components).
  CLI: `--algo {ppo,sac}`, `--policy {mlp,gru}`, `--seed`, `--num_envs`, `--total_timesteps`.
- **T9 — Eval.** Adapt `eval_drone.py` to load the SKRL checkpoint and run the same multi-waypoint
  rollout against the batched env (use 1 world for eval/render).

---

## 8. Verification checklist & gotchas

- **Hover sanity test (do FIRST after T1–T3).** Command hover action (=0) on one world; the drone
  must hold altitude. This catches: (a) the `body_f` spatial convention under `SolverMuJoCo`
  (force in `[0:3]`, torque in `[3:6]`), (b) the rad/s→RPM thrust conversion, (c) mass/inertia
  decomposition errors, (d) `armature`/`target_kd` lag tuning. If it sinks/rockets, fix here before
  anything else.
- **`register_custom_attributes` ordering** — must precede adding bodies/joints, or armature/
  effort_limit/friction silently do nothing.
- **`eval_fk` after every reset** — without it, randomized `joint_q` won't reach `body_q`;
  collision/render/obs read stale poses.
- **CUDA-graph reset rule** — masked in-place resets keep the graph valid; reallocation invalidates
  it. Don't reallocate `state`/`control` mid-training.
- **`contacts=None`** for `SolverMuJoCo.step` — it runs its own contact generation.
- **Yaw balance** — verify the CW/CCW rotor drag-torque signs net to zero at hover (no spurious
  spin). Wrong signs are a classic quad bug.
- **Gyroscopic coupling now active** — behavior under aggressive maneuvers will differ from the old
  force-kernel sim; this is expected/more accurate, not a regression.
- **No GPU↔CPU sync in the hot loop** — if you see `.numpy()` inside `step()`, it's wrong. All
  per-step data stays as GPU tensors via `ArticulationView`.
- **SB3 `.zip` checkpoints are not portable to SKRL** — don't attempt to load them.

---

## 9. Non-goals / do-not

- **Do not** reintroduce `DummyVecEnv` or any per-env Python stepping loop.
- **Do not** keep `SolverSemiImplicit` for the position controller.
- **Do not** add `.numpy()` syncs inside the per-step path.
- **Do not** change the reward weights, curriculum schedule, observation layout, action shape, or
  physical parameters (§6) without flagging it explicitly.
- **Do not** touch `landing/` or `disturbance/` in this pass (out of scope).
- **Do not** add TD3 to the new trainer (PPO + SAC only).