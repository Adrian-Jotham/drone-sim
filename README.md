# Drone RL — End-to-End Quadrotor Navigation with PPO

A reinforcement learning system for training a Crazyflie 2.x quadrotor position
controller entirely inside the [Newton](https://github.com/newton-physics/newton)
GPU-native rigid-body simulator using **Proximal Policy Optimization (PPO)**.

The environment models a real **Crazyflie 2.x** (27 g nano-quadrotor) with correct
mass, inertia, arm length, and **Level-5.1 direct RPM control** — the lowest-level,
most physically accurate action abstraction, exposing the full nonlinear thrust curve,
motor lag, and inertia tensor to the policy.

Physics baseline: Eschmann et al., "Learning to Fly in Seconds", RAL 2024 —
system-identified Crazyflie parameters, Level-5.1 taxonomy, motor LPF model.

---

## Repository Layout

```
dronesim/
├── drone_gym_env.py          # Gymnasium environment (physics + obs + reward)
├── train_drone.py            # Unified training script: PPO | SAC | TD3
├── eval_trajectory.py        # Trajectory evaluation — waypoint chaining
├── eval_drone.py             # Legacy per-episode evaluation (4-waypoint fixed)
├── LearningtoFlyinSeconds.pdf # Reference paper (Eschmann 2024)
├── landing/                  # Separate landing task environment
│   ├── drone_landing_env.py
│   ├── train_landing.py
│   └── eval_landing.py
├── disturbance/              # Hover task with random force/torque disturbances
│   ├── quadrotor_hover_env.py
│   ├── train_hover.py
│   └── eval_hover.py
└── drone_logs/               # TensorBoard logs (created at first training run)
    ├── ppo/
    ├── sac/
    └── td3/
```

---

## Quick Start

```bash
# Train PPO — primary algorithm for this project
python train_drone.py --algo ppo --headless --seed 0

# Long run (40 M steps, overnight)
python train_drone.py --algo ppo --headless --seed 1 \
  --total_timesteps 40_000_000 \
  --curriculum_steps 1_500_000

# Resume from a checkpoint
python train_drone.py --algo ppo --headless \
  --resume checkpoints/ppo_s1_3000000_steps \
  --total_timesteps 40_000_000

# Monitor training
tensorboard --logdir drone_logs

# Evaluate — waypoint chaining (primary evaluation)
python eval_trajectory.py --model ppo_drone_final_s1 --algo ppo

# Clean start (curriculum=0, upright stationary spawn)
python eval_trajectory.py --model ppo_drone_final_s1 --algo ppo --clean_start

# Square / Lissajous trajectories
python eval_trajectory.py --model ppo_drone_final_s1 --algo ppo --traj square
python eval_trajectory.py --model ppo_drone_final_s1 --algo ppo --traj lissajous
```

---

## System Architecture

```
                 ┌──────────────────────────────────────────────────────┐
                 │                DroneEnv (drone_gym_env.py)            │
                 │                                                       │
  action (4D) ──►│  RPM mapping  ──►  Motor LPF  ──►  Newton Physics   │──► obs (22D)
  [-1, 1]        │  hover-centred     τ = 0.15 s     100 Hz rigid-body  │──► reward
                 │  Level-5.1 RPM     n[t+1]=…        Crazyflie 27 g   │
                 └──────────────────────────────────────────────────────┘
                          ▲                                  │
                          │    RL Policy (PPO / SAC / TD3)   │
                          └──────────────────────────────────┘
```

The environment wraps Newton's GPU-accelerated rigid-body simulator as a standard
Gymnasium interface. The RL policy runs on the CPU side and sends RPM setpoints;
Newton propagates physics at 100 Hz using the real Crazyflie physical parameters and
returns the next state.

---

## Physics Simulation

### Crazyflie 2.x Parameters

All physical constants match the real Crazyflie 2.x nano-quadrotor (Förster 2015
system identification + Bitcraze documentation):

| Parameter | Value | Source |
|-----------|-------|--------|
| Total mass | 27 g (0.027 kg) | Bitcraze spec |
| Arm length (centre→motor) | 32.5 mm (0.0325 m) | Bitcraze spec |
| Ixx = Iyy | 1.657 × 10⁻⁵ kg·m² | Förster 2015 |
| Izz | 2.900 × 10⁻⁵ kg·m² | Förster 2015 |
| Thrust constant KT | 3.16 × 10⁻¹⁰ N/RPM² | System ID |
| Drag-torque constant KD | 7.94 × 10⁻¹² N·m/RPM² | System ID |
| Max motor speed | 21 702 RPM | Bitcraze spec |
| Idle motor speed | 1 000 RPM | ESC minimum |
| Hover RPM (each motor) | ≈ 14 476 RPM | derived: mg/4 = KT·n² |
| Thrust-to-weight ratio | 2.25 | at 21 702 RPM |

Mass and inertia are set **directly** on the Newton body (not derived from geometry),
ensuring they match system-identification values exactly regardless of the collision
shape used.

### Drone Body

The quadrotor body is modelled as two thin cross-arm boxes for collision detection.
The collision geometry is scaled proportionally to the real Crazyflie arm length
(32.5 mm); density=0 is assigned so the geometry contributes no additional
inertia beyond the explicitly specified tensor.

```
      prop[0]                prop[2]
        ↑                       ↑
  (0, +0.0325, 0)       (+0.0325, 0, 0)
          \                   /
           ●─────────────────●    ← body (cross, density=0, inertia from Förster 2015)
          /                   \
  (0, -0.0325, 0)       (-0.0325, 0, 0)
        ↓                       ↓
      prop[1]                prop[3]
```

Propellers alternate turning direction (−1, +1, +1, −1) to cancel net yaw torque
at hover.

### Propeller Aerodynamics — Level 5.1 (Direct RPM)

This environment operates at **Level 5.1 — Motor commands/RPM setpoints**, the
lowest and most physically accurate actuation level. Each propeller's thrust and
reaction torque are computed from the filtered motor speed using the quadratic RPM²
model:

```
F_i = KT · n_i²          (thrust, N)
Q_i = KD · n_i²          (drag torque, N·m)
```

where `n_i` is the filtered motor speed in RPM. The full wrench applied to the body is:

```
force_i  = R_body · ẑ · KT · n_i²
torque_i = R_body · ẑ · KD · n_i² · turning_dir_i  +  r_i × force_i
```

with `r_i` the moment arm from the body centre of mass to propeller `i`.

**Why Level 5.1 matters:** Higher abstraction levels (e.g. angular rate commands)
hide the nonlinear RPM²→thrust relationship, motor delay, and rotor speed dynamics.
Direct RPM control forces the policy to model all of them, enabling potential
zero-shot sim-to-real transfer.

### Motor Dynamics (First-Order Low-Pass Filter)

Real brushless motors do not respond instantaneously to commands. The Crazyflie ESC
exhibits τ ≈ 0.15 s response time. This is implemented as a discrete first-order IIR
filter on motor RPM at every step:

```
α = SIM_DT / τ = 0.01 / 0.15 ≈ 0.067

n[t+1] = (1 − α) · n[t] + α · n_sp[t]
```

The filter is applied in **RPM space** (not thrust fraction space), so the full
nonlinear RPM²→thrust mapping is preserved. The drone takes approximately 15 steps
(≈ 0.15 s) to reach a new commanded RPM setpoint.

The policy compensates for this lag using the **action history** in the observation.

### Physics Integrator

Newton's `SolverSemiImplicit` advances the rigid-body dynamics at 100 Hz. All four
propeller wrenches are accumulated into `body_f` by a single Warp GPU kernel before
each solver step. A ground plane provides collision detection.

---

## Observation Space (22-D)

| Slice | Symbol | Dim | Description |
|-------|--------|-----|-------------|
| `[0:3]` | `p_err` | 3 | Position error = `pos − target` (world frame, m) |
| `[3:12]` | `R_flat` | 9 | Drone rotation matrix, row-major flattened |
| `[12:15]` | `v` | 3 | Linear velocity, world frame (m/s) |
| `[15:18]` | `ω` | 3 | Angular velocity, body frame (rad/s) |
| `[18:22]` | `a_prev` | 4 | Previous normalised action (action history N_H=1) |

**Total: 22 dimensions.**

### Why position error, not absolute position?

The policy sees `p_err = pos − target`. The network always "thinks" it is flying to
the origin regardless of where the actual target is. At inference you can move the
target anywhere by calling `env.set_target(new_pos)` — the same trained policy
handles it without retraining.

### Why rotation matrix, not quaternion?

A unit quaternion `q` and `−q` represent the same physical rotation (double-coverage
of SO(3)). A neural network fed raw quaternions must implicitly learn to ignore this
ambiguity. The 3×3 rotation matrix has no such ambiguity.

### Why action history?

The motor LPF introduces ~15 steps of lag. Without knowing what was commanded
recently, the policy cannot predict the drone's near-future response. The previous
action provides a window into the current motor state, partially restoring
observability of the delayed RPM.

### Optional observation noise

When training with `--obs_noise`, Gaussian noise simulates imperfect onboard sensors:

| Component | Noise σ |
|-----------|---------|
| `p_err` | 0.01 m |
| `v` | 0.01 m/s |
| `ω` | 0.05 rad/s |

---

## Action Space (4-D) — Level 5.1 RPM

Each action component is a **normalised RPM setpoint** in `[−1, 1]`.
The hover-centred linear map converts to actual RPM:

```
n_sp_i = clip(CF_HOVER_RPM + action_i × CF_RPM_RANGE,  CF_MIN_RPM,  CF_MAX_RPM)

CF_HOVER_RPM = 14 476 RPM   → action = 0   (stable hover)
CF_RPM_RANGE =  7 226 RPM   → action = ±1  (idle or full throttle)
CF_MIN_RPM   =  1 000 RPM
CF_MAX_RPM   = 21 702 RPM
```

Example mapping:

| action | RPM | Thrust/motor |
|--------|-----|-------------|
| −1.0 | 7 250 | 1.7 g |
| −0.5 | 10 863 | 3.8 g |
| 0.0 | 14 476 | 6.7 g ← hover |
| +0.5 | 18 089 | 10.5 g |
| +1.0 | 21 702 | 15.2 g |

The hover point (action = 0) is set at the exact RPM where thrust equals weight:
`4 × KT × n² = m·g`. Thrust is **nonlinear** in action because `F ∝ n² ∝ (hover + action·range)²`.

The RPM setpoint feeds the motor LPF filter described above.

---

## Reward Function

The reward at every step is the sum of nine components:

```
r = pos_c + orient_c + vel_c + ang_c + act_c + survival + crash + arrival + approach + hover_bonus
```

```python
# drone_gym_env.py — step() method
pos_c    = -C_rp  * ‖p_err‖²
orient_c = -C_rq  * (1 − qw²)
vel_c    = -C_rv  * ‖v‖²
ang_c    = -C_rω  * ‖ω‖²
act_c    = -C_ra  * ‖Δa‖²
survival = +0.50                             # every step (fixed)
crash    = -2.00  if z < 0.05 m             # one-time ground impact
arrival  = +15.00 first time dist < 0.15 m  # one-time per waypoint
approach = _APPROACH_COEF × (last_dist − dist)  if dist > 0.10 m  else 0.0
hover_bonus = +0.15  if dist < 0.15 m AND speed < 0.5 m/s  else 0.0
```

---

### pos_c — Position cost

**Code:** `pos_c = -C_rp * float(np.dot(p_err, p_err))` — curriculum ramp C_rp: 0.05 → 1.00

The cost grows quadratically with distance. At full curriculum a drone 1 m away pays
`−1.0/step`, which exceeds the survival bonus (`+0.50`), forcing the policy to close
the gap. At curriculum = 0 the same error costs only `−0.05/step` so early episodes
are dominated by the survival bonus, keeping the drone airborne while it learns.

**TensorBoard — `reward_components/pos_c`:** A **negative value rising toward 0** as
training progresses. If it stays at −0.3 or below after 2 M steps the drone is
consistently far from the target.

---

### orient_c — Orientation cost

**Code:** `orient_c = -_C_RQ * float(1.0 - qw**2)` — `_C_RQ = 0.10`, fixed

`qw = quaternion_w = cos(θ/2)` where θ is the tilt angle from upright.

| Orientation | qw | orient_c |
|-------------|-----|----------|
| Perfectly upright | 1.0 | 0.00 |
| 45° tilted | 0.924 | −0.015 |
| 90° tilted | 0.707 | −0.050 |
| Inverted | 0.0 | −0.100 |

**TensorBoard — `reward_components/orient_c`:** Healthy: **−0.005 to −0.02**.
Values below −0.05 mean the drone is spending significant time tilted.

---

### vel_c — Linear velocity cost

**Code:** `vel_c = -C_rv * float(np.dot(v, v))` — curriculum ramp C_rv: 0.01 → 0.30

Encourages the drone to slow down and hover. At full curriculum a 1 m/s speed costs
`−0.30/step`. This term is in deliberate tension with `approach` (which rewards moving
toward the target) — a good policy balances them by approaching quickly then braking.

**TensorBoard — `reward_components/vel_c`:** Should show **mild negative values during
navigation** and **near zero when settled**. Large negative spikes that never decay
indicate oscillation.

---

### ang_c — Angular velocity cost

**Code:** `ang_c = -C_rw * float(np.dot(w, w))` — curriculum ramp C_rw: 0.001 → 0.05

Discourages spinning and wobbling. Healthy values: **−0.01 to −0.03** once converged.

---

### act_c — Action-change cost (jerk regularisation)

**Code:** `act_c = -C_ra * float(np.dot(delta_a, delta_a))` — curriculum ramp C_ra: 0.005 → 0.02

`delta_a = action − prev_action` — penalises RPM **changes** (jerk), not magnitude.
A sustained high-RPM climb is not penalised; rapid oscillation between high and low
RPM every step is. Healthy: **−0.02 to −0.06**. Below −0.10 = policy is thrashing.

---

### survival — Constant per-step bonus

**Code:** `survival = _C_RS = 0.50`, fixed constant

A constant `+0.50` every step the drone is alive. Makes staying airborne intrinsically
rewarding throughout training. At curriculum = 0 this dominates, letting early policies
learn to hover before navigating. At curriculum = 1, a drone 1 m from target earns
`+0.50 − 1.00 = −0.50/step`, forcing navigation.

**Note:** The flat survival constant creates a hover attractor — the drone can always
earn positive per-step reward by hovering in place. This is an identified training
limitation; the approach and hover_bonus terms counteract it.

---

### crash — Ground impact penalty

**Code:** `crash = -2.0 if z < 0.05 else 0.0`

A one-time `−2.0` on the step where the drone hits the ground (altitude < 5 cm).
The episode terminates immediately after this step.

---

### arrival — One-time waypoint bonus

**Code:** `arrival = 15.0` the first time `dist < 0.15 m` (per waypoint)

A `+15.0` bonus the first time the drone comes within 0.15 m of the current waypoint.
The flag `self._arrived` prevents repeated claiming on the same target. When
`multi_target=True`, after claiming arrival a new random target is assigned.

**Why 15.0:** Over an 800-step episode the survival bonus accumulates +400 total.
A +1.0 arrival signal is only 0.25% of episode reward — too small for PPO to reliably
credit the approach actions that led to it. At +15.0 the arrival is equivalent to
30 steps of survival bonus, making target-reaching clearly the most valuable single
event per episode. The arrival threshold (0.15 m) aligns with the evaluation success
criterion so training and evaluation use the same gate.

---

### approach — Potential-based distance shaping

**Code:** `drone_gym_env.py`
```python
_APPROACH_COEF = 1.0
_APPROACH_GATE = 0.10  # m — shaping turns off inside this radius

approach = (
    _APPROACH_COEF * (self.last_dist - dist)
    if dist > _APPROACH_GATE else 0.0
)
```

`approach` is positive when the drone gets closer to the target in one step, and
negative when it retreats. It is **gated off when the drone is already within 0.10 m**
of the target.

**Why potential-based shaping:** The quadratic `pos_c` provides a weak gradient far
from the target. The approach term is mathematically equivalent to `Φ(s') − Φ(s)`
with `Φ(s) = −dist`, a reward-shaping-theorem compliant transformation that provides
a **dense directional gradient** toward the target without changing the optimal policy.

**Why gated at 0.10 m:** Without the gate, the drone earns approach reward for any
motion toward the target — even a tiny oscillation near it. With the gate, inside
0.10 m the approach signal is zero and the quadratic `pos_c` (−C_rp × 0.01 at 0.10 m)
combined with `hover_bonus` becomes the only signal, rewarding settlement.

The gate radius (0.10 m) is set inside the success threshold (0.15 m), so approach
reward is still active from 0.15 m down to 0.10 m, providing gradient right up to the
success boundary.

**TensorBoard — `reward_components/approach`:** Converges toward **0** as the drone
learns to hover near the target. A well-tuned policy shows `approach ≈ 0` at
convergence.

---

### hover_bonus — Dense settlement reward

**Code:** `drone_gym_env.py`
```python
_HOVER_BONUS      = 0.15   # per-step reward for settling at target
_HOVER_SPEED_GATE = 0.5    # m/s — must be nearly stopped to earn it

speed = float(np.linalg.norm(v))
hover_bonus = _HOVER_BONUS if (dist < 0.15 and speed < _HOVER_SPEED_GATE) else 0.0
```

A `+0.15/step` bonus when the drone is within 0.15 m of the target **and moving
slower than 0.5 m/s**. Both conditions must hold simultaneously.

**Why the speed gate is critical:** A pure distance gate (`if dist < 0.15`) creates
a discontinuous reward cliff — the policy earns +0.15/step the instant it crosses
0.15 m regardless of how fast it is moving. This teaches the policy to *rush* at the
target: approaching at 3 m/s and entering the zone for one step earns approach reward
(fast closing) plus a hover_bonus step. The result is bimodal behaviour — some episodes
successfully braking, others overshooting and crashing.

The `_HOVER_SPEED_GATE` removes this incentive: you only earn the bonus when
*already slow*, so the optimal strategy becomes approach → decelerate → settle.

**TensorBoard — `reward_components/hover_bonus`:** Should **rise from 0 toward a
positive value** throughout training. Unlike `approach` (which decays to 0), hover_bonus
growing is a healthy sign that the drone is spending increasing time settled at targets.

---

### Reward Curriculum

Weights ramp linearly from conservative → strict over the **first 50% of training**
(i.e., over `curriculum_steps` absolute steps):

| Weight | Init | Target | Controls |
|--------|------|--------|----------|
| `C_rp` | 0.05 | 1.00 | Position precision |
| `C_rv` | 0.010 | 0.30 | Velocity damping |
| `C_rω` | 0.001 | 0.05 | Angular stability |
| `C_ra` | 0.005 | 0.02 | Action smoothness |
| `C_rq` | 0.10 | 0.10 | Orientation (fixed) |
| `C_rs` | 0.50 | 0.50 | Survival (fixed) |

**Why curriculum matters:** At `curriculum = 0` the position cost is tiny and the
survival bonus dominates — the drone gets positive reward just by staying airborne.
At `curriculum = 1` the position penalty grows 20×, forcing tight navigation. Without
curriculum the heavy early position penalty makes crashing at step 1 "optimal" for
the initial random policy.

```python
# train_drone.py — CurriculumCallback._on_step()
t = min(num_timesteps / curriculum_steps, 1.0)
# curriculum_steps default: 1_500_000 (fixed absolute count — see --curriculum_steps)
```

Disable with `--no_curriculum` for ablation experiments.

---

## Termination Conditions

| Condition | Reason |
|-----------|--------|
| `z < 0.05 m` | Ground impact |
| `z > 6.0 m` | Escaped upward |
| `R22 < −0.5` | Drone more than ~120° inverted — unrecoverable |
| `dist > 4.0 m` | Out of arena |

Episodes also truncate after `MAX_EPISODE_STEPS = 800` steps (8.0 s at 100 Hz).

---

## Reinforcement Learning Algorithms

Three algorithms are supported. **PPO is the primary algorithm for this project.**

### PPO — Proximal Policy Optimisation (`stable_baselines3`)

**Type:** On-policy
**Best for:** Stable long-run training with reproducible convergence.

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_steps` | 2048 | Steps per env per rollout; 2048 × 16 envs = 32,768 transitions/rollout. Exceeds MAX_EPISODE_STEPS=800 so complete episodes always fit in one rollout — no GAE truncation mid-episode. |
| `batch_size` | 64 | Mini-batch size; 512 mini-batches per rollout |
| `n_epochs` | 10 | Gradient passes per rollout |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE bias-variance trade-off |
| `clip_range` | 0.2 | PPO trust-region clip (ratio clamped to [0.8, 1.2]) |
| `ent_coef` | 0.005 | Entropy bonus; keeps exploration from collapsing |
| `vf_coef` | 0.5 | Value function loss weight |
| `max_grad_norm` | 0.5 | Gradient clipping for stability |
| `net_arch` | [256, 256] | Two hidden layers, tanh activations |
| `learning_rate` | 3e-4 | Adam learning rate (constant unless `--lr_final` set) |

### SAC — Soft Actor-Critic (`sbx`, JAX)

**Type:** Off-policy, entropy-regularised

| Hyperparameter | Value |
|---|---|
| `buffer_size` | 500 000 |
| `batch_size` | 256 |
| `learning_starts` | 0 |
| `tau` | 0.005 |
| `ent_coef` | 0.005 (fixed, not auto) |
| `net_arch` | [256, 256] |

**Why fixed `ent_coef` for SAC:** SAC's `ent_coef="auto"` targets `H = −dim(action) = −4`
nats, forcing σ ≈ 1.0 per dimension = ±7 200 RPM noise. The drone thrashes between
full throttle and motor stall. Using `ent_coef=0.005` (fixed) keeps entropy as a mild
regulariser without overwhelming the task reward.

### TD3 — Twin Delayed Deep Deterministic (`sbx`, JAX)

**Type:** Off-policy, deterministic

| Hyperparameter | Value |
|---|---|
| `buffer_size` | 500 000 |
| `batch_size` | 256 |
| `learning_starts` | 0 |
| `tau` | 0.005 |
| `policy_delay` | 2 |
| `target_policy_noise` | 0.2 |
| `target_noise_clip` | 0.5 |
| `action_noise σ` | 0.10 → 0.02 (decayed by CurriculumCallback) |
| `net_arch` | [256, 256] |

**Why `learning_starts = 0` for TD3:** With `learning_starts=10_000`, SBX fills the
buffer with uniform random actions in [−1, 1] during warm-up. At Level-5.1 RPM
control that means random RPMs from 7 250 to 21 702 — the drone crashes almost every
episode. Q-networks then learn that all (state, action) pairs lead to crashes, making
the policy pessimistic (outputs below-hover RPMs → sinks).

With `learning_starts=0`, a randomly-initialised MLP with tanh outputs ≈ 0, which
maps to ≈ hover RPM. The drone stays airborne from episode 1, the buffer fills with
useful hovering experience, and Q-values remain optimistic. Action noise σ=0.10
adds ±723 RPM variation — enough to explore without causing immediate crashes.

### Algorithm Comparison

```
Primary algorithm: PPO (this project)
Sample efficiency:  SAC ≈ TD3 >> PPO
Stability:          PPO > TD3 ≈ SAC (at Level 5.1 with this reward shaping)
Wall-clock (GPU):   SBX (SAC/TD3) >> SB3 (PPO)  — JAX vs PyTorch
```

---

## Training

### Basic usage

```bash
# PPO — 3 M steps (baseline)
python train_drone.py --algo ppo --headless --seed 0

# PPO — 40 M steps (converged policy)
python train_drone.py --algo ppo --headless --seed 1 \
  --total_timesteps 40_000_000 \
  --curriculum_steps 1_500_000

# With sensor noise
python train_drone.py --algo ppo --headless --obs_noise
```

### All training arguments

| Argument | Default | Description |
|---|---|---|
| `--algo` | `td3` | Algorithm: `ppo`, `sac`, or `td3` |
| `--seed` | `0` | Global random seed. Each seed produces an independent run. Run name encoded as `{algo}_s{seed}`. |
| `--num_envs` | `16` | Parallel training environments (DummyVecEnv — sequential, not threaded) |
| `--total_timesteps` | `3 000 000` | Total environment steps |
| `--curriculum_steps` | `1 500 000` | Steps over which curriculum ramps 0→1. Fixed absolute count, decoupled from `--total_timesteps`. |
| `--target_success` | `1.0` | Early-stop when rolling 100-episode success rate exceeds this. `1.0` = never stop early. |
| `--lr_final` | `None` | If set, linearly decay LR from `--learning_rate` to this value. E.g., `--lr_final 1e-5` for long runs. |
| `--checkpoint_freq` | `500 000` | Save checkpoint every N global steps |
| `--checkpoint_dir` | `checkpoints` | Directory for checkpoints |
| `--resume` | `None` | Path to a checkpoint `.zip` to resume training from. |
| `--learning_rate` | `3e-4` | Initial Adam learning rate |
| `--gamma` | `0.99` | Discount factor |
| `--render_freq` | `5 000` | Render update frequency (global steps) |
| `--obs_noise` | off | Add Gaussian sensor noise to observations |
| `--multi_target` | off | When the drone reaches a waypoint, immediately assign a new random one without physics reset. Forces the policy to learn repeated target-reaching within one episode. |
| `--no_curriculum` | off | Disable reward curriculum (ablation: degrades reliability) |
| `--headless` | off | No OpenGL viewer |

### Run naming and output files

Every run is named `{algo}_s{seed}`. TensorBoard logs go to `drone_logs/{algo}/{algo}_s{seed}_<timestamp>/`.

```
checkpoints/ppo_s1_500000_steps.zip
checkpoints/ppo_s1_1000000_steps.zip
...
ppo_drone_final_s1.zip           ← final model
```

Checkpoints save every `--checkpoint_freq` steps (default 500 000 global steps).
The final model file includes the seed in its name for unambiguous identification.

### Resuming from a checkpoint

```bash
# Resume from the 3 M step checkpoint and continue to 40 M
python train_drone.py --algo ppo --headless \
  --resume checkpoints/ppo_s1_3000000_steps \
  --total_timesteps 40_000_000 \
  --curriculum_steps 1_500_000
```

`--resume` loads the policy weights, optimizer state, and step counter from the
checkpoint zip. With `reset_num_timesteps=False` (set automatically), the curriculum,
LR schedule, and early-stop window all continue correctly from the saved step count.

| What | Effect at resume |
|---|---|
| **Curriculum** | If `num_timesteps ≥ curriculum_steps` at load, curriculum stays at 1.0 immediately |
| **LR schedule** | `progress_remaining = 1 − done/total`. LR is correct fraction of the way through the decay. |
| **Early stop** | Rolling success window resets — stale pre-resume data does not trigger early stop |

**Replay buffer note:** SAC and TD3 checkpoints do **not** include the replay buffer.
Off-policy methods retrain with an empty buffer for the first ~50 k steps after resume.
PPO is unaffected (no replay buffer).

### Spawn Randomisation

The environment randomises the drone's initial state at every episode reset. All ranges
grow linearly with the curriculum:

| Dimension | Range at c=0 | Range at c=1 | Why |
|-----------|-------------|-------------|-----|
| **Position (XY, Z)** | ±0.1 m from target | ±0.7 m from target | `pos_range = 0.1 + 0.6 × curriculum` |
| **Orientation (roll/pitch)** | 0° | ±15° | Forces policy to learn attitude stabilisation concurrent with navigation |
| **Linear velocity** | 0 m/s | ±1.5 m/s (all axes) | Teaches braking and settling from arbitrary initial velocity |
| **Angular velocity** | 0 rad/s | ±0.5 rad/s (body frame) | Forces the policy to damp oscillations it did not cause itself |

No yaw randomisation — yaw is always initialised to identity.

At c = 0 (clean start) the drone spawns stationary and nearly upright within 0.1 m
of the waypoint. At c = 1.0 (full randomisation) the drone may start with significant
velocity, tilt, and angular rate.

The spawn cap at ±0.7 m ensures the survival bonus (+0.5/step) always exceeds the
position penalty (−C_rp × 0.49 = −0.49/step at full curriculum), keeping episode
returns positive and giving PPO a learnable gradient from the first update.

### Exploration Noise Decay (TD3)

`CurriculumCallback` decays the TD3 action noise σ in lockstep with the reward
curriculum:

```
σ(t) = 0.10  →  0.02   over first curriculum_steps
```

High initial noise allows exploration before reward weights tighten; low final noise
allows precise RPM control once the curriculum is fully ramped.

### TensorBoard monitoring

```bash
tensorboard --logdir drone_logs
```

| Group | Signals |
|---|---|
| `metrics/` | `ep_length`, `ep_reward`, `success_rate`, `terminal_dist`, `terminal_upright`, `mean_motor_rpm`, `rpm_hover_dev` |
| `reward_components/` | `pos_c`, `orient_c`, `vel_c`, `ang_c`, `act_c`, `survival`, `approach`, `hover_bonus` |

---

## PPO Training — How It Works

### Stage 1 — Environment factory: 16 parallel worlds

```python
# train_drone.py
def _make_env(rank: int):
    def _init():
        env = DroneEnv(
            render_mode=None, viewer=None,
            random_targets=True,
            multi_target=args.multi_target,
            obs_noise=args.obs_noise,
            curriculum=0.0,                  # starts easy; CurriculumCallback ramps this
        )
        env.reset(seed=args.seed + rank)     # each env gets an independent offset seed
        return env
    return _init

train_env = DummyVecEnv([_make_env(i) for i in range(args.num_envs)])  # default: 16
```

`DummyVecEnv` creates **16 independent copies** running sequentially in a single Python
process. Each has its own Newton physics state, Crazyflie body, random target, and
curriculum counter.

Why 16 envs? PPO is **on-policy**: it collects a batch of fresh experience, updates
the policy, then discards that experience. More parallel environments means more diverse
experience per rollout — 16 drones explore different starting positions and targets
simultaneously, giving the policy gradient a broader estimate at each update.

---

### Stage 2 — Building the PPO model

```python
# train_drone.py
model = PPO(
    "MlpPolicy", train_env,
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.005,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    policy_kwargs = dict(net_arch=[256, 256]),
)
```

#### The network architecture

`MlpPolicy` with `net_arch=[256, 256]` creates a shared-trunk MLP:

```
obs (22D)
    │
    ▼
Linear(22 → 256) + Tanh
    │
    ▼
Linear(256 → 256) + Tanh
    │
   ┌┴──────────────┐
   ▼               ▼
Actor head      Critic head
Linear(256→4)   Linear(256→1)
Tanh            (no activation)
   │               │
   ▼               ▼
action mean (4D)  V(s)  ← scalar state value
```

The **actor** outputs the mean of a 4-dimensional Gaussian. PPO samples an action from
this distribution during rollout collection and records the log-probability for the
policy gradient. The **log standard deviation** is a separate learned parameter that
decays during training as the policy becomes more confident.

The **critic** outputs a single scalar `V(s)` — the estimated sum of future discounted
rewards from state `s`. Used only during training to compute advantage estimates.

#### Why n_steps = 2048 is the right choice

`n_steps` must exceed `MAX_EPISODE_STEPS = 800`. If `n_steps < 800`, some rollouts
will end mid-episode, causing GAE to truncate returns at an arbitrary step rather than
at a natural episode boundary. This biases the advantage estimates:

- The truncated episode gets a bootstrapped value estimate in place of actual future reward
- The policy receives misleading credit for mid-episode actions near the truncation boundary
- Empirically: `n_steps = 512` (well below 800) yields ~0% success at 3M steps

At `n_steps = 2048`, complete episodes always fit within one rollout:
- 2048 steps / 800 max episode = room for ~2.5 complete episodes per env per rollout
- 16 envs × 2048 = **32,768 total transitions per rollout update**

---

### Stage 3 — The PPO training loop

`model.learn(total_timesteps=N)` repeats the following cycle:

```
┌─────────────────────────────────────────────────────────────────┐
│  ROLLOUT COLLECTION  (n_steps × num_envs = 32 768 steps total)  │
│                                                                  │
│  for step in range(2048):                                        │
│      action, log_prob, value = policy(obs)   ← forward pass     │
│      obs, reward, done, info = env.step(action)  ← 16 envs      │
│      store (obs, action, reward, done, value, log_prob)          │
│      fire callbacks (CurriculumCallback, MetricsCallback, etc.)  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  ADVANTAGE ESTIMATION (GAE)                                      │
│                                                                  │
│  For each step t (backwards):                                    │
│      δ_t = r_t + γ·V(s_{t+1}) − V(s_t)    ← TD error           │
│      A_t = δ_t + γ·λ·A_{t+1}               ← GAE recursion      │
│  returns_t = A_t + V(s_t)                   ← value target       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  POLICY UPDATE  (n_epochs × mini-batches)                        │
│                                                                  │
│  for epoch in range(10):                                         │
│      shuffle all 32 768 transitions                              │
│      for mini_batch in chunks(32768, size=64):                   │
│          r = π_θ(a|s) / π_old(a|s)          ← prob ratio        │
│          L_clip = −mean(min(r·A, clip(r,0.8,1.2)·A))            │
│          L_vf   = 0.5 × (V(s) − returns)²   ← value loss        │
│          L_ent  = −0.005 × H[π]              ← entropy bonus     │
│          loss   = L_clip + L_vf − L_ent                          │
│          loss.backward(); clip_grad; optimizer.step()            │
└─────────────────────────────────────────────────────────────────┘
```

---

### Stage 4 — Callbacks

Four callbacks fire on every environment step:

```python
callbacks = [
    CheckpointCallback(
        save_freq    = max(args.checkpoint_freq // args.num_envs, 1),
        save_path    = args.checkpoint_dir,
        name_prefix  = run_name,  # e.g. "ppo_s1"
    ),
    RenderCallback(viewer=viewer, train_vec_env=train_env, render_freq=5_000),
    MetricsCallback(log_freq=1_000, window=100, target_success=args.target_success),
    CurriculumCallback(curriculum_steps=args.curriculum_steps, noise_init=0.10, noise_final=0.02),
]
```

**CheckpointCallback:** Saves a model zip every `checkpoint_freq` global steps
(default 500 000 global steps = `500_000 // 16 = 31_250` steps per env internally).
Files: `checkpoints/ppo_s1_500000_steps.zip`, etc.

**RenderCallback:** Every 5 000 steps reads the current physics state from all 16
environments and sends them to the Newton viewer as a 4×4 grid with 4 m spacing.
Drones are coloured green (close to target) to red (far). No-op if `--headless`.

**MetricsCallback:** Fires every step. Reads `info["reward_components"]` and
`info["motor_rpms"]` from every environment, stores values in `deque(maxlen=100)`
rolling buffers, flushes means to TensorBoard every 1 000 steps.

Early-stop check: after each flush, if `success_rate ≥ target_success` and the
rolling buffer is full (100 episodes), training stops and the model is saved.

**CurriculumCallback:** Computes `t = min(num_timesteps / curriculum_steps, 1.0)`
and writes it into every environment's `curriculum` attribute. Also decays TD3 action
noise σ from 0.10 → 0.02 over the same window.

---

### Stage 5 — One full training run, by the numbers

With `--algo ppo --num_envs 16 --total_timesteps 40_000_000`:

| Quantity | Value |
|---|---|
| Total env steps | 40 000 000 |
| Steps per rollout (all envs) | 32 768 (= 2048 × 16) |
| Number of rollout cycles | ~1 221 (= 40 000 000 / 32 768) |
| Gradient steps per rollout | 5 120 (= 10 epochs × 512 mini-batches) |
| Total gradient steps | ~6 250 000 |
| Checkpoint saves | 80 (every 500k steps) |
| Curriculum fully ramped at | 1 500 000 steps (~46th rollout cycle) |
| Convergence observed at | ~30 000 000 steps |

---

### Stage 6 — Saving the final model

```python
# train_drone.py
save_path = f"{algo}_drone_final_s{args.seed}"
model.save(save_path)
print(f"\nSaved → {save_path}.zip")
```

After training, SB3 serialises the policy and value networks into
`ppo_drone_final_s{seed}.zip`. Load with:
```python
model = PPO.load("ppo_drone_final_s1")
```

---

## TensorBoard Signal Reference

All signals logged by `MetricsCallback`. Rolled mean over last 100 episodes.

```bash
tensorboard --logdir drone_logs
```

### metrics/ep_length

Mean steps per episode. Hard ceiling is `MAX_EPISODE_STEPS = 800` (8 s at 100 Hz).
Episodes also end early on crash or arena escape.

| Value | Interpretation |
|---|---|
| Rising toward 800 | Drone is staying alive — healthy |
| Suddenly drops | Drone started crashing; cross-check `terminal_upright` |
| Stable at 200–400 | Drone crashes at a consistent point |

### metrics/ep_reward

Total undiscounted return per episode. A drone that only hovers earns ~+400 (survival
0.50 × 800 steps). Positive and growing = healthy.

### metrics/success_rate

Fraction of training episodes where `terminal_dist < 0.15 m`. Training uses one
waypoint per episode; evaluation chains multiple waypoints. Training success is
necessary but not sufficient for evaluation success.

### metrics/terminal_dist

Mean distance to target at episode end. Falling toward 0.15 → curriculum is working.
Stuck above 0.30 → drone is not converging on the target.

### metrics/terminal_upright

`R22 = 1 − 2(qx² + qy²)`: dot product of drone's up-axis with world Z.
1.0 = perfectly level, −1 = inverted (crashed). Should converge toward 1.0.

### metrics/mean_motor_rpm

Average filtered motor speed across the episode. Physical hover RPM ≈ **14 476**.
Values consistently below 14 200 indicate under-thrusting; above 14 800 indicate
over-thrusting.

### metrics/rpm_hover_dev

`|mean_motor_rpm − CF_HOVER_RPM|` — absolute deviation from hover equilibrium.
Captures **low-frequency (slow) RPM drift** — sustained over- or under-thrust
that builds into oscillation. Complementary to `act_c` which captures
**high-frequency (step-to-step) jitter**.

| Value | Interpretation |
|---|---|
| < 100 RPM | Healthy — motors close to hover on average |
| 200–500 RPM | Slow oscillation — drone drifts from hover RPM over many steps |
| > 500 RPM | Severe drift; ungated approach reward or high entropy |

### reward_components/*

See the Reward Function section for per-component descriptions. Key diagnostics:
- `pos_c` should rise toward 0 over training
- `vel_c` should show mild negatives during nav, near 0 when settled
- `hover_bonus` should grow (more time settled at target)
- `approach` should decay toward 0 (drone is already near the gate)
- `survival` is always exactly 0.50 — deviation = reward pipeline bug

### Diagnostic quick-reference

| Symptom | Most informative signals | Likely cause |
|---|---|---|
| Drone oscillates up/down | `rpm_hover_dev` high, `vel_c` spikes | Approach reward driving overshoot |
| Drone does not reach target | `pos_c` flat negative, `approach` > 0 | Pos gradient too weak; check curriculum |
| Drone crashes early | `ep_length` short, `terminal_upright` low | Stability failure |
| Good training, bad eval | `success_rate` high, eval dist large | Train/eval distribution mismatch |
| `ep_reward` negative | short `ep_length` | Entropy too high or crash-filled buffer |
| Drone hovers but does not navigate | `pos_c` flat negative, `hover_bonus` = 0 | Hover attractor from flat survival |

---

## Evaluation

### Primary evaluation: eval_trajectory.py (waypoint chaining)

`eval_trajectory.py` is the main evaluation script. It evaluates a trained policy by
chaining waypoints continuously within a single episode — the drone navigates to each
waypoint in sequence without any physics reset between them. An episode ends when the
drone crashes, exhausts the step budget, or reaches all waypoints.

**Primary metric: leg success rate** — fraction of all attempted waypoint legs in which
the drone came within 0.15 m of the target.

```bash
# Basic usage
python eval_trajectory.py --model ppo_drone_final_s1 --algo ppo

# Clean start (curriculum=0: upright, stationary, within 0.1 m of first wp)
python eval_trajectory.py --model ppo_drone_final_s1 --algo ppo --clean_start

# Randomised spawn (curriculum=1.0: full training distribution)
python eval_trajectory.py --model ppo_drone_final_s1 --algo ppo  # default

# Trajectory shapes
python eval_trajectory.py --model ppo_drone_final_s1 --algo ppo --traj square
python eval_trajectory.py --model ppo_drone_final_s1 --algo ppo --traj lissajous

# Per-attempt success rate (1 wp per episode, many episodes)
python eval_trajectory.py --model ppo_drone_final_s1 --algo ppo \
    --max_waypoints 1 --num_episodes 50
```

#### Trajectory shapes

| `--traj` | Description |
|---|---|
| `random` (default) | Uniform sample: r=0.5–1.5 m, z=0.3–1.2 m — matches training distribution |
| `square` | 4-corner loop, half=0.8 m, z=0.7 m — repeating fixed pattern |
| `lissajous` | Figure-8: x=sin(t), y=0.7·sin(2t), 7 pts/cycle, z=0.7 m |

#### Spawn modes

Two evaluation modes test different aspects of the policy:

| Mode | curriculum | Spawn condition | Tests |
|---|---|---|---|
| Clean start (`--clean_start`) | 0.0 | Upright, stationary, within 0.1 m of first waypoint | Pure navigation capability |
| Randomised (default) | 1.0 | Full training distribution: ±0.7 m offset, ±1.5 m/s velocity, ±15° tilt | Robustness to disturbed initial states |

**Critical: spawn alignment.** Before `env.reset()`, the evaluator sets
`env._target = waypoints[0]` so the drone spawns within curriculum distance of the
first waypoint — matching training conditions exactly. Without this, `reset()` would
spawn near the hardcoded `TARGETS[0]=[1,0,0.5]` instead of the actual first waypoint.

#### All evaluation arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `ppo_drone_final_s1` | Path to model zip (without `.zip`) |
| `--algo` | `ppo` | Algorithm used to train: `ppo`, `sac`, `td3` |
| `--traj` | `random` | Trajectory shape: `random`, `square`, `lissajous` |
| `--num_episodes` | `10` | Number of evaluation episodes |
| `--max_waypoints` | `20` | Waypoint cap per episode |
| `--max_steps` | `8000` | Step cap per episode (~80 s at 100 Hz) |
| `--seed` | `42` | RNG seed for reproducible waypoint generation |
| `--clean_start` | off | Spawn upright with zero velocity (curriculum=0) |
| `--stochastic` | off | Use stochastic (non-deterministic) policy |

#### Evaluation output

```
  ep  1/ 10 | reached= 6/20 | CRASH   | flight= 18.4s  1.2s/wp | rpm≈14092
    ✓✓✓✓✓✓✗
  ep  2/ 10 | reached= 8/20 | CRASH   | flight= 26.3s  1.5s/wp | rpm≈14025
    ✓✓✓✓✓✓✓✓✗
  ...

════════════════════════════════════════════════════════════════════
  EVALUATION REPORT  [PPO]  —  random    r=0.5–1.5 m, z=0.3–1.2 m, uniform
════════════════════════════════════════════════════════════════════
  Episodes run          : 20  (waypoint cap = 20 / ep)

  ── Episode outcomes ────────────────────────────────────────────
  Crash rate            : 100.0%
  Timeout rate          :   0.0%
  Full-success rate     :   0.0%  (all 20 wps reached)
  Mean wps / episode    : 4.5 ± 3.6  (cap 20)
  Mean flight time      : 16.2 s / ep

  ── Per-waypoint-attempt (leg-level) ────────────────────────────
  Leg success rate      : 81.8%  (110 total attempts)
  Mean time to reach wp : 1.59 ± 0.73 s  (successful legs only)
  Mean leg distance     : 1.234 m  (wp-to-wp)
  Mean arrival speed    : 0.294 m/s  (speed when dist < 0.15 m)
  Mean closest (failed) : 0.287 m  (best dist before crash/timeout on failed legs)

  ── Motor / Thrust ──────────────────────────────────────────────
  Mean motor RPM        : 14061 RPM  (97.1% of hover)
  Hover RPM             : 14476 RPM  (required for level flight)
  Mean |RPM − hover|    : 415 RPM  (0 = perfect hover thrust)
  Max RPM (hardware)    : 21702 RPM

  ── Leg success rate by distance (random traj) ──────────────────
   0.0–0.5m  [████████████████████]  100.0%  (n=7)
   0.5–1.0m  [██████████████████░░]   92.9%  (n=28)
   1.0–1.5m  [█████████████████░░░]   87.1%  (n=31)
   1.5–3.0m  [█████████████░░░░░░░]   68.2%  (n=44)
════════════════════════════════════════════════════════════════════
```

**Why 100% crash rate in chaining:** The policy was trained with episode resets after
each crash, but the chaining evaluator does not reset between waypoints. Once the drone
fails a difficult waypoint, it may be in a state outside its recovery envelope. This is
a training-eval mismatch, not a navigation failure — the per-leg success rate is the
correct metric.

---

## Benchmark Results

### Learning progression (PPO, seed 1, clean-start evaluation, 20 episodes)

The policy converges at ~30M steps. Results from `eval_trajectory.py --clean_start`:

| Checkpoint | Leg SR | Wps/ep | Mean RPM | RPM% hover | \|RPM−hover\| | Arrival spd |
|---|---|---|---|---|---|---|
| 3M steps | 73.0% | 2.7 ± 2.3 | 14,303 | 98.8% | 227 RPM | 0.419 m/s |
| 30M steps | **82.9%** | **4.8 ± 3.2** | 14,039 | 97.0% | 437 RPM | 0.326 m/s |
| 40M steps | 81.8% | 4.5 ± 3.6 | 14,061 | 97.1% | 415 RPM | 0.294 m/s |

### Robustness to spawn randomisation (40M checkpoint)

| Condition | Leg SR |
|---|---|
| Clean start (curriculum=0) | 81.8% |
| Randomised (curriculum=1.0) | 50.0% |

The 31.5 pp gap reflects sensitivity to disturbed initial conditions. Policies trained
with curriculum randomisation (which reaches c=1.0 at 1.5M steps) do not fully
generalise to all states in the training distribution — large initial tilts combined
with nonzero velocity can exhaust the episode step budget before recovery.

### Reward shaping ablation (PPO seed 1, 3M steps, fixed 4-waypoint protocol)

| Configuration | Leg Success | Mean Dist (m) | \|RPM−hover\| |
|---|---|---|---|
| (A) Base reward only | 42.5% | 0.433 | 262 |
| (B) + Gated approach term only | 32.5% | 0.345 | 126 |
| (C) Full reward (both terms) | **65.0%** | **0.178** | 257 |

Configuration (B) performs **worse** than baseline despite reducing RPM deviation.
The approach gate removes gradient in the 0.10–0.15 m band without replacing it; the
drone learns to hover steadily at the gate boundary rather than crossing the success
threshold. Configuration (C) resolves this with the velocity-gated hover bonus.

### Algorithm comparison (3M steps, same reward, same environment)

| Metric | PPO | SAC | TD3 |
|---|---|---|---|
| Per-waypoint success | **50%** (base reward) / **65%** (shaped) | 0% | 0% |
| Mean episode length (steps) | **636** | 89 | 167 |
| Mean motor RPM | **≈ 14 476** | 12 890 ↓ | 12 147 ↓↓ |

**SAC diverged:** `ent_coef="auto"` targeted maximum stochasticity (σ ≈ 1.0 → ±7 200
RPM noise). Fixed `ent_coef=0.005` resolves this.

**TD3 — generalisation failure:** `learning_starts=10_000` filled the replay buffer
with crash trajectories from uniform random actions. Q-networks learned pessimistic
values (all states → crash), making the policy output below-hover RPMs. Fixed with
`learning_starts=0` and small initial noise (σ=0.10).

---

## Environment API Reference

`DroneEnv` is a standard `gymnasium.Env`:

```python
env = DroneEnv(
    render_mode    = "human" | None,  # "human" enables Newton viewer
    viewer         = viewer,           # Newton viewer object (or None)
    random_targets = True,             # randomly sample target at each reset
    multi_target   = False,            # reassign a new random target on each waypoint reach
    obs_noise      = False,            # add Gaussian sensor noise
    curriculum     = 0.0,             # reward weight scale [0=easy, 1=hard]
)

obs, info = env.reset()
obs, rew, term, trunc, info = env.step(action)

env.set_target(np.array([x, y, z]))  # switch waypoint mid-episode
obs = env.get_obs()                   # re-read obs after set_target
```

The `info` dict from `step()` always contains:

```python
{
    "dist":       float,        # distance to current target (m)
    "upright":    float,        # R[2,2]: 1=level, −1=inverted
    "z":          float,        # altitude (m)
    "motor_rpms": [float × 4],  # filtered RPM per motor (≈14 476 at hover)
    "reward_components": {
        "pos_c":       float,   # -C_rp * ‖p_err‖²
        "orient_c":    float,   # -C_rq * (1 − qw²)
        "vel_c":       float,   # -C_rv * ‖v‖²
        "ang_c":       float,   # -C_rw * ‖ω‖²
        "act_c":       float,   # -C_ra * ‖Δa‖²
        "survival":    float,   # +0.50 constant
        "approach":    float,   # distance shaping, gated at 0.10 m
        "hover_bonus": float,   # +0.15 when dist < 0.15 m AND speed < 0.5 m/s
    }
}
```

At episode end (`terminated or truncated`), additionally:

```python
{
    "terminal_dist":    float,
    "terminal_upright": float,
    "terminal_ep_len":  int,
    "terminal_reward":  float,
}
```

---

## Design Decisions

### Why Level 5.1 (direct RPM)?

Higher abstraction levels (angular rate commands, CTBR) hide the nonlinear
RPM²→thrust curve, motor delay, and rotor speed dynamics. Controlling directly at
RPM forces the policy to model all of them. The motor LPF (τ = 0.15 s at 100 Hz)
alone introduces ~15 steps of transport lag that must be implicitly compensated.

### Why correct Crazyflie mass and inertia?

Incorrect mass/inertia (e.g., derived from collision geometry density) produces wrong
force/inertia feedback. Any policy trained with incorrect dynamics would not transfer
to real hardware. Mass (27 g) and inertia (Förster 2015) are set explicitly via Newton
`add_body(mass=..., inertia=...)`, bypassing geometry-derived values entirely.

### Why `‖Δa‖²` instead of `‖a‖²` for action cost?

`−C_ra‖a − a_prev‖²` penalises **changes** in the RPM setpoint (jerk) rather than
the absolute setpoint level. A sustained high-RPM climb should not be penalised, but
rapid oscillation between high and low should.

### Why motor LPF in RPM space?

If the LPF were applied to a linear thrust fraction, the nonlinearity `F=KT·n²` would
be hidden from the physics engine. Applying it in RPM space preserves the full
nonlinear relationship in the filtered output.

### Why the approach gate at 0.10 m?

The gate suppresses the per-step approach reward inside 0.10 m of the target. Without
the gate, the drone earns approach reward for any motion toward the target — including
oscillatory movements close to the waypoint. The gate radius is set inside the success
threshold (0.15 m) so approach reward still guides the drone right up to the success
boundary, but is suppressed in the innermost 0.10 m where settlement behaviour is
preferred.

### Why a velocity gate on hover_bonus?

A pure distance gate (`if dist < 0.15`) creates a reward cliff — the policy earns
+0.15/step the instant it crosses 0.15 m regardless of speed. This teaches rushing:
approaching at 3 m/s to enter the zone sooner earns more cumulative approach reward
plus a bonus step. The velocity gate (speed < 0.5 m/s) makes the bonus collectible
only while stationary, enforcing the approach→decelerate→settle strategy.

### Why flat survival = 0.50?

The flat constant makes staying airborne intrinsically rewarding. At low curriculum
it dominates position penalty, bootstrapping stable hovering. At full curriculum the
position penalty exceeds it, forcing navigation. The downside: a hover attractor forms
where the drone earns positive per-step reward by hovering in place. The approach and
hover_bonus terms counteract this by rewarding target-directed motion.

Decaying survival (e.g., exp(−dist)) was tested but caused crashes — it combined with
`vel_c` to create a double-bind (urgency to move + penalty for moving), and the
resulting instability increased crash rate rather than reducing hovering.

---

## Dependencies

```bash
pip install gymnasium numpy warp-lang newton stable-baselines3 sbx-rl tensorboard
```

| Package | Role |
|---|---|
| `newton` | GPU rigid-body simulator (NVIDIA Warp backend) |
| `warp-lang` | CUDA kernel execution via `@wp.kernel` |
| `gymnasium` | Standard RL environment interface |
| `stable_baselines3` | PPO + callbacks, vec envs |
| `sbx` | SAC and TD3 implementations (JAX-accelerated) |
| `tensorboard` | Training monitoring |

---

## References

Eschmann, J., Albani, D., Loianno, G. (2024).
**Learning to Fly in Seconds.**
*IEEE Robotics and Automation Letters.* arXiv:2311.13081.

Förster, D. (2015).
**System Identification of the Crazyflie 2.0 Nano Quadrocopter.**
Bachelor's thesis, ETH Zürich.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017).
**Proximal Policy Optimization Algorithms.**
arXiv:1707.06347.

Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., Dormann, N. (2021).
**Stable-Baselines3: Reliable Reinforcement Learning Implementations.**
*Journal of Machine Learning Research*, 22(268), 1–8.
