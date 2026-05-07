# Drone RL — Learning to Fly with PPO, SAC, and TD3

A reinforcement learning system for training a quadrotor position controller inside the
[Newton](https://github.com/newton-physics/newton) GPU-native rigid-body simulator.
The observation, action, and reward design is based on
**"Learning to Fly in Seconds"** (Eschmann, Albani, Loianno — RAL 2024).

The environment models a real **Crazyflie 2.x** (27 g nano-quadrotor) with correct
mass, inertia, arm length, and **Level-5.1 direct RPM control** — the lowest-level,
most physically accurate action abstraction in the paper's taxonomy.

---

## Repository Layout

```
dronesim/
├── drone_gym_env.py          # Gymnasium environment (physics + obs + reward)
├── train_drone.py            # Unified training script: PPO | SAC | TD3
├── eval_drone.py             # Multi-waypoint evaluation script
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
    ├── ppo_1/
    ├── sac_1/
    └── td3_1/
```

---

## Quick Start

```bash
# Train with TD3 — matches the paper algorithm
python train_drone.py --algo td3

# Headless (no OpenGL window)
python train_drone.py --algo td3 --headless

# Compare all three algorithms
python train_drone.py --algo ppo --headless
python train_drone.py --algo sac --headless
python train_drone.py --algo td3 --headless

# Monitor training
tensorboard --logdir drone_logs

# Evaluate a trained model
python eval_drone.py --model td3_drone_final --algo td3
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
Gymnasium interface. The RL policy runs on the CPU/JAX side and sends RPM setpoints;
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
(32.5 mm); a zero density is assigned so the geometry contributes no additional
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

The paper's taxonomy classifies quadrotor controllers by their control input level.
This environment operates at **Level 5.1 — Motor commands/RPM setpoints**, the
lowest and most physically accurate level.

Each propeller's thrust and reaction torque are computed from the filtered motor
speed using the quadratic RPM² model:

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
The paper's ablation study shows that removing the rotor delay model alone reduces
real-world success from 10/10 to **0/10** flights (Table II).

### Motor Dynamics (First-Order Low-Pass Filter)

Real brushless motors do not respond instantaneously to commands. The paper (§IV)
identifies τ ≈ 0.15 s for the Crazyflie. This is implemented as a discrete
first-order IIR filter on motor RPM at every step:

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

The observation design follows the paper's actor observation
`o_a = {p, R, v, ω, H}`:

| Slice     | Symbol   | Dim | Description |
|-----------|----------|-----|-------------|
| `[0:3]`   | `p_err`  | 3   | Position error = `pos − target` (world frame, m) |
| `[3:12]`  | `R_flat` | 9   | Drone rotation matrix, row-major flattened |
| `[12:15]` | `v`      | 3   | Linear velocity, world frame (m/s) |
| `[15:18]` | `ω`      | 3   | Angular velocity, body frame (rad/s) |
| `[18:22]` | `a_prev` | 4   | Previous normalised action (action history N_H=1) |

**Total: 22 dimensions.**

### Why position error, not absolute position?

The policy sees `p_err = pos − target`. The network always "thinks" it is flying to
the origin regardless of where the actual target is. At inference you can move the
target anywhere by calling `env.set_target(new_pos)` — the same trained policy
handles it without retraining.

### Why rotation matrix, not quaternion?

A unit quaternion `q` and `−q` represent the same physical rotation (double-coverage
of SO(3)). A neural network fed raw quaternions must implicitly learn to ignore this
ambiguity. The 3×3 rotation matrix has no such ambiguity and is the representation
used in the paper.

### Why action history?

The motor LPF introduces ~15 steps of lag. Without knowing what was commanded
recently, the policy cannot predict the drone's near-future response. The previous
action provides a window into the current motor state, partially restoring
observability of the delayed RPM.

### Optional observation noise

When training with `--obs_noise`, Gaussian noise simulates imperfect onboard sensors:

| Component | Noise σ     |
|-----------|-------------|
| `p_err`   | 0.01 m      |
| `v`       | 0.01 m/s    |
| `ω`       | 0.05 rad/s  |

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

| action | RPM    | Thrust/motor |
|--------|--------|--------------|
| −1.0   | 7 250  | 1.7 g        |
| −0.5   | 10 863 | 3.8 g        |
|  0.0   | 14 476 | 6.7 g ← hover|
| +0.5   | 18 089 | 10.5 g       |
| +1.0   | 21 702 | 15.2 g       |

The hover point (action = 0) is set at the exact RPM where thrust equals weight:
`4 × KT × n² = m·g`. The thrust is **nonlinear** in action because `F ∝ n² ∝ (hover + action·range)²`.

The RPM setpoint feeds the motor LPF filter described above.

---

## Reward Function

The reward at every step is the sum of nine components:

```
r = pos_c + orient_c + vel_c + ang_c + act_c + survival + crash + arrival + approach + hover_bonus
```

```python
# drone_gym_env.py — step() method, lines ~462–501
pos_c    = -C_rp  * ‖p_err‖²
orient_c = -C_rq  * (1 − qw²)
vel_c    = -C_rv  * ‖v‖²
ang_c    = -C_rω  * ‖ω‖²
act_c    = -C_ra  * ‖Δa‖²
survival = +0.50                             # every step
crash    = -2.00  if z < 0.05 m             # one-time ground impact
arrival  = +1.00  first time dist < 0.10 m  # one-time per waypoint
approach = _APPROACH_COEF × (last_dist − dist)  if dist > 0.25 m  else 0.0
hover_bonus = +0.15  if dist < 0.15 m AND speed < 0.5 m/s  # settled at target
```

---

### pos_c — Position cost

**Code:** `drone_gym_env.py` — `pos_c = -C_rp * float(np.dot(p_err, p_err))`

`C_rp` ramps from **0.05 → 1.00** with curriculum. The cost grows quadratically with
distance — being 2× as far from the target yields 4× the penalty.

**What it drives:** This is the primary navigation signal. A drone 1 m away pays
`−1.0/step` at full curriculum, which exceeds the survival bonus (`+0.50`), forcing
the policy to close the gap. At curriculum = 0 the same error costs only `−0.05/step`
so early episodes are dominated by the survival bonus instead, keeping the drone
airborne while it learns to fly.

**TensorBoard — `reward_components/pos_c`:** A **negative value rising toward 0** as
training progresses. If it stays at −0.3 or below after 2 M steps the drone is
consistently far from the target. If it rises steeply after 1–1.5 M steps the
curriculum ramp is working.

---

### orient_c — Orientation cost

**Code:** `drone_gym_env.py` — `orient_c = -_C_RQ * float(1.0 - qw**2)`

`_C_RQ = 0.10` is **fixed** (does not ramp with curriculum).
`qw = quaternion_w = cos(θ/2)` where θ is the tilt angle from upright.

| Orientation | qw | orient_c |
|-------------|-----|----------|
| Perfectly upright | 1.0 | 0.00 |
| 45° tilted | 0.924 | −0.015 |
| 90° tilted | 0.707 | −0.050 |
| Inverted | 0.0 | −0.100 |

**What it drives:** Keeps the drone upright throughout the episode. The cost is
deliberately small (`−0.10` worst case) so it does not overwhelm the position signal,
but it provides a continuous tilt penalty that discourages aggressive manoeuvres.

**TensorBoard — `reward_components/orient_c`:** Should converge to **−0.005 to −0.02**
for a drone that flies with small tilts. Values below −0.05 mean the drone is spending
significant time tilted or inverted — usually a sign of unstable control that will
soon lead to crashes.

---

### vel_c — Linear velocity cost

**Code:** `drone_gym_env.py` — `vel_c = -C_rv * float(np.dot(v, v))`

`C_rv` ramps from **0.01 → 0.30** with curriculum. `v` is the linear velocity in
the world frame (m/s).

**What it drives:** Encourages the drone to slow down and hover rather than passing
through the target. At full curriculum a 1 m/s speed costs `−0.30/step`. This signal
is in deliberate tension with the `approach` reward (which rewards moving toward the
target) — a good policy balances them by approaching quickly then braking.

**TensorBoard — `reward_components/vel_c`:** Should show **mild negative values during
navigation** and **near zero when settled**. Large negative spikes (e.g., −0.5) after
waypoint switches indicate the drone is accelerating without braking. **If vel_c
spikes never decay**, the drone is oscillating: it accelerates toward the target,
overshoots, reverses, and repeats. This is the primary oscillation indicator.

---

### ang_c — Angular velocity cost

**Code:** `drone_gym_env.py` — `ang_c = -C_rw * float(np.dot(w, w))`

`C_rw` ramps from **0.001 → 0.05** with curriculum. `w` is angular velocity in the
body frame (rad/s).

**What it drives:** Discourages spinning and wobbling. The very small initial weight
(0.001) means this signal is nearly invisible early in training — the drone first
learns not to crash and not to drift, then later learns to stop spinning.

**TensorBoard — `reward_components/ang_c`:** Healthy values are **−0.01 to −0.03**
once training converges. If ang_c is large and negative late in training despite
stable vel_c, the drone is spinning in place (yaw instability). If it is always near
zero, the curriculum weight is still small — check the curriculum progress.

---

### act_c — Action-change cost (jerk regularisation)

**Code:** `drone_gym_env.py` — `act_c = -C_ra * float(np.dot(delta_a, delta_a))`

`C_ra` ramps from **0.005 → 0.02** with curriculum. `delta_a = action − prev_action`
is the change in normalised RPM setpoint between consecutive steps.

**Key distinction:** This penalises the **change** in action (jerk), not the action
magnitude. A sustained high-RPM climb is not penalised; rapid oscillation between
high and low RPM every step is. This matches paper Eq. 1 and produces smooth motor
commands that are safe to execute on real hardware.

**TensorBoard — `reward_components/act_c`:** Healthy values are **−0.02 to −0.06**.
Values below −0.10 mean the policy is thrashing — changing RPM commands dramatically
every step. act_c captures **step-to-step (high-frequency) jitter**; `rpm_hover_dev`
captures **slow (low-frequency) drift**. Both must be small for a stable policy.

---

### survival — Constant per-step bonus

**Code:** `drone_gym_env.py` — `survival = _C_RS  # 0.50, fixed`

A constant `+0.50` every step that the drone is alive.

**What it drives:** Makes staying airborne intrinsically rewarding throughout training.
At curriculum = 0 with `C_rp = 0.05`, a drone 1 m from target earns `+0.50 − 0.05 = +0.45/step` just by hovering. The survival bonus ensures early training has a positive
gradient even when navigation is poor. At curriculum = 1 the same drone earns
`+0.50 − 1.00 = −0.50/step`, forcing it to navigate.

**TensorBoard — `reward_components/survival`:** A **flat constant 0.50** — logged as
a sanity check. Any deviation indicates a bug in the reward pipeline.

---

### crash — Ground impact penalty

**Code:** `drone_gym_env.py` — `crash = -2.0 if z < 0.05 else 0.0`

A one-time `−2.0` on the step where the drone hits the ground (altitude < 5 cm).
The episode terminates immediately after this step.

**What it drives:** Teaches the drone that crashes are categorically bad, separate
from the continuous position cost. Without this, a drone that crashes near the target
might be incorrectly rewarded.

**Note:** `crash` is not logged as a separate TensorBoard component. Its effect is
visible in `ep_length` (shorter episodes) and `terminal_upright` (near 0 if crashed).

---

### arrival — One-time waypoint bonus

**Code:** `drone_gym_env.py` — `arrival = 1.0` first time `dist < 0.10 m`

A `+1.0` bonus the first time the drone comes within 0.10 m of the current waypoint.
The flag `self._arrived` prevents repeated claiming on the same target.

**What it drives:** Provides a salient one-time signal that clearly marks "you found
the target". The `hover_bonus` (below) then takes over to reward staying there.

**Note:** Arrival is not logged separately in TensorBoard. Its combined effect with
`hover_bonus` is visible as the `success_rate` rising.

---

### approach — Potential-based distance shaping

**Code:** `drone_gym_env.py`
```python
_APPROACH_COEF = 1.0
_APPROACH_GATE = 0.25  # m — shaping turns off inside this radius

approach = (
    _APPROACH_COEF * (self.last_dist - dist)
    if dist > _APPROACH_GATE else 0.0
)
```

`approach` is positive when the drone gets closer to the target in one step, and
negative when it retreats. It is **gated off when the drone is already within 0.25 m**
of the target.

**Why potential-based shaping:** The quadratic `pos_c` provides a weak gradient far
from the target (the derivative `−2·C_rp·dist` is small at large dist with small
C_rp). The approach term is mathematically equivalent to `Φ(s') − Φ(s)` with
`Φ(s) = −dist`, a reward-shaping-theorem compliant transformation that provides a
**dense directional gradient** toward the target without changing the optimal policy.

**Why gated at 0.25 m:** Without the gate, the drone earns approach reward for any
motion toward the target — even a tiny oscillation. A drone that overshoots the
target by 0.1 m and bounces back earns `+0.1 × 1.0 = +0.1` reward, incentivising
the oscillation. With the gate, inside 0.25 m the approach signal is zero and
`hover_bonus` (below) becomes the only positive signal, rewarding settlement instead
of movement.

**TensorBoard — `reward_components/approach`:** Converges toward **0** as the drone
learns to hover. Early training shows positive values (drone approaches from far
away). A well-tuned policy near convergence will show `approach ≈ 0`: the drone is
already close to the target and the gate is active.

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
target: a drone that approaches at 3 m/s and enters the 0.15 m zone for even one
step earns more approach reward (from fast closing) plus a hover_bonus step. The
result is bimodal behaviour — episodes where the drone brakes in time and hovers
(high reward) alternating with episodes where it overshoots and crashes (catastrophic
negative reward from vel_c spikes). The `_HOVER_SPEED_GATE` removes this incentive:
you only earn the bonus when you are *already slow*, so the optimal strategy becomes
approach → decelerate → settle, not approach → sprint.

**What it drives:** Provides a sustained per-step incentive to remain at the target
with low velocity, complementing the one-shot `arrival` bonus. The coefficient (0.15)
is intentionally small relative to `survival` (0.50) so it does not dominate the
reward and cause risky behaviour.

**TensorBoard — `reward_components/hover_bonus`:** Rises from **0 toward a positive
value** as success_rate and episode quality improve. Unlike `approach` (which decays
to 0 at convergence), hover_bonus should *grow* throughout training — a healthy sign
that the drone is spending increasing time settled at targets. If it stays near 0
despite a rising success_rate, the drone is grazing inside 0.15 m at high speed
rather than hovering. Cross-check `vel_c`: a well-settled drone will have hover_bonus
positive and vel_c near zero simultaneously.

---

### Reward Curriculum

Weights ramp linearly from conservative → strict over the **first 50% of training**:

| Weight | Init  | Target | Controls |
|--------|-------|--------|----------|
| `C_rp` | 0.05  | 1.00   | Position precision |
| `C_rv` | 0.010 | 0.30   | Velocity damping |
| `C_rω` | 0.001 | 0.05   | Angular stability |
| `C_ra` | 0.005 | 0.02   | Action smoothness |
| `C_rq` | 0.10  | 0.10   | Orientation (fixed) |
| `C_rs` | 0.50  | 0.50   | Survival (fixed) |

**Why curriculum matters:** At `curriculum = 0` the position cost is tiny and the
survival bonus dominates — the drone gets positive reward just by staying airborne.
At `curriculum = 1` the position penalty grows 20×, forcing tight navigation. Without
curriculum the heavy early position penalty makes crashing at step 1 "optimal" for
the initial random policy.

```python
# train_drone.py — CurriculumCallback._on_step()
curriculum = min(num_timesteps / (total_timesteps × 0.5), 1.0)
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

Episodes also truncate after `MAX_EPISODE_STEPS = 800` steps (8.0 s at 100 Hz),
matching the evaluator's budget of 4 waypoints × 200 steps each.

---

## Reinforcement Learning Algorithms

Three algorithms are supported. **TD3 is the default and matches the paper.**

### TD3 — Twin Delayed Deep Deterministic (`sbx`, JAX) ← paper algorithm

**Type:** Off-policy, deterministic  
**Exploration:** Gaussian action noise in normalised RPM space, decaying with curriculum.

Three stabilising tricks over DDPG:
1. **Twin critics** — minimum of two Q-values reduces overestimation bias.
2. **Delayed policy update** — actor updates every 2 critic steps.
3. **Target policy smoothing** — noise on target actions prevents narrow Q-spikes.

| Hyperparameter | Value | Why |
|----------------|-------|-----|
| `buffer_size` | 500 000 | Replay buffer |
| `batch_size` | 256 | Samples per gradient step |
| `learning_starts` | 10 000 | Fill buffer before first update (RPM crashes in early exploration make noise higher) |
| `tau` | 0.005 | Soft target network update |
| `policy_delay` | 2 | Actor updates per 2 critic steps |
| `target_policy_noise` | 0.2 | Smoothing noise on target actions |
| `target_noise_clip` | 0.5 | Clip on target noise |
| `action_noise σ` | 0.30 → 0.05 | Decayed by CurriculumCallback alongside reward weights |
| `net_arch` | [256, 256] | Two hidden layers (matches paper) |

### SAC — Soft Actor-Critic (`sbx`, JAX)

**Type:** Off-policy, entropy-regularised  
**Exploration:** Automatic entropy tuning — no manual noise schedule required.

| Hyperparameter | Value |
|----------------|-------|
| `buffer_size` | 500 000 |
| `batch_size` | 256 |
| `learning_starts` | 10 000 |
| `tau` | 0.005 |
| `ent_coef` | "auto" |
| `target_entropy` | "auto" (= −4 for 4D action) |

### PPO — Proximal Policy Optimisation (`stable_baselines3`)

**Type:** On-policy  
**Best for:** Baselines and ablations; less sample-efficient than off-policy methods
on continuous control tasks.

| Hyperparameter | Value |
|----------------|-------|
| `n_steps` | 2048 |
| `n_epochs` | 10 |
| `batch_size` | 64 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |

### Algorithm Comparison

```
Paper algorithm:     TD3
Sample efficiency:   SAC ≈ TD3 >> PPO
Stability:           TD3 ≈ PPO > SAC (early training)
Wall-clock (GPU):    SBX (SAC/TD3) >> SB3 (PPO)  — JAX vs PyTorch
```

---

## Training

### Basic usage

```bash
# Default: TD3, 3 M steps, 16 envs (matches paper)
python train_drone.py --algo td3

# With sensor noise for sim-to-real robustness
python train_drone.py --algo td3 --obs_noise

# Headless, custom timesteps
python train_drone.py --algo td3 --headless --total_timesteps 3000000
```

### All training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--algo` | `td3` | Algorithm: `ppo`, `sac`, or `td3` |
| `--num_envs` | `16` | Parallel training environments |
| `--total_timesteps` | `3 000 000` | Total environment steps |
| `--learning_rate` | `3e-4` | Adam learning rate |
| `--gamma` | `0.99` | Discount factor |
| `--checkpoint_freq` | `50 000` | Save checkpoint every N steps |
| `--checkpoint_dir` | `checkpoints` | Directory for checkpoints |
| `--obs_noise` | off | Add Gaussian sensor noise to observations |
| `--no_curriculum` | off | Disable reward curriculum (fixed target weights) |
| `--headless` | off | No OpenGL viewer (Newton built-in flag) |

### Exploration Noise Decay (TD3)

The `CurriculumCallback` decays the TD3 action noise σ in lockstep with the reward
curriculum, matching the paper's exploration schedule:

```
σ(t) = 0.30  →  0.05   over first 50% of training
```

This ensures the policy explores broadly when reward weights are loose, and refines
its RPM control precisely when weights tighten.

### TensorBoard monitoring

```bash
tensorboard --logdir drone_logs
```

Two panel groups are logged. See the **TensorBoard Signal Reference** section below
for a full explanation of every signal, what it measures, and how to interpret it.

| Group | Signals |
|-------|---------|
| `metrics/` | `ep_length`, `ep_reward`, `success_rate`, `terminal_dist`, `terminal_upright`, `mean_motor_rpm`, `rpm_hover_dev` |
| `reward_components/` | `pos_c`, `orient_c`, `vel_c`, `ang_c`, `act_c`, `survival`, `approach`, `hover_bonus` |

### Output files

```
td3_drone_final.zip           ← final model
sac_drone_final.zip
ppo_drone_final.zip

checkpoints/td3_hover_50000_steps.zip
checkpoints/td3_hover_100000_steps.zip
...
```

---

## PPO Training — How It Works

This section walks through `train_drone.py` end-to-end, explaining exactly what
happens at each stage when you run:

```bash
python train_drone.py --algo ppo
```

---

### Stage 1 — Environment factory: 16 parallel worlds

```python
# train_drone.py:265-275
def _make_env():
    def _init():
        return DroneEnv(
            render_mode=None, viewer=None,
            random_targets=True,
            obs_noise=args.obs_noise,
            curriculum=0.0,        # ← starts easy; CurriculumCallback will ramp this
        )
    return _init

train_env = DummyVecEnv([_make_env() for _ in range(args.num_envs)])  # default: 16
```

`DummyVecEnv` creates **16 independent copies** of the drone environment running in
a single Python process. Each copy has its own Newton physics state, its own
Crazyflie body, its own random target, and its own `curriculum` counter. They are
stepped **sequentially** in a loop — not in parallel threads — but because the
Newton physics step is fast (< 1 ms per env on GPU), the serial overhead is
acceptable.

Each env starts with `curriculum = 0.0` (easy reward weights). The
`CurriculumCallback` will raise this to 1.0 over the first 1.5 M steps.

Why 16 envs? PPO is an **on-policy** algorithm: it collects a batch of fresh
experience, updates the policy, then discards that experience and collects again.
More parallel environments means more diverse experience per rollout — the 16 drones
explore different starting positions and targets simultaneously, giving the policy
gradient a broader gradient estimate at each update.

---

### Stage 2 — Building the PPO model

```python
# train_drone.py:281-298
from stable_baselines3 import PPO

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

The **actor** outputs the mean of a 4-dimensional Gaussian. PPO samples an action
from this distribution during rollout collection and records the log-probability for
the policy gradient. The **log standard deviation** is a separate learned parameter
(not a network output) that decays during training as the policy becomes more
confident.

The **critic** outputs a single scalar `V(s)` — the estimated sum of future discounted
rewards from state `s`. This is not used at inference time; it is only used during
training to compute the advantage estimates.

The two heads share the 256-256 trunk, so features learned for "am I near the
target?" are shared between navigation (actor) and value estimation (critic).

#### Hyperparameter explanations

| Parameter | Value | What it controls |
|-----------|-------|-----------------|
| `n_steps` | 2048 | Steps collected per environment before each update. With 16 envs: **2048 × 16 = 32 768 transitions per rollout**. |
| `batch_size` | 64 | Mini-batch size for gradient steps. The 32 768 transitions are shuffled and split into **512 mini-batches of 64**. |
| `n_epochs` | 10 | Number of complete passes through the rollout buffer. Each rollout yields **10 × 512 = 5 120 gradient steps**. |
| `gamma` | 0.99 | Discount factor. A reward 800 steps away (end of an 8 s episode) is worth `0.99^800 ≈ 0.0003` today — the agent is effectively planning over ~100 steps at full weight. |
| `gae_lambda` | 0.95 | Generalised Advantage Estimation λ. Closer to 1 = lower bias, higher variance (uses more of the actual trajectory). Closer to 0 = higher bias, lower variance (relies more on the value function). 0.95 is the standard trade-off. |
| `clip_range` | 0.2 | PPO's trust-region clip. The policy ratio `π_θ(a\|s) / π_old(a\|s)` is clipped to `[0.8, 1.2]`, preventing any single update from moving the policy too far. |
| `ent_coef` | 0.005 | Entropy bonus weight. Adds `0.005 × H[π]` to the objective, slightly rewarding action uncertainty. Without this, PPO collapses to a deterministic policy that stops exploring. |
| `vf_coef` | 0.5 | Value function loss weight in the combined objective. Balances how much the critic learns relative to the actor. |
| `max_grad_norm` | 0.5 | Gradient clipping. If the gradient norm exceeds 0.5, all gradients are scaled down proportionally. Prevents large parameter updates when advantages are noisy. |

---

### Stage 3 — The PPO training loop

`model.learn(total_timesteps=3_000_000)` drives the main loop. Internally, SB3
repeats the following cycle until 3 M environment steps have been collected:

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

#### What the GAE advantage means for drone training

The advantage `A_t` answers: *"How much better was taking action `a_t` at state `s_t`
than the policy's average expected outcome?"*

A positive `A_t` means the action led to more reward than the critic predicted — the
policy gradient increases the probability of that action. A negative `A_t` means the
action underperformed — the policy gradient decreases its probability.

For the drone this is especially important at waypoint reach events: the moment the
drone enters the 0.15 m hover zone at low speed and earns `hover_bonus = +0.15`, the
TD error `δ_t` spikes positively (unexpected reward). GAE propagates this spike
backwards through the trajectory, increasing the probability of all the approach and
deceleration actions that led to that success. This is how the drone gradually learns
to navigate and settle.

#### Why on-policy data is discarded after each update

PPO uses the old policy `π_old` for the probability ratio `r = π_θ / π_old`. After
10 epochs of updates, `π_θ` has moved away from `π_old` and the ratio `r` would grow
unconstrained, producing biased gradients. This is why the entire rollout buffer is
discarded after each update cycle and fresh experience is collected with the new
policy. It is the fundamental cost of on-policy algorithms: **sample efficiency is
lower than off-policy methods** (TD3, SAC), but the guarantee that data comes from
the current policy makes training more stable.

---

### Stage 4 — Callbacks: what runs between steps

Four callbacks fire on every environment step. They are listed in the order they
were appended to the callback list:

```python
# train_drone.py:348-370
callbacks = [
    CheckpointCallback(...),   # 1. saves model checkpoints
    RenderCallback(...),       # 2. updates the Newton viewer
    MetricsCallback(...),      # 3. logs to TensorBoard
    CurriculumCallback(...),   # 4. ramps reward weights
]
```

#### CheckpointCallback

```python
CheckpointCallback(
    save_freq = max(50_000 // 16, 1),   # = 3125 steps per env = 50k global steps
    save_path = "checkpoints/",
    name_prefix = "ppo_hover",
)
```

Saves a model zip every **50 000 global steps**. The `save_freq` is divided by
`num_envs` because SB3 counts steps per-environment internally — 3 125 steps in
each of the 16 envs equals 50 000 total environment interactions.

Saved files: `checkpoints/ppo_hover_50000_steps.zip`,
`checkpoints/ppo_hover_100000_steps.zip`, …

Use these to resume training from a checkpoint or to evaluate intermediate policies
during a long run.

#### RenderCallback

```python
RenderCallback(
    viewer       = viewer,      # None if --headless
    train_vec_env = train_env,
    render_freq  = 5_000,       # render every 5k global steps
)
```

Every 5 000 steps it reads the current physics state from all 16 environments and
sends them to the Newton viewer as a spatial **grid of drones** (4 × 4 layout with
4 m spacing). Each drone is coloured by its current distance to its target:

```python
# train_drone.py:125-127
def _dist_color(self, dist: float) -> wp.vec3:
    t = float(np.clip(dist / 2.0, 0.0, 1.0))
    return wp.vec3(t, 1.0 - t * 0.8, 0.0)   # green = close, red = far
```

Green drones are close to their targets; yellow/red drones are far. This gives a
real-time visual diagnostic of training progress without slowing down the loop (the
render only writes to the viewer — it does not block physics).

If `--headless` is passed, `viewer = None` and this callback is a no-op.

#### MetricsCallback

```python
MetricsCallback(log_freq=1_000, window=100)
```

Fires on every step. It reads `info["reward_components"]` and `info["motor_rpms"]`
from every environment at every step, and reads the terminal metrics
(`terminal_dist`, `terminal_upright`, `terminal_ep_len`, `terminal_reward`) whenever
an episode finishes (`done=True`).

All values are stored in `deque(maxlen=100)` rolling buffers. Every 1 000 steps it
flushes the mean of each buffer to TensorBoard via `self.logger.record(...)`.

```python
# train_drone.py:202-217  — _on_step()
for done, info in zip(dones, infos):
    if done and "terminal_dist" in info:
        self._dists.append(info["terminal_dist"])       # at episode end
        self._successes.append(float(d < 0.15))
    for k in self._RC_KEYS:
        v = info.get("reward_components", {}).get(k)
        if v is not None:
            self._rc[k].append(v)                       # every step
    rpms = info.get("motor_rpms")
    if rpms is not None:
        self._mean_rpms.append(float(np.mean(rpms)))    # every step
```

Note that reward components and motor RPMs are logged **every step** (all 3 M of
them contribute to the rolling mean), while terminal metrics are only logged at
**episode boundaries** (roughly every 600–800 steps per env).

#### CurriculumCallback

```python
CurriculumCallback(
    total_timesteps = 3_000_000,
    noise_init      = 0.10,
    noise_final     = 0.02,
)
```

Fires on every step. Computes the curriculum progress `t` and writes it into
every environment:

```python
# train_drone.py:72-77  — _on_step()
t = min(self.num_timesteps / (self._total * 0.5), 1.0)
for env in self.training_env.envs:
    env.curriculum = t
```

`t` reaches 1.0 at **1.5 M steps** (50% of 3 M) and stays there for the rest of
training. This means the reward weights spend the first half of training ramping from
easy to hard, and the second half of training at full precision — giving the policy
time to master tight hovering under the full weight regime.

The `noise_init` / `noise_final` arguments only apply to TD3 (which has explicit
action noise). For PPO the callback still fires, but the `action_noise` branch is
skipped because `model.action_noise` is `None`. PPO's exploration is handled
entirely by the stochastic policy (Gaussian actor) and `ent_coef`.

---

### Stage 5 — One full training run, by the numbers

With the default settings (`--algo ppo --num_envs 16 --total_timesteps 3_000_000`):

| Quantity | Value |
|----------|-------|
| Total env steps | 3 000 000 |
| Steps per rollout (all envs) | 32 768 (= 2048 × 16) |
| Number of rollout cycles | ~91 (= 3 000 000 / 32 768) |
| Gradient steps per rollout | 5 120 (= 10 epochs × 512 mini-batches) |
| Total gradient steps | ~466 560 |
| Checkpoint saves | 60 (every 50k steps) |
| TensorBoard flushes | 3 000 (every 1k steps) |
| Curriculum fully ramped at | 1 500 000 steps (step 46 of 91 rollouts) |
| Estimated wall time (GPU) | ~30 min headless |

---

### Stage 6 — Saving the final model

```python
# train_drone.py:388-390
save_path = f"{algo}_drone_final"
model.save(save_path)
print(f"\nSaved → {save_path}.zip")
```

After all 3 M steps, SB3 serialises the policy and value networks (weights, optimizer
state, hyperparameters) into `ppo_drone_final.zip`. This file is self-contained: the
evaluator loads it with `PPO.load("ppo_drone_final.zip")` and runs the actor network
deterministically (no sampling, no exploration).

---

### Data flow summary

```
                   reset()
                      │ obs (22D)
                      ▼
  ┌──────────────────────────────────┐
  │  PPO actor network               │
  │  obs → [256] → [256] → action μ │
  │  sample: a ~ N(μ, σ)             │
  └──────────────────┬───────────────┘
                     │ action (4D ∈ [-1,1])
                     ▼
  ┌──────────────────────────────────────────────────────┐
  │  DroneEnv.step(action)                               │
  │                                                      │
  │  1. RPM mapping:  n_sp = hover + action × range      │
  │  2. Motor LPF:    n[t] = (1-α)·n[t-1] + α·n_sp     │
  │  3. Warp kernel:  apply thrust + torque to body      │
  │  4. Newton step:  advance rigid-body physics 10ms    │
  │  5. Obs:          [p_err, R_flat, v, ω, a_prev]      │
  │  6. Reward:       sum of 9 components                │
  └──────────────────────────────────────────────────────┘
                     │ (obs, reward, done, info)
                     ▼
  Rollout buffer  ──────►  GAE  ──────►  PPO loss  ──────►  Adam
  (32 768 steps)           A_t           L_clip              Δθ
                                         L_vf
                                         L_ent
```

---

## TensorBoard Signal Reference

All signals are logged by `MetricsCallback` in `train_drone.py`. The callback
accumulates data from the `info` dict returned by `DroneEnv.step()` and flushes
averaged values every 1 000 steps. Each metric is a rolling mean over the last
100 episodes.

```bash
tensorboard --logdir drone_logs
```

---

### metrics/ep_length

**Source:** `info["terminal_ep_len"] = self._step_count` → `MetricsCallback` line `np.mean(self._ep_lens)`

The mean number of steps in an episode. The hard ceiling is `MAX_EPISODE_STEPS = 800`
(8 s at 100 Hz); episodes also end early on crash or arena escape.

| Value | Interpretation |
|-------|---------------|
| Rising toward 800 | Drone is staying alive and approaching the step budget — healthy |
| Suddenly drops | Drone started crashing; cross-check `terminal_upright` |
| Stable at 200–400 | Drone crashes at a consistent point; examine `terminal_dist` to see if it is a navigation or stability failure |

---

### metrics/ep_reward

**Source:** `self._ep_reward` accumulated in `step()` → `np.mean(self._ep_rewards)`

The total undiscounted return per episode (sum of all per-step rewards). With
`MAX_EPISODE_STEPS = 800` and survival = +0.50/step, a drone that only hovers earns
+400. A drone that also hits all waypoints adds arrival (+4.0), hover_bonus (up to
+240), and approach terms on top.

| Value | Interpretation |
|-------|---------------|
| Positive and growing | Survival bonus dominates — drone is alive and making progress |
| Negative | The drone is crashing too fast for survival bonus to accumulate, or entropy is too high (SAC) |
| Plateaus without `success_rate` rising | Drone found a hovering local optimum but is not navigating to the target |

---

### metrics/success_rate

**Source:** `float(terminal_dist < 0.15)` → `np.mean(self._successes)`

Fraction of completed training episodes where the drone ended within 0.15 m of its
single training waypoint. **Training episodes use one waypoint; eval uses four
sequential waypoints.** Training success is a necessary but not sufficient condition
for eval success.

| Value | Interpretation |
|-------|---------------|
| > 0.40 by 2–3 M steps | Well-converging policy |
| Stays near 0 past 1 M steps | Drone is not reaching the target; check `approach` and `pos_c` |
| High here but low in eval | Drone reaches targets from its spawn position but fails after waypoint switches; verify spawn/eval distribution match |

---

### metrics/terminal_dist

**Source:** `info["terminal_dist"] = dist` at episode end → `np.mean(self._dists)`

Mean distance to target at episode end — either from a crash or from the step budget
being exhausted. This is the most direct measure of whether the drone is navigating
to the target.

| Value | Interpretation |
|-------|---------------|
| Falling from ~1.0 toward 0.15 | Curriculum is working; drone converging on target |
| Stuck above 0.30 | Drone is orbiting rather than settling — check `vel_c` spikes and `approach` |
| Bounces between values | Curriculum ramp may be too fast; try `--no_curriculum` |

---

### metrics/terminal_upright

**Source:** `info["terminal_upright"] = R22` where `R22 = 1 − 2(qx² + qy²)`, the
dot product of the drone's up-axis with world Z.

How upright the drone is at episode end. 1.0 = perfectly level, 0 = 90° tilted,
−1 = inverted (crashed upside-down).

| Value | Interpretation |
|-------|---------------|
| Converging toward 1.0 | Drone is learning stable flight |
| Near 0 or negative | Episodes are ending with drone tilted or crashed; cross-check `ep_length` |
| Near 1.0 but `success_rate` is low | Stable flight but poor navigation — increase pos_c weight or check target distribution |

---

### metrics/mean_motor_rpm

**Source:** `info["motor_rpms"]` (4-element list of filtered RPMs) → mean across
motors and steps → `np.mean(self._mean_rpms)`

Average filtered motor speed across the episode. Physical hover RPM ≈ **14 476**.

| Value | Interpretation |
|-------|---------------|
| ≈ 14 476 | Drone is hovering correctly |
| Consistently below 14 200 | Under-thrusting — policy outputs below-hover actions; check `learning_starts` (TD3) or `ent_coef` (SAC) |
| Consistently above 14 800 | Over-thrusting — drone is climbing; check z-escape terminations |
| High variance across runs | Policy not converged |

---

### metrics/rpm_hover_dev

**Source:** `abs(np.mean(self._mean_rpms) - CF_HOVER_RPM)` in `MetricsCallback._flush()`

The absolute deviation of mean motor RPM from the hover equilibrium. Captures
**low-frequency (slow) RPM drift** — sustained over- or under-thrust that builds
up over many steps into an oscillation. Complementary to `act_c` which captures
**high-frequency (step-to-step) jitter**.

| Value | Interpretation |
|-------|---------------|
| < 100 RPM | Healthy — motors close to hover on average |
| 200–500 RPM | Slow oscillation — drone drifts from hover RPM over many steps |
| > 500 RPM | Severe drift; the "up and down everywhere" symptom — likely caused by ungated approach reward or high entropy |

**Diagnosis pair:** If `act_c` is healthy (−0.02 to −0.06) but `rpm_hover_dev` is
high (> 200), the policy makes small consistent RPM changes every step that
accumulate into a slow oscillation. The fix is gating the approach reward and adding
`hover_bonus` to reward settlement.

---

### reward_components/pos_c

**Code:** `pos_c = -C_rp * np.dot(p_err, p_err)` — curriculum ramp C_rp: 0.05 → 1.00

Mean position cost per step. A negative value that should rise toward 0 as the drone
closes in on targets. At convergence with a drone settling at 0.1 m from target, the
expected value is `−1.00 × 0.01 = −0.01`.

Stays deeply negative (−0.3 or below after 2 M steps) → drone is consistently far
from target. Rises steeply after 1–1.5 M steps → curriculum ramp is working.

---

### reward_components/orient_c

**Code:** `orient_c = -_C_RQ * (1.0 - qw**2)` — `_C_RQ = 0.10`, fixed

Mean orientation cost per step. Ranges from 0 (upright) to −0.10 (inverted).
Healthy range: **−0.005 to −0.02**. Values below −0.05 mean the drone spends
significant time tilted and is at risk of crashing.

---

### reward_components/vel_c

**Code:** `vel_c = -C_rv * np.dot(v, v)` — curriculum ramp C_rv: 0.01 → 0.30

Mean velocity cost per step. The clearest oscillation indicator: **large negative
spikes that do not decay** over training mean the drone is bouncing between
overshoot and correction rather than braking before the target. Healthy: mild
negatives during approach, near 0 when settled.

---

### reward_components/ang_c

**Code:** `ang_c = -C_rw * np.dot(w, w)` — curriculum ramp C_rw: 0.001 → 0.05

Mean angular velocity cost per step. Healthy range: **−0.01 to −0.03** once
converged. Large negative values late in training with stable `vel_c` indicate
yaw instability — the drone is spinning in place near the target.

---

### reward_components/act_c

**Code:** `act_c = -C_ra * np.dot(delta_a, delta_a)` — curriculum ramp C_ra: 0.005 → 0.02

Mean action-change cost per step. Penalises RPM jerk (step-to-step change), not
absolute RPM level. Healthy: **−0.02 to −0.06**. Below −0.10 means the policy is
thrashing — large RPM swings every step. Pairs with `rpm_hover_dev` (slow drift vs.
fast jitter).

---

### reward_components/survival

**Code:** `survival = _C_RS = 0.50`, fixed constant

Always **0.50** — logged as a sanity check that the reward pipeline is functioning
correctly. Any deviation indicates a code bug.

---

### reward_components/approach

**Code:**
```python
approach = _APPROACH_COEF * (self.last_dist - dist)  if dist > _APPROACH_GATE  else 0.0
# _APPROACH_COEF = 1.0,  _APPROACH_GATE = 0.25 m
```

The decrease in distance from the previous step, scaled by 1.0, and gated off when
the drone is already within 0.25 m of the target.

Converges toward **0** as training progresses: the drone spends more time near the
target where the gate is active. A policy that oscillates around the target will
also show approach ≈ 0 (approach and retreat cancel) — disambiguate using `vel_c`
(oscillating drone has large `vel_c`; settled drone has `vel_c` near 0) and
`hover_bonus` (settled drone earns positive `hover_bonus`).

---

### reward_components/hover_bonus

**Code:**
```python
speed = float(np.linalg.norm(v))
hover_bonus = _HOVER_BONUS if (dist < 0.15 and speed < _HOVER_SPEED_GATE) else 0.0
# _HOVER_BONUS = 0.15,  _HOVER_SPEED_GATE = 0.5 m/s
```

Per-step bonus for being within 0.15 m of the target **and moving slower than 0.5 m/s**.
Both conditions must hold. Rises from **0 toward a positive value** as the policy
learns to both navigate to and settle at waypoints.

A pure distance gate (`if dist < 0.15`) caused the policy to rush at targets to
collect the bonus (high-speed approaches → vel_c spikes → crash episodes). The speed
gate forces deceleration: you earn the bonus only once you have braked.

If `hover_bonus` is near 0 despite a rising `success_rate`, the drone is passing
through the 0.15 m zone at speed > 0.5 m/s — check `vel_c` for persistent spikes.
If `hover_bonus` is positive but not growing, the drone is settling but coverage of
the training target distribution is still incomplete.

---

### Diagnostic quick-reference

| Symptom | Most informative signals | Likely cause |
|---------|--------------------------|--------------|
| Drone oscillates up/down | `rpm_hover_dev` high, `vel_c` spikes | Approach reward driving overshoot; gate approach at 0.25 m |
| Drone does not reach target | `pos_c` flat and negative, `approach` > 0 | Pos gradient too weak; check curriculum or spawn range |
| Drone crashes early | `ep_length` short, `terminal_upright` low | Stability failure; lower initial noise or `ent_coef` |
| Good training, bad eval | `success_rate` high, eval dist large | Train/eval distribution mismatch; increase spawn range |
| `ep_reward` negative | `ep_reward` < 0, short `ep_length` | Entropy too high (SAC) or `learning_starts` crash-filling buffer (TD3) |
| Drone hovers but does not navigate | `pos_c` flat negative, `hover_bonus` = 0 | Local optimum at spawn position; increase pos_c weight or reduce survival |

---

## Benchmark Results & Algorithm Analysis

### Baseline results — 3 M steps, 16 envs

Evaluated after training each algorithm for 3 M steps on the Level-5.1 RPM
environment with correct Crazyflie physics (27 g, Förster 2015 inertia).
10 episodes × 4 random waypoints (radius 0.5–1.5 m, alt 0.3–1.2 m).

| Metric | PPO | SAC | TD3 |
|--------|-----|-----|-----|
| Per-waypoint success (dist < 0.15 m) | **50 %** | 0 % | 0 % |
| Mean waypoint dist (m) | **0.29** | 1.41 | 1.19 |
| Mean episode length (steps) | **636** | 89 | 167 |
| Mean motor RPM | **14 527** ≈ hover | 12 890 ↓ | 12 147 ↓↓ |
| Training success rate (TensorBoard) | **46 %** | 0 % | 16 % |
| TensorBoard ep_reward (final) | **+122** | −73 | +139 |

### Failure mode diagnostics

The mean motor RPM column is the clearest indicator:

**PPO** — RPM ≈ 14 527 (100.4 % of hover 14 476). The drone learned to hover
precisely. 50 % waypoint success with many near-misses at 0.17–0.27 m; limited by
network capacity and rollout length rather than a fundamental failure.

**SAC — diverged** (mean RPM 12 890, episodes avg 89 steps). Root cause:
`ent_coef="auto"` targets `H = −dim(action) = −4` nats. For a 4-D Gaussian that
corresponds to σ ≈ 1.0 per dimension — RPM noise of ±7 200 RPM. The drone thrashed
between full throttle and near-idle, crashed within seconds, and the TensorBoard
`ep_reward` curve went **negative** throughout training. SAC's entropy term actively
prevented the policy from learning to hover.

**TD3 — generalisation failure** (mean RPM 12 147, 16 % training success but 0 %
eval success). Two compounding problems:
1. `learning_starts=10 000` with uniform random actions in `[−1, 1]` filled the
   replay buffer with crash trajectories. Q-networks learned that all states lead
   to crashes, making the policy pessimistic (outputs below-hover RPM → sinks).
2. Training spawned the drone 0.1–0.5 m from a fixed target; eval places waypoints
   0.5–1.5 m from the drone's reset position. The policy generalised to 16 % in the
   easy training distribution but completely failed on the harder eval range.

### Fixes applied (v2)

#### Fix 1 — Training distribution now matches eval (`drone_gym_env.py`)

`_sample_random_target()` draws from `radius ∈ [0.5, 1.5] m, alt ∈ [0.3, 1.2] m`,
identical to the evaluator. Spawn range scales with curriculum:

```
pos_range = 0.1 + 1.4 × curriculum   (was 0.1 + 0.4 ×)
```

At `curriculum = 0` the drone spawns 0.1 m from target (safe hover).
At `curriculum = 1` it spawns up to 1.5 m away — the same maximum the evaluator uses.

#### Fix 2 — SAC entropy pinned (`train_drone.py`)

```python
# Before (diverges):
ent_coef="auto", target_entropy="auto"   # → σ ≈ 1.0 per dim, RPM thrash

# After (stable):
ent_coef=0.005                           # fixed small; entropy stays low
learning_starts=1_000                    # use policy early, before buffer fills
```

#### Fix 3 — TD3 hover bootstrap (`train_drone.py`)

```python
# Before (crash-fills buffer):
learning_starts=10_000    # 10k uniform-random actions → many crashes
action_noise σ = 0.30     # ±2 170 RPM noise → frequent motor stall

# After (stays near hover from step 1):
learning_starts=0         # MLP tanh outputs ≈ 0 → hover RPM from episode 1
action_noise σ = 0.10     # ±723 RPM — enough to explore, not enough to crash
```

With `learning_starts=0` the randomly-initialised policy outputs near-zero actions
(tanh of small random weights ≈ 0), which map to ≈ hover RPM. The drone stays
airborne from episode 1, buffer fills with useful hovering experience, and Q-values
remain optimistic.

#### Fix 4 — PPO capacity and credit assignment (`train_drone.py`)

| Hyperparameter | Before | After | Why |
|----------------|--------|-------|-----|
| `n_steps` | 2 048 | **4 096** | More on-policy data before each update |
| `batch_size` | 64 | **256** | Larger mini-batches reduce gradient noise |
| `n_epochs` | 10 | **20** | More gradient passes per rollout |
| `gae_lambda` | 0.95 | **0.98** | Better long-horizon credit assignment |
| `net_arch` | [256, 256] | **[512, 512]** | More capacity for navigation |
| `ent_coef` | 0 | **0.005** | Keeps exploration alive in later training |
| `device` | gpu | **cpu** | SB3 MLP policy is faster on CPU |

### Diagnosing your own runs with TensorBoard

If you observe a new training run failing, use these signals:

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `mean_motor_rpm` < 14 000 and falling | Policy outputting below-hover RPMs | Check ent_coef (SAC) or learning_starts (TD3) |
| `ep_length` < 100 steps from the start | Buffer filled with crashes | Reduce learning_starts; reduce initial noise σ |
| `ep_reward` goes negative | Entropy too high (SAC) | Set `ent_coef` to a fixed small value |
| `success_rate` in training > 0 but eval = 0 % | Training/eval distribution mismatch | Verify spawn range covers eval waypoint distances |
| PPO `success_rate` plateaus early | Network too small or rollouts too short | Increase `n_steps`, `net_arch`, `n_epochs` |

---

## Evaluation

### Basic usage

```bash
# Evaluate TD3 final model
python eval_drone.py --model td3_drone_final --algo td3

# Evaluate a checkpoint
python eval_drone.py --model checkpoints/td3_hover_500000_steps --algo td3

# More episodes, fixed seed
python eval_drone.py --model td3_drone_final --algo td3 --num_episodes 20 --seed 0
```

### Multi-waypoint episodes

Each episode visits **4 random waypoints** without resetting the drone between them.
Waypoints are sampled from the training distribution:

```
angle    ∈ [0, 2π)
radius   ∈ [0.5, 1.5] m
altitude ∈ [0.3, 1.2] m
```

A waypoint slot ends when:
- `dist < 0.15 m` → **success**, move to next waypoint immediately
- Step budget exhausted (default **200 steps = 2.0 s**) → **fail**, move on
- Drone crashes → **fail**, episode ends early

> The step budget was increased from 1.5 s to **2.0 s** to account for the motor
> LPF lag at Level-5.1 control: the drone requires ~15 steps just to reach the
> commanded RPM after a waypoint switch.

### Evaluation output

```
CF hover ≈ 14476 RPM  |  max 21702 RPM  |  action space: Level-5.1 (direct RPM)

  ep   1/10 | rew=  312.44 | len= 600 | rpm≈14501 | wp1(✓,0.08m)  wp2(✗,0.31m)  wp3(✓,0.12m)  wp4(✓,0.07m) | [3/4]
  ep   2/10 | rew=  401.17 | len= 750 | rpm≈14489 | wp1(✓,0.06m)  wp2(✓,0.09m)  wp3(✓,0.11m)  wp4(✓,0.08m) | [ALL✓]
  ...

  ──────────────────────────────────────────────────────────────────
    Evaluation summary  [TD3]  — Level-5.1 RPM control
  ──────────────────────────────────────────────────────────────────
    Episodes            : 10
    Waypoints / ep      : 4  (random, r=0.5–1.5 m, z=0.3–1.2 m)
    Step budget / wp    : 200 steps = 2.0 s
    Mean reward         : 374.21 ± 48.33
    Mean ep length      : 571.3 steps
    Mean wpts reached   : 3.40 / 4
    Per-waypoint succ   : 85.0%  (dist < 0.15 m)
    All-waypoints succ  : 40.0%  (all wpts hit)
    Mean waypoint dist  : 0.0921 m
    Mean motor RPM      : 14501 RPM  (100.2% of hover RPM 14476)
    Max motor RPM       : 21702 RPM
  ──────────────────────────────────────────────────────────────────
```

### All evaluation arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `td3_drone_final` | Path to model zip (without `.zip`) |
| `--algo` | `td3` | Algorithm used to train: `ppo`, `sac`, `td3` |
| `--num_episodes` | `10` | Number of evaluation episodes |
| `--waypoints_per_ep` | `4` | Waypoints per episode (1–8) |
| `--steps_per_wp` | `200` | Step budget per waypoint (2.0 s at 100 Hz) |
| `--seed` | `42` | RNG seed for reproducible waypoint generation |
| `--stochastic` | off | Use stochastic (non-deterministic) policy |

---

## Environment API Reference

`DroneEnv` is a standard `gymnasium.Env`:

```python
env = DroneEnv(
    render_mode    = "human" | None,  # "human" enables Newton viewer
    viewer         = viewer,           # Newton viewer object (or None)
    random_targets = True,             # randomly pick from TARGETS at reset
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
        "approach":    float,   # distance shaping, gated at 0.25 m
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
RPM²→thrust curve, motor delay, and rotor speed dynamics. The paper shows these
are the primary sources of the reality gap. Controlling directly at RPM forces the
policy to model all of them, enabling zero-shot sim-to-real transfer without domain
randomisation.

### Why correct Crazyflie mass and inertia?

The previous version used cross-arm geometry with density=1750 kg/m³, producing a
simulated mass of ~560 g — **20× heavier than the real drone**. Any policy trained
in that environment would receive incorrect force/inertia feedback and would not
transfer to real hardware. Fixing mass (27 g) and inertia (Förster 2015) via explicit
Newton `add_body(mass=..., inertia=...)` removes this gap entirely.

### Why `‖Δa‖²` instead of `‖a‖²` for action cost?

The paper's reward term is `−C_ra‖a − a_prev‖²`, penalising **changes** in the RPM
setpoint (jerk) rather than the absolute setpoint level. With RPM-level control this
is more meaningful: a sustained high-RPM climb command should not be penalised, but
a rapid oscillation between high and low should.

### Why motor LPF in RPM space?

The filter is applied to RPM (not thrust fraction). This preserves the nonlinear
F=KT·n² relationship in the filtered output. If the LPF were applied to a linear
thrust fraction, the nonlinearity would be hidden from the physics engine.

### Why `learning_starts = 0` for TD3?

The intuitive fix for a crash-heavy environment is a large `learning_starts`
buffer — but the opposite is true here. SBX's TD3 fills the buffer with
**uniform random actions** in `[−1, 1]` during `learning_starts`. With RPM-level
control that means random RPMs from 7 250 to 21 702 — the drone crashes almost
every episode. Q-networks then learn that all (state, action) pairs lead to
crashes, making the policy pessimistic: it outputs below-hover RPMs to "avoid"
the bad states, which causes more crashes.

The fix is `learning_starts=0` with **small initial noise (σ=0.10)**. A randomly
initialised MLP with tanh outputs ≈ 0, which maps to ≈ hover RPM. The drone stays
airborne from episode 1, the buffer fills with useful hovering experience, and
Q-values start optimistic. Exploration noise σ=0.10 adds ±723 RPM variation —
enough to discover control directions without causing immediate crashes.

### Why fixed `ent_coef` for SAC?

SAC's `ent_coef="auto"` targets `H = −dim(action) = −4` nats, which forces the
policy toward maximum stochasticity regardless of the task. For RPM control this
means σ ≈ 1.0 per action dimension = ±7 200 RPM noise — the drone thrashes between
full throttle and motor stall. Using `ent_coef=0.005` (fixed) keeps entropy as a
mild regulariser without overwhelming the task reward.

### Why action noise decays with curriculum?

High initial noise allows the policy to discover diverse control strategies before
the reward weights tighten. For TD3, σ decays from 0.10 → 0.02 in lockstep with
the curriculum ramp: as position penalties grow and the drone must navigate precisely,
large RPM noise would cause crashes and push the policy off the precision it has
learned.

---

## Dependencies

```bash
pip install gymnasium numpy warp-lang newton stable-baselines3 sbx-rl tensorboard
```

| Package | Role |
|---------|------|
| `newton` | GPU rigid-body simulator (NVIDIA Warp backend) |
| `warp-lang` | CUDA kernel execution via `@wp.kernel` |
| `gymnasium` | Standard RL environment interface |
| `stable_baselines3` | PPO + `NormalActionNoise`, callbacks, vec envs |
| `sbx` | SAC and TD3 implementations (JAX-accelerated) |
| `tensorboard` | Training monitoring |

---

## Reference

Eschmann, J., Albani, D., Loianno, G. (2024).  
**Learning to Fly in Seconds.**  
*IEEE Robotics and Automation Letters.* arXiv:2311.13081.

See `LearningtoFlyinSeconds.pdf` for the full paper.
