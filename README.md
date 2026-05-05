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

The reward follows paper Eq. 1 — quadratic costs plus a survival bonus:

```
r(s, a, s') = − C_rp ‖p_err‖²
              − C_rq (1 − qw²)
              − C_rv ‖v‖²
              − C_rω ‖ω‖²
              − C_ra ‖Δa‖²        ← action-change penalty (jerk)
              + C_rs
              + crash_penalty      (−2.0 when z < 0.05 m)
              + arrival_bonus      (+1.0 first time dist < 0.1 m)
```

### Term-by-term explanation

| Term | Purpose |
|------|---------|
| `−C_rp ‖p_err‖²` | Penalises distance from target |
| `−C_rq (1 − qw²)` | Penalises tilt; `qw = cos(θ/2)`, zero when upright |
| `−C_rv ‖v‖²` | Encourages zero velocity (stable hover at target) |
| `−C_rω ‖ω‖²` | Discourages spinning and oscillation |
| `−C_ra ‖Δa‖²` | Penalises **changes** between consecutive actions (jerk regularisation) |
| `+C_rs` | Constant `+0.5` per step; survival is always better than crashing |
| `−2.0` | One-time crash penalty when `z < 0.05 m` |
| `+1.0` | One-time arrival bonus first time `dist < 0.1 m` per target |

> **Note:** The action cost penalises `‖Δa‖²` (action *change*, i.e. jerk) rather
> than `‖a‖²` (action magnitude). This matches paper Eq. 1 (`‖a − a_rab‖²` where
> `a_rab` is the previous action) and promotes smooth RPM transitions.

### Reward Curriculum

Weights ramp linearly from conservative → strict over the first 50 % of training:

| Weight | Init  | Target | Controls |
|--------|-------|--------|----------|
| `C_rp` | 0.05  | 1.00   | Position precision |
| `C_rv` | 0.010 | 0.30   | Velocity damping |
| `C_rω` | 0.001 | 0.05   | Angular stability |
| `C_ra` | 0.005 | 0.02   | Action smoothness |
| `C_rq` | 0.10  | 0.10   | Orientation (fixed) |
| `C_rs` | 0.50  | 0.50   | Survival (fixed) |

**Why curriculum matters:** At `curriculum = 0` the position cost is tiny. The
survival bonus dominates — the drone gets positive reward just by staying airborne.
At `curriculum = 1` the position penalty grows 20×, forcing tight navigation. Without
curriculum the heavy early position penalty makes crashing at step 1 "optimal".

The `CurriculumCallback` updates `env.curriculum` every step:

```python
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

Episodes also truncate after `MAX_EPISODE_STEPS = 500` steps (5 s at 100 Hz).

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

| Panel | Metric | Meaning |
|-------|--------|---------|
| `metrics/` | `success_rate` | Fraction of episodes ending with dist < 0.15 m |
| `metrics/` | `terminal_dist` | Distance to target at episode end |
| `metrics/` | `terminal_upright` | `R[2,2]` at end (1=level, −1=inverted) |
| `metrics/` | `ep_reward` | Total return per episode |
| `metrics/` | `ep_length` | Steps per episode |
| `metrics/` | `mean_motor_rpm` | Average filtered motor speed across episode |
| `metrics/` | `rpm_hover_dev` | Absolute deviation from hover RPM (≈14 476) |
| `reward_components/` | `pos_c` | Mean position cost per step |
| `reward_components/` | `orient_c` | Mean orientation cost per step |
| `reward_components/` | `vel_c` | Mean velocity cost per step |
| `reward_components/` | `ang_c` | Mean angular velocity cost per step |
| `reward_components/` | `act_c` | Mean action-change cost per step |
| `reward_components/` | `survival` | Constant 0.5 (sanity check) |

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
        "pos_c": ..., "orient_c": ..., "vel_c": ...,
        "ang_c": ..., "act_c":    ..., "survival": ...
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

### Why `learning_starts = 10 000`?

With Level-5.1 RPM control and random initial actions, the early exploration
phase produces many crashes (motors at full throttle or near-zero). A larger
initial buffer ensures the replay buffer is not dominated by crash trajectories
before learning begins.

### Why action noise decays with curriculum?

High exploration noise early on is essential for discovering how to maintain
altitude with the RPM²→thrust relationship. As the policy matures and the
reward weights tighten, large random RPM deviations cause crashes. Decaying
σ from 0.30 to 0.05 in lockstep with the curriculum keeps exploration
productive throughout training.

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
