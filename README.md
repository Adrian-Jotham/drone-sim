# Drone RL — Learning to Fly with PPO, SAC, and TD3

A reinforcement learning system for training a quadrotor position controller inside the
[Newton](https://github.com/newton-physics/newton) GPU-native rigid-body simulator.
The observation and action design is based on
**"Learning to Fly in Seconds"** (Eschmann, Albani, Loianno — RAL 2024).

---

## Repository Layout

```
dronesim/
├── drone_gym_env.py          # Gymnasium environment (physics + obs + reward)
├── train_drone.py            # Unified training script: PPO | SAC | TD3
├── eval_drone.py             # Multi-waypoint evaluation script
├── LearningtoFlyinSeconds.pdf # Reference paper
└── drone_logs/               # TensorBoard logs (created at first training run)
    ├── ppo_1/
    ├── sac_1/
    └── td3_1/
```

---

## Quick Start

```bash
# Train SAC (recommended — best sample efficiency)
python train_drone.py --algo sac

# Train all three for comparison
python train_drone.py --algo ppo  --headless
python train_drone.py --algo sac  --headless
python train_drone.py --algo td3  --headless

# Compare convergence in a single TensorBoard session
tensorboard --logdir drone_logs

# Evaluate a trained model
python eval_drone.py --model sac_drone_final --algo sac
```

---

## System Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │              DroneEnv (drone_gym_env.py)      │
                 │                                               │
  action (4D) ──►│  Motor LPF  ──►  Newton Physics  ──►  Obs   │──► reward
  [-1, 1]        │  τ = 0.15 s      100 Hz rigid-body   (22D)  │
                 └──────────────────────────────────────────────┘
                          ▲                        │
                          │    RL Policy           │
                          │  (PPO / SAC / TD3)     │
                          └────────────────────────┘
```

The environment wraps Newton's GPU-accelerated rigid-body simulator as a standard
Gymnasium interface. The RL policy runs on the CPU/JAX side and sends motor commands;
Newton propagates physics at 100 Hz and returns the next state.

---

## Physics Simulation

### Drone Model

The quadrotor body is two thin carbon-fibre cross-bars modelled as box shapes:

```
      prop[0]            prop[2]
        ↑                  ↑
  (0, +s, 0)          (+s, 0, 0)
        \                /
         ●──────────────●     ← drone body (two intersecting boxes)
        /                \
  (0, -s, 0)          (-s, 0, 0)
        ↓                  ↓
      prop[1]            prop[3]
```

- **Half-span** `s = 0.2 m`
- **Body density** `ρ_body = 1750 kg/m³`  →  total mass ≈ 0.56 kg
- Propellers alternate turning direction (±1) to cancel net yaw torque at hover

### Propeller Aerodynamics

Each propeller's maximum thrust and torque follow the actuator-disk aerodynamic scaling law:

```
F_max = C_T · ρ_air · n² · D⁴
τ_max = C_P · ρ_air · n² · D⁵ / (2π)
```

where:
- `C_T = 0.109919` — thrust coefficient
- `C_P = 0.040164` — power coefficient
- `D   = 0.2286 m` — propeller diameter
- `n   = 6396.667 / 60 ≈ 106.6 rev/s` — physical RPS at max RPM
- `ρ_air = 1.225 kg/m³` — air density at sea level

At these values `F_max ≈ 4.17 N` per propeller. Hover requires `mg/4 ≈ 1.37 N` per
propeller, giving **hover motor fraction ≈ 0.33** (33 % of maximum thrust).

### Motor Dynamics (First-Order Low-Pass Filter)

Real brushless motors do not respond instantaneously to commands. The paper
(§IV) identifies a time constant τ ≈ 0.15 s for the Crazyflie's motors.
This is implemented as a discrete first-order IIR filter applied to every step:

```
α = SIM_DT / τ = 0.01 / 0.15 ≈ 0.067

motor_frac[t+1] = (1 − α) · motor_frac[t] + α · setpoint[t]
```

This causes the actual motor speed to lag behind the commanded setpoint by roughly
15 steps (≈ 0.15 s). The policy must therefore learn to issue commands in advance —
otherwise it will always overshoot or undershoot. Including the previous action in
the observation (action history) is what gives the policy the information it needs
to compensate for this delay.

### Physics Integrator

Newton's `SolverSemiImplicit` advances the rigid-body dynamics at 100 Hz:

```
v[t+1] = v[t] + dt · M⁻¹ · (F_prop + F_gravity − D · v[t])
q[t+1] = q[t] ⊕ (dt · v[t+1])
```

where `q ∈ SE(3)` is the body transform (position + quaternion) and `⊕` is the
SE(3) integration operator. All four propeller wrenches are accumulated into
`body_f` by a single GPU kernel before each solver step.

---

## Observation Space (22-D)

The observation design closely follows the paper's actor observation
`o_a = {p, R, v, ω, H}`:

| Slice    | Symbol   | Dim | Description |
|----------|----------|-----|-------------|
| `[0:3]`  | `p_err`  | 3   | Position error = `pos − target` (world frame, metres) |
| `[3:12]` | `R_flat` | 9   | Drone rotation matrix, row-major flattened |
| `[12:15]`| `v`      | 3   | Linear velocity, world frame (m/s) |
| `[15:18]`| `ω`      | 3   | Angular velocity, body frame (rad/s) |
| `[18:22]`| `a_prev` | 4   | Previous action sent to the motors |

**Total: 22 dimensions.**

### Why position error, not absolute position?

The policy sees `p_err = pos − target` instead of raw position. This means the
network always "thinks" it is flying to the origin regardless of where the actual
target is. During evaluation or deployment you can move the target anywhere by
calling `env.set_target(new_pos)` — the same trained policy handles it without
retraining.

### Why rotation matrix, not quaternion?

A unit quaternion `q` and its negation `−q` represent the same physical rotation
(double-coverage of SO(3)). A neural network fed raw quaternions must implicitly
learn to handle this ambiguity, which wastes capacity. The 3×3 rotation matrix
has no such ambiguity and is the representation used in the paper.

```python
# Quaternion [qx, qy, qz, qw]  →  9-D row-major rotation matrix
R = [[1−2(qy²+qz²),  2(qxqy−qzqw),  2(qxqz+qyqw)],
     [2(qxqy+qzqw),  1−2(qx²+qz²),  2(qyqz−qxqw)],
     [2(qxqz−qyqw),  2(qyqz+qxqw),  1−2(qx²+qy²)]]
```

### Why action history?

The motor LPF introduces a delay of ~15 steps between a command and its full
physical effect. Without knowing what was commanded recently, the policy cannot
predict the drone's near-future response. Adding the last action (N_H = 1) gives
the policy a window into what thrust is currently "in flight" through the filter,
partially restoring observability of the delayed motor state.

### Optional observation noise

When training with `--obs_noise`, Gaussian noise is added to simulate imperfect
onboard sensors (IMU noise, position estimation error):

| Component | Noise σ |
|-----------|---------|
| `p_err`   | 0.01 m  |
| `v`       | 0.01 m/s|
| `ω`       | 0.05 rad/s |

---

## Action Space (4-D)

Each action component is a **normalised motor setpoint** in `[−1, 1]`.
The mapping to physical motor fraction uses a hover-centred linear map:

```
setpoint_i = clip(HOVER_FRAC + action_i × THRUST_RANGE, 0.05, 1.0)

HOVER_FRAC   = 0.33   # action = 0 → stable hover
THRUST_RANGE = 0.33   # action = ±1 → 0 % or 66 % thrust
```

This ensures the exploration noise used by SAC/TD3 is centred around hover rather
than zero thrust, making early random policies much more likely to stay airborne.
The setpoint then feeds the motor LPF filter described above.

---

## Reward Function

The reward follows paper Eq. 1 — a sum of quadratic costs plus a constant survival
bonus:

```
r(s, a, s') = − C_rp ‖p_err‖²
              − C_rq (1 − qw²)
              − C_rv ‖v‖²
              − C_rω ‖ω‖²
              − C_ra ‖a‖²
              + C_rs
              + crash_penalty
              + arrival_bonus
```

### Term-by-term explanation

| Term | Symbol | Purpose |
|------|--------|---------|
| `−C_rp ‖p_err‖²` | Position cost | Penalises distance from target; zero only at exact target |
| `−C_rq (1 − qw²)` | Orientation cost | `qw = cos(θ/2)`; equals 0 when upright, 1 when 180° flipped; punishes tilt |
| `−C_rv ‖v‖²` | Velocity cost | Encourages zero velocity at target (stable hover) |
| `−C_rω ‖ω‖²` | Angular velocity cost | Discourages spinning and oscillation |
| `−C_ra ‖a‖²` | Action regularisation | Penalises aggressive motor commands; promotes smooth flight |
| `+C_rs` | Survival bonus | Constant `+0.5` per step; makes surviving clearly better than crashing |
| `−2.0` (crash) | Crash penalty | One-time penalty when `z < 0.05 m` |
| `+1.0` (arrival) | Arrival bonus | One-time bonus first time `dist < 0.1 m` per target |

### Reward Curriculum

The C values are not fixed — they are linearly ramped from conservative initial
values to strict target values over the first 50 % of training:

| Weight  | Init  | Target | Controls |
|---------|-------|--------|----------|
| `C_rp`  | 0.05  | 1.00   | Position precision |
| `C_rv`  | 0.010 | 0.30   | Velocity damping |
| `C_rω`  | 0.001 | 0.05   | Angular stability |
| `C_ra`  | 0.005 | 0.02   | Action smoothness |
| `C_rq`  | 0.10  | 0.10   | Orientation (fixed) |
| `C_rs`  | 0.50  | 0.50   | Survival (fixed) |

**Why curriculum matters:**

At `curriculum = 0`, position cost is tiny. The survival bonus dominates, so even
a drone that hovers far from the target still gets net positive reward and has no
incentive to crash. This allows the policy to learn basic flight stability before
worrying about navigation accuracy.

As `curriculum → 1`, the position penalty grows by 20× and velocity penalty by 30×.
Now the policy is forced to fly precisely to the target and hold position with low
velocity. Without curriculum, the high position cost at early training typically
causes the drone to crash immediately (it is penalised so heavily for being far away
that crashing early to end the episode is "optimal").

The `CurriculumCallback` in `train_drone.py` updates `env.curriculum` every step:

```python
curriculum = min(num_timesteps / (total_timesteps * 0.5), 1.0)
```

Disable with `--no_curriculum` for ablation experiments.

---

## Termination Conditions

An episode ends early (`terminated = True`) when:

| Condition | Reason |
|-----------|--------|
| `z < 0.05 m` | Ground impact |
| `z > 6.0 m`  | Escaped upward |
| `R22 < −0.5` | Drone more than ~120° inverted — unrecoverable |
| `dist > 4.0 m` | Flew out of the arena |

Episodes also truncate (`truncated = True`) after `MAX_EPISODE_STEPS = 500`
steps (5 seconds at 100 Hz) regardless of position.

---

## Reinforcement Learning Algorithms

Three algorithms are supported, each with different trade-offs:

### PPO — Proximal Policy Optimisation (`stable_baselines3`)

**Type:** On-policy  
**Update rule:** Collects `n_steps × n_envs` transitions, then performs multiple
gradient descent epochs with a clipped surrogate objective to prevent destructively
large policy updates.

| Hyperparameter | Value | Why |
|----------------|-------|-----|
| `n_steps`      | 2048  | Rollout buffer per env before each update |
| `n_epochs`     | 10    | Gradient passes over each rollout batch |
| `batch_size`   | 64    | Mini-batch size during each epoch |
| `gae_lambda`   | 0.95  | GAE advantage estimator bias-variance trade-off |
| `clip_range`   | 0.2   | Maximum relative policy change per update |
| `net_arch`     | [256, 256] | Two hidden layers |

**Pros:** Stable training, good wall-clock efficiency on GPU with many parallel envs.  
**Cons:** Requires more total environment interactions than off-policy methods; cannot
reuse experience.

### SAC — Soft Actor-Critic (`sbx`, JAX-accelerated)

**Type:** Off-policy, entropy-regularised  
**Update rule:** Maximises a modified objective that includes an entropy bonus
`α · H(π)`, encouraging the policy to remain stochastic and explore diverse
behaviours. The entropy temperature `α` is tuned automatically.

| Hyperparameter | Value | Why |
|----------------|-------|-----|
| `buffer_size`  | 500 000 | Replay buffer — large enough to break temporal correlation |
| `batch_size`   | 256   | Samples per gradient step |
| `learning_starts` | 5 000 | Random exploration before learning begins |
| `tau`          | 0.005 | Soft target network update rate |
| `ent_coef`     | "auto" | Automatic entropy temperature tuning |
| `target_entropy` | "auto" | Set to −dim(A) = −4 |

**Pros:** Excellent sample efficiency, smooth training, naturally handles
multi-modal action distributions.  
**Cons:** More hyperparameters; occasional instability at very early training.

### TD3 — Twin Delayed Deep Deterministic (`sbx`, JAX-accelerated)

**Type:** Off-policy, deterministic  
**Update rule:** Improves DDPG with three stabilising tricks:
1. **Twin critics** — uses the minimum of two Q-value estimates to reduce
   overestimation bias.
2. **Delayed policy update** — updates the actor less frequently than the critics
   (every 2 critic steps) to reduce variance in policy gradient estimates.
3. **Target policy smoothing** — adds noise to target actions during critic updates,
   preventing the policy from exploiting narrow Q-value spikes.

| Hyperparameter | Value | Why |
|----------------|-------|-----|
| `buffer_size`  | 500 000 | Same as SAC |
| `batch_size`   | 256   | Same as SAC |
| `tau`          | 0.005 | Same as SAC |

**Pros:** Stable training for continuous control, low variance.  
**Cons:** No automatic exploration tuning; deterministic policy can get stuck in
local optima in early training.

### Algorithm Comparison

```
Sample efficiency:   SAC > TD3 >> PPO
Stability:           TD3 ≈ PPO > SAC (early training)
Final performance:   SAC ≈ TD3 > PPO (empirically on hover tasks)
Wall-clock (GPU):    SBX (SAC/TD3) >> SB3 (PPO)  — JAX vs PyTorch
```

From our training runs (1 M steps, 4 envs):

| Algo | Success rate | Terminal dist | Ep length |
|------|-------------|---------------|-----------|
| SAC  | ~91 %       | ~0.09 m       | 400 steps |
| TD3  | ~5 %        | ~0.96 m       | 396 steps |
| PPO  | TBD         | TBD           | TBD       |

---

## Training

### Basic usage

```bash
# Recommended: SAC with curriculum (default settings)
python train_drone.py --algo sac

# Longer run for better convergence
python train_drone.py --algo sac --total_timesteps 3000000

# All three algorithms headless (no OpenGL window)
python train_drone.py --algo ppo --headless
python train_drone.py --algo sac --headless
python train_drone.py --algo td3 --headless
```

### All training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--algo` | `sac` | Algorithm: `ppo`, `sac`, or `td3` |
| `--num_envs` | `4` | Parallel training environments |
| `--total_timesteps` | `1 500 000` | Total environment steps |
| `--learning_rate` | `3e-4` | Adam learning rate |
| `--gamma` | `0.99` | Discount factor |
| `--checkpoint_freq` | `50 000` | Save checkpoint every N steps |
| `--checkpoint_dir` | `checkpoints` | Directory for checkpoints |
| `--obs_noise` | off | Add Gaussian sensor noise |
| `--no_curriculum` | off | Disable reward curriculum |
| `--headless` | off | No OpenGL viewer (newton built-in flag) |

### TensorBoard monitoring

All algorithms write to `./drone_logs/` with automatic subdirectory naming
(`sac_1/`, `td3_1/`, `ppo_1/`). Open a single TensorBoard to compare all three:

```bash
tensorboard --logdir drone_logs
```

Tracked metrics:

| Panel | Metric | Meaning |
|-------|--------|---------|
| `metrics/` | `success_rate` | Fraction of episodes ending with dist < 0.15 m |
| `metrics/` | `terminal_dist` | Final distance to target at episode end |
| `metrics/` | `terminal_upright` | `R[2,2]` at episode end (1 = level, −1 = inverted) |
| `metrics/` | `ep_reward` | Total return per episode |
| `metrics/` | `ep_length` | Steps per episode |
| `reward_components/` | `pos_c` | Mean position cost per step |
| `reward_components/` | `orient_c` | Mean orientation cost per step |
| `reward_components/` | `vel_c` | Mean velocity cost per step |
| `reward_components/` | `ang_c` | Mean angular velocity cost per step |
| `reward_components/` | `act_c` | Mean action regularisation per step |
| `reward_components/` | `survival` | Constant 0.5 (sanity check) |

### Output files

After training completes, the model is saved as a zip:
```
sac_drone_final.zip
td3_drone_final.zip
ppo_drone_final.zip
```

Mid-training checkpoints are saved to `checkpoints/`:
```
checkpoints/sac_drone_50000_steps.zip
checkpoints/sac_drone_100000_steps.zip
...
```

---

## Evaluation

### Basic usage

```bash
# Evaluate SAC final model
python eval_drone.py --model sac_drone_final --algo sac

# Evaluate a mid-training checkpoint
python eval_drone.py --model checkpoints/sac_drone_500000_steps --algo sac

# More episodes, fixed seed for reproducibility
python eval_drone.py --model sac_drone_final --algo sac --num_episodes 20 --seed 0
```

### Multi-waypoint episodes

By default each episode visits **4 random waypoints** without resetting the drone
between them. This tests the policy's ability to navigate a sequence of positions
from whatever state it arrives in — a much harder test than single-target evaluation.

Waypoints are sampled uniformly from the training distribution:
```
angle  ∈ [0, 2π)
radius ∈ [0.5, 1.5] m
altitude ∈ [0.3, 1.2] m
```

A waypoint slot ends when:
- `dist < 0.15 m` → **success**, immediately move to next waypoint
- Step budget exhausted (default 150 steps = 1.5 s) → **fail**, move on
- Drone crashes → **fail**, episode ends early

### Evaluation output

```
  ep   1/10 | rew=  312.44 | len= 520 | wp1(✓,0.08m)  wp2(✗,0.31m)  wp3(✓,0.12m)  wp4(✓,0.07m) | [3/4]
  ep   2/10 | rew=  401.17 | len= 600 | wp1(✓,0.06m)  wp2(✓,0.09m)  wp3(✓,0.11m)  wp4(✓,0.08m) | [ALL✓]
  ...

  ────────────────────────────────────────────────────────────────
    Evaluation summary  [SAC]
  ────────────────────────────────────────────────────────────────
    Episodes            : 10
    Waypoints / ep      : 4  (random, r=0.5–1.5 m, z=0.3–1.2 m)
    Mean reward         : 374.21 ± 48.33
    Mean ep length      : 571.3 steps
    Mean wpts reached   : 3.40 / 4
    Per-waypoint succ   : 85.0%  (dist < 0.15 m)
    All-waypoints succ  : 40.0%  (all wpts hit)
    Mean waypoint dist  : 0.0921 m
  ────────────────────────────────────────────────────────────────
```

### All evaluation arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `sac_drone_final` | Path to model zip (without `.zip`) |
| `--algo` | `sac` | Algorithm used to train: `ppo`, `sac`, `td3` |
| `--num_episodes` | `10` | Number of evaluation episodes |
| `--waypoints_per_ep` | `4` | Waypoints per episode (1–8) |
| `--steps_per_wp` | `150` | Step budget per waypoint (1.5 s at 100 Hz) |
| `--seed` | `42` | RNG seed for reproducible waypoint generation |
| `--stochastic` | off | Use stochastic (non-deterministic) policy |

---

## Environment API Reference

`DroneEnv` is a standard `gymnasium.Env` with a few extra methods:

```python
env = DroneEnv(
    render_mode   = "human" | None,  # "human" enables Newton viewer
    viewer        = viewer,           # Newton viewer object (or None)
    random_targets = True,            # randomly pick from TARGETS at reset
    obs_noise     = False,            # add Gaussian sensor noise
    curriculum    = 0.0,             # reward weight scale [0=easy, 1=hard]
)

obs, info = env.reset()              # standard Gymnasium reset
obs, rew, term, trunc, info = env.step(action)  # standard step

env.set_target(np.array([x, y, z])) # switch waypoint mid-episode
obs = env.get_obs()                  # re-read obs after set_target
```

The `info` dict returned by `step()` always contains:

```python
{
    "dist":    float,   # distance to current target (m)
    "upright": float,   # R[2,2] — 1=level, −1=inverted
    "z":       float,   # altitude (m)
    "reward_components": {
        "pos_c": ..., "orient_c": ..., "vel_c": ...,
        "ang_c": ..., "act_c": ..., "survival": ...
    }
}
```

At episode end (`terminated or truncated`), it additionally contains:
```python
{
    "terminal_dist":    float,
    "terminal_upright": float,
    "terminal_ep_len":  int,
    "terminal_reward":  float,
}
```

---

## Dependencies

```bash
pip install gymnasium numpy warp-lang newton stable-baselines3 sbx-rl
```

| Package | Role |
|---------|------|
| `newton` | GPU rigid-body simulator (uses NVIDIA Warp) |
| `warp-lang` | CUDA kernel execution and automatic differentiation |
| `gymnasium` | Standard RL environment interface |
| `stable_baselines3` | PPO implementation |
| `sbx` | SAC and TD3 implementations (JAX-accelerated) |
| `tensorboard` | Training monitoring |

---

## Design Decisions

### Why `p_err` and not goal-conditioned RL?

Goal-conditioned RL typically concatenates the goal position to the observation.
Here the goal is implicit: since the obs is always relative to the current target
(`p_err = pos − target`), the policy learns a single function "go to where p_err = 0"
that generalises to any target position at inference time by simply changing
`env._target` or calling `env.set_target(...)`.

### Why motor LPF instead of direct thrust?

Without motor delay the policy can make perfectly sharp throttle transitions. The real
Crazyflie's motors take ~0.15 s to spin up or down. Training without this delay
produces policies that issue commands the hardware physically cannot follow, causing
large tracking errors when deployed. The LPF during training forces the policy to
account for this inertia.

### Why curriculum?

Without curriculum, the quadratic position penalty dominates immediately. A drone at
2 m distance from target receives `−C_rp × 4 = −4` per step, while the survival bonus
gives only `+0.5`. Total is `−3.5` per step. After 500 steps that is `−1750` total,
far worse than crashing at step 1 (only `−2`). The optimal early policy is to crash
immediately. Curriculum prevents this by starting with a tiny `C_rp = 0.05`, making
survival the dominant signal until the drone learns basic flight.

### Why SAC outperforms TD3 here?

SAC's automatic entropy tuning provides adaptive exploration throughout training.
Early on, the high entropy pushes the policy to try many motor combinations, which
is essential for discovering how to maintain altitude. TD3 uses fixed Gaussian
exploration noise that can be too conservative in early training and too aggressive
later. The entropy bonus in SAC also naturally regularises the policy, producing
smoother flight behaviour without explicit action smoothness penalties.

---

## Reference

Eschmann, J., Albani, D., Loianno, G. (2024).
**Learning to Fly in Seconds.**
*IEEE Robotics and Automation Letters.* arXiv:2311.13081.

See `LearningtoFlyinSeconds.pdf` for the full paper.
