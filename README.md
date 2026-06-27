# Drone RL — GPU-Parallel Quadrotor Navigation (Newton/MuJoCo + SKRL)

A reinforcement learning system for training a Crazyflie 2.x quadrotor position
controller **massively in parallel** inside the
[Newton](https://github.com/newton-physics/newton) GPU-native simulator, using
**SKRL** (PPO and SAC, with MLP or GRU policies).

The drone is modelled as a generalized-coordinate articulation — a `FREE` airframe +
**4 physical `REVOLUTE` rotor joints** (with `armature` / `effort_limit` / `friction`) —
stepped by **`SolverMuJoCo`**. One model holds **N worlds** (default 4096) via
`builder.replicate`; all per-step I/O stays on the GPU (no `.numpy()` syncs), and the
substep loop is captured as a CUDA graph. A batched Warp kernel converts rotor ω into
axial thrust + drag torque, so roll/pitch moments and the yaw reaction emerge from the
physics automatically.

The physics models a real **Crazyflie 2.x** (27 g nano-quadrotor) with correct mass,
inertia, arm length, thrust/torque constants, motor lag (τ = 0.15 s), and 100 Hz control.
Action remains the hover-centred 4-D `[-1, 1]` setpoint (0 → hover); observation remains
the 22-D layout; reward and curriculum follow the paper.

Physics baseline: Eschmann et al., "Learning to Fly in Seconds", RAL 2024 —
system-identified Crazyflie parameters, motor LPF model.

> **Migrated from** a single-body, CPU-bound Stable-Baselines3 pipeline. The old
> `.zip` SB3 checkpoints are archival and **cannot** be loaded by the SKRL trainer.
> See [CLAUDE.md](CLAUDE.md) for the full migration rationale.

---

## Repository Layout

```
dronesim/
├── drone_env_batched.py      # ★ Batched N-world Newton/MuJoCo env (build, aero,
│                             #   actuation, obs/reward/reset kernels, CUDA graph)
├── train_drone.py            # ★ SKRL trainer: PPO|SAC × MLP|GRU (+ SKRL env wrapper)
├── skrl_models.py            # ★ Gaussian/Deterministic/Q models — MLP + GRU
├── eval_drone.py             # ★ Multi-waypoint eval against the batched env + SKRL ckpt
├── hover_test_batched.py     # ★ Hover sanity gate (run first — see "Verify")
├── drone_gym_env.py          #   Single-env reference + the shared constants/config
├── LearningtoFlyinSeconds.pdf #  Reference paper (Eschmann 2024)
├── CLAUDE.md                 #   Migration spec / architecture rationale
├── landing/  disturbance/    #   Out-of-scope sibling tasks (still SB3)
└── runs/                     #   SKRL experiment dirs (TensorBoard + checkpoints)
    └── <algo>_<policy>_s<seed>/
        ├── events.out.tfevents…       # TensorBoard
        └── checkpoints/
            ├── best_agent.pt          # best by tracked reward
            └── agent_<step>.pt        # periodic
```

★ = the GPU-parallel pipeline. `★` files are the ones you run.

---

## Quick Start

```bash
# 0. Sanity-check the physics first (must PASS before training)
python hover_test_batched.py

# 1. Train PPO with an MLP policy at 4096 parallel worlds
python train_drone.py --algo ppo --policy mlp --num_envs 4096 \
  --total_timesteps 50_000_000 --seed 0

# 2. Monitor
tensorboard --logdir runs

# 3. Evaluate the best checkpoint on random multi-waypoint episodes
python eval_drone.py --algo ppo --policy mlp \
  --model runs/ppo_mlp_s0/checkpoints/best_agent.pt --num_episodes 20
```

> Run everything with the env that has Newton + SKRL installed. On this machine:
> `/home/adrian/miniconda3/envs/newton/bin/python` (the `newton` conda env).

---

## End-to-End Usage (Train → Evaluate)

This is the full workflow in detail. The four algorithm/policy combinations
— **PPO/SAC × MLP/GRU** — all share the same env, CLI, and checkpoint format.

### Step 0 — Verify the simulator (do this first)

`hover_test_batched.py` builds a small batch, commands the hover action (`a = 0`),
and checks the drone holds altitude. It catches mass/inertia, thrust-conversion, and
actuator-tuning bugs before you waste a training run.

```bash
python hover_test_batched.py
```

Expected tail:

```
hover thrust check: 4*KT_SI*ω² = 0.2649 N  vs m*g = 0.2649 N
|ω| settled ~ 1516 rad/s (target hover 1515.9)
altitude drift over 3 s: 0.020 m
HOVER: PASS
```

If it prints `FAIL` (sinks or rockets), fix the physics before training — see
[CLAUDE.md §8](CLAUDE.md).

### Step 1 — Train

```bash
# General form
python train_drone.py --algo {ppo,sac} --policy {mlp,gru} \
  --num_envs 4096 --total_timesteps 50_000_000 --seed 0

# Examples
python train_drone.py --algo ppo --policy mlp                 # PPO + MLP (default)
python train_drone.py --algo sac --policy mlp                 # SAC + MLP
python train_drone.py --algo ppo --policy gru --num_envs 2048 # recurrent PPO
python train_drone.py --algo sac --policy gru --num_envs 2048 # recurrent SAC
```

**CLI arguments**

| Flag | Default | Meaning |
|---|---|---|
| `--algo` | `ppo` | `ppo` or `sac` |
| `--policy` | `mlp` | `mlp` or `gru` (GRU → `PPO_RNN` / `SAC_RNN`) |
| `--num_envs` | `4096` | parallel worlds (the one intended change vs the paper config) |
| `--total_timesteps` | `50_000_000` | **environment frames** (`agent_steps = total / num_envs`) |
| `--seed` | `0` | RNG seed; also names the run |
| `--rollouts` | `24` | PPO steps/env per update (`rollouts × num_envs` = batch) |
| `--buffer_size` | `500_000` | SAC replay capacity (total; sized per-env internally) |
| `--curriculum_steps` | `5_000_000` | frames to ramp curriculum `c: 0 → 1` |
| `--entropy_decay_steps` | `20_000_000` | frames to decay PPO entropy `0.02 → 5e-4` |
| `--logdir` | `runs` | experiment root |
| `--device` | `cuda` | compute device |
| `--no_graph` | off | disable CUDA-graph capture (slower; for debugging) |

**What it does:** builds one Newton model with `--num_envs` worlds, wraps it for SKRL,
constructs the agent (PPO/SAC, MLP/GRU), and runs `SequentialTrainer`. A thin agent
subclass ramps the curriculum, decays PPO entropy, and logs success metrics — all keyed
on environment frames.

**Output:** `runs/<algo>_<policy>_s<seed>/` containing TensorBoard events and
`checkpoints/{best_agent.pt, agent_<step>.pt}`.

> **GPU memory:** 4096 envs (esp. SAC, or any GRU) can exceed a small card.
> If you hit `CUDA out of memory`, drop `--num_envs` (e.g. 2048 → 1024). The 8 GiB
> laptop GPU here comfortably runs PPO/MLP at 4096 and the recurrent/SAC variants at
> ~1024–2048.

### Step 2 — Monitor

```bash
tensorboard --logdir runs
```

Key scalars (mirroring the old logging):

| Tag | Meaning |
|---|---|
| `Reward / Instantaneous reward (mean)` | per-step reward across all worlds |
| `Reward / Total reward (mean)` | cumulative episode return (mean over finished episodes) |
| `Episode / success_rate` | fraction of finished episodes with terminal `dist < 0.15 m` |
| `Episode / terminal_dist` | mean distance to target at episode end |
| `Curriculum / c` | curriculum scalar `0 → 1` |
| `Curriculum / entropy_scale` | PPO entropy coefficient (decaying) |

A healthy PPO/MLP run: reward climbs as `c` ramps, `success_rate` rises past ~0.5 once
the curriculum saturates. Convergence takes tens of millions of frames (minutes-to-hours
depending on `--num_envs` and GPU).

### Step 3 — Evaluate

`eval_drone.py` loads a checkpoint and flies **random multi-waypoint episodes** on the
batched env (1 world): the drone visits each waypoint in sequence *without* resetting
between them; a slot succeeds at `dist < 0.15 m`.

```bash
python eval_drone.py --algo ppo --policy mlp \
  --model runs/ppo_mlp_s0/checkpoints/best_agent.pt \
  --num_episodes 20 --waypoints_per_ep 4

# Recurrent policy — match --algo/--policy to how the checkpoint was trained
python eval_drone.py --algo sac --policy gru \
  --model runs/sac_gru_s0/checkpoints/best_agent.pt
```

**Eval arguments:** `--model` (checkpoint `.pt`, required), `--algo`, `--policy`
(must match training), `--num_episodes` (10), `--waypoints_per_ep` (4),
`--steps_per_wp` (200), `--seed` (42), `--device` (`cuda`).

The policy runs **deterministically** (uses the action mean); GRU hidden state is
carried within an episode and reset between episodes. Output is a per-episode log plus a
summary (mean reward, per-waypoint & all-waypoint success, mean waypoint distance, mean
motor RPM).

> The eval reconstructs the *same* agent as training and calls `agent.load(...)`, so
> `--algo` and `--policy` **must** match the checkpoint. The old SB3 `.zip` files are not
> loadable here.

### Real-time visualization (OpenGL)

Both training and eval can open a live **Newton GL viewer** (CUDA/OpenGL interop — the
sim state is uploaded GPU→GPU each frame). The viewer renders a chosen subset of worlds
in a grid (display-only offsets; the physics is unaffected) plus a sphere marker at each
world's current target. Requires a display (`$DISPLAY`).

```bash
# Watch training — a 9-world grid, one frame every 4 control steps
python train_drone.py --algo ppo --policy mlp --num_envs 4096 \
  --render --render_worlds 9 --render_every 4

# Watch a trained policy fly, paced to wall-clock 100 Hz
python eval_drone.py --algo ppo --policy mlp \
  --model runs/ppo_mlp_s0/checkpoints/best_agent.pt --render --realtime
```

| Flag | Where | Meaning |
|---|---|---|
| `--render` | train + eval | open the GL window |
| `--render_worlds N` | train | how many worlds to show in the grid (default 4) |
| `--render_every K` | train | draw one frame per K control steps (default 4; throttles overhead) |
| `--realtime` | eval | sleep so playback matches wall-clock 100 Hz |

Notes:
- `log_state` synchronizes the device each rendered frame, so rendering throttles
  throughput — keep `--render` for **watching**, drop it for fast headless training.
- Training runs faster than real time, so the grid fast-forwards; closing the window
  stops rendering but lets training continue headless.
- The drone shows as its cross-arm collision shapes; the orange sphere is the target.

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
                          │    RL Policy (PPO / SAC)         │
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

The single-env reference (`drone_gym_env.py`) can add Gaussian sensor noise to the
observation (the batched trainer keeps it off by default):

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
# train_drone.py — agent pre_interaction hook
c = min(env_frames / curriculum_steps, 1.0)   # env_frames = timestep × num_envs
# --curriculum_steps default: 5_000_000 frames (fixed absolute count)
```

The curriculum is always on; control its length with `--curriculum_steps`.

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

Two algorithms × two policy backbones, all from **SKRL** (PyTorch, GPU-native):
**PPO** / **SAC**, each with an **MLP** or a **GRU** policy (GRU → `PPO_RNN` / `SAC_RNN`).
TD3 was dropped in the migration. Models live in `skrl_models.py`; head is `[256, 256]`
with `Tanh` (matching the old config). Hyperparameters are mapped from the preserved
SB3 config and set in `train_drone.py` (`build_ppo_cfg` / `build_sac_cfg`).

### PPO — Proximal Policy Optimisation (`skrl.agents.torch.ppo`)

On-policy. Default policy. Maps to `PPO_CFG`:

| SKRL cfg key | Value | Old SB3 equivalent |
|---|---|---|
| `rollouts` | `--rollouts` (24) | `n_steps` (per env); batch = `rollouts × num_envs` |
| `mini_batches` | `rollouts·N // 16384` | preserves the effective minibatch size |
| `learning_epochs` | 10 | `n_epochs` |
| `discount_factor` | 0.99 | `gamma` |
| `gae_lambda` | 0.95 | `gae_lambda` |
| `ratio_clip` / `value_clip` | 0.3 | `clip_range` |
| `entropy_loss_scale` | 0.02 → 5e-4 (decayed) | `ent_coef` schedule |
| `value_loss_scale` | 0.3 | `vf_coef` |
| `grad_norm_clip` | 0.5 | `max_grad_norm` |
| `learning_rate` | 3e-4 | `learning_rate` |
| `observation_preprocessor` | `RunningStandardScaler` | obs normalisation |

### SAC — Soft Actor-Critic (`skrl.agents.torch.sac`)

Off-policy, entropy-regularised. Models: Gaussian policy + twin Q-critics + targets.
Maps to `SAC_CFG`:

| SKRL cfg key | Value | Old SB3 equivalent |
|---|---|---|
| `batch_size` | 256 | `batch_size` |
| `polyak` | 0.005 | `tau` |
| `discount_factor` | 0.99 | `gamma` |
| `learn_entropy` | `True` (auto) | `ent_coef="auto"` |
| `learning_starts` | 1000 | warm-up before updates |
| `learning_rate` | 3e-4 | actor/critic/entropy LR |
| replay capacity | `--buffer_size` (500 000) | `buffer_size`, sized per-env internally |

### MLP vs GRU

- **MLP** — fast, the default; the 22-D obs is Markov enough for position control.
- **GRU** — a single GRU (hidden 256) in front of the head; implements SKRL's RNN
  contract (`get_specification` + hidden-state passing + per-episode resets). Use it for
  partial-observability / sim2real robustness experiments. Costs more memory — reduce
  `--num_envs` accordingly.

---

## Training

Full command reference and the workflow are in
[End-to-End Usage](#end-to-end-usage-train--evaluate) above. This section adds detail on
output layout and the curriculum/entropy schedules.

### Basic usage

```bash
python train_drone.py --algo ppo --policy mlp --num_envs 4096 \
  --total_timesteps 50_000_000 --seed 0
```

See the CLI table in [Step 1 — Train](#step-1--train) for every flag.

### Run naming and output files

Each run is named `{algo}_{policy}_s{seed}` and written under `--logdir` (default `runs/`):

```
runs/ppo_mlp_s0/
├── events.out.tfevents…             ← TensorBoard
└── checkpoints/
    ├── best_agent.pt                ← best by tracked reward (use this for eval)
    └── agent_<step>.pt              ← periodic snapshots
```

Checkpoints are SKRL `.pt` files (a dict of model + preprocessor state). They are **not**
interchangeable with the old SB3 `.zip` files.

> **Resuming:** SKRL's `SequentialTrainer` runs a fixed `timesteps` loop and does not
> expose a `--resume` flag here. To continue training, load a checkpoint into a freshly
> built agent (`agent.load(path)` — see `eval_drone.py` for the load pattern) before
> calling `trainer.train()`; the SAC replay buffer is not saved.

### Curriculum & entropy schedules

Both are driven by **environment frames** (`timestep × num_envs`) inside the agent's
`pre_interaction` hook:

- **Curriculum** `c: 0 → 1` linearly over `--curriculum_steps` (default 5 M frames),
  ramping reward weights *and* spawn extremes (position ±0.15→±1.5 m, tilt ±15°→±90°,
  velocity/rate, and initial rotor-speed band). Logged as `Curriculum / c`.
- **PPO entropy** decays `0.02 → 5e-4` over `--entropy_decay_steps` (default 20 M).
  Logged as `Curriculum / entropy_scale`. (SAC learns its entropy automatically.)

### Spawn Randomisation

The environment randomises the drone's initial state at every episode reset. All ranges
grow linearly with the curriculum:

| Dimension | Range at c=0 | Range at c=1 | Curriculum law (per axis) |
|-----------|-------------|-------------|-----|
| **Position (XY, Z)** | ±0.15 m from target | ±1.5 m from target | `0.15 + 1.35 × c` |
| **Orientation (tilt)** | ±15° | ±90° (full SO(3) cap) | `15° + 75° × c`, area-uniform on the cap |
| **Linear velocity** | ±0.1 m/s | ±1.0 m/s | `0.1 + 0.9 × c` |
| **Angular velocity** | ±0.1 rad/s | ±1.0 rad/s | `0.1 + 0.9 × c` |
| **Initial rotor speed** | ~hover band | 0 → MAX/2 band | widens with `c` |

A fixed **10% "guidance" branch** (curriculum-independent) always spawns the drone at the
target with identity attitude, supplying a steady stream of hold-at-target data. Yaw is
sampled uniformly. All of this runs inside the batched `reset_kernel` (GPU), masked to the
done envs each step.

---

## PPO Training — How It Works (legacy SB3 internals)

> ⚠️ **The sections from here down describe the original single-body Stable-Baselines3
> pipeline** (`DummyVecEnv`, 16 sequential envs, `metrics/…` tags, `eval_trajectory.py`).
> They are kept for historical reference. For the current GPU-parallel SKRL pipeline use
> [End-to-End Usage](#end-to-end-usage-train--evaluate). The physics/observation/reward/
> curriculum sections above remain accurate.

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
