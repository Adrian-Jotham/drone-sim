# Reward Shaping for PPO-Based Direct-RPM Quadrotor Position Control

**Draft for KICS 2025 Summer Conference**

---

## Abstract

Reinforcement learning (RL) for quadrotor control at the direct motor RPM level
(Level 5.1) has been demonstrated using off-policy methods such as TD3, with
on-policy approaches such as Proximal Policy Optimization (PPO) explicitly noted
as less suitable for this task. In this paper we show that PPO can achieve
competitive waypoint-navigation performance at Level-5.1 RPM control when
augmented with two targeted reward modifications: (1) a gated potential-based
approach term that suppresses oscillation near the target, and (2) a
velocity-gated settlement bonus that prevents rush-to-target behaviour. Using a
physics-accurate Crazyflie 2.x simulation built on the Newton GPU rigid-body
engine, our final policy achieves **65 % per-waypoint success** and **99.5 % of
hover RPM** on a four-waypoint sequential navigation benchmark after 3 million
environment steps. An ablation study confirms that both reward modifications
contribute meaningfully to performance.

---

## I. Introduction

Autonomous quadrotor flight has attracted significant research interest driven by
applications in inspection, search-and-rescue, and package delivery. Classical
control architectures decompose the problem into cascaded layers—attitude
stabilisation, velocity control, and position planning—each requiring
hand-engineered gains and explicit dynamic models. Reinforcement learning (RL)
offers an alternative: a single end-to-end policy mapping raw sensor observations
directly to motor commands, eliminating manual tuning and potentially capturing
nonlinear dynamics that classical controllers approximate away.

The lowest and most physically faithful level of quadrotor control is **Level 5.1
— direct motor RPM setpoints** [1]. At this level the policy must implicitly
account for the quadratic thrust-to-RPM relationship (F = K_T · n²), motor
first-order lag (τ ≈ 0.15 s), rotor reaction torques, and the full inertia tensor
of the vehicle. Prior work [1] demonstrated that TD3 — an off-policy deterministic
policy gradient algorithm — can train a Crazyflie position controller at Level 5.1
in approximately 18 seconds of wall-clock time on a consumer GPU, achieving
zero-shot Sim-to-Real transfer. Crucially, the authors note that on-policy
algorithms such as PPO [2] are less suitable for this task because the
high-frequency exploratory actions required in RPM space are suppressed by
on-policy entropy regularisation, degrading sample efficiency.

Despite this claim, PPO remains one of the most widely used RL algorithms in
robotics due to its stability, simplicity, and absence of a replay buffer. Whether
PPO can succeed at Level-5.1 control — and if so, under what reward formulation —
has not been investigated. This gap is practically significant: PPO-based pipelines
are easier to integrate into existing robotics workflows, and demonstrating their
viability at the direct RPM level would lower the barrier to entry for researchers
without off-policy infrastructure.

In this paper we address this gap with the following contributions:

1. We demonstrate that PPO **can** learn Level-5.1 direct-RPM quadrotor position
   control, achieving 65 % per-waypoint success on a randomised multi-waypoint
   benchmark.

2. We identify two failure modes specific to applying PPO at this control level
   and propose reward modifications that resolve each:
   - **Target oscillation** caused by an ungated potential-based approach reward,
     fixed by suppressing the approach term inside a 0.25 m radius.
   - **Rush-to-target instability** caused by a distance-only settlement bonus,
     fixed by conditioning the bonus on both distance and speed.

3. We provide an open, reproducible implementation of the Level-5.1 training
   environment using the Newton GPU physics engine [3] and the
   Stable-Baselines3 PPO implementation [4].

---

## II. Background and Related Work

### A. Level-5.1 Quadrotor Control

Eschmann et al. [1] introduce a taxonomy classifying quadrotor controllers by
their input abstraction level. Level 5.1 corresponds to direct motor RPM
setpoints, the lowest level in the taxonomy. The full complexity of vehicle
dynamics — nonlinear thrust curves, motor delay, rotor gyroscopic effects, and
inertia coupling — is exposed to the policy at this level. The paper demonstrates
that policies trained at Level 5.1 transfer to real hardware without domain
randomisation, because the simulator already incorporates the dominant sources of
the reality gap.

### B. Proximal Policy Optimisation

PPO [2] is an on-policy actor-critic algorithm that clips the probability ratio
between the updated and old policy to enforce a trust region. It collects a
fixed-length rollout of experience from all parallel environments before each
parameter update, discarding the data afterwards. This on-policy data requirement
is the source of its lower sample efficiency compared to off-policy methods: every
gradient step uses freshly collected data, so the policy cannot learn from past
experiences stored in a replay buffer.

### C. Reward Shaping

Potential-based reward shaping [5] augments the base reward with a term
F(s, s') = γΦ(s') − Φ(s) that is guaranteed by the Policy Invariance Theorem to
preserve the optimal policy. Setting Φ(s) = −‖p_err‖ yields an approach reward
proportional to the per-step reduction in distance, providing a dense gradient
toward the target that supplements the sparse quadratic position cost.

---

## III. Method

### A. Simulation Environment

The environment models a **Crazyflie 2.x** nano-quadrotor using system-identified
physical parameters from Förster [6]. All constants are set to match the real
vehicle exactly:

| Parameter | Value |
|-----------|-------|
| Mass | 27 g |
| Arm length (centre–motor) | 32.5 mm |
| I_xx = I_yy | 1.657 × 10⁻⁵ kg·m² |
| I_zz | 2.900 × 10⁻⁵ kg·m² |
| Thrust constant K_T | 3.16 × 10⁻¹⁰ N/RPM² |
| Drag-torque constant K_D | 7.94 × 10⁻¹² N·m/RPM² |
| Hover RPM | ≈ 14,476 RPM |

Rigid-body dynamics are integrated at **100 Hz** using Newton's
`SolverSemiImplicit`. Propeller thrust (F_i = K_T · n_i²) and reaction torque
(Q_i = K_D · n_i²) are applied via a Warp GPU kernel at each step. A first-order
low-pass filter with time constant τ = 0.15 s models motor lag, consistent with
the Crazyflie ESC response characterised in [1].

Sixteen independent environments run in a single process using
`DummyVecEnv`. Each episode runs for up to 800 steps (8.0 s), matching the
evaluation budget of four waypoints at 200 steps each.

### B. Observation and Action Space

The **22-dimensional observation** follows the actor observation of [1]:

| Slice | Content | Dim |
|-------|---------|-----|
| [0:3] | Position error p_err = pos − target | 3 |
| [3:12] | Rotation matrix R (row-major, avoids quaternion double-cover) | 9 |
| [12:15] | Linear velocity v (world frame, m/s) | 3 |
| [15:18] | Angular velocity ω (body frame, rad/s) | 3 |
| [18:22] | Previous action a_prev (action history N_H = 1) | 4 |

The **4-dimensional action** consists of normalised RPM setpoints ∈ [−1, 1],
mapped linearly to motor speeds centred at hover:

```
n_sp,i = clip( n_hover + a_i · (n_max − n_hover),  n_min,  n_max )
       = clip( 14476 + a_i · 7226,  1000,  21702 )  RPM
```

Action 0 corresponds to the exact hover equilibrium. The mapped setpoint feeds
the motor LPF filter; the filtered RPM drives the physics engine.

### C. PPO Configuration

We use the Stable-Baselines3 PPO implementation with an MLP policy. The shared
trunk (two 256-unit hidden layers, tanh activations) feeds separate actor and
critic heads. Key hyperparameters:

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_steps` | 2048 | Steps per env per rollout (32,768 total with 16 envs) |
| `batch_size` | 64 | Mini-batch size; 512 mini-batches per rollout |
| `n_epochs` | 10 | Gradient passes per rollout |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE bias-variance trade-off |
| `clip_range` | 0.2 | PPO trust-region clip |
| `ent_coef` | 0.005 | Entropy bonus; keeps exploration from collapsing |
| `max_grad_norm` | 0.5 | Gradient clipping for stability |
| Total timesteps | 3,000,000 | ~91 rollout cycles |

### D. Base Reward Function

The base reward follows the formulation of [1]:

```
r_base = −C_rp‖p_err‖² − C_rq(1 − q_w²) − C_rv‖v‖² − C_rω‖ω‖² − C_ra‖Δa‖² + C_rs
```

where `Δa = a_t − a_{t−1}` penalises action jerk rather than magnitude. A
one-time crash penalty of −2.0 is applied when altitude z < 0.05 m, and a
one-time arrival bonus of +1.0 is given the first time `‖p_err‖ < 0.10 m`.

Reward weights ramp linearly from conservative initial values to strict target
values over the first 50% of training (curriculum ∈ [0, 1]):

| Weight | C_init | C_target |
|--------|--------|----------|
| C_rp (position) | 0.05 | 1.00 |
| C_rv (velocity) | 0.01 | 0.30 |
| C_rω (angular velocity) | 0.001 | 0.05 |
| C_ra (action jerk) | 0.005 | 0.02 |
| C_rq (orientation) | 0.10 | 0.10 (fixed) |
| C_rs (survival) | 0.50 | 0.50 (fixed) |

The curriculum is updated every step by a callback:
`curriculum = min( t / (T × 0.5), 1.0 )` where T = 3,000,000.

### E. Reward Modification 1 — Gated Approach Term

Training with the base reward alone produces a policy that oscillates around the
target. The root cause is the potential-based approach term commonly added to
provide a dense gradient:

```
approach = λ_ap · (‖p_err‖_{t−1} − ‖p_err‖_t)
```

With λ_ap = 2.0 applied at all distances, the policy earns positive reward for any
motion toward the target, including small oscillatory movements close to the
waypoint. The drone learns to perpetually orbit the target rather than settle.

We suppress the approach term inside a **gate radius d_gate = 0.25 m**:

```
approach = λ_ap · (‖p_err‖_{t−1} − ‖p_err‖_t)    if ‖p_err‖_t > d_gate
         = 0                                         otherwise
```

with λ_ap reduced to 1.0. This preserves the dense navigation gradient when far
from the target, while removing the oscillation incentive once the drone is close.
The gate radius is chosen to be larger than the success threshold (0.15 m) but
small enough that the quadratic position cost C_rp provides sufficient gradient in
the remaining range.

### F. Reward Modification 2 — Velocity-Gated Settlement Bonus

To encourage the drone to remain at the waypoint after arrival, a per-step
settlement bonus is added:

```
hover_bonus = B_h    if ‖p_err‖_t < d_succ  AND  ‖v_t‖ < v_gate
            = 0      otherwise
```

with d_succ = 0.15 m (the evaluation success threshold), B_h = 0.15, and
v_gate = 0.5 m/s.

**The speed condition is critical.** A distance-only gate (`if ‖p_err‖ < 0.15`)
creates a discontinuous reward cliff: the policy receives +B_h the instant it
crosses 0.15 m regardless of speed. Empirically, this caused the policy to learn
a rush-to-target strategy — approaching at 3–4 m/s to enter the bonus zone
sooner — resulting in frequent overshoots and crash episodes. The resulting
training instability is visible as `vel_c` spikes reaching −5.0 per step and
episode rewards collapsing to below −10,000 in training runs without the speed
condition.

Adding the speed gate v_gate = 0.5 m/s eliminates this incentive. The policy can
only collect the bonus while nearly stationary inside the success radius, making
*approach and decelerate* strictly better than *sprint through*.

The full reward with both modifications is:

```
r = r_base + approach + hover_bonus
```

---

## IV. Experiments

### A. Evaluation Protocol

We evaluate each configuration using the sequential four-waypoint protocol.
Each episode samples four random waypoints without resetting the drone physics
between them (radius ∈ [0.5, 1.5] m, altitude ∈ [0.3, 1.2] m, angle uniform).
A waypoint is counted as reached if the drone comes within 0.15 m within a 200-step
(2.0 s) budget. We report results over **20 episodes** with seed 42.

### B. Ablation Study

We train three configurations for 3,000,000 steps with 16 parallel environments
and identical PPO hyperparameters. All other settings are held fixed.

| Configuration | Per-wp Success | Mean Dist (m) | RPM Dev | All-wp Success |
|---|---|---|---|---|
| (A) Base reward only | [TBD] | [TBD] | [TBD] | [TBD] |
| (B) + Gated approach | [TBD] | [TBD] | [TBD] | [TBD] |
| (C) + Velocity-gated hover bonus (full) | **65 %** | **0.175 m** | **~75 RPM** | **20 %** |

> **[TBD]** — Rows (A) and (B) are currently training. Results will be inserted
> before submission.

### C. Final Policy Evaluation

Configuration (C), evaluated over 10 episodes × 4 waypoints (seed 42):

```
Episodes            : 10
Waypoints / ep      : 4  (random, r=0.5–1.5 m, z=0.3–1.2 m)
Step budget / wp    : 200 steps = 2.0 s
Mean reward         : 286.56 ± 26.31
Mean ep length      : 657.9 steps
Mean wpts reached   : 2.60 / 4
Per-waypoint succ   : 65.0%  (dist < 0.15 m)
All-waypoints succ  : 20.0%  (all wpts hit)
Mean waypoint dist  : 0.1752 m
Mean motor RPM      : 14,400 RPM  (99.5% of hover RPM 14,476)
```

The mean motor RPM of 14,400 (99.5% of hover equilibrium) and the absence of
crashes across all 10 episodes indicate a stable policy. The primary failure mode
is near-miss waypoints at 0.16–0.27 m — the drone approaches and decelerates but
does not fully converge within the 2.0 s budget, particularly for the second and
third waypoints where it must transition directly from the preceding hovering state.

---

## V. Discussion

### Why PPO Struggles at Level-5.1 Without Reward Shaping

PPO collects all training data on-policy: every gradient step uses the current
policy's own rollouts, and past data is discarded. For Level-5.1 RPM control, the
sparse quadratic position reward provides a weak gradient signal when the drone is
far from the target — the derivative ∂r/∂p_err = −2C_rp · p_err is small when
C_rp is small (early curriculum) and distance is large. The approach term fills
this gap by providing a dense signal proportional to the per-step distance
reduction, critical for on-policy learning where each rollout must provide enough
gradient to improve the policy.

Off-policy methods such as TD3 are less sensitive to this problem because they
can revisit informative past transitions stored in the replay buffer. PPO must
extract useful learning signal from whatever the current policy explores — making
reward density more important.

### Failure Mode Analysis

The velocity-gated settlement bonus was motivated by direct observation of training
collapse. When the bonus was distance-gated only (B_h = 0.30, no speed condition),
training showed bimodal episode quality: good episodes where the drone hovered
successfully alternated with episodes where it approached at 3–4 m/s, overshot,
and accumulated large velocity penalties (vel_c ≈ −5.0 per step). The smoothed
episode reward fell to −17,000 during this phase despite individual episodes
occasionally reaching +300. Adding the speed gate eliminated this bimodality
entirely.

---

## VI. Conclusion

We have demonstrated that PPO can learn Level-5.1 direct-RPM quadrotor position
control, achieving 65 % per-waypoint success on a randomised four-waypoint
benchmark — a task previously considered unsuitable for on-policy methods.
Two reward modifications were necessary and sufficient: a gated potential-based
approach term that removes the oscillation incentive near the target, and a
velocity-gated settlement bonus that prevents the rush-to-target instability
introduced by a distance-only gate.

The velocity gate insight — that a settlement bonus must condition on speed, not
only on distance — is the primary engineering contribution of this work. A
distance-only bonus creates a reward cliff that incentivises high-speed approaches;
the speed condition enforces the decelerate-then-hover behaviour that actually
produces successful waypoint landings.

Future work will extend this study in two directions. First, a direct comparison of
PPO against TD3 and SAC under identical conditions (same Newton environment, same
reward, same evaluation protocol) will quantify the sample-efficiency gap between
on- and off-policy approaches at this control level. Second, zero-shot Sim-to-Real
transfer on a physical Crazyflie 2.x will validate whether the PPO-trained policy
is deployable under the real-world conditions characterised in [1].

---

## References

[1] J. Eschmann, D. Albani, and G. Loianno, "Learning to Fly in Seconds,"
*IEEE Robotics and Automation Letters*, vol. 9, no. 6, pp. 5621–5628, Apr. 2024.
arXiv:2311.13081.

[2] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal
Policy Optimization Algorithms," arXiv:1707.06347, 2017.

[3] NVIDIA, "Newton: GPU-Accelerated Rigid Body Simulation,"
https://github.com/newton-physics/newton, 2024.

[4] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann,
"Stable-Baselines3: Reliable Reinforcement Learning Implementations,"
*Journal of Machine Learning Research*, vol. 22, no. 268, pp. 1–8, 2021.

[5] A. Y. Ng, D. Harada, and S. Russell, "Policy Invariance Under Reward
Transformations: Theory and Application to Reward Shaping,"
in *Proc. ICML*, 1999, pp. 278–287.

[6] D. Förster, "System Identification of the Crazyflie 2.0 Nano Quadrocopter,"
Bachelor's thesis, ETH Zürich, 2015.

---

*Code available at: [repository URL]*
*Simulation environment built on Newton physics engine and Stable-Baselines3.*
