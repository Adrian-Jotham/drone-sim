# End-to-End Quadrotor Position Control via Proximal Policy Optimization with Direct Motor Actuation

**Draft for KICS 2025 Summer Conference**

---

## Abstract

Reinforcement learning (RL) offers a path to fully autonomous quadrotor control
by mapping raw sensor observations directly to motor commands without any
hand-engineered control hierarchy. In this paper we present a complete end-to-end
learning pipeline for quadrotor position control at the direct motor RPM level —
the lowest and most physically demanding actuation abstraction — using Proximal
Policy Optimization (PPO). Training entirely within a physics-accurate Crazyflie
2.x simulation built on the Newton GPU rigid-body engine, our policy achieves
**82.9 % per-leg waypoint success** after 30 million environment steps, converging
to stable performance that is maintained at 40 million steps (81.8 %). We
characterise learning progression from 3M to 40M steps, provide distance-stratified
success rates across four distance bands, and quantify robustness to spawn
randomisation — a 31.5-percentage-point gap between clean-start and fully
randomised initialisation that highlights a key challenge for Sim-to-Real transfer.
A reward shaping analysis shows that coupled potential-based approach and velocity-gated
settlement terms are jointly necessary for convergence; neither alone is sufficient.

---

## I. Introduction

Autonomous quadrotor navigation is a long-standing challenge at the intersection of
control theory, robotics, and machine learning. Classical approaches decompose the
problem into a cascade of independently tuned controllers: a high-level trajectory
planner, a position controller, an attitude stabiliser, and low-level motor
mixing — each layer requiring explicit dynamic models and hand-tuned gains.
While such architectures have demonstrated excellent performance on real hardware,
they depend on accurate system identification and are brittle to unmodelled
dynamics or hardware variation.

Reinforcement learning offers an alternative paradigm: a single neural network
policy learns to map raw sensor readings directly to motor commands through
trial-and-error interaction with a simulator. If the simulator is physically
faithful, the resulting policy can generalise to real hardware without requiring
separate tuning of each control layer. End-to-end RL policies have been
demonstrated for a range of platforms — from simulated cars to legged robots —
but applying them to nano-quadrotors at the direct motor RPM level presents
specific challenges that are not yet fully understood.

Direct motor RPM control exposes the policy to the full complexity of quadrotor
dynamics: the quadratic thrust-to-RPM relationship (F = K_T · n²), first-order
motor lag (τ ≈ 0.15 s), reaction torques, and the complete inertia tensor. A
single policy must simultaneously learn attitude stabilisation, position tracking,
and multi-waypoint navigation — tasks that are handled by separate subsystems in
classical architectures. The reward function must therefore guide learning across
all of these coupled objectives simultaneously, making reward design the central
engineering challenge.

In this work we investigate PPO — one of the most widely used on-policy RL
algorithms — for this task, with a focus on reward engineering. Our contributions
are:

1. We design and evaluate a reward function with coupled potential-based approach
   and velocity-gated settlement terms, demonstrating that both components are
   jointly necessary. The gated approach term suppresses oscillation near the
   target; the velocity-gated bonus incentivises deceleration and stable hover
   inside the success radius. Applied individually, each modification is
   insufficient or counterproductive.

2. We characterise training progression from 3M to 40M environment steps on a
   multi-waypoint chaining benchmark, identifying convergence at approximately
   30M steps and a training-induced attractor toward over-cautious flight.

3. We provide distance-stratified success rates and a clean vs. randomised spawn
   robustness comparison, identifying specific failure modes that must be addressed
   for robust real-world deployment.

4. We provide an open, reproducible training environment for the Crazyflie 2.x
   nano-quadrotor using the Newton GPU physics engine and the Stable-Baselines3
   PPO implementation.

---

## II. Background

### A. End-to-End RL for Direct Motor Control

Reinforcement learning policies for quadrotors have been demonstrated at multiple
levels of actuation abstraction. High-level policies command desired position or
velocity setpoints to an underlying stabilisation controller; lower-level policies
command attitude setpoints or body-rate targets; and at the lowest level, policies
directly command individual motor speeds or RPMs. This lowest abstraction level —
which we refer to as **direct RPM control** — is the most demanding because the
policy must simultaneously learn all control layers from a single reward signal,
but it is also the most expressive: it places no restrictions on what motor
patterns the policy can discover and removes the intermediate control layers that
may constrain performance or introduce additional tuning parameters.

The key challenge at this level is reward density. The position error between the
drone and its target is a sparse signal: it changes slowly and provides weak
gradient when the drone is far away. An effective reward function must supplement
this signal with dense terms that guide exploration during the early stages of
training, without creating incentives that redirect the policy toward undesirable
local optima (e.g., hovering in place, oscillating near the target, or sprinting
through waypoints without decelerating).

### B. Proximal Policy Optimisation

PPO [1] is an on-policy actor-critic algorithm that collects a fixed-length rollout
of experience from all parallel environments, then performs multiple gradient
passes over the collected data using a clipped probability ratio to enforce a
trust region. After each update cycle the collected data is discarded and a fresh
rollout is collected from the updated policy.

Because all training data is freshly collected, the policy must extract sufficient
gradient from each rollout before it is discarded. This places stronger demands on
reward density than off-policy methods, which can revisit informative transitions
stored in a replay buffer. Dense, well-shaped rewards are therefore especially
important for PPO in high-dimensional control tasks where the unmodified reward is
sparse or weakly structured.

### C. Curriculum Learning

Curriculum learning [2] schedules the difficulty of training examples to accelerate
learning. In continuous control tasks this commonly takes the form of randomising
spawn positions, velocities, and orientations uniformly at first with a small
range, then gradually expanding the range as the policy improves. This allows the
policy to learn simple hovering behaviour early in training, which serves as a
bootstrap for learning navigation.

We apply curriculum learning to spawn randomisation: the position offset, linear
velocity, angular velocity, and tilt angle at spawn are all scaled by a curriculum
coefficient c ∈ [0, 1] that increases linearly over the first 50% of training.
Reward weights are simultaneously annealed from conservative to strict values.

---

## III. Method

### A. Simulation Environment

The environment models a **Crazyflie 2.x** nano-quadrotor using system-identified
physical parameters [3]. Rigid-body dynamics are integrated at **100 Hz** using
Newton's `SolverSemiImplicit`. Propeller thrust and reaction torque are applied via
a Warp GPU kernel at each step. A first-order low-pass filter with time constant
τ = 0.15 s models motor lag consistent with the Crazyflie ESC response.

| Parameter | Value |
|-----------|-------|
| Mass | 27 g |
| Arm length (centre–motor) | 32.5 mm |
| I_xx = I_yy | 1.657 × 10⁻⁵ kg·m² |
| I_zz | 2.900 × 10⁻⁵ kg·m² |
| Thrust constant K_T | 3.16 × 10⁻¹⁰ N/RPM² |
| Drag-torque constant K_D | 7.94 × 10⁻¹² N·m/RPM² |
| Hover RPM | ≈ 14,476 RPM |

Sixteen independent environments run in a single process using `DummyVecEnv`.
Each episode runs for up to 800 steps (8.0 s).

### B. Observation and Action Space

The **22-dimensional observation** encodes full rigid-body state relative to the
current target waypoint:

| Slice | Content | Dim |
|-------|---------|-----|
| [0:3] | Position error p_err = pos − target | 3 |
| [3:12] | Rotation matrix R (row-major) | 9 |
| [12:15] | Linear velocity v (world frame, m/s) | 3 |
| [15:18] | Angular velocity ω (body frame, rad/s) | 3 |
| [18:22] | Previous action a_prev | 4 |

The rotation matrix representation avoids the quaternion double-cover and
discontinuities near ±180°, providing a smooth manifold for the neural network to
learn attitude representations.

The **4-dimensional action** consists of normalised RPM setpoints ∈ [−1, 1],
mapped linearly to motor speeds centred at hover:

```
n_sp,i = clip( n_hover + a_i · (n_max − n_hover),  n_min,  n_max )
       = clip( 14476 + a_i · 7226,  1000,  21702 )  RPM
```

Action 0 corresponds exactly to the hover equilibrium. The mapped setpoint feeds
the motor LPF; the filtered RPM drives the physics engine.

### C. PPO Configuration

We use the Stable-Baselines3 PPO implementation [4] with an MLP policy. The shared
trunk (two 256-unit hidden layers, tanh activations) feeds separate actor and
critic heads. The rollout length is set to 2,048 steps per environment, giving a
total rollout buffer of 32,768 transitions per update cycle. This exceeds the
maximum episode length of 800 steps, ensuring that complete episodes are captured
within each rollout and GAE returns are never truncated mid-episode.

| Hyperparameter | Value |
|---|---|
| `n_steps` | 2048 (rollout per env) |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.005 |
| `max_grad_norm` | 0.5 |
| Total timesteps | 40,000,000 |

### D. Reward Function

The reward at each timestep combines five per-step cost terms, two shaped bonus
terms, and one-time event rewards:

```
r = pos_c + orient_c + vel_c + ang_c + act_c + survival
  + approach + hover_bonus
```

**Per-step cost terms:**

```
pos_c    = −C_rp · ‖p_err‖²
orient_c = −C_rq · (1 − q_w²)
vel_c    = −C_rv · ‖v‖²
ang_c    = −C_rω · ‖ω‖²
act_c    = −C_ra · ‖Δa‖²      (Δa = a_t − a_{t−1})
survival =  C_rs               (flat per-step bonus)
```

Reward weights ramp linearly from conservative to strict values over the first 50%
of training as the curriculum coefficient c increases:

| Term | C_init | C_final |
|------|--------|---------|
| C_rp (position) | 0.05 | 1.00 |
| C_rv (velocity) | 0.01 | 0.30 |
| C_rω (angular rate) | 0.001 | 0.05 |
| C_ra (action jerk) | 0.005 | 0.02 |
| C_rq (orientation) | 0.10 | 0.10 (fixed) |
| C_rs (survival) | 0.50 | 0.50 (fixed) |

**One-time event rewards:**
- Crash (altitude z < 0.05 m): −2.0
- Arrival (‖p_err‖ < 0.10 m): +1.0

**Shaped approach term (gated):**

```
approach = λ_ap · ( ‖p_err‖_{t−1} − ‖p_err‖_t )    if ‖p_err‖_t > d_gate
         = 0                                          if ‖p_err‖_t ≤ d_gate
```

with λ_ap = 1.0 and d_gate = 0.25 m. This term provides a dense per-step signal
proportional to distance reduction during transit, suppressed inside the gate
radius to prevent oscillation incentives near the target.

**Velocity-gated settlement bonus:**

```
hover_bonus = B_h    if ‖p_err‖_t < d_succ  AND  ‖v_t‖ < v_gate
            = 0      otherwise
```

with B_h = 0.15, d_succ = 0.15 m, and v_gate = 0.5 m/s.

### E. Spawn Curriculum

At the start of each episode the drone spawns near the first waypoint with
randomised offset, velocity, and tilt angle. The spawn distribution scales with
curriculum coefficient c ∈ [0, 1]:

| Parameter | Range at c |
|-----------|------------|
| Position offset | ‖Δpos‖ ≤ 0.1 + 0.6c m |
| Linear velocity | ‖v‖ ≤ 1.5c m/s |
| Angular velocity | ‖ω‖ ≤ 0.5c rad/s |
| Tilt angle | ≤ 15c° |

At c = 0 (clean start) the drone spawns stationary and nearly upright within 0.1 m
of the waypoint. At c = 1.0 (full randomisation) the drone may start with
significant velocity and tilt. This curriculum allows early-stage policies to learn
stable hovering before being challenged by recovery from disturbed initial states.

---

## IV. Experiments

### A. Evaluation Protocol

All evaluations use a **multi-waypoint chaining** protocol: waypoints are generated
sequentially and the drone navigates to each without episode reset between waypoints.
A leg is considered succeeded if ‖p_err‖ < 0.15 m within a 800-step (8.0 s)
episode budget shared across all legs. Results are reported over 20 episodes.

Primary metric: **leg success rate** — the fraction of all attempted waypoint legs
in which the drone came within 0.15 m of the target.

Secondary metrics: waypoints reached per episode (mean ± std), mean motor RPM as
percentage of hover RPM, |RPM − hover| deviation, and arrival speed (velocity at
first entry into the 0.15 m radius).

Two evaluation modes:
- **Clean start** (c = 0): drone spawns nearly stationary within 0.1 m of the first
  waypoint. Tests navigation capability under ideal initialisation.
- **Randomised** (c = 1.0): drone spawns with full curriculum randomisation.
  Tests robustness to disturbed initial conditions.

### B. Reward Shaping Ablation

To establish that both reward components are jointly necessary, we evaluate three
reward configurations after 3M training steps using a fixed four-waypoint protocol
(200-step budget per waypoint, 20 episodes):

| Configuration | Leg Success | Mean Dist (m) | |RPM − hover| |
|---|---|---|---|
| (A) Base reward only | 42.5 % | 0.433 | 262 |
| (B) + Gated approach term only | 32.5 % | 0.345 | 126 |
| (C) Full reward (both terms) | **65.0 %** | **0.178** | 257 |

Configuration (B) performs **worse** than the base reward despite reducing RPM
deviation. This is because the approach gate removes the dense gradient in the
0.15–0.25 m band without replacing it: the drone learns to hover steadily at the
gate boundary (~0.20–0.35 m) rather than closing the final gap. Configuration (C)
resolves this by adding the settlement bonus inside 0.15 m, giving the policy an
explicit incentive to cross and hold the success threshold.

The speed condition on the hover bonus (‖v‖ < 0.5 m/s) proved critical. A
distance-only bonus without the speed gate caused training instability: the policy
learned to sprint toward the target at 3–4 m/s to enter the bonus zone sooner,
producing frequent overshoots. The velocity gate eliminates this by making the
bonus collectible only while nearly stationary.

### C. Learning Progression

Table II shows policy performance at three checkpoints during a 40M-step training
run (clean-start evaluation, 20 episodes each):

**Table II. Learning progression (clean-start evaluation, 20 episodes)**

| Checkpoint | Leg SR | Wps/ep | Mean RPM | RPM % hover | \|RPM−hover\| | Arrival spd |
|---|---|---|---|---|---|---|
| 3M steps | 73.0 % | 2.7 ± 2.3 | 14,303 | 98.8 % | 227 RPM | 0.419 m/s |
| 30M steps | **82.9 %** | **4.8 ± 3.2** | 14,039 | 97.0 % | 437 RPM | 0.326 m/s |
| 40M steps | 81.8 % | 4.5 ± 3.6 | 14,061 | 97.1 % | 415 RPM | 0.294 m/s |

Performance converges at approximately **30M steps** with only marginal change
to 40M. The 3M-to-30M improvement (+9.9 pp in leg success rate, +2.1 waypoints/ep)
reflects the policy transitioning from single-hop navigation to sustained
multi-waypoint chaining. The 30M-to-40M plateau confirms that 30M steps is
sufficient for convergence on this task.

Arrival speed decreases monotonically (0.419 → 0.326 → 0.294 m/s), indicating
that the policy learns progressively smoother approaches over training. However,
motor RPM deviation from hover increases from 3M to 30M (227 → 437 RPM), reflecting
a shift from cautious single-step hovering to more dynamic transit manoeuvres.

### D. Distance-Stratified Performance

Table III stratifies the 40M-step clean-start results by waypoint distance to
characterise how success rate degrades with navigation difficulty:

**Table III. Distance-stratified leg success rate (40M, clean start)**

| Distance Band | Leg SR | Legs |
|---|---|---|
| 0.0 – 0.5 m | 100.0 % | 7 |
| 0.5 – 1.0 m | 92.9 % | 28 |
| 1.0 – 1.5 m | 87.1 % | 31 |
| 1.5 – 3.0 m | 68.2 % | 44 |

The policy achieves near-perfect success for short-range navigation (< 1.0 m) and
degrades gracefully with distance. For the hardest band (1.5–3.0 m), 68.2 % success
indicates that the policy has learned to navigate at range but fails on
approximately one-third of distant legs. These failures correspond to cases where
the drone overshoots the waypoint and exhausts the episode step budget attempting
recovery, rather than crashes.

### E. Robustness to Spawn Randomisation

**Table IV. Clean vs. randomised spawn evaluation (40M checkpoint, 20 episodes)**

| Condition | Leg SR | Wps/ep |
|---|---|---|
| Clean start (c = 0) | 81.8 % | 4.5 ± 3.6 |
| Randomised (c = 1.0) | 50.0 % | — |

The 31.5-percentage-point gap between clean and randomised evaluation reveals a
significant robustness deficit. The policy trained with curriculum randomisation
(which reaches c = 1.0 at 20M steps) does not fully generalise to all disturbed
initial states encountered during randomised evaluation. The most common failure
mode is that the policy cannot recover from large initial tilts (> 10°) combined
with nonzero translational velocity before the step budget is consumed.

This gap is the primary challenge for Sim-to-Real transfer: real-world launches
involve disturbed initial conditions, and a policy that performs at 81.8 % under
ideal initialisation but only 50.0 % under the level of randomisation it was
trained with leaves open the question of whether it can reliably handle real
hardware conditions.

---

## V. Discussion

### A. Convergence and the Hover Attractor

Training converges at approximately 30M steps, as confirmed by entropy loss
stabilising at −0.72 and clip fraction falling to 0.044 at 30M steps. The flat
survival bonus (C_rs = 0.50/step) creates an attractor toward hovering: a
stationary drone with zero position error earns positive net per-step reward from
the survival term alone. This attractor benefits early-stage stability learning
but competes with the approach gradient during navigation — the drone can always
do better in the short term by hovering than by moving. The plateau from 30M to
40M steps (81.8 % vs. 82.9 %) may partly reflect this attractor preventing further
improvement, as the policy has learned a balance between hovering and navigating
rather than a fully committed navigation strategy.

### B. Motor RPM Characteristics

All three checkpoints operate within 1–3% of hover RPM on average, consistent with
stable flight throughout training. The increasing RPM deviation from 3M to 30M
(227 → 437 RPM) is expected: early policies hover conservatively near each waypoint,
while mature policies execute more aggressive transits that require larger RPM
excursions to generate thrust asymmetry for attitude changes. The policy at 40M
uses 415 RPM of deviation — approximately 2.9% of hover RPM — which is well within
the linear operating range of the motor model.

### C. Implications for Sim-to-Real Transfer

The 31.5-percentage-point robustness gap identified in Section IV.E suggests that
the policy is sensitive to initial conditions in ways that may not be fully
observable from training metrics. The motor LPF (τ = 0.15 s at 100 Hz) amplifies
the effect of aggressive attitude changes: a rapid RPM command generates a delayed
thrust response that, combined with large initial tilts, can result in the drone
entering a recovery trajectory that the policy has not encountered often enough to
handle reliably. Closing this gap likely requires either longer training under full
randomisation, or domain randomisation of motor lag parameters to encourage more
robust recovery strategies.

### D. Episode-Level Failure Mode

In the chaining evaluation, 100% of episodes end with a crash (the drone eventually
crashes rather than reaching the episode step limit). This is an artefact of the
multi-waypoint chaining protocol without episode resets: once the policy fails a
difficult waypoint, the drone is in a position and velocity state that may be
outside its recovery envelope. This is a training mismatch — the policy was
trained with episode resets after each crash, but evaluated with persistent state
across failures. A curriculum that explicitly trains recovery from difficult
mid-chain states could address this.

---

## VI. Conclusion

We have presented a complete end-to-end learning pipeline for direct motor RPM
control of a nano-quadrotor using Proximal Policy Optimization. The central
finding is that reward shaping determines whether a PPO policy converges to
navigation or hovering behaviour at this control level: a gated potential-based
approach term combined with a velocity-gated settlement bonus is jointly necessary
for convergence to successful waypoint navigation, while either term alone is
insufficient or counterproductive.

Training to 40 million steps achieves 81.8% per-leg waypoint success under
clean-start conditions, with convergence occurring at approximately 30M steps.
Distance-stratified analysis shows degradation from 100% at < 0.5 m to 68.2% at
1.5–3.0 m, identifying long-range navigation as the primary remaining challenge.
The 31.5-percentage-point gap between clean and randomised spawn conditions
quantifies the robustness deficit that must be addressed before Sim-to-Real transfer
can be attempted.

Future work will target three directions: (1) improving robustness under disturbed
initialisation through domain randomisation of motor parameters and explicit
recovery training; (2) replacing the flat survival bonus with a shaped term that
does not create a hover attractor while remaining stable to train; and (3)
zero-shot Sim-to-Real validation on a physical Crazyflie 2.x to determine whether
the 40M-step policy is deployable under real hardware conditions.

---

## References

[1] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal
Policy Optimization Algorithms," arXiv:1707.06347, 2017.

[2] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, "Curriculum Learning,"
in *Proc. ICML*, 2009, pp. 41–48.

[3] D. Förster, "System Identification of the Crazyflie 2.0 Nano Quadrocopter,"
Bachelor's thesis, ETH Zürich, 2015.

[4] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann,
"Stable-Baselines3: Reliable Reinforcement Learning Implementations,"
*Journal of Machine Learning Research*, vol. 22, no. 268, pp. 1–8, 2021.

[5] NVIDIA, "Newton: GPU-Accelerated Rigid Body Simulation,"
https://github.com/newton-physics/newton, 2024.

[6] A. Y. Ng, D. Harada, and S. Russell, "Policy Invariance Under Reward
Transformations: Theory and Application to Reward Shaping,"
in *Proc. ICML*, 1999, pp. 278–287.

---

*Code available at: [repository URL]*
*Simulation environment: Newton physics engine + Stable-Baselines3.*
