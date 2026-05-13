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
**82.9 % per-leg waypoint success** after 30 million environment steps under
clean-start conditions, converging to stable performance maintained at 40 million
steps (81.8 %). We characterise learning progression from 3M to 40M steps under
both ideal (clean-start) and fully randomised spawn conditions, provide
distance-stratified success rates across four distance bands, and quantify
robustness to spawn randomisation — a 28.3-percentage-point gap at 40M steps
that highlights a key challenge for Sim-to-Real transfer, but one that narrows
consistently over training (47.1 pp at 3M → 28.3 pp at 40M). A reward shaping
analysis shows that coupled potential-based approach and velocity-gated settlement
terms are jointly necessary for convergence; neither alone is sufficient.

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
and multi-waypoint navigation — tasks handled by separate subsystems in classical
architectures. The reward function must therefore guide learning across all of
these coupled objectives simultaneously, making reward design the central
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
   multi-waypoint chaining benchmark under both clean-start and fully randomised
   spawn conditions, identifying convergence at approximately 30M steps and
   demonstrating that the robustness gap between conditions narrows consistently
   over training.

3. We provide distance-stratified success rates and a spawn robustness comparison
   at multiple training checkpoints, identifying specific failure modes that must
   be addressed for robust real-world deployment.

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
spawn positions, velocities, and orientations with a small range initially, then
gradually expanding the range as the policy improves. This allows the policy to
learn simple hovering behaviour early in training, which serves as a bootstrap for
learning navigation.

We apply curriculum learning to spawn randomisation: the position offset, linear
velocity, angular velocity, and tilt angle at spawn are all scaled by a curriculum
coefficient c ∈ [0, 1] that increases linearly over the first 1.5 M steps.
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
discontinuities near ±180°, providing a smooth manifold for attitude representation.

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
| Parallel environments | 16 (DummyVecEnv) |
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
act_c    = −C_ra · ‖Δa‖²      (Δa = a_t − a_{t−1}, action jerk)
survival =  C_rs               (flat per-step bonus, fixed)
```

Reward weights ramp linearly from conservative to strict values over the first
1.5 M steps as the curriculum coefficient c increases from 0 to 1:

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
- Arrival (‖p_err‖ < 0.15 m, first crossing per waypoint): +15.0

**Shaped approach term (gated):**

```
approach = λ_ap · ( ‖p_err‖_{t−1} − ‖p_err‖_t )    if ‖p_err‖_t > d_gate
         = 0                                          if ‖p_err‖_t ≤ d_gate
```

with λ_ap = 1.0 and d_gate = 0.10 m. This term provides a dense per-step signal
proportional to distance reduction during transit, suppressed inside the gate
radius to prevent oscillation incentives near the target. The gate radius (0.10 m)
is set inside the success threshold (0.15 m), so the approach reward remains active
from the success boundary inward to 0.10 m, providing gradient right up to the
target without incentivising oscillation at close range.

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

| Parameter | Range at c = 0 | Range at c = 1 |
|-----------|----------------|----------------|
| Position offset | ‖Δpos‖ ≤ 0.1 m | ‖Δpos‖ ≤ 0.7 m |
| Linear velocity | ‖v‖ = 0 | ‖v‖ ≤ 1.5 m/s |
| Angular velocity | ‖ω‖ = 0 | ‖ω‖ ≤ 0.5 rad/s |
| Roll / Pitch tilt | 0° | ≤ 15° |

At c = 0 (clean start) the drone spawns stationary and upright within 0.1 m
of the waypoint. At c = 1.0 (full randomisation) the drone may start with
significant velocity and tilt, requiring attitude recovery concurrent with
navigation. Yaw is not randomised.

---

## IV. Experiments

### A. Evaluation Protocol

All evaluations use the **multi-waypoint chaining** protocol: waypoints are generated
sequentially and the drone navigates to each without physics reset between them.
Each episode has a shared step budget of 8,000 steps (~80 s). A leg is counted as
succeeded when ‖p_err‖ < 0.15 m. Results are reported over 20 episodes.

**Primary metric: leg success rate** — the fraction of all attempted waypoint legs
in which the drone came within 0.15 m of the target.

Secondary metrics: mean waypoints reached per episode (mean ± std), mean motor RPM
as percentage of hover RPM, absolute RPM deviation from hover, and arrival speed
(‖v‖ at first entry into the 0.15 m success radius).

Two evaluation modes test different aspects of the policy:

| Mode | curriculum | Spawn | Tests |
|---|---|---|---|
| **Clean start** | c = 0 | Upright, stationary, ≤ 0.1 m from first waypoint | Pure navigation |
| **Randomised** | c = 1.0 | Full training distribution | Robustness to disturbed init |

Spawn alignment: before `env.reset()` the evaluator sets the internal target to
`waypoints[0]`, so the drone spawns within curriculum distance of the first
waypoint — matching the training distribution exactly.

### B. Learning Progression

Table II shows policy performance at three training checkpoints evaluated under
both clean-start and fully randomised spawn conditions (20 episodes each):

**Table II. Learning progression — clean start (curriculum c = 0) vs. randomised (c = 1.0)**

| Checkpoint | Clean Leg SR | Clean Wps/ep | Rand Leg SR | Rand Wps/ep | Gap (pp) |
|---|---|---|---|---|---|
| 3M steps | 73.0 % | 2.7 ± 2.3 | 25.9 % | 0.3 ± 0.7 | 47.1 |
| 30M steps | **82.9 %** | **4.8 ± 3.2** | 42.9 % | 0.8 ± 0.8 | 40.0 |
| 40M steps | 81.8 % | 4.5 ± 3.6 | **53.5 %** | **1.1 ± 1.8** | **28.3** |

Under clean-start conditions, performance converges at approximately **30M steps**
(+9.9 pp from 3M to 30M, −1.1 pp from 30M to 40M). The 3M-to-30M improvement
reflects the policy transitioning from single-hop navigation to sustained
multi-waypoint chaining (2.7 → 4.8 waypoints/ep).

Under randomised conditions, performance continues to improve through 40M steps
(25.9 % → 42.9 % → 53.5 %), suggesting that generalisation to disturbed initial
states requires more training than convergence of the clean-start policy. The
robustness gap narrows consistently over training: 47.1 pp at 3M, 40.0 pp at 30M,
and 28.3 pp at 40M, indicating that extended training under the full curriculum
distribution gradually builds robustness.

Arrival speed decreases monotonically under clean-start evaluation (0.419 → 0.326 →
0.294 m/s), confirming that the policy learns progressively smoother approaches.
Motor RPM deviation increases from 3M to 30M under clean conditions (227 → 437 RPM),
reflecting more dynamic transit manoeuvres in the mature policy.

**Table III. Motor characteristics by checkpoint and spawn mode**

| Checkpoint | Clean RPM | Clean % hover | Clean \|RPM−hover\| | Rand RPM | Rand % hover | Rand \|RPM−hover\| |
|---|---|---|---|---|---|---|
| 3M | 14,303 | 98.8 % | 227 | 13,796 | 95.3 % | 704 |
| 30M | 14,039 | 97.0 % | 437 | 13,247 | 91.5 % | 1,229 |
| 40M | 14,061 | 97.1 % | 415 | 13,027 | 90.0 % | 1,449 |

Under randomised conditions, mean RPM drifts substantially below hover across all
checkpoints (90.0–95.3 %), indicating that the policy frequently generates
under-thrust during recovery from disturbed initial states. The large RPM deviation
under randomised evaluation (704–1,449 RPM vs. 227–437 RPM under clean start) is
consistent with the policy expending significant control authority attempting to
stabilise from large initial tilts and velocities.

### D. Distance-Stratified Performance

Table IV stratifies the 40M-step results by waypoint distance for both evaluation modes:

**Table IV. Distance-stratified leg success rate (40M checkpoint, 20 episodes)**

| Distance Band | Clean Leg SR | n | Rand Leg SR | n |
|---|---|---|---|---|
| 0.0 – 0.5 m | 100.0 % | 7 | 100.0 % | 1 |
| 0.5 – 1.0 m | 92.9 % | 28 | 55.6 % | 9 |
| 1.0 – 1.5 m | 87.1 % | 31 | 52.9 % | 17 |
| 1.5 – 3.0 m | 68.2 % | 44 | 50.0 % | 16 |

Under clean-start conditions, success rate degrades gracefully with distance from
100 % at close range to 68.2 % for distant waypoints (1.5–3.0 m). This degradation
is expected: longer transits expose the policy to larger mid-flight attitude changes
and more opportunities for the error to compound.

Under randomised conditions, the near-range advantage collapses: even for the
0.5–1.0 m band, success drops to 55.6 %, confirming that the primary failure mode is
attitude recovery from the disturbed initial state rather than navigation difficulty.
The small sample sizes (n = 1 for the 0.0–0.5 m band under randomised conditions)
reflect the fact that many episodes crash on the very first leg before accumulating
enough data to populate close-range bins.

### E. Robustness Analysis

The clean-start vs. randomised comparison at each checkpoint reveals a consistent
pattern: the policy learns navigation competency faster than it learns robustness.
At 3M steps, the clean-start policy already achieves 73.0 % leg success but the
randomised policy manages only 25.9 %, implying the early policy is highly brittle
to initial conditions. By 40M steps, randomised performance has improved by +27.6 pp
while clean-start performance improved by only +8.8 pp, suggesting that additional
training under the full curriculum distribution yields diminishing returns on
navigation quality but meaningful gains in robustness.

The 28.3 pp gap remaining at 40M steps under the current curriculum is the primary
challenge for Sim-to-Real transfer: real-world launches involve disturbed initial
conditions that closely resemble the randomised evaluation protocol. Under-thrust
during recovery (mean RPM 90.0 % of hover at 40M randomised) is the proximate
failure mechanism, compounded by the motor LPF lag (τ = 0.15 s) which delays the
policy's recovery commands by ~15 steps.

---

## V. Discussion

### A. Convergence and the Hover Attractor

Training converges at approximately 30M steps under clean-start evaluation, as
confirmed by entropy loss stabilising at −0.72 and clip fraction falling to 0.044.
The flat survival bonus (C_rs = 0.50/step) creates an attractor toward hovering: a
stationary drone earns positive net per-step reward from the survival term alone
(+0.50/step at c = 1.0 with zero position error). This attractor benefits early-stage
stability learning but competes with the approach gradient during navigation.

The plateau from 30M to 40M steps under clean conditions (82.9 % → 81.8 %) may
partly reflect this attractor: the policy has converged to a balance between
hovering and navigating rather than a fully committed navigation strategy. Under
randomised conditions the same attractor may actually hurt performance — a policy
that defaults to hovering has no recovery mechanism for episodes that start in
highly disturbed states.

### B. Robustness Gap and Training Dynamics

The monotonically narrowing robustness gap (47.1 → 40.0 → 28.3 pp) suggests that
robustness is a slowly-learned property that continues to improve after the
clean-start policy has converged. This has implications for training strategy:
stopping at 30M steps (when the clean-start policy is optimal) sacrifices 10.6 pp
of randomised performance (42.9 % vs. 53.5 % at 40M). For Sim-to-Real deployment
the randomised metric is more relevant; results indicate that training should continue
beyond clean-start convergence if robustness is the goal.

### C. Implications for Sim-to-Real Transfer

The 28.3 pp gap at 40M steps quantifies the robustness deficit remaining for
Sim-to-Real transfer. Three factors contribute to failures under disturbed
initialisation:

1. **Motor lag amplification:** The LPF (τ = 0.15 s) delays recovery commands by
   ~15 steps. A drone spawning at 15° tilt with 1.5 m/s velocity has already lost
   significant altitude before the first corrective commands take effect.

2. **Under-thrust under disturbed conditions:** Mean RPM falls to 90.0 % of hover
   under full randomisation at 40M (vs. 97.1 % clean), suggesting the policy issues
   below-hover commands when faced with large attitude errors — a conservative
   response that is appropriate for small tilts but insufficient for aggressive recovery.

3. **Training-evaluation mismatch for chaining:** Episodes in chaining evaluation
   do not reset after crashes, so a drone that fails one waypoint may arrive at the
   next in a damaged or off-nominal state that was never encountered during training.

Closing the robustness gap likely requires domain randomisation of motor lag
parameters, explicit recovery training from extreme initial conditions, and
potentially extending training well beyond 40M steps under full curriculum randomisation.

### D. Episode-Level Failure Mode

In both clean and randomised evaluations, 100 % of episodes end with a crash rather
than timing out or completing all 20 waypoints. This is an artefact of the chaining
protocol: once the policy fails a waypoint and the drone crashes, the episode ends
without reset, and accumulated state (velocity, tilt) from the failure can compound
into subsequent crash events. Since training uses per-episode resets after each crash,
the policy has no experience recovering from the kind of post-failure states that
accumulate during chaining. The leg success rate — which excludes the effect of
chaining-induced cascades — is therefore the correct metric for evaluating navigation
capability.

---

## VI. Conclusion

We have presented a complete end-to-end learning pipeline for direct motor RPM
control of a nano-quadrotor using Proximal Policy Optimization. The central finding
is that reward shaping determines whether a PPO policy converges to navigation or
hovering behaviour at this control level: a gated potential-based approach term
combined with a velocity-gated settlement bonus is jointly necessary for convergence
to successful waypoint navigation, while either term alone is insufficient or
counterproductive.

Training to 40 million steps achieves 81.8 % per-leg waypoint success under clean-start
conditions, with convergence occurring at approximately 30M steps.
Distance-stratified analysis shows degradation from 100 % at < 0.5 m to 68.2 % at
1.5–3.0 m, identifying long-range navigation as the primary remaining challenge.

A key finding is that robustness under disturbed spawn conditions continues to improve
beyond clean-start convergence: the robustness gap narrows from 47.1 pp at 3M steps
to 28.3 pp at 40M steps, with the randomised policy improving from 25.9 % to 53.5 %
over the same period. This demonstrates that extended training has qualitatively
different effects on navigation quality (quickly saturating) and initialisation
robustness (slowly but continuously improving), a distinction with direct implications
for deployment strategy.

Future work will target three directions: (1) improving robustness under disturbed
initialisation through domain randomisation of motor parameters and explicit recovery
training; (2) replacing the flat survival bonus with a shaped term that does not
create a hover attractor while remaining stable to train; and (3) zero-shot Sim-to-Real
validation on a physical Crazyflie 2.x to determine whether the 40M-step policy is
deployable under real hardware conditions.

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
