# TensorBoard Metrics — DroneSimPPO Analysis Guide

Simulation: Newton/Warp · Crazyflie 2.x · 100 Hz · MAX_EPISODE_STEPS = 800 (8 s)  
Algorithm: PPO (SB3) · obs = 22-D · action = 4-D normalised RPM setpoints

---

## 1. metrics/ — Flight Performance

These are logged per **episode end** (rolling window of last 100 episodes).

---

### `metrics/success_rate`
**What it is:** Fraction of the last 100 episodes where `terminal_dist < 0.15 m`.

**Formula:**
```
success = 1  if  ‖pos − target‖ < 0.15 m at episode end
        = 0  otherwise
```

**Healthy progression:**
- `0.0` early → climbing toward `1.0` by end of training
- PPO on this task: expect slow rise, often stuck near 0 without entropy tuning
- TD3 (paper): reaches reliable success by ~300K steps

**If stuck at 0:** drone never gets within 15 cm of target. Check `terminal_dist` — if it's plateaued above 1.0 m, policy is in a local minimum (survival bonus dominating).

---

### `metrics/terminal_dist`
**What it is:** Mean distance (metres) between drone and target **at episode end** (crash or timeout).

**Formula:**
```
terminal_dist = ‖pos_final − target‖  [metres]
```

**Healthy progression:**
- Starts ~0.8–1.2 m (random spawn near target at curriculum=0)
- Rises as curriculum increases spawn range (up to 0.7 m offset + random target)
- Should fall below 0.15 m for successful episodes

**Interpretation table:**

| Value | Meaning |
|---|---|
| > 2.0 m | Drone barely reacts to target — random or collapsed policy |
| 0.5 – 2.0 m | Drone moving toward target but crashing before arriving |
| 0.15 – 0.5 m | Getting close — near-success, refine further |
| < 0.15 m | Success threshold crossed |

**Causal link to reward:** `pos_c = -C_rp × ‖p_err‖²`. If `terminal_dist` is high, `pos_c` is large-negative throughout episodes — the main loss driver that should motivate learning.

---

### `metrics/terminal_upright`
**What it is:** R₂₂ value at episode end — the dot product of drone's Z-axis with world Z-axis.

**Formula:**
```
R22 = 1 - 2(qx² + qy²)   (from quaternion)
R22 = +1.0  → perfectly upright
R22 =  0.0  → 90° tilted
R22 = -1.0  → fully inverted
```
Episode terminates if `R22 < -0.5` (severely inverted).

**Healthy progression:**
- Should rise from ~0 toward `0.9+`
- If `terminal_upright` is high but `success_rate = 0`: drone maintains orientation but can't reach target (position local minimum)
- If `terminal_upright` stays low: drone tumbles — orientation cost `orient_c` is not being minimised

**Causal link to reward:** `orient_c = -0.10 × (1 - qw²)`. When R22 = 1.0, qw = 1.0, orient_c = 0 (no penalty). When inverted, orient_c ≈ -0.10.

---

### `metrics/ep_length`
**What it is:** Mean number of steps per episode (rolling 100 ep). Max = 800 steps = 8 seconds.

**Formula:**
```
ep_length = steps until terminated OR truncated (≥ 800 steps)
```

**Termination triggers (premature):**
- `z < 0.05 m` — ground impact
- `z > 6.0 m` — escaped upward
- `R22 < -0.5` — severely inverted
- `dist > 4.0 m` — left arena

**Healthy progression:**
- Starts low (~60–100 steps) as drone crashes quickly
- Should rise toward 800 as drone learns to stay alive
- Once near 800 consistently: drone survives full episodes, `success_rate` becomes the key metric

**Warning sign:** `ep_length` declining after initially rising = policy destabilising (entropy collapse, overfitting, or curriculum shock).

---

### `metrics/ep_reward`
**What it is:** Total cumulative reward summed over the full episode.

**Formula:**
```
ep_reward = Σ(pos_c + orient_c + vel_c + ang_c + act_c + survival + crash + approach + hover_bonus)
```

**Reference values at curriculum=1.0, hovering 1m from target:**
```
survival  = +0.50 / step × 800 steps = +400  (best case, full episode)
pos_c     ≈ -1.00 × 1.0²  = -1.00 / step × 800 = -800  (1m away, strict penalty)
orient_c  ≈  0.00          (upright)
vel_c     ≈ -0.30 × 0.0   =  0.00  (hovering, v≈0)
Total/ep  ≈ -400  (bad — survival barely covers position penalty)
```

**Good sign:** ep_reward trending upward. Does NOT directly mean success — the drone can improve ep_reward by simply crashing less (longer episodes × survival bonus) without ever reaching the target.

---

### `metrics/mean_motor_rpm`
**What it is:** Mean of all 4 motor RPMs averaged over the last 100 episodes, logged per step.

**Reference values:**
```
CF_HOVER_RPM ≈ 14,476 RPM   (thrust exactly balances gravity)
CF_MIN_RPM   =  1,000 RPM   (ESC idle)
CF_MAX_RPM   = 21,702 RPM   (full throttle)
```

**Critical diagnostic:**

| Value | Meaning |
|---|---|
| ≈ 14,476 | Policy commanding hover thrust — healthy baseline |
| < 13,000 | Under-thrusting → drone falling → short episodes |
| > 16,000 | Over-thrusting → drone climbing → terminal at z > 6m |
| Declining over training | Policy learned to "give up" on position, reduce thrust |

**Root cause when below hover:** The action smoothness penalty (`act_c`) discourages large RPM changes. If the policy initialises below hover and is penalised for changing, it can lock in at low RPMs. Also caused by the survival bonus being achievable without maintaining altitude.

---

### `metrics/rpm_hover_dev`
**What it is:** Absolute deviation of mean motor RPM from hover RPM.

**Formula:**
```
rpm_hover_dev = |mean_motor_rpm - 14476|
```

**Healthy value:** Should decrease over training toward ~0–500 RPM as the policy learns to maintain hover thrust near the target.

**Warning:** If this is **increasing** throughout training (as seen in the failed runs), the policy is diverging from stable hover — strong signal of a local minimum where the drone trades off flight stability for some other reward component.

---

## 2. reward_components/ — Per-Step Reward Breakdown

All components are **mean per-step values** over the last 100 episodes. They sum to the per-step reward signal the policy optimises.

---

### `reward_components/pos_c`
**What it is:** Position error penalty — the primary navigation objective.

**Formula:**
```
C_rp = 0.05 + curriculum × (1.00 - 0.05)   [ramps 0.05 → 1.00]
pos_c = -C_rp × ‖p_err‖²
```

**Range:** `0.0` (at target) → large negative (far from target at curriculum=1.0)

**At 1 m distance, curriculum=1.0:** `pos_c = -1.00 × 1.0² = -1.0 / step`

**Healthy progression:** Should rise (become less negative) as drone gets closer to target. If flat at -0.5 to -1.0 for millions of steps → drone is stuck at a fixed distance.

**Curriculum effect:** At `curriculum=0` the penalty weight is 0.05 — very gentle. At `curriculum=1.0` it's 20× stricter. This is intentional: early training allows the drone to learn basic flight before position precision is demanded.

---

### `reward_components/orient_c`
**What it is:** Orientation penalty — penalises tilting away from upright.

**Formula:**
```
orient_c = -0.10 × (1 - qw²)     [fixed weight, no curriculum]
```

**Range:**
- `0.0` — perfectly upright (qw = ±1.0)
- `-0.10` — fully inverted (qw = 0.0)
- `-0.075` — 90° tilted

**Healthy value:** Should approach `0.0` as drone learns attitude stabilisation. Usually converges quickly (within 1–2M steps for PPO).

**Note:** This is fixed-weight (does not ramp with curriculum) — orientation stability is expected from the start.

---

### `reward_components/vel_c`
**What it is:** Linear velocity penalty — penalises high-speed motion.

**Formula:**
```
C_rv = 0.01 + curriculum × (0.30 - 0.01)   [ramps 0.01 → 0.30]
vel_c = -C_rv × ‖v‖²    (v = linear velocity, world frame)
```

**At 1 m/s speed, curriculum=1.0:** `vel_c = -0.30 × 1.0 = -0.30 / step`

**Healthy progression:** Should improve (less negative) as drone slows down near the target. A policy that rushes the target will have poor `vel_c` but good `approach`. Balance between the two is desired.

**Interaction with approach reward:** `approach` rewards getting closer; `vel_c` penalises moving fast. The policy must learn to decelerate before arrival — matching real flight.

---

### `reward_components/ang_c`
**What it is:** Angular velocity penalty — penalises spinning/tumbling.

**Formula:**
```
C_rw = 0.001 + curriculum × (0.05 - 0.001)   [ramps 0.001 → 0.05]
ang_c = -C_rw × ‖ω‖²    (ω = angular velocity, body frame)
```

**At 1 rad/s spin, curriculum=1.0:** `ang_c = -0.05 / step`

**Healthy value:** Should stay close to `0.0`. If strongly negative, drone is spinning — usually caused by unbalanced RPM commands or tumbling during recovery.

---

### `reward_components/act_c`
**What it is:** Action smoothness penalty — penalises rapid RPM changes between steps.

**Formula:**
```
C_ra = 0.005 + curriculum × (0.02 - 0.005)   [ramps 0.005 → 0.02]
delta_a = action_t - action_{t-1}
act_c = -C_ra × ‖delta_a‖²    (action ∈ [-1,1]⁴)
```

**Purpose:** Prevents jerky motor commands that would stress real hardware and cause oscillations. Mimics the paper's smooth control requirement for Sim2Real.

**Warning:** If `act_c` is overly negative AND `mean_motor_rpm` is below hover, the smoothness penalty may be preventing the policy from learning to increase thrust. The drone is "penalised into passivity." This is a known failure mode for PPO on this task.

**Healthy value:** Small negative, e.g. `-0.05` to `-0.08`. If larger than `-0.2`, the policy is making excessively large RPM jumps.

---

### `reward_components/survival`
**What it is:** Fixed per-step bonus for staying alive (not terminating).

**Formula:**
```
survival = +0.50   (constant, every step the episode continues)
```

**This is the most dangerous metric.** At `curriculum=0`, spawn is only 0.1 m from target. Survival bonus (0.50/step) is designed to exceed the position penalty at spawn (`-C_rp × 0.1² = -0.0005`), giving positive returns from the first update.

**The local minimum trap:** At `curriculum=1.0`, spawn is up to 0.7 m from target:
```
survival = +0.50
pos_c    = -1.00 × 0.7² = -0.49
net      = +0.01  (barely positive — drone survives without flying)
```
The drone can collect 0.50/step just by hovering in place, making position improvement marginally rewarding. This is why `survival = 0.5` flat with `success_rate = 0` is the classic PPO failure signature on this task.

**If this is 0.5 constant and ep_reward is flat:** policy is locked into survival-without-navigation local minimum.

---

### `reward_components/approach`
**What it is:** Potential-based shaping reward — rewards getting closer to target each step.

**Formula:**
```
approach = 1.0 × (last_dist - dist)    if dist > 0.10 m
         = 0.0                          if dist ≤ 0.10 m  (suppressed near target)
```
Gate at 0.10 m prevents oscillation reward (drone bouncing around target to generate approach signal).

**Healthy value:** Small positive when drone is approaching (`+0.001` to `+0.01` per step means ~0.01–10 cm/step progress). Negative means drone is moving away.

**Important:** This is not curriculum-ramped — it provides dense navigation signal from step 0, which is critical for PPO to receive a gradient before the position penalty becomes strict.

---

### `reward_components/hover_bonus`
**What it is:** Dense bonus for settling at the target with low speed.

**Formula:**
```
hover_bonus = +0.15   if dist < 0.15 m  AND  ‖v‖ < 0.5 m/s
            =  0.0    otherwise
```

**Purpose:** Rewards deceleration at target, not just passing through it. Without the speed gate, a policy could rush through the waypoint at full speed and collect the bonus.

**If hover_bonus ≈ 0.0 throughout training:** drone never reaches within 15 cm of target. This confirms `success_rate = 0` is not a scoring artifact.

**If hover_bonus is positive but success_rate is 0:** the 100-episode window definition mismatch — check `terminal_dist` directly.

---

## 3. train/ — PPO Algorithm Internals

These are SB3's standard PPO diagnostics, logged per policy update.

---

### `train/explained_variance`
**What it is:** How well the value function predicts actual returns.

**Formula:**
```
explained_variance = 1 - Var(returns - values) / Var(returns)
```

**Range:** `−∞` to `1.0`

| Value | Meaning |
|---|---|
| `> 0.8` | Value function is well calibrated — advantage estimates are meaningful |
| `0.5 – 0.8` | Moderate — value function still learning |
| `< 0.0` | Value function is worse than predicting the mean — broken critic |

**Implication:** High `explained_variance` does NOT mean the policy is good. In the failed runs, `explained_variance = 0.86` while `success_rate = 0` — the critic accurately predicted the bad returns from the stuck policy. It only means PPO can compute accurate advantages from whatever trajectory it is currently executing.

---

### `train/entropy_loss`
**What it is:** Negative entropy of the policy's action distribution (continuous Gaussian).

**Formula:**
```
entropy_loss = -H(π)   where H = sum of Gaussian entropies per action dim
```
More negative = **lower entropy = more deterministic policy**.

**Critical diagnostic:**

| Value | Meaning |
|---|---|
| `-2` to `-3` | High entropy — lots of exploration |
| `-3` to `-5` | Moderate — policy converging |
| `< -5` | Low entropy — policy nearly deterministic, **exploration collapsed** |

**The failure mode:** When `entropy_loss` drops below `-5`, the policy has committed to one behavior pattern. If that pattern is the local minimum (hover in place), it will never escape. This is why `ent_coef = 0.02` was increased from `0.005` — the entropy regularisation term in the loss directly resists this collapse.

**Relationship to `ent_coef`:**
```
PPO loss += ent_coef × entropy_loss
```
Higher `ent_coef` adds a larger gradient push to maintain diversity in actions.

---

### `train/clip_fraction`
**What it is:** Fraction of policy update steps where the probability ratio was clipped.

**Formula:**
```
ratio = π(a|s) / π_old(a|s)
clipped when |ratio - 1| > clip_range
```

**Healthy range:** `0.05 – 0.10` (5–10%)

| Value | Meaning |
|---|---|
| `< 0.05` | Policy barely changing — too conservative or gradient too weak |
| `0.05 – 0.10` | Healthy — clip_range is appropriate |
| `> 0.15` | Policy trying to change too fast — consider reducing `learning_rate` or `clip_range` |

**In the failed runs:** `clip_fraction ≈ 0.12–0.13` → policy was consistently hitting the clip ceiling. This contributed to slow learning — the policy wanted to make bigger updates but was being held back. Increasing `clip_range` to `0.3` directly addresses this.

---

### `train/approx_kl`
**What it is:** KL divergence between old and new policy (approximation).

**Formula:**
```
approx_kl ≈ mean((ratio - 1) - log(ratio))
```

**Healthy range:** `0.005 – 0.02`

| Value | Meaning |
|---|---|
| `< 0.005` | Tiny policy update — barely learning |
| `0.005 – 0.02` | Good — policy changing at a reasonable rate |
| `> 0.05` | Large policy shift — training might be unstable |

Closely related to `clip_fraction`. If KL is low but `clip_fraction` is high, many small individual updates are being clipped (lots of actions near the clip boundary).

---

### `train/policy_gradient_loss`
**What it is:** The clipped surrogate policy gradient loss value.

**Formula (PPO clip objective):**
```
L_PG = -E[min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)]
```
Negative because SB3 minimises loss (gradient ascent on reward).

**Healthy progression:** Should start more negative (large updates) and rise toward 0 as policy converges.

**Warning:** If `policy_gradient_loss ≈ 0` very early and stays there, the policy gradient signal has vanished:
- Advantages `A ≈ 0` (all trajectories give similar returns — stuck in local min)
- OR ratio stays at 1 (policy not updating)

This was the key diagnostic in the failed runs — `policy_gradient_loss ≈ -0.002` while `success_rate = 0`. The policy had no gradient signal to escape the local minimum.

---

### `train/value_loss`
**What it is:** MSE between predicted values and actual returns.

**Formula:**
```
value_loss = MSE(V(s), R_t)   where R_t = discounted return
```

**Healthy progression:** Should decrease over training. High value loss early is normal — critic is learning. Converging to near 0 means critic accurately predicts returns.

**Interaction:** As `value_loss` decreases, `explained_variance` increases. Both should move together. Divergence (value_loss high, explained_variance high) would indicate a bug in return computation.

---

### `train/std`
**What it is:** Mean standard deviation of the policy's Gaussian action distribution.

**Range:** Action space is `[-1, 1]⁴` (normalised RPM). Std starts at SB3 default `~1.0`.

| Value | Meaning |
|---|---|
| `> 0.8` | High exploration — lots of RPM variation |
| `0.4 – 0.8` | Moderate exploration |
| `< 0.3` | Low exploration — policy nearly deterministic |

**Relationship to entropy_loss:** Both measure the same thing. `std` is more intuitive — `std = 0.5` means the policy samples RPM offsets ±0.5 × `CF_RPM_RANGE` ≈ ±3,600 RPM from its mean. Enough to explore different flight regimes.

---

### `train/clip_range`
**What it is:** The PPO ε parameter — just confirms your configured value (0.3 after our change).

Constant unless you use a decaying clip schedule. Not a diagnostic on its own.

---

### `train/learning_rate`
**What it is:** Current learning rate. Useful to confirm `--lr_final` decay is working.

With `--learning_rate 1e-3 --lr_final 1e-5`, this should decay linearly from `1e-3` at step 0 to `1e-5` at step 10M. If it stays flat, `--lr_final` was not set.

---

## 4. Quick Diagnosis Reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `success_rate = 0`, `survival = 0.5` flat | Survival bonus local minimum | Increase `ent_coef`, extend `curriculum_steps` |
| `mean_motor_rpm` declining | Under-thrust, policy giving up on flight | Increase `ent_coef`, check `act_c` magnitude |
| `entropy_loss < -5` | Entropy collapsed, policy stuck | Increase `ent_coef` (current: 0.02) |
| `clip_fraction > 0.15` | Policy hitting clip ceiling | Increase `clip_range` (current: 0.3) |
| `policy_gradient_loss ≈ 0` | No advantage signal, stuck in flat reward basin | Entropy fix + `clip_range` increase |
| `ep_length` declining after peak | Policy destabilising (entropy too high or LR too high) | Reduce `learning_rate` |
| `explained_variance < 0.5` | Value function not converging | Reduce `vf_coef`, check `batch_size` |
| `terminal_dist` plateaued > 1.0 m | Drone not navigating at all | Check `approach` component — should be > 0 |
| `approach ≈ 0` throughout | Drone not moving toward target | Reward collapse — re-check curriculum and ent_coef |

---

## 5. Reward Equation Summary

```
r(s,a,s') = pos_c + orient_c + vel_c + ang_c + act_c
           + survival + crash + arrival + approach + hover_bonus

pos_c     = -C_rp(t) × ‖p_err‖²          C_rp: 0.05 → 1.00
orient_c  = -0.10    × (1 − qw²)
vel_c     = -C_rv(t) × ‖v‖²              C_rv: 0.01 → 0.30
ang_c     = -C_rw(t) × ‖ω‖²             C_rw: 0.001 → 0.05
act_c     = -C_ra(t) × ‖Δa‖²            C_ra: 0.005 → 0.02
survival  = +0.50    (every step)
crash     = -2.00    if z < 0.05 m
arrival   = +15.0    first time dist < 0.15 m (one-shot)
approach  = +1.0 × (last_dist − dist)     if dist > 0.10 m
hover_bonus = +0.15  if dist < 0.15 m AND ‖v‖ < 0.5 m/s
```

Curriculum ramps linearly from 0 → 1 over `--curriculum_steps` (default 1.5M, recommended 3M for 10M PPO runs).

---

## 8. Survival Bonus Redesign — Breaking the Hover Local Minimum

### Why the flat survival bonus fails

At `curriculum=1.0` with spawn at 0.7 m from target, per-step net reward doing nothing:

```
survival (flat)  = +0.50
pos_c            = -1.00 × 0.7² = -0.49
─────────────────────────────────────────
net              = +0.01 / step  ← hovering costs almost nothing
```

The position penalty and survival bonus nearly cancel out. `approach` (+0.01/step for 1 cm/step motion) is too small to overcome PPO's gradient noise. The drone sits still, collects +0.01/step, and never moves.

### Solution: Distance × Time decaying survival

**Formula:**
```python
dist_factor = exp(-SURVIVAL_DIST_SCALE × dist)     # = exp(-1.0 × dist)
time_factor = TIME_MIN + (1 - TIME_MIN) × (1 - step / MAX_STEPS)
            = 0.2 + 0.8 × (1 - step/800)           # 1.0 → 0.2 over episode
survival    = 0.50 × dist_factor × time_factor
```

### Net reward at key scenarios (curriculum=1.0)

| Situation | survival | pos_c | net/step |
|---|---|---|---|
| Hovering at spawn (dist=0.7m, t=0) | +0.25 | -0.49 | **-0.24** — must move |
| Approaching (dist=0.3m, t=400) | +0.19 | -0.09 | **+0.10** — rewarded |
| At target (dist=0m, any time) | +0.50→+0.10 | 0 | **positive always** |
| Early training spawn (dist=0.1m, curriculum=0) | +0.45 | -0.0005 | **+0.45** — safe ✓ |

### What changed behaviourally

| Behaviour | Old survival | New survival |
|---|---|---|
| Hover 0.7m from target | net ≈ 0 — stable local min | net = −0.24 — unstable, must move |
| Approach over 400 steps | marginal gain | clearly positive gradient |
| Reach and stay at target | +0.5+0.15/step | +0.5→+0.3/step + hover_bonus (still best strategy) |
| Crash early (t=0→50) | loses future survival | loses future survival (same) |

### Time pressure effect

The `time_factor` decays from 1.0 → 0.2 over the episode:
- **Early in episode:** full survival bonus if near target → drone learns to fly there fast
- **Late in episode:** survival shrinks even at target → drone must arrive early to maximise cumulative reward
- **Implicit urgency:** a drone that reaches the target at step 100 earns ~5× more total survival than one arriving at step 700

Constants in [drone_gym_env.py](drone_gym_env.py):
```python
_SURVIVAL_DIST_SCALE = 1.0   # tune higher (e.g. 2.0) to decay faster with distance
_SURVIVAL_TIME_MIN   = 0.2   # minimum floor at episode end — keeps stability incentive alive
```

---

## 6. SB3 PPO Constructor Parameters

Reference: [train_drone.py:380-394](train_drone.py#L380-L394)

These parameters control the PPO optimisation loop — separate from the reward design above.

---

### `"MlpPolicy"`
**What it is:** Policy architecture type. `MlpPolicy` = fully-connected neural network for both actor and critic.

SB3 options: `MlpPolicy` (flat obs), `CnnPolicy` (image obs), `MultiInputPolicy` (dict obs).

For this task, obs is a 22-D vector `[p_err(3), R_flat(9), v(3), ω(3), a_hist(4)]` — `MlpPolicy` is correct. Both actor and critic share the same `net_arch` backbone then split into separate heads.

---

### `n_steps = 2048`
**What it is:** Number of steps each environment collects before a policy update. Also called the **rollout length**.

**Formula:**
```
rollout_buffer_size = n_steps × n_envs
                    = 2048   × 64    = 131,072 transitions per update
```

**Effect on training:**

| Value | Effect |
|---|---|
| Low (256–512) | More frequent updates, less diverse data per update, poorer GAE estimates |
| Medium (1024–2048) | Balanced — SB3 default, good for most continuous control |
| High (4096+) | Fewer updates, very diverse rollout, better advantage estimates for long horizons |

**Drone-specific:** `MAX_EPISODE_STEPS = 800`. With `n_steps=2048` and 64 envs, each rollout covers roughly `2048/800 ≈ 2.5` full episodes per env on average. This ensures GAE has complete episodes to work with.

**Why not lower:** Setting `n_steps=512` (tried earlier) caused near-zero success rate. The drone's task requires the policy to see the full approach-hover sequence (~200–500 steps) in a single rollout for the advantage function to assign credit correctly.

---

### `batch_size = 64`
**What it is:** Number of transitions fed into the network per gradient update step (mini-batch size).

**Formula:**
```
mini_batches_per_epoch = rollout_buffer_size / batch_size
                       = 131,072 / 64 = 2,048 mini-batches per epoch
total_gradient_steps   = n_epochs × mini_batches_per_epoch × n_rollouts
                       = 10 × 2,048 × 76 ≈ 1,556,480  (for 10M steps)
```

**Effect:**

| batch_size | Mini-batches/epoch | Gradient steps total | GPU utilisation |
|---|---|---|---|
| 64 (current) | 2,048 | 1,556,480 | Poor (tensor cores idle) |
| 512 | 256 | 194,560 | Better |
| 2048 | 64 | 48,828 | Good — fills GPU |

**Constraint:** `batch_size` must divide `n_steps × n_envs` evenly.
```
131,072 % 64   = 0  ✓
131,072 % 512  = 0  ✓
131,072 % 2048 = 0  ✓
```

**Drone recommendation:** `512` — reduces wasted gradient steps while keeping enough mini-batches per epoch (256) for stable optimisation.

---

### `n_epochs = 10`
**What it is:** How many full passes over the rollout buffer per policy update. Each pass reshuffles into `batch_size` mini-batches.

**PPO tradeoff:** More epochs = more gradient steps per experience batch = better data efficiency. But PPO's clipping is designed to prevent over-optimising on stale data. Too many epochs → policy drifts far from the data-collection policy → clipping becomes ineffective.

**Effect:**

| n_epochs | Behaviour |
|---|---|
| 3–5 | Conservative, stable but less data-efficient |
| 10 (current) | SB3 default, good balance for most tasks |
| 20+ | Risk of over-fitting to rollout — KL divergence may spike |

**Drone-specific:** 10 is appropriate. Reducing to 5 would be safer if `approx_kl` spikes are observed.

---

### `gamma = 0.99`
**What it is:** Discount factor — how much future rewards are worth relative to immediate ones.

**Formula:**
```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
```

**Effect:**

| gamma | Effective horizon | Behaviour |
|---|---|---|
| 0.90 | ~10 steps | Short-sighted — optimises immediate reward only |
| 0.99 (current) | ~100 steps | Balanced — considers ~1 second of future at 100Hz |
| 0.999 | ~1000 steps | Long-horizon — 10 seconds of future |

**Drone-specific:** At 100Hz, `gamma=0.99` gives an effective horizon of `1/(1-0.99) = 100 steps = 1 second`. The approach-to-hover sequence takes 2–5 seconds, so the drone needs to "see" further ahead. `gamma=0.995` (horizon ~200 steps = 2s) could help for long approach trajectories, but increases variance in advantage estimates.

---

### `gae_lambda = 0.95`
**What it is:** GAE (Generalised Advantage Estimation) λ — controls bias/variance tradeoff in advantage estimates.

**Formula:**
```
A_t^GAE = Σ_{k=0}^{∞} (γλ)^k · δ_{t+k}
where δ_t = r_t + γ·V(s_{t+1}) - V(s_t)  (TD error)
```

**Effect:**

| gae_lambda | Bias | Variance | Behaviour |
|---|---|---|---|
| 0.0 | Low | Low | Pure TD(0) — uses only 1-step lookahead |
| 0.95 (current) | Medium | Medium | SB3 default, good balance |
| 1.0 | High | High | Monte Carlo returns — uses full episode |

**Drone-specific:** `0.95` is standard. Lowering to `0.90` reduces variance in advantage estimates at the cost of more bias — useful if training is unstable. Raising toward `1.0` makes the advantage signal more accurate but noisier.

---

### `clip_range = 0.2`
**What it is:** PPO's ε clipping parameter — the maximum allowed change in the policy per update step.

**Formula:**
```
L_CLIP = E[min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)]
ratio  = π_θ(a|s) / π_θ_old(a|s)
```

With `clip_range=0.2`, the policy probability ratio is constrained to `[0.8, 1.2]` per step.

**Effect:**

| clip_range | Policy update size | Risk |
|---|---|---|
| 0.1 | Very conservative | Slow learning, may never escape local minima |
| 0.2 (current) | Standard | Good default, but clip_fraction >0.12 means updates hit ceiling |
| 0.3 (recommended) | Moderate | Allows bigger escapes from local minima |
| 0.5+ | Aggressive | Training instability risk |

**Drone diagnosis:** In failed runs, `clip_fraction = 0.12–0.13` with `clip_range=0.2` means the policy was consistently hitting the ceiling — it wanted to make larger updates but couldn't. Setting `clip_range=0.3` gives 50% more room before clipping triggers.

---

### `ent_coef = 0.005`
**What it is:** Entropy regularisation coefficient — adds a bonus to the loss for maintaining action diversity.

**Formula:**
```
L_total = L_CLIP - vf_coef·L_VF + ent_coef·H(π)
H(π)    = entropy of the action distribution (higher = more random)
```

**Effect:**

| ent_coef | Behaviour |
|---|---|
| 0.0 | No entropy bonus — policy free to collapse to deterministic |
| 0.005 (current) | Weak regularisation — insufficient for this task (entropy collapsed to −5.5) |
| 0.02 (recommended) | Moderate — keeps policy exploring, resists local minimum trapping |
| 0.1+ | Very high — policy stays random too long, slow convergence |

**This is the most critical parameter for this task.** The drone's local minimum (hover in place, collect survival bonus) is stable and attractive. Without sufficient entropy pressure, the policy converges there permanently. `ent_coef=0.02` is the primary fix for `success_rate=0`.

---

### `vf_coef = 0.5`
**What it is:** Value function loss coefficient — relative weight of critic loss vs policy loss.

**Formula:**
```
L_total = L_CLIP - vf_coef·L_VF + ent_coef·H(π)
L_VF    = MSE(V(s), R_t)
```

**Effect:**

| vf_coef | Behaviour |
|---|---|
| 0.1 | Weak critic training — value function lags, poor advantage estimates |
| 0.5 (current) | SB3 default — equal weight to both actor and critic losses |
| 1.0 | Strong critic training — faster value convergence but dominates policy gradient |

**Drone-specific:** In the failed runs, `explained_variance=0.86` shows the critic is already well-trained. Reducing to `vf_coef=0.3` shifts the gradient budget back toward the policy, which was barely updating (`policy_gradient_loss ≈ 0`).

---

### `max_grad_norm = 0.5`
**What it is:** Gradient clipping threshold — limits the L2 norm of all gradients before the optimiser step.

**Formula:**
```
if ‖∇θ‖₂ > max_grad_norm:
    ∇θ ← ∇θ × (max_grad_norm / ‖∇θ‖₂)
```

**Purpose:** Prevents exploding gradients during large policy updates. Critical for stability when `clip_range` is increased.

**Effect:**

| max_grad_norm | Behaviour |
|---|---|
| 0.1 | Very conservative — safe but slow |
| 0.5 (current) | SB3 default — good for most tasks |
| 1.0 | Less restriction — allows larger steps, higher instability risk |

**Keep at 0.5.** With `clip_range=0.3` and `ent_coef=0.02`, the larger updates are already handled by those parameters. Raising `max_grad_norm` further could cause training instability.

---

### `policy_kwargs = dict(net_arch=[256, 256])`
**What it is:** Neural network architecture for both actor and critic.

`[256, 256]` = two hidden layers of 256 neurons each, with shared trunk then separate actor/critic heads.

```
obs (22-D) → Linear(22, 256) → Tanh → Linear(256, 256) → Tanh
                                                         ↓ actor head → Linear(256, 4) → action mean
                                                         ↓ critic head → Linear(256, 1) → V(s)
```

**Effect on capacity vs speed:**

| net_arch | Parameters | Capacity | Wall-clock |
|---|---|---|---|
| [64, 64] | ~12K | Low — may underfit complex dynamics | Fastest |
| [128, 128] | ~34K | Medium | Fast |
| [256, 256] (current) | ~133K | Good — sufficient for 22-D drone task | Moderate |
| [512, 512] | ~528K | High — overkill for this task | Slower |

**Drone-specific:** `[256, 256]` is appropriate. The task has 22-D obs → 4-D action. The dynamics are nonlinear (RPM² thrust, Coriolis effects) but not highly complex. Going larger provides no meaningful gain on an RTX 4070 mobile.

**Activation:** SB3's `MlpPolicy` uses `Tanh` by default for continuous control (bounded activations prevent exploding features).

---

### `learning_rate = 3e-4`
**What it is:** Adam optimiser step size — controls how large a parameter update is made per gradient step.

**Formula (Adam update):**
```
θ ← θ - lr × m̂_t / (√v̂_t + ε)
where m̂, v̂ are bias-corrected first/second moment estimates
```

**Effect:**

| learning_rate | Behaviour |
|---|---|
| 1e-4 | Conservative — stable but slow, good for fine-tuning |
| 3e-4 (current) | Adam default, SB3 default — good starting point |
| 1e-3 (recommended early) | Aggressive — escapes local minima faster, risk of instability |
| 1e-2+ | Too high — training diverges |

**Drone recommendation:** Use `--learning_rate 1e-3 --lr_final 1e-5` to decay from aggressive early exploration to fine-tuning. The `--lr_final` flag in `train_drone.py` implements a linear schedule passed as a callable to SB3:
```python
learning_rate = lambda p: lr_end + (lr_init - lr_end) * p
# p = progress_remaining: 1.0 at start → 0.0 at end
```

---

## 7. Parameter Interaction Summary

How the parameters interact with each other and the drone task:

```
ent_coef  ──→  entropy of π  ──→  exploration breadth
                                   └─ too low → local minimum trap (main failure)

clip_range ──→  max ratio change  ──→  policy update size
                                        └─ too low → updates hit ceiling, slow escape

n_steps  ──→  rollout length  ──→  GAE quality + update frequency
                                    └─ too low → poor advantage estimates, no convergence

batch_size ──→  mini-batches/epoch  ──→  GPU utilisation + gradient steps
                                          └─ too low → wasted compute (1.5M vs 50K steps)

gamma    ──→  effective horizon  ──→  how far ahead policy "sees"
                                       └─ 0.99 → 1s horizon, drone needs 2–5s for approach

vf_coef  ──→  critic loss weight  ──→  gradient budget split actor vs critic
                                        └─ too high → critic dominates, policy barely updates

learning_rate ──→  step size  ──→  speed of escape from local minima
                                    └─ use decay: 1e-3 → 1e-5 over 10M steps
```

**Recommended settings for this task (10M PPO on Crazyflie RPM control):**

| Parameter | Default | Recommended | Why |
|---|---|---|---|
| `n_steps` | 2048 | 2048 | Needs full episode coverage for GAE |
| `batch_size` | 64 | 512 | 32× fewer gradient steps, better GPU use |
| `n_epochs` | 10 | 10 | Keep standard |
| `gamma` | 0.99 | 0.99 | 1s horizon sufficient |
| `gae_lambda` | 0.95 | 0.95 | Standard, stable |
| `clip_range` | 0.2 | 0.3 | Allows larger escapes from local minima |
| `ent_coef` | 0.005 | 0.02 | Prevents entropy collapse — most critical fix |
| `vf_coef` | 0.5 | 0.3 | Critic already converges fast, free budget for policy |
| `max_grad_norm` | 0.5 | 0.5 | Keep — stability guard |
| `net_arch` | [256,256] | [256,256] | Sufficient capacity |
| `learning_rate` | 3e-4 | 1e-3 → 1e-5 | Aggressive early, fine-tune late |
