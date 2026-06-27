# Parameter-Conformance Report: `drone_gym_env.py` + `train_drone.py` vs. *Learning to Fly in Seconds*

**Subject:** Comparison between the Crazyflie quadrotor environment in [drone_gym_env.py](drone_gym_env.py), the trainer in [train_drone.py](train_drone.py), and the parameter tables in [parameters.pdf](parameters.pdf) (Tables 1–6) / *Learning to Fly in Seconds* (Eschmann et al., RAL 2024). Recommendations tailored to the **PPO** training path (since the checkpoints in the repo are `ppo_*`).

**Verdict:** The implementation reproduces the *structure* of the paper (Crazyflie geometry, 100 Hz integration, paper-style reward terms, 1st-order motor LPF) but **deviates numerically in almost every quantitative parameter**. The inertia tensor, drag-torque constant, observation-noise stds, termination thresholds, reward weights, curriculum schedule, action-history length, and the entire initial-state distribution all differ from the paper. Some deviations are deliberate adaptations to the PPO training stack used here; others appear to be unit/scaling errors that meaningfully change the physics.

---

## 1. Dynamics (paper Table 1)

| Parameter | Paper | Code ([drone_gym_env.py:124-159](drone_gym_env.py#L124-L159)) | Match? | Notes |
|---|---|---|---|---|
| Integration Δt | 0.01 s | `SIM_DT = 1/100` | ✅ | identical |
| Vehicle mass `m` | 0.027 kg | `CF_MASS = 0.027` | ✅ | identical |
| Gravity | −9.81 m/s² | Newton default | ✅ | identical |
| `Tm` (motor LPF τ) | 0.15 s | `MOTOR_TAU = 0.15` | ✅ | identical |
| RPM range | [0, 21 702] | [`CF_MIN_RPM=1000`, `CF_MAX_RPM=21702`] | ⚠️ | code clamps to 1000 RPM idle (anti-cutoff); paper allows zero |
| Rotor positions `r_p` | ±0.028 m, X-config (diagonals) | `±al=±0.0325` m, **+-config** (along ±X and ±Y axes) at [drone_gym_env.py:338-343](drone_gym_env.py#L338-L343) | ❌ | **Geometry mismatch**: paper places rotors on the diagonals (X-frame), code places them on the axes (+-frame). Arm length also differs (0.028 vs 0.0325). |
| Rotor thrust dir `r_f` | [0,0,1] | `wp.vec3(0,0,1)` | ✅ | identical |
| Rotor torque sign `r_τ` | [−1, +1, −1, +1] | [−1, +1, +1, −1] at [drone_gym_env.py:339-342](drone_gym_env.py#L339-L342) | ❌ | **Spin pattern differs**. With code's +-frame, adjacent rotors must counter-rotate; the assigned pattern may be self-consistent for control, but it does not correspond to the paper's labelling. |
| Thrust model `[Kf0,Kf1,Kf2]` | [0, 0, 3.16e-10] → `f = 3.16e-10·ω²` | `CF_KT = 3.16e-10`, `F = KT·n²` | ✅ | identical (note: paper's `K_f0`, `K_f1` static/linear terms are zero, so the quadratic-only form is equivalent). |
| Drag-torque `Kd` | 0.005 964 552 (dimensionless, used as `τ = r_τ·Kd·f`) | `CF_KD = 7.94e-12` N·m/RPM², used as `Q = KD·n²` | ❌ | **Numerical mismatch.** Translating paper to code form: `Kd_code = Kd_paper × Kf2 = 0.005964552 × 3.16e-10 ≈ 1.88e-12` N·m/RPM². Code uses **≈4.2× that value** (7.94e-12). Yaw authority and propeller-drag yaw coupling will be ~4× too strong. |
| `Ixx`, `Iyy` | 3.85e-6 kg·m² | `CF_IXX=CF_IYY = 1.657e-5` | ❌ | **Code is 4.30× larger.** Roll/pitch angular acceleration for a given torque is ~4.3× lower than the paper's Crazyflie. (Note: the code's value matches Förster 2015 system-ID; the paper uses a smaller idealised value.) |
| `Izz` | 5.9675e-6 kg·m² | `CF_IZZ = 2.9e-5` | ❌ | **Code is 4.86× larger.** Combined with the inflated `Kd`, the net yaw response is roughly similar to paper, but for the wrong reasons. |
| External disturbances `f_r`, `τ_r` | Random per Table 3 | Not modelled | ❌ | No process-noise disturbance in physics step. |

**Implication.** The 4× discrepancy in `Ixx/Iyy` is the most significant physical deviation: roll/pitch authority is ~4× lower in our simulator than in the paper's simulator. Combined with code's larger arm length (0.0325 vs 0.028 m → ~1.16× longer moment arm) the net rotational dynamics are still off. Policies trained here are tuned to a heavier (rotationally) Crazyflie than the paper's, and may fail to transfer to real hardware whose true inertia is closer to 3.85e-6.

---

## 2. Control parameterisation (paper §1 dynamics, `ω_sp := a`)

| Aspect | Paper | Code | Match? |
|---|---|---|---|
| Action meaning | `ω_sp = a` (raw RPM setpoint per rotor) | `n_sp = HOVER_RPM + a·(MAX−HOVER)` (hover-centred) at [drone_gym_env.py:453-456](drone_gym_env.py#L453-L456) | ❌ |
| Action range | implicit [0, 21702] | `[-1, 1]` symmetric, then mapped | ❌ |
| Action baseline `C_rab` | 0.334 (penalises drift from baseline) | Not implemented | ❌ |

**Implication.** The paper's action is the unscaled RPM setpoint, and the action-penalty term in the reward is computed against a fixed baseline `C_rab = 0.334` (i.e. roughly the hover ratio of full RPM). Our code uses hover-centred normalised actions and penalises only `‖Δa‖²` (change in consecutive actions), not deviation from a baseline. This is a different shape of regularisation — paper discourages high RPM, we discourage rapid RPM changes.

---

## 3. Reward function (paper Table 2)

The code reward at [drone_gym_env.py:492-530](drone_gym_env.py#L492-L530) follows the paper's term structure (`pos²`, `1−qw²`, `‖v‖²`, `‖ω‖²`, `‖Δa‖²`, survival) but **with different weights and several extra shaping terms** that the paper does not have.

### Weights — initial curriculum

| Weight | Paper `C_init` | Code init ([:187-193](drone_gym_env.py#L187-L193)) | Ratio (code/paper) |
|---|---|---|---|
| `C_rs` survival | **2.0** | 0.5 | 0.25× |
| `C_rp` position | **2.5** | 0.05 | 0.02× |
| `C_rq` orientation | **2.5** | 0.10 | 0.04× |
| `C_rv` lin. vel | 0.005 | 0.01 | 2.0× |
| `C_rω` ang. vel | **0** | 0.001 | non-zero |
| `C_ra` action | 0.005 | 0.005 | 1.0× |

### Weights — target (end-of-curriculum)

| Weight | Paper `C_target` | Code target ([:187-193](drone_gym_env.py#L187-L193)) | Ratio |
|---|---|---|---|
| `C_rs` survival | **2.0** | 0.5 | 0.25× |
| `C_rp` position | **20** | 1.0 | 0.05× |
| `C_rq` orientation | **2.5** | 0.10 | 0.04× |
| `C_rv` lin. vel | 0.5 | 0.30 | 0.6× |
| `C_rω` ang. vel | **0** | 0.05 | non-zero |
| `C_ra` action | **0.5** | 0.02 | 0.04× |

### Curriculum schedule

| Aspect | Paper | Code |
|---|---|---|
| Update interval `N_C` | 100 000 env steps | external (`self.curriculum` set by training callback) |
| Schedule shape | **Multiplicative** (`C_cp=1.2×`, `C_cv=1.4×`, `C_ca=1.4×` per `N_C` steps until limits) | **Linear interpolation** between `_INIT` and `_TGT` via `curriculum ∈ [0,1]` |
| `C_rp` limit | 20 | 1.0 (target) — **20× smaller** |
| `C_ra` limit | 0.5 | 0.02 — **25× smaller** |

### Extra terms not in paper

- `_APPROACH_COEF = 1.0` — potential-based shaping `(last_dist − dist)` at [drone_gym_env.py:519-522](drone_gym_env.py#L519-L522)
- `_HOVER_BONUS = 0.15` — dense bonus when within 0.15 m and speed < 0.5 m/s at [drone_gym_env.py:527-528](drone_gym_env.py#L527-L528)
- `arrival = +15` one-shot bonus on first reach at [drone_gym_env.py:511-513](drone_gym_env.py#L511-L513)
- `crash = −2` when `z<0.05` at [drone_gym_env.py:508](drone_gym_env.py#L508)

**Implication.**
- Code's reward is **far less penalty-heavy** than the paper. The target `C_rp` is 1.0 vs paper's 20 — position error costs 20× less per unit `‖p_err‖²` here. Combined with the +0.5 survival and +0.15 hover bonus, the optimal policy is biased toward "loiter near target" rather than the paper's "precision station-keeping".
- The non-zero `C_rω` adds an angular-rate damping pressure the paper deliberately omits — explains why the policy here may produce smoother but less agile attitude responses.
- The extra approach/arrival/hover shaping terms are typical for PPO with sparse position targets but **change the optimisation landscape entirely**: this is no longer the paper's reward, just inspired by it. Any comparison of learning curves vs. the paper is apples-to-oranges.
- The schedule difference (linear vs. multiplicative) is qualitatively different too: paper ramps weights up *aggressively* late in training (1.2^k grows fast), whereas linear interpolation gives a smooth proportional sweep.

---

## 4. Initial-state distribution (paper Table 3)

| Aspect | Paper | Code ([:399-442](drone_gym_env.py#L399-L442)) | Match? |
|---|---|---|---|
| Guidance prob. | 0.10 (spawn at origin, zero attitude, random vel) | not implemented | ❌ |
| Position | Uniform(−0.2, +0.2) m on all axes | Curriculum-scaled `Uniform(−pos_range, +pos_range)` where `pos_range = 0.1 + 0.6·curriculum` (0.1 → 0.7 m) | ❌ |
| Orientation | Uniform(SO3) with cone angle α ≤ 90° | Roll/pitch only, Uniform(±15°) at full curriculum, no yaw | ❌ (much narrower) |
| Linear velocity | Uniform(±1 m/s) | Curriculum-scaled to `±1.5·c` m/s (0 → ±1.5) | ⚠️ different schedule, similar max |
| Angular velocity | Uniform(±1 rad/s) | Curriculum-scaled to `±0.5·c` rad/s (0 → ±0.5) | ❌ half the paper's range |
| Initial RPM `ω_m` | Uniform(0, 21702/2) (i.e. roughly half-throttle) | Always `CF_HOVER_RPM` (~14 476) | ❌ deterministic start |

**Implication.** The paper deliberately stress-tests recovery from highly-inverted attitudes (`α ≤ 90°`) and arbitrary motor states; our code uses gentle ±15° tilts and always-hovering initial RPMs. **The trained policy here has not been exposed to anywhere near the same attitude or motor-state distribution as the paper's**, so robustness to upset conditions will be much weaker. Also, the absence of a "guidance" spawn (zero pose, random velocities) removes the explicit recovery-from-perturbation subtask.

---

## 5. Observation noise (paper Table 4)

| Channel | Paper σ | Code σ ([:380-385](drone_gym_env.py#L380-L385)) | Ratio |
|---|---|---|---|
| Position | 0.001 m | 0.01 m | **10×** |
| Orientation | 0.001 | not applied | n/a |
| Linear velocity | 0.002 m/s | 0.01 m/s | **5×** |
| Angular velocity | 0.002 rad/s | 0.05 rad/s | **25×** |

**Implication.** Code noise is one to two orders of magnitude larger than the paper, and orientation is left clean. This is a *much* harder observation problem on velocity/rate channels and a *much* easier one on the rotation matrix. Net effect: the policy learns to lean heavily on the (clean) attitude observation while ignoring noisy rate signals — opposite of the paper's tuning.

---

## 6. Termination (paper Table 5)

| Condition | Paper | Code ([:533-538](drone_gym_env.py#L533-L538)) |
|---|---|---|
| Max position error | 0.6 m | **4.0 m** |
| Max linear vel error | 1000 m/s (effectively off) | not bounded |
| Max angular vel error | 1000 rad/s (effectively off) | not bounded |
| Ground impact | — | `z < 0.05` ✅ |
| Escape upward | — | `z > 6.0` ✅ |
| Inverted | — | `R22 < -0.5` (≈±120° tilt) ✅ |

**Implication.** Paper terminates episodes the moment the drone wanders more than 60 cm from the target — strongly funnelling the policy toward staying close. Code allows up to 4 m of drift before termination, which combined with the much smaller `C_rp` makes wandering vastly less costly. This explains why a policy trained here may exhibit large overshoots that the paper's setup would have killed early.

---

## 7. Network / RL setup (paper Table 6)

Environment-side parameters live in [drone_gym_env.py](drone_gym_env.py); RL hyperparameters live in [train_drone.py](train_drone.py).

### Environment-side

| Parameter | Paper | Code |
|---|---|---|
| Algorithm (default) | **Asymmetric Actor-Critic (TD3-style off-policy)** | `--algo td3` (default); PPO and SAC also offered — but `ppo_drone_final_s3.zip` is what's checkpointed |
| `N_H` action-history length | **32** | `N_ACTION_HIST = 1` at [drone_gym_env.py:162](drone_gym_env.py#L162) |
| Observation dim | 3 + 9 + 3 + 3 + 4·32 = **146** | 22 |
| Env step limit | 500 (5 s @ 100 Hz) | `MAX_EPISODE_STEPS = 800` (8 s) |
| γ | 0.99 | `args.gamma = 0.99` at [train_drone.py:302](train_drone.py#L302) ✅ |

### Trainer-side ([train_drone.py](train_drone.py))

| Parameter | Paper (Table 6, TD3) | TD3 in code ([:419-448](train_drone.py#L419-L448)) | PPO in code ([:380-396](train_drone.py#L380-L396)) |
|---|---|---|---|
| Actor net | [64, 64] Tanh | `net_arch=[256, 256]` (ReLU default) | `net_arch=[256, 256]` |
| Critic net | [64, 64] Tanh | same | same |
| Critic input | **Privileged (asymmetric, 28-D)** | same as actor (22-D) — flagged in module docstring | same as actor (22-D) |
| Batch size | 256 | 256 ✅ | 64 ❌ (paper TD3 uses 256) |
| Actor/critic warmup | 30 000 / 15 000 | `learning_starts=0` ❌ | n/a |
| Train interval | 20 | `train_freq=1` ❌ (50× more often) | `n_steps=2048`, `n_epochs=10` |
| Polyak τ | 0.995 | `tau=0.005` (SB3 uses 1−τ_paper convention, so equivalent) ✅ | n/a |
| Replay capacity | 300 k or 3 M | `buffer_size=500_000` ⚠️ (between the two paper values) | n/a |
| Target action noise / clip | 0.5 / 0.5 | `target_policy_noise=0.2`, `target_noise_clip=0.5` ❌ (noise σ much smaller) | n/a |
| Exploration noise σ | **0.5**, decay start @ 500 k, factor 0.9 every 100 k | `action_noise σ=0.10` initial, callback decays 0.10 → 0.02 over `curriculum_steps` ❌ (5× smaller, different schedule) | n/a (PPO is on-policy stochastic) |
| LR | unspecified in PDF | `3e-4` constant or linear-decay via `--lr_final` | `3e-4` |
| Reward curriculum | multiplicative every 100 k steps | linear ramp via `CurriculumCallback` ([:73-110](train_drone.py#L73-L110)) over `--curriculum_steps=1.5M` ❌ | same callback |
| Total timesteps | 3 M (position control) | `--total_timesteps=3_000_000` ✅ | same |

**Implication.**
- **Algorithm mismatch (PPO path).** Paper uses an off-policy actor-critic with a replay buffer of up to 3 M transitions; the checkpointed model is PPO (on-policy). Many design choices in the paper's reward (e.g. zero `C_rω`, large `C_rp`) are tuned for off-policy gradient estimators and behave poorly under PPO's clipped-surrogate updates — which is the most plausible reason the reward weights in this repo were re-tuned downward.
- **Action history (`N_H=32`).** The paper feeds 32 past actions to the actor, giving it a memory of motor dynamics over 0.32 s. Code feeds only **1** prior action. This dramatically reduces the policy's ability to compensate for motor LPF delay, and (combined with the deterministic-hover initial RPM) means the policy has no explicit observation of motor state at all. **This is the single largest architectural deviation from the paper that PPO would benefit from closing.**
- **TD3-specific gaps in trainer.** Even on the TD3 path, the trainer drifts from the paper: small exploration σ (0.10 vs 0.5), no warmup, training every step instead of every 20, no asymmetric critic. The `train_drone.py` docstring at [:16-22](train_drone.py#L16-L22) already calls out the asymmetric-critic gap and the missing external disturbances; the noise/warmup/interval gaps are not currently flagged.
- **PPO batch size 64.** Combined with 16 parallel envs × `n_steps=2048` = 32 768 samples per rollout, 64 is unusually small (= 512 minibatches per epoch × 10 epochs). Larger minibatches (256–512) are typical for PPO on continuous control and reduce gradient noise.
- **Episode length (800 vs 500).** Longer episodes inflate survival rewards but also raise per-episode variance; together with the 5× larger survival bonus per step relative to position penalty, this further biases the trained policy toward "stay alive, don't navigate".

---

## 8. Summary of deviations and severity

| # | Deviation | Severity | Why it matters |
|---|---|---|---|
| 1 | `Ixx`, `Iyy` are 4.30× too large; `Izz` is 4.86× too large | **High (physics)** | Roll/pitch authority is much lower; sim2real transfer to a real Crazyflie will be poor. |
| 2 | `Kd` (yaw drag coefficient) is ~4.2× too large | **High (physics)** | Yaw response over-damped; reaction torque from propellers exaggerated. |
| 3 | Rotor layout is +-frame instead of paper's X-frame | **High (physics)** | Different control allocation; sign pattern of torque commands is not directly comparable. |
| 4 | `C_rp` target is 1.0 vs paper's 20 (20× weaker position penalty) | **High (objective)** | Different optimal policy; not a faithful reproduction of the paper's reward. |
| 5 | Action history `N_H = 1` vs paper's 32 | **High (policy)** | Policy cannot model motor dynamics from observation history. |
| 6 | Initial-state distribution: ±15° tilt vs paper's full SO(3) ≤ 90° cone | **High (robustness)** | Trained policy will not recover from large attitude upsets. |
| 7 | Termination at 4 m vs paper's 0.6 m | **Medium (objective)** | Allows much more drift; weakens precision pressure. |
| 8 | Extra shaping (approach, hover bonus, arrival, crash) not in paper | **Medium (objective)** | Reward landscape differs; learning dynamics not comparable. |
| 9 | Observation noise σ 5–25× larger than paper, orientation noise absent | **Medium (observability)** | Different sensor model; policy learns to rely on clean attitude. |
| 10 | Initial RPM deterministic (hover) vs paper's `Uniform(0, 21702/2)` | **Medium (robustness)** | No motor-state randomisation. |
| 11 | Curriculum schedule linear vs paper's multiplicative | **Medium (training)** | Different progression of difficulty; not strictly worse. |
| 12 | Action parameterisation hover-centred vs paper's raw RPM setpoint + `C_rab=0.334` baseline | **Medium (control)** | Equivalent expressive power; different regularisation surface. |
| 13 | Algorithm: PPO vs paper's asymmetric actor-critic | **Medium (algorithmic)** | All paper hyperparameters (replay capacity, exploration noise, Polyak) are inapplicable. |
| 14 | Episode limit 800 vs paper's 500 | **Low** | Slightly inflates undiscounted return; combine with γ=0.99 it has marginal effect. |
| 15 | RPM floor 1000 vs paper's 0 | **Low** | Anti-cutoff guard; physically reasonable for real ESC. |
| 16 | No `f_r`/`τ_r` external force/torque disturbance | **Low–Medium** | Less robustness against process noise; less domain randomisation. |
| 17 | No "guidance" spawn (10% start at origin with random vel) | **Low** | Removes one curriculum subtask. |

---

## 9. Recommended next steps if the goal is paper-faithful reproduction

1. **Fix the inertia tensor.** Change `CF_IXX/IYY = 3.85e-6` and `CF_IZZ = 5.9675e-6`. If matching real Crazyflie hardware is preferred over matching the paper, document the choice — but right now the code claims "Förster 2015" while also claiming "matches the paper", and those two are inconsistent.
2. **Fix `KD`.** Use `CF_KD = Kd_paper × CF_KT = 1.88e-12` N·m/RPM² to match the paper's drag torque.
3. **Switch to X-frame rotor layout** with `±0.028 m` diagonals and torque pattern `[−1, +1, −1, +1]`.
4. **Restore paper reward weights** (`C_rp` target 20, `C_ra` target 0.5, `C_rs` 2.0, `C_rq` 2.5, `C_rω` 0). Remove or gate the extra approach/hover/arrival shaping behind a flag.
5. **Expand `N_ACTION_HIST` to 32** and update `OBS_DIM` (will become 3+9+3+3+32·4 = 146).
6. **Widen initial-state distribution** to Uniform(SO3) with α ≤ 90°, and randomise initial RPM in `Uniform(0, 21702/2)`.
7. **Tighten observation noise** to paper values (0.001, 0.001, 0.002, 0.002) and add orientation noise.
8. **Set termination at 0.6 m** position error.
9. **Add external `f_r`, `τ_r` disturbance** sampled per step from Table 3.
10. **If PPO is non-negotiable**, accept that the reward must be re-tuned, but separate "paper-faithful physics" (items 1–3, 5–9) from "training-stack-specific reward shaping" (items 4, 8, extra terms), and document the latter explicitly.

If, however, the goal is **not** paper-faithful reproduction but rather a working PPO policy on Newton's Crazyflie that flies well, then most of these deviations are defensible — but in that case the docstring's "Based on Eschmann et al." framing oversells the fidelity and should be softened.
