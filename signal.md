# PPO Training Signal Guide — DroneSimPPO

Crazyflie 2.x · Newton/Warp sim · 100 Hz · MAX_EPISODE_STEPS=800
PPO: n_steps=2048 · batch=256 · ent_coef=0.02 · clip=0.3 · vf_coef=0.3
Run: --num_envs 64 · --curriculum_steps 3M · --lr 1e-3 → 1e-5 · --headless

---

## Phase 1 — Stabilisation (0 → ~500K steps)

Curriculum ≈ 0. Spawn 0.1 m from target. Gentle penalties.
Goal: learn to not crash and stay upright.
Distance-decaying survival gives +0.45/step at 0.1 m — strongly positive returns from step 1.

### Expected metrics

| Metric | Expected |
|---|---|
| `ep_length` | Rising: 50 → 200+ |
| `terminal_upright` | Rising: 0.0 → 0.6+ |
| `entropy_loss` | Stable: −2 to −3 (ent_coef=0.02 holding it back) |
| `explained_variance` | Rising: 0.1 → 0.6 |
| `mean_motor_rpm` | Near hover: ~14,476 RPM |
| `reward_components/survival` | ~0.43–0.48/step (near target, high bonus) |
| `success_rate` | Brief flicker > 0 (curriculum=0 spawns within 0.15 m) |

### Red flags — Phase 1

| Signal | Meaning | Action |
|---|---|---|
| `entropy_loss` < −4 before 500K steps | Entropy collapsing too early, policy committing before sufficient exploration | Increase `ent_coef` to 0.03 |
| `mean_motor_rpm` < 13,000 | Under-thrust from step 1 — bad weight init | Restart with different seed |
| `ep_length` never rises above 80 | Drone crashing immediately every episode | Check spawn bounds; reduce `curriculum_steps` |

---

## Phase 2 — Curriculum Shock (500K → 3M steps)

Curriculum 0→1. Spawn grows 0.1 m→0.7 m. Penalties ramp 20×.
Goal: learn to navigate toward target, not just hover.
New survival reward actively punishes staying far away — net reward is negative at spawn distance until the drone moves.

### Expected metrics

| Metric | Expected |
|---|---|
| `ep_length` | Dips temporarily (harder spawns), then recovers |
| `terminal_dist` | Rises initially, then **must fall by ~1.5M steps** |
| `reward_components/survival` | Declining: ~0.45 → ~0.20–0.25 (spawning further from target) |
| `reward_components/approach` | Turns and stays **positive** (drone moving toward target) |
| `reward_components/pos_c` | More negative (stricter weight), then slowly recovers |
| `rpm_hover_dev` | Flat or decreasing |
| `clip_fraction` | 0.08–0.14 (clip_range=0.3 giving room for larger steps) |
| `success_rate` | First sustained non-zero values: 0.01–0.05 |

### Critical checkpoint

> **By step ~2M:** `terminal_dist` must be trending downward.
> Flat at 1.5 m+ after 2M steps = policy stuck in local minimum.

### Red flags — Phase 2

| Signal | Meaning | Action |
|---|---|---|
| `survival` flat AND low (~0.10) | Drone stuck far from target every episode, not improving | New reward is the detector: low survival = far away → restart with different seed |
| `approach` ≈ 0 or negative | Drone not moving toward target at all | Local minimum confirmed |
| `rpm_hover_dev` increasing | Under-thrust starting — classic collapse precursor | Check `act_c`; increase `ent_coef` |
| `entropy_loss` < −5 | Policy fully committed to bad behaviour | Irreversible — restart |
| `terminal_dist` still rising at 2M steps | Curriculum too fast for current policy | Increase `--curriculum_steps` to 4M |
| `ep_length` declining after first peak | Policy destabilising under stricter penalties | Reduce `learning_rate`; check `clip_fraction` |

---

## Phase 3 — Refinement (3M → 10M steps)

Curriculum = 1.0 fixed. LR decaying 1e-3 → 1e-5.
Goal: reliable navigation and precision hovering at the target.

### Expected metrics

| Metric | Expected |
|---|---|
| `success_rate` | Climbing: 0.05 → 0.3 → 0.8+ |
| `terminal_dist` | Falling: 0.5 m → 0.2 m → < 0.15 m |
| `ep_length` | Approaching 800 (surviving full episodes) |
| `reward_components/hover_bonus` | Non-zero and growing (drone settling at target) |
| `reward_components/survival` | Recovering upward (drone spending more time near target) |
| `entropy_loss` | Slowly declining: −3 → −4 (appropriate late convergence) |
| `policy_gradient_loss` | Small but non-zero (still updating, just fine steps) |
| `train/learning_rate` | Visibly decaying in TensorBoard |
| `rpm_hover_dev` | Near 0 — motors at hover RPM while stationary at target |

### Red flags — Phase 3

| Signal | Meaning | Action |
|---|---|---|
| `success_rate` flat at 0 past 5M steps | Never escaped Phase 2 local minimum | Restart, different seed |
| `hover_bonus` = 0 at 7M+ steps | Drone reaching vicinity but not decelerating | `vel_c` weight may be too low; check `approach` gate |
| `entropy_loss` < −6 | Policy fully deterministic — no more improvement possible | Accept result or restart with higher `ent_coef` |
| `ep_length` = 800 but `success_rate` = 0 | Drone survives full episodes but never reaches target | Position penalty too weak — check curriculum reached 1.0 |
| `value_loss` spiking after 5M | LR still too high for fine-tuning phase | Verify `--lr_final` was set |

---

## The Two-Metric Early Warning System

With the distance-decaying survival reward, `survival` and `approach` together
give you the clearest signal of training health:

```
survival DECLINING  +  approach POSITIVE   →  HEALTHY
                                               Drone moving toward target.
                                               Survival shrinks as harder spawns kick in.

survival FLAT LOW   +  approach NEAR ZERO  →  LOCAL MINIMUM
                                               Drone stuck far from target, not moving.
                                               Restart with different seed.

survival RECOVERING +  approach POSITIVE   →  LATE TRAINING WORKING
                                               Drone spending more time near target.
                                               survival rises as dist falls.
```

---

## Five-Second Dashboard Check

Open TensorBoard. Check in this order:

```
1. entropy_loss > −4 ?          YES → exploring       NO → potential collapse
2. approach > 0 ?               YES → navigating      NO → stuck
3. rpm_hover_dev flat/falling ? YES → healthy thrust  NO → under-thrust collapse
4. terminal_dist falling after 1.5M steps ?
                                YES → learning        NO → local minimum
5. survival declining in Phase 2 ?
                                YES → new reward OK   NO → something is wrong
```

**If metrics 1, 2, and 3 are all bad at the same time → restart with a different seed.**
The policy is irreversibly stuck. PPO cannot recover from a fully collapsed entropy + under-thrust + no-navigation state.

---

## Seed Reliability Expectation

With all current fixes (ent_coef=0.02, clip=0.3, distance-decaying survival):

| Seeds that get stuck (est.) | Seeds that converge (est.) |
|---|---|
| ~3–4 out of 10 | ~6–7 out of 10 |

The paper achieves 10/10 using TD3 + asymmetric critic. PPO without the asymmetric
critic will have higher seed variance. Run at least 3 seeds and report the mean.
The missing component (critic seeing privileged state: motor RPMs + disturbances)
is the main remaining gap vs. the paper's reliability.
