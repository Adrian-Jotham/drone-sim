# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Hover sanity test for the batched env (CLAUDE.md §8 — do FIRST).

Commands the hover action (=0) and checks the drone holds altitude, that rotor
speeds settle near ω_hover, and that obs/reward/reset machinery runs without sync
errors. Spawns at curriculum=0 (near-target, mild tilt) so a working controller
should stay aloft for the whole window.
"""
import numpy as np
import torch
import drone_env_batched as E


def main():
    N = 64
    env = E.BatchedDroneEnv(num_envs=N, device="cuda", curriculum=0.0, seed=0,
                            capture_graph=True)
    obs, _ = env.reset()
    print(f"obs shape {tuple(obs.shape)}  dtype {obs.dtype}  device {obs.device}")
    print(f"OMEGA_HOVER={E.OMEGA_HOVER:.1f} rad/s  KT_SI={E.KT_SI:.3e}  KD_SI={E.KD_SI:.3e}")
    print(f"hover thrust check: 4*KT_SI*ω² = {4*E.KT_SI*E.OMEGA_HOVER**2:.4f} N  "
          f"vs m*g = {E.CF_MASS*9.81:.4f} N")

    act = torch.zeros(N, 4, device="cuda")
    z_hist, w_hist, rew_hist = [], [], []
    for t in range(300):
        obs, rew, term, trunc, info = env.step(act)
        # absolute z of airframe (read sparingly for logging only)
        bq = env.state_0.body_q.numpy()
        z = bq[0::E.BODIES_PER_WORLD, 2]   # airframe z per world
        omega = env.motor_omega.numpy().reshape(N, 4)
        z_hist.append(z.mean())
        w_hist.append(np.abs(omega).mean())
        rew_hist.append(rew.mean().item())
        if t % 50 == 0:
            print(f"t={t:3d}  z_mean={z.mean():.3f}  z_std={z.std():.3f}  "
                  f"|ω|_mean={np.abs(omega).mean():.1f}  rew={rew.mean().item():.3f}  "
                  f"term={term.sum().item():.0f}")

    z_hist = np.array(z_hist)
    print("\n=== summary ===")
    print(f"z start {z_hist[0]:.3f} -> end {z_hist[-1]:.3f}  (min {z_hist.min():.3f}, max {z_hist.max():.3f})")
    print(f"|ω| settled ~ {w_hist[-1]:.1f} rad/s (target hover {E.OMEGA_HOVER:.1f})")
    drift = abs(z_hist[-1] - z_hist[0])
    print(f"altitude drift over 3 s: {drift:.3f} m")
    ok = z_hist[-1] > 0.1 and z_hist.min() > 0.05 and drift < 0.6
    print("HOVER:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
