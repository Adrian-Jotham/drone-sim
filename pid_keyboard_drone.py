#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Keyboard-controlled Crazyflie using a near-hovering cascaded PID.

Control pipeline:
  keys → body-yaw velocity setpoints
       → roll/pitch angle commands  (near-hover linearisation: ẍ_bdy≈gθ, ÿ_bdy≈−gφ)
       → body torques               (attitude PD)
       → motor RPMs                 (+ -config mixer)
       → normalised action          (DroneEnv API)

Keyboard layout (cross around H, non-conflicting with viewer):
        T  Y  U
        G  H  J
           B  N

  Y / H   fly forward / backward  (body-yaw frame)
  G / J   fly left / right
  U / T   climb / descend
  B / N   yaw left / right
  R       reset episode
  ESC     quit
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import threading
import numpy as np
import newton.examples

from drone_gym_env import (
    DroneEnv, SIM_DT,
    CF_MASS, CF_IXX, CF_IYY, CF_IZZ,
    CF_KT, CF_KD, CF_ARM,
    CF_HOVER_RPM, CF_MAX_RPM, CF_MIN_RPM,
)

# ── Physical constants ────────────────────────────────────────────────────────
G            = 9.81
CF_RPM_RANGE = CF_MAX_RPM - CF_HOVER_RPM

# ── Velocity command limits ───────────────────────────────────────────────────
MAX_VXY      = 1.5   # m/s  horizontal
MAX_VZ       = 0.8   # m/s  vertical
MAX_YAW_RATE = 1.0   # rad/s

MAX_TILT = np.radians(25.0)   # clamp on roll/pitch angle commands

# ── Cascaded PID gains ────────────────────────────────────────────────────────
#
# Outer:  body-yaw velocity error → roll/pitch angle command
#         near-hover: ẍ_bdy ≈ g·θ  →  θ_cmd = KP_VXY · v_err
#         ω_n_vel ≈ g·KP_VXY ≈ 2 rad/s  (≈ 0.3 Hz, slower than attitude loop)
KP_VXY = 0.20   # rad / (m/s)

# Altitude:  vz error → extra vertical acceleration
KP_VZ  = 3.00   # (m/s²) / (m/s)

# Inner:  attitude PD   (equation of motion: I·α = τ  →  α = Kp·e − Kd·ω)
#         ω_n_att ≈ sqrt(KP_RP) ≈ 5 rad/s,  ζ ≈ KD_RP/(2·ω_n) ≈ 0.8
#         Motor LPF (τ=0.15 s) limits effective bandwidth; gains are conservative.
KP_RP  = 25.0;  KD_RP  = 8.0   # roll / pitch
KP_YAW = 12.0;  KD_YAW = 4.0   # yaw


# ── Motor mixer (+ configuration) ────────────────────────────────────────────
#
# Propeller layout (from drone_gym_env.py _build_sim):
#   prop 0: pos=(0, +L, 0)  turning_dir=−1  (CCW)
#   prop 1: pos=(0, −L, 0)  turning_dir=+1  (CW)
#   prop 2: pos=(+L, 0, 0)  turning_dir=+1  (CW)
#   prop 3: pos=(−L, 0, 0)  turning_dir=−1  (CCW)
#
# Thrust:      F_i = KT·n_i²
# Yaw reaction: Q_i = KD·n_i²·dir_i
#
# Body torques (arm cross thrust + reaction):
#   τ_x = KT·L·(n0²−n1²)            roll
#   τ_y = KT·L·(n3²−n2²)            pitch
#   τ_z = KD·(−n0²+n1²+n2²−n3²)    yaw
#
# Inverse mixer (solving 4×4 linear system):
#   n0² = (T/KT  + τx/(KT·L)            − τz/KD) / 4
#   n1² = (T/KT  − τx/(KT·L)            + τz/KD) / 4
#   n2² = (T/KT            − τy/(KT·L)  + τz/KD) / 4
#   n3² = (T/KT            + τy/(KT·L)  − τz/KD) / 4

def mix_to_rpms(thrust: float, tx: float, ty: float, tz: float) -> np.ndarray:
    L  = CF_ARM
    T  = thrust / CF_KT
    rx = tx / (CF_KT * L)
    ry = ty / (CF_KT * L)
    rz = tz / CF_KD
    n2 = np.array([
        (T + rx      - rz) / 4.0,   # prop 0
        (T - rx      + rz) / 4.0,   # prop 1
        (T      - ry + rz) / 4.0,   # prop 2
        (T      + ry - rz) / 4.0,   # prop 3
    ])
    rpms = np.sqrt(np.clip(n2, 0.0, None))
    return np.clip(rpms, CF_MIN_RPM, CF_MAX_RPM).astype(np.float32)


def rpms_to_action(rpms: np.ndarray) -> np.ndarray:
    return np.clip((rpms - CF_HOVER_RPM) / CF_RPM_RANGE, -1.0, 1.0).astype(np.float32)


# ── Quaternion → ZYX Euler ────────────────────────────────────────────────────

def quat_to_rpy(q: np.ndarray):
    """[qx, qy, qz, qw] → (roll, pitch, yaw) in radians, ZYX convention."""
    qx, qy, qz, qw = q.astype(np.float64)
    roll  = np.arctan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx*qx + qy*qy))
    pitch = np.arcsin(np.clip(2.0*(qw*qy - qz*qx), -1.0, 1.0))
    yaw   = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
    return float(roll), float(pitch), float(yaw)


# ── Keyboard listener ─────────────────────────────────────────────────────────

class KeyState:
    def __init__(self):
        self._lock  = threading.Lock()
        self.held: set[str] = set()
        self.quit   = False
        self.reset  = False

    def _name(self, key) -> str | None:
        c = getattr(key, 'char', None)
        if c:
            return c.lower()
        n = getattr(key, 'name', None)
        return n.lower() if n else None

    def on_press(self, key):
        name = self._name(key)
        if name == 'esc':
            self.quit = True
        elif name == 'r':
            self.reset = True
        elif name:
            with self._lock:
                self.held.add(name)

    def on_release(self, key):
        name = self._name(key)
        if name:
            with self._lock:
                self.held.discard(name)

    def snapshot(self) -> set[str]:
        with self._lock:
            return set(self.held)


def _start_listener(ks: KeyState):
    from pynput import keyboard
    lst = keyboard.Listener(on_press=ks.on_press, on_release=ks.on_release)
    lst.daemon = True
    lst.start()
    return lst


# ── Keys → velocity setpoints (body-yaw frame) ───────────────────────────────

def key_velocity(keys: set[str]):
    """Return (vx_fwd, vy_left, vz_up, yaw_rate) all in body-yaw frame.

    Cross layout (non-conflicting with viewer camera controls):
        T  Y  U
        G  H  J
           B  N
    """
    vx = vy = vz = yr = 0.0
    if 'y' in keys: vx += MAX_VXY   # forward
    if 'h' in keys: vx -= MAX_VXY   # backward
    if 'g' in keys: vy += MAX_VXY   # left
    if 'j' in keys: vy -= MAX_VXY   # right
    if 'u' in keys: vz += MAX_VZ    # climb
    if 't' in keys: vz -= MAX_VZ    # descend
    if 'b' in keys: yr += MAX_YAW_RATE   # yaw left
    if 'n' in keys: yr -= MAX_YAW_RATE   # yaw right
    return vx, vy, vz, yr


# ── Near-hovering cascaded PID ────────────────────────────────────────────────

class NearHoverPID:
    """
    Cascaded PID controller for near-hovering flight.

    Stage 1 (outer): body-yaw velocity setpoint → roll/pitch command
      Near-hover linearisation (ZYX Euler, Z-up ENU):
        ẍ_bdy ≈ g·θ   →  θ_cmd =  KP_VXY · (vx_sp − vx_bdy)
        ÿ_bdy ≈ −g·φ  →  φ_cmd = −KP_VXY · (vy_sp − vy_bdy)
      Valid because ZYX pitch/roll are exactly the body-yaw-frame tilt angles.

    Stage 2 (inner): attitude PD
      τ = I · (Kp·e_angle − Kd·ω)

    Stage 3: motor mixer → normalised action
    """

    def __init__(self):
        self.yaw_sp = 0.0

    def update(
        self,
        quat:   np.ndarray,   # [qx, qy, qz, qw] body-to-world
        vel_w:  np.ndarray,   # [vx, vy, vz] world frame
        omega:  np.ndarray,   # [p, q, r] body frame angular velocity
        vx_sp: float,
        vy_sp: float,
        vz_sp: float,
        yr_sp: float,
    ) -> np.ndarray:
        """Returns normalised action ∈ [-1, 1]^4."""

        roll, pitch, yaw = quat_to_rpy(quat)

        # ── Altitude: P on vertical velocity error ────────────────────────
        vz_err = vz_sp - vel_w[2]
        thrust = float(np.clip(CF_MASS * (G + KP_VZ * vz_err), 0.0, CF_MASS * G * 3.0))

        # ── Horizontal velocity → roll/pitch command ──────────────────────
        # Rotate world velocity into body-yaw frame so setpoints are
        # heading-relative and both signals are in the same frame.
        cy, sy     = np.cos(yaw), np.sin(yaw)
        vx_bdy     =  cy * vel_w[0] + sy * vel_w[1]   # forward (body-yaw x)
        vy_bdy     = -sy * vel_w[0] + cy * vel_w[1]   # left    (body-yaw y)

        vx_err     = vx_sp - vx_bdy
        vy_err     = vy_sp - vy_bdy

        pitch_cmd  = float(np.clip( KP_VXY * vx_err, -MAX_TILT, MAX_TILT))
        roll_cmd   = float(np.clip(-KP_VXY * vy_err, -MAX_TILT, MAX_TILT))

        # ── Attitude PD ───────────────────────────────────────────────────
        p, q, r = omega

        self.yaw_sp += yr_sp * SIM_DT
        yaw_err = float(((self.yaw_sp - yaw) + np.pi) % (2.0 * np.pi) - np.pi)

        tau_x = CF_IXX * (KP_RP  * (roll_cmd  - roll)  - KD_RP  * p)
        tau_y = CF_IYY * (KP_RP  * (pitch_cmd - pitch) - KD_RP  * q)
        tau_z = CF_IZZ * (KP_YAW * yaw_err             - KD_YAW * r)

        # ── Motor mix ─────────────────────────────────────────────────────
        rpms = mix_to_rpms(thrust, float(tau_x), float(tau_y), float(tau_z))
        return rpms_to_action(rpms)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)

    env = DroneEnv(
        render_mode="human",
        viewer=viewer,
        random_targets=False,
        curriculum=0.0,
    )

    ks  = KeyState()
    _start_listener(ks)

    print()
    print("─── Near-Hover PID Keyboard Controller ────────────────────────")
    print("  Cross layout (non-conflicting with viewer):")
    print()
    print("        T  Y  U")
    print("        G  H  J")
    print("           B  N")
    print()
    print("  Y / H   fly forward / backward  (body-yaw frame)")
    print("  G / J   fly left / right")
    print("  U / T   climb / descend")
    print("  B / N   yaw left / right")
    print("  R       reset episode")
    print("  ESC     quit")
    print("───────────────────────────────────────────────────────────────")
    print()

    obs, _ = env.reset()
    pid    = NearHoverPID()
    step   = 0

    while not ks.quit:

        if ks.reset:
            obs, _ = env.reset()
            pid    = NearHoverPID()
            step   = 0
            ks.reset = False
            print("  [reset]")
            continue

        # Read sim state (avoid going through obs to get un-noised values)
        bq    = env._state.body_q.numpy()[0].astype(np.float32)
        bqd   = env._state.body_qd.numpy()[0].astype(np.float32)
        pos   = bq[:3]
        quat  = bq[3:]    # [qx, qy, qz, qw]
        vel_w = bqd[3:]   # linear velocity, world frame
        omega = bqd[:3]   # angular velocity, body frame

        vx_sp, vy_sp, vz_sp, yr_sp = key_velocity(ks.snapshot())
        action = pid.update(quat, vel_w, omega, vx_sp, vy_sp, vz_sp, yr_sp)

        obs, _, terminated, truncated, info = env.step(action)

        if step % 100 == 0:
            roll, pitch, yaw = quat_to_rpy(quat)
            keys = ks.snapshot()
            print(
                f"  t={step * SIM_DT:6.2f}s | "
                f"xyz=({pos[0]:+.2f} {pos[1]:+.2f} {pos[2]:+.2f}) | "
                f"rpy=({np.degrees(roll):+5.1f}° {np.degrees(pitch):+5.1f}° {np.degrees(yaw):+5.1f}°) | "
                f"rpm≈{np.mean(env._motor_rpms):.0f} | "
                f"keys={sorted(keys) or '—'}"
            )

        if terminated:
            print(f"  Crashed at z={info['z']:.2f} m — auto-resetting …")
            obs, _ = env.reset()
            pid  = NearHoverPID()
            step = 0
            continue

        step += 1

    env.close()
    print("Goodbye.")


if __name__ == "__main__":
    main()
