# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
SKRL trainer for the GPU-parallel drone position controller (CLAUDE.md T7+T8).

Replaces the conflicted SB3 trainer. One Newton model with N worlds is driven by a
single SKRL agent (PPO or SAC), with MLP or GRU policies. Hyperparameters are mapped
from the preserved SB3 config (CLAUDE.md §6). The reward (drone_env_batched.py) follows the
RAPTOR recipe (arXiv 2509.11481): dense L2-norm costs + a terminal penalty, and NO curriculum
(the spawn distribution is held fixed at --spawn_difficulty). SAC is the recommended algorithm
for this recipe — RAPTOR uses SAC because PPO is "more unstable and prone to local minima".
Entropy decay is applied via a thin agent subclass; success/terminal-distance metrics mirror
the old TensorBoard logging.

Usage:
    python train_drone.py --algo sac --policy mlp --num_envs 1024 --total_timesteps 20_000_000
    python train_drone.py --algo sac --policy mlp --num_envs 1024 --spawn_difficulty 0.5
    python train_drone.py --algo ppo --policy mlp --num_envs 4096 --total_timesteps 50_000_000

TD3 (off-policy, deterministic actor + twin critics) mirrors the converging RK4Dynamics
reference: 64×64 nets, exploration via additive action noise, tight termination box, and
ground-truth rotor speeds in the observation.

IMPORTANT — TD3 is a LOW-env-count algorithm (unlike PPO). Its learning is measured in
gradient updates, not frames. Running it at thousands of envs with gradient_steps=1 starves
the critic (~1 update per N fresh transitions) and it will NOT converge. Use few envs
(~32-64) and/or raise --gradient_steps so updates-per-transition ≈ the reference's ~1:10.
"""

from __future__ import annotations

import argparse

import gymnasium
import numpy as np
import torch

import skrl
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_RNN, PPO_CFG
from skrl.agents.torch.sac import SAC, SAC_RNN, SAC_CFG
from skrl.agents.torch.td3 import TD3, TD3_CFG
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.noises.torch import GaussianNoise

from drone_env_batched import BatchedDroneEnv, OBS_DIM, SUCCESS_R
import skrl_models as M


# ── SKRL env wrapper (T7) ─────────────────────────────────────────────────

class SkrlDroneWrapper(Wrapper):
    """Thin SKRL wrapper over the batched Newton env (GPU tensors throughout)."""

    def __init__(self, env: BatchedDroneEnv):
        super().__init__(env)
        self._obs_space = gymnasium.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self._act_space = gymnasium.spaces.Box(-1.0, 1.0, (env.num_act,), np.float32)
        self._last_info: dict = {}

    @property
    def num_envs(self) -> int:          return self._env.num_envs
    @property
    def observation_space(self):        return self._obs_space
    @property
    def action_space(self):             return self._act_space
    @property
    def state_space(self):              return None
    @property
    def unwrapped(self):                return self._env

    def reset(self):
        obs, info = self._env.reset()
        return obs, info

    def step(self, actions: torch.Tensor):
        obs, rew, term, trunc, info = self._env.step(actions)
        self._last_info = info
        return (obs,
                rew.view(-1, 1),
                term.view(-1, 1).to(torch.bool),
                trunc.view(-1, 1).to(torch.bool),
                info)

    def state(self):                    return None
    def render(self, *a, **k):          return None
    def close(self):                    return None


# ── Curriculum + entropy-decay + metric logging mix-in ────────────────────

def _make_agent_class(base):
    """Subclass a SKRL agent to add curriculum ramp, entropy decay, custom metrics."""

    class _Agent(base):
        def configure_schedule(self, env, spawn_difficulty, ent_start, ent_end, ent_steps):
            self._cur_env = env
            self._ent0, self._ent1 = ent_start, ent_end
            self._ent_steps = max(int(ent_steps), 1)
            self._has_entropy = hasattr(self.cfg, "entropy_loss_scale")
            # RAPTOR recipe: NO curriculum. The dense L2 reward stabilizes training on its
            # own, so the spawn distribution is held fixed at `spawn_difficulty` for the
            # whole run (1.0 = full distribution). Set once here.
            self._spawn_difficulty = float(min(max(spawn_difficulty, 0.0), 1.0))
            self._cur_env.unwrapped.set_curriculum(self._spawn_difficulty)

        def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
            # timestep is per-env agent steps; convert to environment frames.
            frames = timestep * self._cur_env.num_envs
            if self._has_entropy:
                frac = min(frames / self._ent_steps, 1.0)
                self.cfg.entropy_loss_scale = self._ent0 + frac * (self._ent1 - self._ent0)
                self.track_data("Schedule / entropy_scale", float(self.cfg.entropy_loss_scale))
            super().pre_interaction(timestep=timestep, timesteps=timesteps)

        def record_transition(self, **kw):
            super().record_transition(**kw)
            terminated = kw.get("terminated"); truncated = kw.get("truncated")
            infos = kw.get("infos")
            done = (terminated | truncated)
            if self.training and torch.any(done) and isinstance(infos, dict) and "dist" in infos:
                d = infos["dist"][done.view(-1)]
                if d.numel():
                    self.track_data("Episode / terminal_dist", d.mean().item())
                    self.track_data("Episode / success_rate",
                                    (d < SUCCESS_R).float().mean().item())

    return _Agent


# ── Config mapping (CLAUDE.md §6) ─────────────────────────────────────────

def build_ppo_cfg(num_envs: int, rollouts: int, experiment_dir: str, name: str) -> PPO_CFG:
    cfg = PPO_CFG(
        rollouts=rollouts,                 # steps/env (rollouts×N ≈ SB3 batch)
        learning_epochs=10,                # n_epochs
        mini_batches=max(1, rollouts * num_envs // 16384),
        discount_factor=0.99,              # gamma
        gae_lambda=0.95,                   # gae_lambda
        learning_rate=3e-4,                # lr
        ratio_clip=0.3,                    # clip_range
        value_clip=0.3,
        entropy_loss_scale=0.02,           # ent_coef (decayed to 5e-4 via schedule)
        value_loss_scale=0.3,              # vf_coef
        grad_norm_clip=0.5,                # max_grad_norm
        kl_threshold=0.0,
        learning_starts=0,
        observation_preprocessor=RunningStandardScaler,
        observation_preprocessor_kwargs={"size": OBS_DIM},
        value_preprocessor=RunningStandardScaler,
        value_preprocessor_kwargs={"size": 1},
    )
    cfg.experiment.directory = experiment_dir
    cfg.experiment.experiment_name = name
    return cfg


def build_sac_cfg(gradient_steps: int, experiment_dir: str, name: str) -> SAC_CFG:
    cfg = SAC_CFG(
        gradient_steps=gradient_steps,
        batch_size=256,                    # SAC batch
        discount_factor=0.99,              # gamma
        polyak=0.005,                      # tau
        learning_rate=3e-4,                # lr (actor+critic+entropy)
        # FIXED entropy temperature (no auto-tuning). With auto-entropy, the target
        # entropy (-|A|=-4) is below the policy's natural entropy, so α collapsed to ~0;
        # that made the policy deterministic and removed the target-action noise that is
        # SAC's only guard against critic overestimation → Q diverged to NaN. A fixed α
        # keeps the policy stochastic (≈ TD3's target-policy smoothing) and stable.
        learn_entropy=False,               # fixed ent_coef (was auto → α-collapse → NaN)
        initial_entropy_value=0.1,
        random_timesteps=0,
        learning_starts=1000,
        grad_norm_clip=0.5,
        observation_preprocessor=RunningStandardScaler,
        observation_preprocessor_kwargs={"size": OBS_DIM},
    )
    cfg.experiment.directory = experiment_dir
    cfg.experiment.experiment_name = name
    return cfg


def build_td3_cfg(num_envs: int, gradient_steps: int, experiment_dir: str, name: str,
                  device: str) -> TD3_CFG:
    # Hyperparameters from the converging RK4Dynamics TD3 reference (rl-tools): γ=0.99,
    # polyak τ=0.005, batch 256, lr 1e-3, exploration noise σ=0.1, target-policy
    # smoothing σ=0.2 clipped at 0.5, policy delay 2. Warmup seeds the replay buffer
    # with random actions before learning starts (reference uses 5000 single-env steps;
    # here scaled to agent steps since each agent step collects num_envs transitions).
    #
    # gradient_steps is the updates-per-agent-step knob. TD3 is a low-env-count algorithm:
    # its learning is measured in gradient updates, not frames. The reference does ~1 update
    # per 10 env transitions; with N envs each agent step yields N transitions, so set
    # gradient_steps ≈ N/10 (or just run few envs) to avoid update starvation.
    warmup = max(5000 // max(num_envs, 1), 64)
    cfg = TD3_CFG(
        gradient_steps=gradient_steps,
        batch_size=256,                    # TD3 batch
        discount_factor=0.99,              # gamma
        polyak=0.005,                      # tau
        learning_rate=1e-3,                # actor+critic lr (reference / TD3 default)
        random_timesteps=warmup,           # random actions to seed the buffer
        learning_starts=warmup,
        grad_norm_clip=0.0,
        exploration_noise=GaussianNoise,
        exploration_noise_kwargs={"mean": 0.0, "std": 0.1, "device": device},
        smooth_regularization_noise=GaussianNoise,
        smooth_regularization_noise_kwargs={"mean": 0.0, "std": 0.2, "device": device},
        smooth_regularization_clip=0.5,
        policy_delay=2,
        observation_preprocessor=RunningStandardScaler,
        observation_preprocessor_kwargs={"size": OBS_DIM},
    )
    cfg.experiment.directory = experiment_dir
    cfg.experiment.experiment_name = name
    return cfg


# ── Agent / model factories ───────────────────────────────────────────────

def make_models(algo: str, policy: str, obs_space, act_space, device, num_envs):
    gru = policy == "gru"
    if algo == "td3":
        if gru:
            raise ValueError("TD3 GRU is not wired up; use --policy mlp with --algo td3")
        return {
            "policy":          M.DeterministicActorMLP(obs_space, act_space, device, clip_actions=True),
            "target_policy":   M.DeterministicActorMLP(obs_space, act_space, device, clip_actions=True),
            "critic_1":        M.QMLP(obs_space, act_space, device),
            "critic_2":        M.QMLP(obs_space, act_space, device),
            "target_critic_1": M.QMLP(obs_space, act_space, device),
            "target_critic_2": M.QMLP(obs_space, act_space, device),
        }
    if algo == "ppo":
        if gru:
            models = {
                "policy": M.GaussianGRU(obs_space, act_space, device, num_envs, clip_actions=False),
                "value":  M.DeterministicGRU(obs_space, act_space, device, num_envs),
            }
        else:
            models = {
                "policy": M.GaussianMLP(obs_space, act_space, device, clip_actions=False),
                "value":  M.DeterministicMLP(obs_space, act_space, device),
            }
    else:  # sac
        if gru:
            models = {
                "policy": M.GaussianGRU(obs_space, act_space, device, num_envs, clip_actions=True),
                "critic_1": M.QGRU(obs_space, act_space, device, num_envs),
                "critic_2": M.QGRU(obs_space, act_space, device, num_envs),
                "target_critic_1": M.QGRU(obs_space, act_space, device, num_envs),
                "target_critic_2": M.QGRU(obs_space, act_space, device, num_envs),
            }
        else:
            models = {
                "policy": M.GaussianMLP(obs_space, act_space, device, clip_actions=True),
                "critic_1": M.QMLP(obs_space, act_space, device),
                "critic_2": M.QMLP(obs_space, act_space, device),
                "target_critic_1": M.QMLP(obs_space, act_space, device),
                "target_critic_2": M.QMLP(obs_space, act_space, device),
            }
    return models


def make_viewer(render: bool, headless: bool = False):
    """Create a Newton GL viewer for real-time visualisation, or None."""
    if not render:
        return None
    import newton.viewer
    return newton.viewer.ViewerGL(headless=headless)


def build(args):
    device = args.device
    viewer = make_viewer(getattr(args, "render", False))
    env = BatchedDroneEnv(num_envs=args.num_envs, device=device,
                          curriculum=args.spawn_difficulty,
                          seed=args.seed, capture_graph=not args.no_graph,
                          viewer=viewer,
                          render_worlds=getattr(args, "render_worlds", 1),
                          render_every=getattr(args, "render_every", 4))
    wrapped = SkrlDroneWrapper(env)

    obs_space, act_space = wrapped.observation_space, wrapped.action_space
    models = make_models(args.algo, args.policy, obs_space, act_space, device, args.num_envs)

    name = f"{args.algo}_{args.policy}_s{args.seed}"
    if args.algo == "ppo":
        cfg = build_ppo_cfg(args.num_envs, args.rollouts, args.logdir, name)
        memory = RandomMemory(memory_size=args.rollouts, num_envs=args.num_envs, device=device)
        agent_cls = _make_agent_class(PPO_RNN if args.policy == "gru" else PPO)
    elif args.algo == "td3":
        cfg = build_td3_cfg(args.num_envs, args.gradient_steps, args.logdir, name, device)
        # RandomMemory allocates memory_size × num_envs transitions; size per-env so the
        # total replay capacity ≈ buffer_size regardless of world count (as for SAC).
        per_env = max(args.buffer_size // args.num_envs, 1)
        memory = RandomMemory(memory_size=per_env, num_envs=args.num_envs, device=device)
        agent_cls = _make_agent_class(TD3)
    else:
        cfg = build_sac_cfg(args.gradient_steps, args.logdir, name)
        # RandomMemory allocates memory_size × num_envs transitions; size per-env so the
        # total replay capacity ≈ buffer_size regardless of world count.
        per_env = max(args.buffer_size // args.num_envs, 1)
        memory = RandomMemory(memory_size=per_env, num_envs=args.num_envs, device=device)
        agent_cls = _make_agent_class(SAC_RNN if args.policy == "gru" else SAC)

    agent = agent_cls(models=models, memory=memory, cfg=cfg,
                      observation_space=obs_space, action_space=act_space, device=device)
    # Keep entropy small so it doesn't inflate the (deliberately small) action std back
    # up — fine quadrotor control needs low-variance exploration.
    agent.configure_schedule(wrapped, args.spawn_difficulty, args.ent_start, args.ent_end,
                             args.entropy_decay_steps)
    return wrapped, agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "sac", "td3"], default="ppo")
    p.add_argument("--policy", choices=["mlp", "gru"], default="mlp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--total_timesteps", type=int, default=50_000_000)
    p.add_argument("--rollouts", type=int, default=24, help="PPO steps per env per update")
    p.add_argument("--gradient_steps", type=int, default=1,
                   help="TD3 updates per agent step (set ~num_envs/10 to match the "
                        "reference update:data ratio; or just use few --num_envs)")
    p.add_argument("--buffer_size", type=int, default=500_000, help="SAC/TD3 replay (per env)")
    p.add_argument("--spawn_difficulty", type=float, default=1.0,
                   help="fixed spawn-distribution difficulty in [0,1] (RAPTOR uses no "
                        "curriculum; 1.0 = full distribution). Lower it if SAC struggles.")
    p.add_argument("--entropy_decay_steps", type=int, default=20_000_000)
    p.add_argument("--ent_start", type=float, default=0.003,
                   help="initial PPO entropy_loss_scale (raise for more exploration)")
    p.add_argument("--ent_end", type=float, default=1e-4,
                   help="final PPO entropy_loss_scale after decay")
    p.add_argument("--logdir", default="runs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_graph", action="store_true", help="disable CUDA-graph capture")
    p.add_argument("--render", action="store_true",
                   help="open a real-time OpenGL window to watch training")
    p.add_argument("--render_worlds", type=int, default=4,
                   help="number of worlds to display in a grid when --render")
    p.add_argument("--render_every", type=int, default=4,
                   help="render one frame every N control steps when --render")
    args = p.parse_args()

    skrl.config.torch.device = args.device
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    wrapped, agent = build(args)

    # total_timesteps is environment frames; SKRL counts agent steps (×num_envs).
    timesteps = max(args.total_timesteps // args.num_envs, 1)
    trainer = SequentialTrainer(env=wrapped, agents=agent,
                                cfg={"timesteps": timesteps, "headless": True})
    print(f"[train] algo={args.algo} policy={args.policy} num_envs={args.num_envs} "
          f"agent-steps={timesteps} (~{timesteps*args.num_envs:,} frames)")
    trainer.train()


if __name__ == "__main__":
    main()
