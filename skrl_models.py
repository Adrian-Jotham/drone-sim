# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
SKRL policy / value / critic models for the batched drone env (CLAUDE.md §4.4).

Head architecture is ``[256, 256]`` with ``Tanh`` (matches the SB3 setup). GRU
variants prepend a single GRU layer (hidden 256) feeding the same head and implement
``get_specification()`` + the SKRL RNN ``compute`` contract (hidden-state passing and
per-episode resets), following SKRL's reference GRU example.

  * PPO  → GaussianMLP (policy) + DeterministicMLP (value)
  * SAC  → GaussianMLP (policy) + QMLP ×2 (+ targets)
  * TD3  → DeterministicActorMLP (policy) + QMLP ×2 (+ targets)
  * GRU  → GaussianGRU / DeterministicGRU / QGRU

Net size is the small ``[64, 64]`` of the converging RK4Dynamics TD3 reference
(rl-tools), not the previous ``[256, 256]`` — a tiny net is plenty for the 22-D
hover/recovery task and trains faster.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

HIDDEN = [64, 64]
GRU_HIDDEN = 64
GRU_LAYERS = 1
SEQUENCE_LENGTH = 16

# Quadrotor control needs FINE actions (~±0.2 around hover), but a Gaussian policy with
# log_std=0 (std=1) explores with violent ±1 swings that thrash the drone through the
# 0.15 s motor lag and never samples the stable regime. Start exploration much smaller.
INIT_LOG_STD = -1.6          # std ≈ 0.20


def _mlp(in_dim: int, out_dim: int, hidden=HIDDEN, final=None) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    if final is not None:
        layers += [final()]
    return nn.Sequential(*layers)


# ── MLP models ────────────────────────────────────────────────────────────

class GaussianMLP(GaussianMixin, Model):
    """Stochastic policy: state → action mean (+ state-independent log_std)."""

    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, min_log_std=-20.0, max_log_std=2.0):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=True,
                               min_log_std=min_log_std, max_log_std=max_log_std)
        self.net = _mlp(self.num_observations, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions) * INIT_LOG_STD)

    def compute(self, inputs, role=""):
        x = inputs["observations"]
        return self.net(x), {"log_std": self.log_std_parameter}


class DeterministicMLP(DeterministicMixin, Model):
    """Value function V(s) (PPO critic)."""

    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.net = _mlp(self.num_observations, 1)

    def compute(self, inputs, role=""):
        return self.net(inputs["observations"]), {}


class QMLP(DeterministicMixin, Model):
    """State-action value Q(s, a) (SAC critic)."""

    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.net = _mlp(self.num_observations + self.num_actions, 1)

    def compute(self, inputs, role=""):
        x = torch.cat([inputs["observations"], inputs["taken_actions"]], dim=-1)
        return self.net(x), {}


class DeterministicActorMLP(DeterministicMixin, Model):
    """Deterministic Tanh-bounded actor μ(s) → action in [-1, 1] (TD3 policy).

    Unlike the Gaussian policy, exploration for TD3 is injected by the agent as
    additive Gaussian action noise (cfg.exploration_noise); the network output is
    the greedy action, squashed by a final Tanh into the [-1, 1] action box.
    """

    def __init__(self, observation_space, action_space, device, clip_actions=True):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.net = _mlp(self.num_observations, self.num_actions, final=nn.Tanh)

    def compute(self, inputs, role=""):
        return self.net(inputs["observations"]), {}


# ── GRU models ────────────────────────────────────────────────────────────

class _GRUBase(Model):
    """Shared GRU front-end + SKRL RNN contract (hidden states + episode resets)."""

    def _init_gru(self, input_size: int, num_envs: int):
        self.num_envs = num_envs
        self.sequence_length = SEQUENCE_LENGTH
        self.gru = nn.GRU(input_size=input_size, hidden_size=GRU_HIDDEN,
                          num_layers=GRU_LAYERS, batch_first=True)

    def get_specification(self):
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(GRU_LAYERS, self.num_envs, GRU_HIDDEN)]}}

    def _gru_forward(self, inputs):
        """Returns (rnn_output[flat], {"rnn": [hidden]}) per SKRL's GRU reference."""
        states = inputs["observations"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])
            hidden_states = hidden_states.view(GRU_LAYERS, -1, self.sequence_length,
                                               GRU_HIDDEN)[:, :, 0, :].contiguous()
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                idx = (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                idx = [0] + idx + [self.sequence_length]
                for i in range(len(idx) - 1):
                    i0, i1 = idx[i], idx[i + 1]
                    out, hidden_states = self.gru(rnn_input[:, i0:i1, :], hidden_states)
                    hidden_states[:, (terminated[:, i1 - 1]), :] = 0.0
                    rnn_outputs.append(out)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, hidden_states = self.gru(rnn_input, hidden_states)
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])
            rnn_output, hidden_states = self.gru(rnn_input, hidden_states)

        rnn_output = rnn_output.flatten(start_dim=0, end_dim=1)
        return rnn_output, {"rnn": [hidden_states]}


class GaussianGRU(GaussianMixin, _GRUBase):
    def __init__(self, observation_space, action_space, device, num_envs,
                 clip_actions=False, min_log_std=-20.0, max_log_std=2.0):
        _GRUBase.__init__(self, observation_space=observation_space,
                          action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=True,
                               min_log_std=min_log_std, max_log_std=max_log_std)
        self._init_gru(self.num_observations, num_envs)
        self.head = _mlp(GRU_HIDDEN, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions) * INIT_LOG_STD)

    def compute(self, inputs, role=""):
        feat, rnn_out = self._gru_forward(inputs)
        out = self.head(feat)
        return out, {"log_std": self.log_std_parameter, **rnn_out}


class DeterministicGRU(DeterministicMixin, _GRUBase):
    def __init__(self, observation_space, action_space, device, num_envs, clip_actions=False):
        _GRUBase.__init__(self, observation_space=observation_space,
                          action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self._init_gru(self.num_observations, num_envs)
        self.head = _mlp(GRU_HIDDEN, 1)

    def compute(self, inputs, role=""):
        feat, rnn_out = self._gru_forward(inputs)
        return self.head(feat), rnn_out


class QGRU(DeterministicMixin, _GRUBase):
    """Q(s, a) with a GRU over observations; action concatenated at the head."""

    def __init__(self, observation_space, action_space, device, num_envs, clip_actions=False):
        _GRUBase.__init__(self, observation_space=observation_space,
                          action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self._init_gru(self.num_observations, num_envs)
        self.head = _mlp(GRU_HIDDEN + self.num_actions, 1)

    def compute(self, inputs, role=""):
        feat, rnn_out = self._gru_forward(inputs)
        x = torch.cat([feat, inputs["taken_actions"]], dim=-1)
        return self.head(x), rnn_out
