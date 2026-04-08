"""
Actor-Critic network architectures for policy-gradient algorithms (e.g. A3C).

Each network exposes:
  forward(x) -> (policy_logits, value)
  get_action(obs) -> int   (greedy argmax, no grad)

Two variants:
  ActorCriticCNN  — Atari (4, 84, 84) input, CNN backbone from Mnih et al. [2016]
  ActorCriticMLP  — low-dimensional obs (CartPole, MuJoCo)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticCNN(nn.Module):
    """
    Two-headed CNN for Atari.

    Shared backbone (3 conv layers, identical to Mnih et al. DQN/A3C):
        conv × 3  →  Linear(3136, 512)  →  ReLU
    Separate heads:
        policy_head:  Linear(512, n_actions)  — logits (unnormalised)
        value_head:   Linear(512, 1)          — scalar V(s)

    Input:  (B, 4, 84, 84) float32 in [0, 1]
    Output: (policy_logits, value)  shapes (B, n_actions), (B,)
    """

    def __init__(self, n_actions: int):
        super().__init__()
        self.use_vbn = False
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.fc     = nn.Linear(3136, 512)
        self.policy = nn.Linear(512, n_actions)
        self.value  = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor):
        """x: (B, 4, 84, 84) float32 in [0, 1]"""
        h = F.relu(self.fc(self.conv(x).flatten(1)))
        return self.policy(h), self.value(h).squeeze(-1)

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> int:
        """Greedy (argmax) action.  obs: (4, 84, 84) uint8."""
        x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0) / 255.0
        logits, _ = self.forward(x)
        return int(logits.argmax(dim=1).item())


class ActorCriticMLP(nn.Module):
    """
    Two-headed MLP for low-dimensional observation spaces (CartPole, MuJoCo).

    Input:  flat float32 obs vector
    Output: (policy_logits, value)  shapes (B, n_actions), (B,)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.use_vbn = False
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, n_actions)
        self.value  = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.policy(h), self.value(h).squeeze(-1)

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> int:
        x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
        logits, _ = self.forward(x)
        return int(logits.argmax(dim=1).item())
