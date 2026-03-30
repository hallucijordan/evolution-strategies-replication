import numpy as np
import torch
import torch.nn as nn
from .atari_cnn import get_flat_params, set_flat_params, count_params


class MLP(nn.Module):
    """
    Small MLP policy for low-dimensional environments (e.g. CartPole, MuJoCo).
    Input:  flat observation vector
    Output: logits over discrete actions (or raw values for continuous)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.use_vbn = False   # no VBN needed for low-dim obs
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> int:
        x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
        logits = self.forward(x)
        return int(logits.argmax(dim=1).item())
