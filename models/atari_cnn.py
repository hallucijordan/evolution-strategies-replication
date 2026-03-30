import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Virtual Batch Normalization
# ---------------------------------------------------------------------------

class VBNLayer(nn.Module):
    """
    Virtual Batch Normalization — Salimans et al. [2017], Section 2.2

    Problem: a randomly-initialized CNN maps almost any input to nearly the
    same activation vector, so all 100 perturbed policies behave identically
    and return -21. No diversity => no gradient signal.

    Fix: compute mean/std from a fixed reference batch once at training start,
    then always normalize using those frozen statistics. Different input frames
    now produce different activations => policies diverge => learning begins.

    Difference from standard BatchNorm:
        BN:   normalize(x, mean(x_current), std(x_current))  <- changes each step
        VBN:  normalize(x, ref_mean,        ref_std)          <- frozen after init

    Parameters:
        gamma / beta   — learnable scale/shift, perturbed by ES  (nn.Parameter)
        ref_mean / std — fixed after set_reference_batch()       (register_buffer)
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Learnable scale and shift — included in get_flat_params(), perturbed by ES
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta  = nn.Parameter(torch.zeros(num_features))
        # Reference statistics — buffers are NOT parameters, ES will not touch them
        self.register_buffer("ref_mean", torch.zeros(num_features))
        self.register_buffer("ref_std",  torch.ones(num_features))

    @torch.no_grad()
    def set_reference_batch(self, activations: torch.Tensor) -> None:
        """
        Compute and freeze normalization statistics from a reference batch.
        Called exactly once at the start of training.
        activations: (N, num_features) — conv features from N reference observations
        """
        self.ref_mean.copy_(activations.mean(0))
        # clamp prevents division by zero on all-black frames
        self.ref_std.copy_(activations.std(0).clamp(min=self.eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_features)
        return self.gamma * (x - self.ref_mean) / self.ref_std + self.beta


# ---------------------------------------------------------------------------
# CNN Policy
# ---------------------------------------------------------------------------

class AtariCNN(nn.Module):
    """
    CNN policy for Atari games.
    Architecture from Mnih et al. [2016] (DQN/A3C), same as used in Salimans et al. [2017].

    Input:  (4, 84, 84) stacked grayscale frames — uint8 [0, 255]
    Output: logits over discrete actions

    Data flow with VBN:
        (B, 4, 84, 84)
            -> conv x 3
        (B, 3136)
            -> VBNLayer        <- normalize with frozen reference stats
        (B, 3136)
            -> Linear(3136->512) + ReLU
        (B, 512)
            -> Linear(512->n_actions)
        (B, n_actions)
    """

    def __init__(self, n_actions: int, use_vbn: bool = True):
        super().__init__()
        self.use_vbn = use_vbn
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # After 3 conv layers on (4, 84, 84): feature map is (64, 7, 7) = 3136
        if use_vbn:
            self.vbn = VBNLayer(3136)
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 4, 84, 84) float32 in [0, 1]"""
        features = self.conv(x).flatten(1)   # (B, 3136)
        if self.use_vbn:
            features = self.vbn(features)
        return self.fc(features)

    @torch.no_grad()
    def set_reference_batch(self, obs_batch: np.ndarray) -> None:
        """
        Collect reference observations, compute and freeze VBN statistics.
        Must be called once before training starts.
        obs_batch: (N, 4, 84, 84) uint8
        """
        assert self.use_vbn, "set_reference_batch called but use_vbn=False"
        x = torch.from_numpy(obs_batch.astype(np.float32)) / 255.0
        activations = self.conv(x).flatten(1)   # (N, 3136)
        self.vbn.set_reference_batch(activations)

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> int:
        """
        Greedy action selection (deterministic policy, as used in ES).
        obs: (4, 84, 84) uint8 numpy array
        """
        x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0) / 255.0
        logits = self.forward(x)
        return int(logits.argmax(dim=1).item())


# ---------------------------------------------------------------------------
# Parameter utilities (ES operates directly on flat numpy parameter vectors)
# ---------------------------------------------------------------------------

def get_flat_params(model: nn.Module) -> np.ndarray:
    """Extract all trainable parameters as a flat float32 numpy array.
    Note: register_buffer values (VBN ref_mean/ref_std) are excluded — ES will not perturb them.
    """
    return np.concatenate(
        [p.detach().cpu().numpy().ravel() for p in model.parameters()]
    ).astype(np.float32)


def set_flat_params(model: nn.Module, flat_params: np.ndarray) -> None:
    """Set model parameters from a flat numpy array (in-place, no grad)."""
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(
            torch.from_numpy(flat_params[idx : idx + n].reshape(p.shape))
        )
        idx += n


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
