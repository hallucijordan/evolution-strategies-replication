"""
Deep Q-Network (DQN) — Mnih et al. [2015], Nature 518, 529–533.

Implements:
  - Experience replay (uniform sampling)
  - Target network with periodic hard updates
  - Epsilon-greedy exploration with linear annealing
  - Huber (SmoothL1) loss
  - Gradient norm clipping

Works with any policy whose forward(x) returns per-action Q-values:
AtariCNN and MLP from this repo both satisfy this interface.
"""

from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseAlgorithm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DQNConfig:
    lr:                  float = 1e-4
    gamma:               float = 0.99
    buffer_size:         int   = 100_000  # replay buffer capacity (uint8 for Atari)
    batch_size:          int   = 32
    target_update_freq:  int   = 1_000    # env steps between hard target copies
    epsilon_start:       float = 1.0
    epsilon_end:         float = 0.01
    epsilon_decay_steps: int   = 100_000
    learning_starts:     int   = 10_000   # random play before training starts
    train_freq:          int   = 4        # gradient update every N env steps
    max_grad_norm:       float = 10.0
    seed:                int   = 0


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Circular replay buffer.  Stores observations in their native dtype
    (uint8 for Atari — avoids 4× memory inflation from float32 conversion).
    Normalisation to [0, 1] is applied at sample time.
    """

    def __init__(self, capacity: int):
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        self._buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        obs_arr  = np.array(obs)
        next_arr = np.array(next_obs)
        # Normalise uint8 Atari frames to [0, 1]; leave float envs unchanged
        if obs_arr.dtype == np.uint8:
            obs_arr  = obs_arr.astype(np.float32)  / 255.0
            next_arr = next_arr.astype(np.float32) / 255.0
        else:
            obs_arr  = obs_arr.astype(np.float32)
            next_arr = next_arr.astype(np.float32)
        return (
            obs_arr,
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            next_arr,
            np.array(dones,   dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# DQN algorithm
# ---------------------------------------------------------------------------

class DQN(BaseAlgorithm):
    """
    DQN agent.

    Args:
        policy:  Any nn.Module whose forward(x) returns Q-values (B, n_actions).
                 AtariCNN and MLP from this repo work out-of-the-box.
        config:  DQNConfig.  Defaults are used if None.
    """

    def __init__(self, policy: nn.Module, config: DQNConfig | None = None):
        self.config = config or DQNConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = policy.to(self.device)
        self.target = copy.deepcopy(policy).to(self.device)
        self.target.eval()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _epsilon(self, step: int) -> float:
        cfg  = self.config
        frac = min(step / max(cfg.epsilon_decay_steps, 1), 1.0)
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert a single observation to a normalised float32 tensor."""
        x = torch.from_numpy(obs).to(self.device).float()
        return x / 255.0 if obs.dtype == np.uint8 else x

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env_fn: Callable, total_steps: int, logger) -> None:
        cfg = self.config
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        loss_fn   = nn.SmoothL1Loss()       # Huber loss
        buf       = ReplayBuffer(cfg.buffer_size)

        env       = env_fn()
        obs, _    = env.reset()
        steps     = 0
        episode   = 0
        ep_return = 0.0

        while steps < total_steps:

            # --- ε-greedy action selection ---
            eps = self._epsilon(steps)
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = self.policy(self._to_tensor(obs).unsqueeze(0))
                action = int(q_vals.argmax(dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buf.push(obs, action, float(reward), next_obs, done)

            obs        = next_obs
            ep_return += float(reward)
            steps     += 1

            if done:
                obs, _ = env.reset()
                episode += 1
                logger.log(
                    steps=steps,
                    episode=episode,
                    mean_return=ep_return,
                    epsilon=round(eps, 4),
                )
                ep_return = 0.0

            # --- Learning update ---
            if (
                steps >= cfg.learning_starts
                and steps % cfg.train_freq == 0
                and len(buf) >= cfg.batch_size
            ):
                obs_b, act_b, rew_b, nxt_b, don_b = buf.sample(cfg.batch_size)

                obs_t = torch.from_numpy(obs_b).to(self.device)
                act_t = torch.from_numpy(act_b).to(self.device)
                rew_t = torch.from_numpy(rew_b).to(self.device)
                nxt_t = torch.from_numpy(nxt_b).to(self.device)
                don_t = torch.from_numpy(don_b).to(self.device)

                # Q(s, a) for actions taken
                q_pred = self.policy(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)

                # r + γ · max_{a'} Q_target(s', a')   (masked by done)
                with torch.no_grad():
                    q_next = self.target(nxt_t).max(1).values
                    q_targ = rew_t + cfg.gamma * q_next * (1.0 - don_t)

                loss = loss_fn(q_pred, q_targ)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                optimizer.step()

            # --- Periodic hard target update ---
            if steps % cfg.target_update_freq == 0:
                self.target.load_state_dict(self.policy.state_dict())

        env.close()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, env, n_episodes: int = 10) -> float:
        self.policy.eval()
        total = 0.0
        with torch.no_grad():
            for _ in range(n_episodes):
                obs, _ = env.reset()
                done   = False
                while not done:
                    q_vals = self.policy(self._to_tensor(obs).unsqueeze(0))
                    action = int(q_vals.argmax(dim=1).item())
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total += float(reward)
                    done = terminated or truncated
        self.policy.train()
        return total / n_episodes

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        path = path if path.endswith(".pt") else path + ".pt"
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        path = path if path.endswith(".pt") else path + ".pt"
        state = torch.load(path, map_location="cpu")
        self.policy.load_state_dict(state)
        self.target.load_state_dict(state)
        self.policy.to(self.device)
        self.target.to(self.device)
