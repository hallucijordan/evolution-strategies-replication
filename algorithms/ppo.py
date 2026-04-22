"""
Proximal Policy Optimization (PPO-Clip) — Schulman et al. [2017], arXiv:1707.06347.

Implements:
  - Clipped surrogate objective (PPO-Clip)
  - GAE (Generalized Advantage Estimation) — Schulman et al. [2015], arXiv:1506.02438
  - n_envs parallel environments (SyncVectorEnv) for diverse, correlated-free rollouts
  - Multiple epochs of mini-batch updates per rollout
  - Entropy regularisation for exploration
  - Gradient norm clipping

Design:
  n_envs environments run in the main process (SyncVectorEnv).
  Each rollout collects n_steps * n_envs transitions, which are flattened, shuffled
  into mini-batches of size (n_steps * n_envs) // n_minibatches, and optimized for
  n_epochs passes before the buffer is discarded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .base import BaseAlgorithm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    lr:            float = 2.5e-4
    gamma:         float = 0.99
    gae_lambda:    float = 0.95    # GAE lambda
    clip_eps:      float = 0.1     # PPO clip range
    n_steps:       int   = 128     # steps per rollout per env
    n_envs:        int   = 8       # parallel environments
    n_epochs:      int   = 4       # optimization epochs per rollout
    n_minibatches: int   = 4       # mini-batches per epoch
    entropy_coef:  float = 0.01
    value_coef:    float = 0.5
    max_grad_norm: float = 0.5
    seed:          int   = 0
    ckpt_interval: int   = 50_000


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores n_steps × n_envs transitions and computes GAE advantages."""

    def __init__(self, n_steps: int, n_envs: int, obs_shape, device: torch.device):
        self.n_steps = n_steps
        self.n_envs  = n_envs
        self.device  = device

        self.obs        = torch.zeros((n_steps, n_envs, *obs_shape), dtype=torch.float32)
        self.actions    = torch.zeros((n_steps, n_envs), dtype=torch.long)
        self.log_probs  = torch.zeros((n_steps, n_envs), dtype=torch.float32)
        self.rewards    = torch.zeros((n_steps, n_envs), dtype=torch.float32)
        self.values     = torch.zeros((n_steps, n_envs), dtype=torch.float32)
        self.dones      = torch.zeros((n_steps, n_envs), dtype=torch.float32)
        self.advantages = torch.zeros((n_steps, n_envs), dtype=torch.float32)
        self.returns    = torch.zeros((n_steps, n_envs), dtype=torch.float32)
        self.ptr        = 0

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr]   = reward
        self.values[self.ptr]    = value
        self.dones[self.ptr]     = done
        self.ptr += 1

    def compute_gae(self, last_values: torch.Tensor, gamma: float, gae_lambda: float):
        """
        Compute GAE advantages and discounted returns in-place.
        last_values: (n_envs,) bootstrap values for the observation after the last step.
                     Must be 0 for envs whose last step was terminal.
        """
        gae = torch.zeros(self.n_envs)
        for t in reversed(range(self.n_steps)):
            next_non_terminal = 1.0 - self.dones[t]          # (n_envs,)
            next_val = self.values[t + 1] if t + 1 < self.n_steps else last_values
            delta = (
                self.rewards[t]
                + gamma * next_val * next_non_terminal
                - self.values[t]
            )
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def get_batches(self, n_minibatches: int, device: torch.device):
        """Yield shuffled mini-batches from flattened (n_steps * n_envs) transitions."""
        assert self.ptr == self.n_steps, "Buffer not full"
        total      = self.n_steps * self.n_envs
        indices    = torch.randperm(total)
        batch_size = total // n_minibatches

        obs       = self.obs.flatten(0, 1).to(device)         # (total, *obs_shape)
        actions   = self.actions.flatten(0, 1).to(device)     # (total,)
        log_probs = self.log_probs.flatten(0, 1).to(device)   # (total,)
        adv       = self.advantages.flatten(0, 1).to(device)  # (total,)
        ret       = self.returns.flatten(0, 1).to(device)     # (total,)

        for start in range(0, total, batch_size):
            idx = indices[start: start + batch_size]
            yield obs[idx], actions[idx], log_probs[idx], adv[idx], ret[idx]

    def reset(self):
        self.ptr = 0


# ---------------------------------------------------------------------------
# PPO algorithm
# ---------------------------------------------------------------------------

def _obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """obs: (n_envs, *obs_shape) uint8 or float → float32 in [0, 1]."""
    x = torch.from_numpy(obs.astype(np.float32)).to(device)
    return x / 255.0 if obs.dtype == np.uint8 else x


class PPO(BaseAlgorithm):

    def __init__(self, policy: nn.Module, config: PPOConfig | None = None):
        self.policy = policy
        self.config = config or PPOConfig()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env_fn: Callable, total_steps: int, logger) -> None:
        import gymnasium as gym
        from collections import deque

        cfg    = self.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(device)
        self.policy.train()

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr, eps=1e-5)

        # n_envs parallel environments; SyncVectorEnv runs them sequentially
        # in the main process — simple and avoids subprocess pickling issues.
        env = gym.vector.SyncVectorEnv([env_fn] * cfg.n_envs)
        obs, _ = env.reset(seed=cfg.seed)

        obs_shape = obs.shape[1:]  # (4, 84, 84)
        buffer    = RolloutBuffer(cfg.n_steps, cfg.n_envs, obs_shape, device)

        steps           = 0
        generation      = 0
        episode         = 0
        ep_accum        = np.zeros(cfg.n_envs, dtype=np.float32)
        recent_returns  = deque(maxlen=20)
        last_ckpt_steps = 0

        while steps < total_steps:
            t0 = time.time()
            buffer.reset()

            # ---- Rollout collection ----
            for _ in range(cfg.n_steps):
                x = _obs_to_tensor(obs, device)  # (n_envs, 4, 84, 84)

                with torch.no_grad():
                    logits, value = self.policy(x)    # (n_envs, n_act), (n_envs,)
                    dist   = Categorical(logits=logits)
                    action = dist.sample()             # (n_envs,)

                next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
                done = terminated | truncated  # (n_envs,)

                buffer.add(
                    obs      = x.cpu(),
                    action   = action.cpu(),
                    log_prob = dist.log_prob(action).cpu(),
                    reward   = torch.from_numpy(reward.astype(np.float32)),
                    value    = value.cpu(),
                    done     = torch.from_numpy(done.astype(np.float32)),
                )

                ep_accum += reward
                for i in range(cfg.n_envs):
                    if done[i]:
                        recent_returns.append(float(ep_accum[i]))
                        episode     += 1
                        ep_accum[i]  = 0.0

                obs    = next_obs
                steps += cfg.n_envs

            # Bootstrap: V(next_obs) for non-terminal envs, 0 for terminal.
            # SyncVectorEnv auto-resets terminated envs, so obs[i] is the new
            # episode start when done[i]=True — we must zero those out.
            with torch.no_grad():
                _, last_val = self.policy(_obs_to_tensor(obs, device))
                last_values = last_val.cpu() * (1.0 - torch.from_numpy(done.astype(np.float32)))

            buffer.compute_gae(last_values, cfg.gamma, cfg.gae_lambda)

            # Normalize advantages over all (n_steps * n_envs) transitions
            adv = buffer.advantages
            buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

            # ---- Optimization ----
            total_pg_loss = total_vf_loss = total_ent = 0.0
            n_updates = 0

            for _ in range(cfg.n_epochs):
                for obs_b, act_b, old_lp_b, adv_b, ret_b in buffer.get_batches(
                    cfg.n_minibatches, device
                ):
                    logits_b, values_b = self.policy(obs_b)
                    dist_b  = Categorical(logits=logits_b)
                    new_lp  = dist_b.log_prob(act_b)
                    entropy = dist_b.entropy().mean()

                    ratio    = (new_lp - old_lp_b).exp()
                    pg_loss1 = -adv_b * ratio
                    pg_loss2 = -adv_b * ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps)
                    pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                    vf_loss = 0.5 * (ret_b - values_b).pow(2).mean()

                    loss = pg_loss + cfg.value_coef * vf_loss - cfg.entropy_coef * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                    total_pg_loss += pg_loss.item()
                    total_vf_loss += vf_loss.item()
                    total_ent     += entropy.item()
                    n_updates     += 1

            generation += 1

            mean_entropy = total_ent / max(n_updates, 1)
            logger.log(
                steps        = steps,
                generation   = generation,
                episode      = episode,
                mean_return  = float(np.mean(recent_returns)) if recent_returns else float("nan"),
                mean_entropy = round(mean_entropy, 4),
                gen_time     = round(time.time() - t0, 2),
            )

            if steps - last_ckpt_steps >= cfg.ckpt_interval:
                ckpt   = logger.checkpoint_dir / f"step_{steps}.pt"
                latest = logger.checkpoint_dir / "latest.pt"
                torch.save(self.policy.state_dict(), ckpt)
                torch.save(self.policy.state_dict(), latest)
                print(f"[ckpt] saved → {ckpt}")
                last_ckpt_steps = steps

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
                    action = self.policy.get_action(obs)
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
