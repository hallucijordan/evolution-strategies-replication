"""
Advantage Actor-Critic (A2C/A3C) — Mnih et al. [2016], arXiv:1602.01783.

Implements synchronous A2C using SyncVectorEnv:
  - n_workers parallel envs collected in the main process (no subprocess overhead)
  - GAE (Generalized Advantage Estimation) — Schulman et al. [2015], arXiv:1506.02438
  - Entropy regularisation for exploration
  - Gradient norm clipping
  - Single optimizer step per rollout (A2C style, no PPO clipping)

Why SyncVectorEnv instead of spawn multiprocessing:
  spawn processes have large inter-process communication overhead per round.
  With small t_max (5-20 steps), communication cost >> computation cost, making
  learning extremely inefficient. SyncVectorEnv runs all envs in the main process
  with negligible overhead, allowing small t_max and high update frequency.
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
class A3CConfig:
    lr:            float = 7e-4
    gamma:         float = 0.99
    gae_lambda:    float = 1.0    # original paper uses 1.0 (n-step returns)
    t_max:         int   = 5      # original paper uses 5; small = frequent updates
    entropy_coef:  float = 0.0    # SB3 A2C default for Atari; non-zero entropy fights actor
    value_coef:    float = 0.25   # SB3 A2C default; reduces critic domination of actor gradient
    max_grad_norm: float = 0.5
    n_workers:     int   = 16     # compensates for small t_max: 5×16=80 transitions/update
    seed:          int   = 0
    ckpt_interval: int   = 50_000


# ---------------------------------------------------------------------------
# A3C/A2C algorithm
# ---------------------------------------------------------------------------

def _obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """obs: (n_envs, *obs_shape) uint8 -> float32 in [0, 1]."""
    x = torch.from_numpy(obs.astype(np.float32)).to(device)
    return x / 255.0 if obs.dtype == np.uint8 else x


class A3C(BaseAlgorithm):

    def __init__(self, policy: nn.Module, config: A3CConfig | None = None):
        self.policy = policy
        self.config = config or A3CConfig()

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

        optimizer = optim.RMSprop(
            self.policy.parameters(), lr=cfg.lr, alpha=0.99, eps=1e-5
        )

        # n_workers envs running in the main process — zero IPC overhead
        env = gym.vector.SyncVectorEnv([env_fn] * cfg.n_workers)
        obs, _ = env.reset(seed=cfg.seed)

        steps           = 0
        generation      = 0
        episode         = 0
        ep_accum        = np.zeros(cfg.n_workers, dtype=np.float32)
        recent_returns  = deque(maxlen=20)
        last_ckpt_steps = 0

        while steps < total_steps:
            t0 = time.time()

            # Storage for rollout (obs/actions/rewards as numpy; no computation graph kept)
            obs_list  = []
            act_list  = []
            rew_buf   = []
            val_buf   = []   # detached values used for GAE
            done_buf  = []

            # ---- Rollout collection (no_grad: build computation graph only at update time) ----
            with torch.no_grad():
                for _ in range(cfg.t_max):
                    x = _obs_to_tensor(obs, device)  # (n_workers, 4, 84, 84)

                    logits, value = self.policy(x)   # (n_workers, n_act), (n_workers,)
                    dist   = Categorical(logits=logits)
                    action = dist.sample()           # (n_workers,)

                    obs_list.append(x.cpu())
                    act_list.append(action.cpu())
                    val_buf.append(value.cpu())

                    next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
                    done = terminated | truncated  # (n_workers,)

                    rew_buf.append(torch.from_numpy(reward.astype(np.float32)))
                    done_buf.append(torch.from_numpy(done.astype(np.float32)))

                    ep_accum += reward
                    for i in range(cfg.n_workers):
                        if done[i]:
                            recent_returns.append(float(ep_accum[i]))
                            episode     += 1
                            ep_accum[i]  = 0.0

                    obs    = next_obs
                    steps += cfg.n_workers

                # Bootstrap: V(next_obs) for non-terminal envs, 0 for terminal
                _, last_val = self.policy(_obs_to_tensor(obs, device))
                last_val = last_val.cpu() * (1.0 - done_buf[-1])

            # ---- GAE computation (numpy, no grad needed) ----
            values_np  = torch.stack(val_buf).numpy()   # (t_max, n_workers)
            rewards_np = torch.stack(rew_buf).numpy()
            dones_np   = torch.stack(done_buf).numpy()
            last_np    = last_val.numpy()               # (n_workers,)

            gae_np  = np.zeros(cfg.n_workers, dtype=np.float32)
            adv_np  = np.zeros((cfg.t_max, cfg.n_workers), dtype=np.float32)

            for t in reversed(range(cfg.t_max)):
                next_nont = 1.0 - dones_np[t]
                next_v    = values_np[t + 1] if t + 1 < cfg.t_max else last_np
                delta     = rewards_np[t] + cfg.gamma * next_v * next_nont - values_np[t]
                gae_np    = delta + cfg.gamma * cfg.gae_lambda * next_nont * gae_np
                adv_np[t] = gae_np

            ret_np = adv_np + values_np   # (t_max, n_workers)

            # No advantage normalization: with sparse rewards and short rollouts the std
            # can be near-zero, turning normalization into amplified noise.

            # ---- Multiple gradient steps per rollout ----
            obs_t = torch.stack(obs_list).to(device)        # (t_max, n_workers, 4, 84, 84)
            act_t = torch.stack(act_list).to(device)        # (t_max, n_workers)
            adv_t = torch.from_numpy(adv_np).to(device)     # (t_max, n_workers)
            ret_t = torch.from_numpy(ret_np).to(device)     # (t_max, n_workers)

            obs_flat = obs_t.flatten(0, 1)   # (t_max * n_workers, 4, 84, 84)
            act_flat = act_t.flatten(0, 1)   # (t_max * n_workers,)
            adv_flat = adv_t.flatten()
            ret_flat = ret_t.flatten()

            logits_all, values_all = self.policy(obs_flat)
            dist_all    = Categorical(logits=logits_all)
            log_probs   = dist_all.log_prob(act_flat)
            entropy     = dist_all.entropy().mean()

            actor_loss  = -(log_probs * adv_flat).mean()
            critic_loss = 0.5 * (ret_flat - values_all).pow(2).mean()
            loss        = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            generation += 1

            logger.log(
                steps        = steps,
                generation   = generation,
                episode      = episode,
                mean_return  = float(np.mean(recent_returns)) if recent_returns else float("nan"),
                mean_entropy = round(float(entropy.item()), 4),
                gen_time     = round(time.time() - t0, 2),
            )

            if steps - last_ckpt_steps >= cfg.ckpt_interval:
                ckpt   = logger.checkpoint_dir / f"step_{steps}.pt"
                latest = logger.checkpoint_dir / "latest.pt"
                torch.save(self.policy.state_dict(), ckpt)
                torch.save(self.policy.state_dict(), latest)
                print(f"[ckpt] saved -> {ckpt}")
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
