"""
Advantage Actor-Critic (A3C / A2C) — Mnih et al. [2016], arXiv:1602.01783.

Implements:
  - n_workers parallel rollout workers using multiprocessing.Pool
  - n-step return with advantage estimation
  - Entropy regularisation for exploration
  - Gradient norm clipping
  - Main process owns the Adam optimizer and applies averaged gradients

Worker design:
  Each worker process is initialised ONCE via Pool initializer with its own
  env and network copy.  Subsequent iterations only send flat_params (a small
  numpy array) — the full model is never pickled after startup.
  The env persists across iterations so episodes can complete naturally.

Note: synchronous gradient aggregation (A2C style).
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .base import BaseAlgorithm
from models.atari_cnn import get_flat_params, set_flat_params


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class A3CConfig:
    lr:            float = 1e-4
    gamma:         float = 0.99
    t_max:         int   = 128    # steps per worker rollout before gradient update
    entropy_coef:  float = 0.01
    value_coef:    float = 0.5
    max_grad_norm: float = 40.0
    n_workers:     int   = 18     # set to CPU count - 2
    seed:          int   = 0
    ckpt_interval: int   = 50_000


# ---------------------------------------------------------------------------
# Persistent worker state (one copy per worker process)
# ---------------------------------------------------------------------------

_worker_net:      nn.Module | None = None
_worker_env                        = None
_worker_obs:      np.ndarray | None = None
_worker_ep_accum: float            = 0.0
_worker_device:   str              = "cpu"


def _worker_init(env_fn: Callable, policy_template: nn.Module) -> None:
    """Initialise persistent state for this worker process (called once)."""
    global _worker_net, _worker_env, _worker_obs, _worker_ep_accum, _worker_device

    seed = os.getpid()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)   # prevent CPU thread contention between workers

    _worker_device = "cuda" if torch.cuda.is_available() else "cpu"
    _worker_net = copy.deepcopy(policy_template).to(_worker_device)
    _worker_net.train()

    _worker_env = env_fn()
    _worker_obs, _ = _worker_env.reset()
    _worker_ep_accum = 0.0


# ---------------------------------------------------------------------------
# Worker task — only receives flat_params (small numpy array) each iteration
# ---------------------------------------------------------------------------

def _obs_to_tensor(obs: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
    return x / 255.0 if obs.dtype == np.uint8 else x


def _a3c_worker_task(args):
    """
    Run one t_max-step rollout using the persistent env and network.

    Args (tuple):
        flat_params:  current global policy parameters (float32 numpy)
        cfg:          A3CConfig

    Returns:
        flat_grads:  float32 numpy array
        n_steps:     int
        ep_return:   float if an episode completed this rollout, else None
    """
    global _worker_net, _worker_env, _worker_obs, _worker_ep_accum, _worker_device

    flat_params, cfg = args

    set_flat_params(_worker_net, flat_params)
    _worker_net.train()

    rewards, values, log_probs, entropies = [], [], [], []
    ep_return = None
    done = False

    for _ in range(cfg.t_max):
        x           = _obs_to_tensor(_worker_obs).to(_worker_device)
        logits, val = _worker_net(x)
        dist        = Categorical(logits=logits.squeeze(0))
        action      = dist.sample()

        next_obs, reward, terminated, truncated, _ = _worker_env.step(action.item())
        done = terminated or truncated

        rewards.append(float(reward))
        values.append(val.squeeze(0))
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())

        _worker_ep_accum += float(reward)
        _worker_obs = next_obs

        if done:
            ep_return        = _worker_ep_accum
            _worker_obs, _   = _worker_env.reset()
            _worker_ep_accum = 0.0
            break

    n_steps = len(rewards)

    # Bootstrap value for the last state
    if done:
        R: float = 0.0
    else:
        with torch.no_grad():
            _, R_t = _worker_net(_obs_to_tensor(_worker_obs).to(_worker_device))
            R = float(R_t.item())

    # Compute A3C losses
    actor_loss  = torch.zeros(1, device=_worker_device)
    critic_loss = torch.zeros(1, device=_worker_device)
    entropy_sum = torch.zeros(1, device=_worker_device)

    for i in reversed(range(n_steps)):
        R           = rewards[i] + cfg.gamma * R
        advantage   = R - values[i].item()
        actor_loss  = actor_loss  - log_probs[i] * advantage
        critic_loss = critic_loss + 0.5 * (R - values[i]) ** 2
        entropy_sum = entropy_sum + entropies[i]

    total_loss = (
        actor_loss
        + cfg.value_coef   * critic_loss
        - cfg.entropy_coef * entropy_sum
    )

    _worker_net.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(_worker_net.parameters(), cfg.max_grad_norm)

    flat_grads = np.concatenate([
        p.grad.detach().cpu().numpy().ravel()
        if p.grad is not None
        else np.zeros(p.numel(), dtype=np.float32)
        for p in _worker_net.parameters()
    ]).astype(np.float32)

    mean_entropy = float(entropy_sum.item() / max(n_steps, 1))
    return flat_grads, n_steps, ep_return, mean_entropy


# ---------------------------------------------------------------------------
# A3C algorithm
# ---------------------------------------------------------------------------

class A3C(BaseAlgorithm):

    def __init__(self, policy: nn.Module, config: A3CConfig | None = None):
        self.policy = policy
        self.config = config or A3CConfig()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env_fn: Callable, total_steps: int, logger) -> None:
        import multiprocessing as mp
        from collections import deque

        cfg       = self.config
        optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)

        steps           = 0
        generation      = 0
        episode         = 0
        recent_returns  = deque(maxlen=20)
        last_ckpt_steps = 0

        ctx = mp.get_context("spawn")

        # Workers are initialised ONCE with env + model; only flat_params
        # is sent on each subsequent pool.map call.
        pool = ctx.Pool(
            cfg.n_workers,
            initializer=_worker_init,
            initargs=(env_fn, self.policy),
        ) if cfg.n_workers > 1 else None

        try:
            while steps < total_steps:
                t0          = time.time()
                flat_params = get_flat_params(self.policy)

                tasks = [(flat_params, cfg)] * cfg.n_workers

                if pool is not None:
                    results = pool.map(_a3c_worker_task, tasks)
                else:
                    # Single-worker fallback: initialise once then loop
                    if generation == 0:
                        _worker_init(env_fn, self.policy)
                    results = [_a3c_worker_task(tasks[0])]

                all_grads    = np.stack([r[0] for r in results])
                batch_steps  = sum(r[1] for r in results)
                ep_returns   = [r[2] for r in results if r[2] is not None]
                mean_entropy = float(np.mean([r[3] for r in results]))

                avg_grads = all_grads.mean(axis=0)

                optimizer.zero_grad()
                idx = 0
                for p in self.policy.parameters():
                    n      = p.numel()
                    p.grad = torch.from_numpy(
                        avg_grads[idx: idx + n].reshape(p.shape)
                    )
                    idx += n
                optimizer.step()

                steps      += batch_steps
                generation += 1
                episode    += len(ep_returns)
                recent_returns.extend(ep_returns)

                logger.log(
                    steps=steps,
                    generation=generation,
                    episode=episode,
                    mean_return=float(np.mean(recent_returns)) if recent_returns else float("nan"),
                    mean_entropy=round(mean_entropy, 4),
                    gen_time=round(time.time() - t0, 2),
                )

                if steps - last_ckpt_steps >= cfg.ckpt_interval:
                    ckpt   = logger.checkpoint_dir / f"step_{steps}.pt"
                    latest = logger.checkpoint_dir / "latest.pt"
                    torch.save(self.policy.state_dict(), ckpt)
                    torch.save(self.policy.state_dict(), latest)
                    print(f"[ckpt] saved → {ckpt}")
                    last_ckpt_steps = steps

        finally:
            if pool is not None:
                pool.close()
                pool.join()

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
