"""
Advantage Actor-Critic (A3C / A2C) — Mnih et al. [2016], arXiv:1602.01783.

Implements:
  - n_workers parallel rollout workers using multiprocessing.Pool
    (same pattern as ES — workers receive numpy arrays, return numpy arrays)
  - n-step return with advantage estimation
  - Entropy regularisation for exploration
  - Gradient norm clipping
  - Main process owns the Adam optimizer and applies averaged gradients

Worker design mirrors algorithms/es.py:
  _a3c_worker_task(args) takes plain picklable args and returns plain numpy.
  No shared PyTorch state, no locks, no shared memory needed.
  This avoids fork/spawn incompatibilities with PyTorch internal threads.

Note: synchronous gradient aggregation (A2C style). Equivalent to A3C in
practice and significantly more stable on single-machine setups.
"""

from __future__ import annotations

import copy
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
    t_max:         int   = 20     # steps per worker rollout before gradient update
    entropy_coef:  float = 0.01   # weight on entropy bonus (exploration)
    value_coef:    float = 0.5    # weight on critic loss
    max_grad_norm: float = 40.0
    n_workers:     int   = 4
    seed:          int   = 0


# ---------------------------------------------------------------------------
# Worker (module-level, mirrors ES's _worker_task — picklable for Pool.map)
# ---------------------------------------------------------------------------

def _obs_to_tensor(obs: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
    return x / 255.0 if obs.dtype == np.uint8 else x


def _a3c_worker_task(args):
    """
    Single rollout for one worker.  Mirrors ES's _worker_task:
      - receives only plain picklable objects (numpy arrays, config, env factory)
      - returns plain numpy arrays (gradients) and Python scalars

    Args (packed as tuple for pool.map compatibility):
        env_fn:           callable returning a fresh gym.Env
        flat_params:      current global policy parameters (float32 numpy)
        policy_template:  nn.Module used as architecture blueprint (deepcopied)
        cfg:              A3CConfig
        worker_seed:      int

    Returns:
        flat_grads:  float32 numpy array, same shape as flat_params
        n_steps:     int, environment steps taken in this rollout
        ep_return:   float if an episode completed, else None
    """
    env_fn, flat_params, policy_template, cfg, worker_seed = args

    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)

    # Rebuild local network from the architecture template + current params
    local_net = copy.deepcopy(policy_template)
    set_flat_params(local_net, flat_params)
    local_net.train()

    env       = env_fn()
    obs, _    = env.reset()
    ep_return = None
    ep_accum  = 0.0

    rewards, values, log_probs, entropies = [], [], [], []
    done = False

    for _ in range(cfg.t_max):
        x           = _obs_to_tensor(obs)
        logits, val = local_net(x)
        dist        = Categorical(logits=logits.squeeze(0))
        action      = dist.sample()

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        rewards.append(float(reward))
        values.append(val.squeeze(0))
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())

        obs        = next_obs
        ep_accum  += float(reward)

        if done:
            ep_return = ep_accum
            break

    env.close()
    n_steps = len(rewards)

    # Bootstrap value for the last state
    if done:
        R: float = 0.0
    else:
        with torch.no_grad():
            _, R_t = local_net(_obs_to_tensor(obs))
            R = float(R_t.item())

    # Compute A3C losses (reverse through rollout for n-step returns)
    actor_loss  = torch.zeros(1)
    critic_loss = torch.zeros(1)
    entropy_sum = torch.zeros(1)

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

    local_net.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(local_net.parameters(), cfg.max_grad_norm)

    # Return gradients as flat numpy array (same convention as flat_params)
    flat_grads = np.concatenate([
        p.grad.detach().cpu().numpy().ravel()
        if p.grad is not None
        else np.zeros(p.numel(), dtype=np.float32)
        for p in local_net.parameters()
    ]).astype(np.float32)

    return flat_grads, n_steps, ep_return


# ---------------------------------------------------------------------------
# A3C algorithm
# ---------------------------------------------------------------------------

class A3C(BaseAlgorithm):
    """
    A3C / A2C agent.

    Uses multiprocessing.Pool (same as ES) — workers are stateless, receive
    numpy arrays, and return numpy arrays.  The main process owns the optimizer
    and applies the averaged gradients after each batch of parallel rollouts.

    Args:
        policy:  An ActorCriticCNN or ActorCriticMLP instance.
        config:  A3CConfig.  Defaults are used if None.
    """

    def __init__(self, policy: nn.Module, config: A3CConfig | None = None):
        self.policy = policy
        self.config = config or A3CConfig()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env_fn: Callable, total_steps: int, logger) -> None:
        import multiprocessing as mp

        cfg       = self.config
        optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)

        from collections import deque

        steps          = 0
        generation     = 0
        episode        = 0
        recent_returns = deque(maxlen=20)   # rolling window for mean_return

        ctx  = mp.get_context("spawn")
        pool = ctx.Pool(cfg.n_workers) if cfg.n_workers > 1 else None

        try:
            while steps < total_steps:
                t0 = time.time()

                # Snapshot current parameters as numpy (passed to workers)
                flat_params = get_flat_params(self.policy)

                tasks = [
                    (env_fn, flat_params, self.policy,
                     cfg, cfg.seed + generation * cfg.n_workers + rank)
                    for rank in range(cfg.n_workers)
                ]

                if pool is not None:
                    results = pool.map(_a3c_worker_task, tasks)
                else:
                    results = [_a3c_worker_task(t) for t in tasks]

                # --- Aggregate gradients across workers ---
                all_grads   = np.stack([r[0] for r in results])   # (n_workers, D)
                batch_steps = sum(r[1] for r in results)
                ep_returns  = [r[2] for r in results if r[2] is not None]

                avg_grads = all_grads.mean(axis=0)                # (D,)

                # --- Apply averaged gradients with Adam ---
                optimizer.zero_grad()
                idx = 0
                for p in self.policy.parameters():
                    n = p.numel()
                    p.grad = torch.from_numpy(
                        avg_grads[idx: idx + n].reshape(p.shape)
                    )
                    idx += n
                optimizer.step()

                steps      += batch_steps
                generation += 1
                episode    += len(ep_returns)
                recent_returns.extend(ep_returns)

                # Only log when at least one episode has completed — avoids NaN
                # rows that would make benchmark.py skip this run entirely.
                if recent_returns:
                    logger.log(
                        steps=steps,
                        generation=generation,
                        episode=episode,
                        mean_return=float(np.mean(recent_returns)),
                        gen_time=round(time.time() - t0, 2),
                    )

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
