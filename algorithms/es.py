"""
Evolution Strategies (ES) — Salimans et al. [2017], arXiv:1703.03864.

Implements Algorithm 1 (sequential) with:
  - Mirrored / antithetic sampling  (Brockhoff et al. [2010])
  - Rank-based fitness shaping      (Wierstra et al. [2014])
  - Weight decay
  - Optional multiprocessing for parallel rollouts

The ES update rule (after rank normalization):
    θ_{t+1} ← θ_t * (1 - lr * wd)
              + lr * (1 / (n * σ)) * Σ_i  shaped_F_i * ε_i
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .base import BaseAlgorithm
from models.atari_cnn import AtariCNN, get_flat_params, set_flat_params
from utils.fitness_shaping import rank_normalize


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ESConfig:
    sigma: float         = 0.05    # Noise std — perturbation scale
    lr: float            = 0.01    # Learning rate (Adam-style tuning not needed for ES)
    population_size: int = 100     # Must be even (mirrored sampling uses n/2 noise vectors)
    weight_decay: float  = 0.005   # L2 regularisation coefficient
    n_workers: int       = 1       # >1 enables multiprocessing (set to CPU count for speed)
    seed: int            = 0       # Master random seed


# ---------------------------------------------------------------------------
# Rollout helpers (module-level so they can be pickled for multiprocessing)
# ---------------------------------------------------------------------------

def _run_episode(env_fn: Callable, flat_params: np.ndarray, policy_template) -> tuple[float, int]:
    """Evaluate a single policy (given as flat param vector). Returns (return, steps)."""
    policy = copy.deepcopy(policy_template)
    set_flat_params(policy, flat_params)
    env = env_fn()
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action = policy.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated
    env.close()
    return total_reward, steps


def _worker_task(args):
    env_fn, params_pos, params_neg, policy_template = args
    r_pos, s_pos = _run_episode(env_fn, params_pos, policy_template)
    r_neg, s_neg = _run_episode(env_fn, params_neg, policy_template)
    return r_pos, s_pos, r_neg, s_neg


# ---------------------------------------------------------------------------
# ES algorithm
# ---------------------------------------------------------------------------

class ES(BaseAlgorithm):

    def __init__(self, policy, config: ESConfig | None = None):
        self.policy = policy
        self.config = config or ESConfig()
        assert self.config.population_size % 2 == 0, \
            "population_size must be even (mirrored sampling)"
        self.theta = get_flat_params(policy)   # 1-D float32 numpy array

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def _collect_reference_batch(self, env_fn: Callable, n_obs: int = 128) -> np.ndarray:
        """
        Run random actions to collect N diverse observations for VBN initialization.
        Returns (N, 4, 84, 84) uint8 array.
        """
        obs_list = []
        env = env_fn()
        obs, _ = env.reset()
        while len(obs_list) < n_obs:
            obs_list.append(obs)
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()
        return np.stack(obs_list[:n_obs])

    def train(self, env_fn: Callable, total_steps: int, logger) -> None:
        cfg = self.config
        half_pop = cfg.population_size // 2
        rng = np.random.default_rng(cfg.seed)

        # --- VBN: collect reference batch and freeze normalization stats ---
        if self.policy.use_vbn:
            print("Collecting reference batch for VBN (128 observations)...")
            ref_batch = self._collect_reference_batch(env_fn)
            self.policy.set_reference_batch(ref_batch)
            self.theta = get_flat_params(self.policy)  # re-sync theta after vbn init
            print(f"VBN ready. theta dim = {len(self.theta):,}")

        steps = 0
        generation = 0

        if cfg.n_workers > 1:
            import multiprocessing as mp
            pool = mp.Pool(cfg.n_workers)
        else:
            pool = None

        try:
            while steps < total_steps:
                t0 = time.time()

                # --- Sample noise (half_pop vectors; evaluated as ±ε pairs) ---
                epsilons = rng.standard_normal(
                    (half_pop, len(self.theta))
                ).astype(np.float32)

                params_pos = [self.theta + cfg.sigma * eps for eps in epsilons]
                params_neg = [self.theta - cfg.sigma * eps for eps in epsilons]

                # --- Evaluate population ---
                returns_pos = np.zeros(half_pop, dtype=np.float32)
                returns_neg = np.zeros(half_pop, dtype=np.float32)
                gen_steps = 0

                if pool is not None:
                    tasks = [
                        (env_fn, params_pos[i], params_neg[i], self.policy)
                        for i in range(half_pop)
                    ]
                    results = pool.map(_worker_task, tasks)
                    for i, (rp, sp, rn, sn) in enumerate(results):
                        returns_pos[i] = rp
                        returns_neg[i] = rn
                        gen_steps += sp + sn
                else:
                    for i, eps in enumerate(epsilons):
                        rp, sp = _run_episode(env_fn, params_pos[i], self.policy)
                        rn, sn = _run_episode(env_fn, params_neg[i], self.policy)
                        returns_pos[i] = rp
                        returns_neg[i] = rn
                        gen_steps += sp + sn

                steps += gen_steps
                generation += 1

                # --- Fitness shaping: rank-normalize all 2*half_pop returns ---
                all_returns = np.concatenate([returns_pos, returns_neg])
                all_epsilons = np.vstack([epsilons, -epsilons])   # (n, D)
                shaped = rank_normalize(all_returns)               # (n,)

                # --- Gradient estimate ---
                n = cfg.population_size
                grad = (1.0 / (n * cfg.sigma)) * (all_epsilons.T @ shaped)

                # --- Parameter update with weight decay ---
                self.theta = (
                    self.theta * (1.0 - cfg.lr * cfg.weight_decay)
                    + cfg.lr * grad
                )

                set_flat_params(self.policy, self.theta)

                logger.log(
                    steps=steps,
                    generation=generation,
                    mean_return=float(np.mean(all_returns)),
                    max_return=float(np.max(all_returns)),
                    min_return=float(np.min(all_returns)),
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
        set_flat_params(self.policy, self.theta)
        total = 0.0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = self.policy.get_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                done = terminated or truncated
        return total / n_episodes

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        np.save(path, self.theta)

    def load(self, path: str) -> None:
        path = path if path.endswith(".npy") else path + ".npy"
        self.theta = np.load(path).astype(np.float32)
        set_flat_params(self.policy, self.theta)
