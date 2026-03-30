"""
Random policy baseline — takes uniformly random actions every step.
Useful as a sanity-check lower bound when comparing algorithms.
"""

import numpy as np
from .base import BaseAlgorithm


class RandomPolicy(BaseAlgorithm):

    def train(self, env_fn, total_steps: int, logger) -> None:
        steps = 0
        episode = 0
        while steps < total_steps:
            env = env_fn()
            obs, _ = env.reset()
            ep_return = 0.0
            ep_steps = 0
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += float(reward)
                ep_steps += 1
                done = terminated or truncated
            env.close()
            steps += ep_steps
            episode += 1
            logger.log(steps=steps, episode=episode, mean_return=ep_return)

    def evaluate(self, env, n_episodes: int = 10) -> float:
        total = 0.0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                done = terminated or truncated
        return total / n_episodes

    def save(self, path: str) -> None:
        pass  # nothing to persist

    def load(self, path: str) -> None:
        pass
