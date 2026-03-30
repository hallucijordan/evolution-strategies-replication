"""
Abstract base class for all algorithms.
To add a new algorithm:
  1. Subclass BaseAlgorithm
  2. Implement train(), evaluate(), save(), load()
  3. Register it in algorithms/__init__.py → REGISTRY
"""

from abc import ABC, abstractmethod
from typing import Callable


class BaseAlgorithm(ABC):

    @abstractmethod
    def train(self, env_fn: Callable, total_steps: int, logger) -> None:
        """
        Train for at least `total_steps` environment steps.

        Args:
            env_fn:       Callable with no args that returns a fresh gym.Env.
            total_steps:  Total environment steps budget.
            logger:       utils.logger.Logger instance — call logger.log(**kwargs).
        """
        ...

    @abstractmethod
    def evaluate(self, env, n_episodes: int = 10) -> float:
        """
        Run n_episodes with the current (greedy) policy.

        Returns:
            Mean undiscounted episode return.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the policy to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore the policy from disk."""
        ...
