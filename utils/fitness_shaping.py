"""
Fitness shaping (rank-based return normalization).
From Wierstra et al. [2014], used in Salimans et al. [2017] Section 2.

Rank-transforming returns before computing the gradient update:
  - Removes the influence of outlier individuals in each population
  - Reduces tendency for ES to fall into local optima early in training
  - Makes the algorithm invariant to monotone transformations of the fitness
"""

import numpy as np


def rank_normalize(returns: np.ndarray) -> np.ndarray:
    """
    Map returns to centered ranks in [-0.5, 0.5].

    Args:
        returns: 1-D array of episode returns, length n.

    Returns:
        Normalized array of same shape, where the lowest return maps to
        -(n-1)/(2(n-1)) = -0.5 and the highest maps to +0.5.
    """
    n = len(returns)
    assert n > 1, "Need at least 2 returns to rank-normalize."
    ranks = np.empty(n, dtype=np.float32)
    ranks[np.argsort(returns)] = np.arange(n, dtype=np.float32)
    return ranks / (n - 1) - 0.5
