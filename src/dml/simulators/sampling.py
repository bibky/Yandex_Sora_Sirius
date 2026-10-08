"""Samplers for initial states X (the regression inputs)."""

import numpy as np


def grid(lower: float, upper: float, m: int) -> np.ndarray:
    """Evenly spaced 1-d grid, shape (m, 1)."""
    return np.linspace(lower, upper, m).reshape(-1, 1)


def uniform(lower: float, upper: float, m: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Independent uniform draws, shape (m, n)."""
    return rng.uniform(lower, upper, size=(m, n))
