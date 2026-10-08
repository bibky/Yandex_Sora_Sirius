"""Data normalization (paper Appx 2; normalize_data in the main notebook)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Scaler:
    """Stores the statistics needed to scale inputs and unscale predictions."""
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray


def normalize(x, y, dydx=None, eps: float = 1e-8):
    """Standardize X and Y, rescale dY/dX, and compute derivative weights lambda_j.

    Returns (scaler, x_n, y_n, dydx_n, lambda_j).
    """
    raise NotImplementedError
