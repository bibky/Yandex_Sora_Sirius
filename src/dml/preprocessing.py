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

def normalize_data(x_raw, y_raw, dydx_raw=None, crop=None, eps: float = 1e-8):
    """Standardize inputs and labels, rescale pathwise derivatives, compute derivative weights.

    crop: use only the first `crop` examples (None = all).
    Returns x_mean, x_std, x, y_mean, y_std, y, dy_dx, lambda_j
    (dy_dx and lambda_j are None when no derivative labels are given).
    """
    m = crop if crop is not None else x_raw.shape[0]

    xc = x_raw[:m]; yc = y_raw[:m]
    dyc = dydx_raw[:m] if dydx_raw is not None else None

    x_mean = xc.mean(axis=0); x_std = xc.std(axis=0) + eps
    x = (xc - x_mean) / x_std

    y_mean = yc.mean(axis=0); y_std = yc.std(axis=0) + eps
    y = (yc - y_mean) / y_std

    if dyc is not None:
        dy_dx = dyc / y_std * x_std
        # weight of each derivative in the loss = 1 / its root mean square
        lambda_j = 1.0 / np.sqrt((dy_dx ** 2).mean(axis=0)).reshape(1, -1)
    else:
        dy_dx = None; lambda_j = None

    return x_mean, x_std, x, y_mean, y_std, y, dy_dx, lambda_j
