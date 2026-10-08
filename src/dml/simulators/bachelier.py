"""Basket call in a correlated Bachelier (normal) model — paper section 2.1."""

import numpy as np

from .. import analytics


def random_correlation(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random correlation matrix (same recipe as the authors' genCorrel)."""
    r = rng.uniform(-1.0, 1.0, size=(2 * n, n))
    cov = r.T @ r
    inv = np.diag(1.0 / np.sqrt(np.diag(cov)))
    return inv @ cov @ inv


class BachelierBasket:
    """Call on the basket sum_i w_i S_i, with dS_i = vol_i dW_i (correlated).

    The basket is itself normal, so its price and deltas are exact (Bachelier formula).
    """

    def __init__(self, weights, correl, vols, K: float = 110.0, T: float = 3.0):
        self.weights = np.asarray(weights, dtype=float).ravel()
        self.correl = np.asarray(correl, dtype=float)
        self.vols = np.asarray(vols, dtype=float).ravel()
        self.K, self.T = K, T
        self.n = self.weights.size
        cov = np.diag(self.vols) @ self.correl @ np.diag(self.vols) * T
        self._chol = np.linalg.cholesky(cov)
        wv = self.weights * self.vols
        self.basket_vol = float(np.sqrt(wv @ self.correl @ wv))

    @classmethod
    def random(cls, n: int, rng: np.random.Generator, basket_vol: float = 20.0,
               K: float = 110.0, T: float = 3.0):
        """Random weights, correlation and vols, rescaled so the basket vol equals `basket_vol`
        (as in DifferentialRegression.ipynb)."""
        correl = random_correlation(n, rng)
        weights = rng.uniform(size=n)
        weights /= weights.sum()
        vols = rng.uniform(size=n)
        wv = weights * vols
        vols *= basket_vol / np.sqrt(wv @ correl @ wv)
        return cls(weights, correl, vols, K, T)

    def basket(self, s0):
        return np.asarray(s0) @ self.weights.reshape(-1, 1)

    def price(self, s0):
        return analytics.bachelier_price(self.basket(s0), self.K, self.basket_vol, self.T)

    def delta(self, s0):
        """Deltas to each asset, shape (m, n)."""
        b = self.basket(s0)
        return analytics.bachelier_delta(b, self.K, self.basket_vol, self.T) * self.weights.reshape(1, -1)

    def sample_payoffs(self, s0, rng: np.random.Generator, antithetic: bool = False):
        """One simulated payoff per initial state and its pathwise derivatives.

        s0: shape (m, n). Returns y (m, 1), z (m, n).
        """
        s0 = np.asarray(s0, dtype=float)
        inc = rng.standard_normal(size=s0.shape) @ self._chol.T
        w = self.weights.reshape(1, -1)

        def one(sign):
            bT = self.basket(s0 + sign * inc)
            return np.maximum(bT - self.K, 0.0), (bT > self.K) * w

        y, z = one(+1.0)
        if antithetic:
            ya, za = one(-1.0)
            y, z = 0.5 * (y + ya), 0.5 * (z + za)
        return y, z
