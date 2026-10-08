"""European call in Black-Scholes (zero rates)."""

import numpy as np

from .. import analytics


class BlackScholesCall:
    """Call with strike K and maturity T on a lognormal stock with volatility `vol`."""

    def __init__(self, K: float = 110.0, vol: float = 0.2, T: float = 2.0):
        self.K, self.vol, self.T = K, vol, T

    def price(self, s0):
        return analytics.bs_price(s0, self.K, self.vol, self.T)

    def delta(self, s0):
        return analytics.bs_delta(s0, self.K, self.vol, self.T)

    def sample_payoffs(self, s0, rng: np.random.Generator, antithetic: bool = False):
        """One simulated payoff per initial state and its pathwise derivative dPayoff/dS0.

        s0: shape (m, 1). Returns y (m, 1), z (m, 1).
        """
        s0 = np.asarray(s0, dtype=float).reshape(-1, 1)
        w = rng.standard_normal(size=s0.shape)
        drift, diff = -0.5 * self.vol**2 * self.T, self.vol * np.sqrt(self.T)

        def one(sign):
            growth = np.exp(drift + sign * diff * w)        # S_T / S_0
            sT = s0 * growth
            return np.maximum(sT - self.K, 0.0), np.where(sT > self.K, growth, 0.0)

        y, z = one(+1.0)
        if antithetic:
            ya, za = one(-1.0)
            y, z = 0.5 * (y + ya), 0.5 * (z + za)
        return y, z
