"""LSM training-set generators of the main experiment (DifferentialML notebook of the authors).

Training inputs are states at T1 (simulated with vol * volMult for wider coverage), labels are
payoffs at T2 on one path (two antithetic paths averaged) with their pathwise derivatives.
Test sets use exact prices and deltas. Moved without changes from DifferentialML_PyTorch.ipynb.
"""

import numpy as np

from ..analytics import (bachelier_delta, bachelier_price, bachelier_vega,
                         bs_delta, bs_price, bs_vega)


class BlackScholes:
    def __init__(self, vol=0.2, T1=1, T2=2, K=1.10, volMult=1.5):
        self.spot = 1
        self.vol, self.T1, self.T2, self.K, self.volMult = vol, T1, T2, K, volMult

    def trainingSet(self, m, anti=True, seed=None):
        """Returns S1 (m,1), payoff (m,1), dPayoff/dS1 (m,1)."""
        rng = np.random.default_rng(seed)
        returns = rng.normal(size=[m, 2])

        vol0 = self.vol * self.volMult
        R1 = np.exp(-0.5 * vol0 * vol0 * self.T1
                    + vol0 * np.sqrt(self.T1) * returns[:, 0])
        R2 = np.exp(-0.5 * self.vol * self.vol * (self.T2 - self.T1)
                    + self.vol * np.sqrt(self.T2 - self.T1) * returns[:, 1])
        S1 = self.spot * R1; S2 = S1 * R2
        pay = np.maximum(0, S2 - self.K)

        if anti:
            R2a = np.exp(-0.5 * self.vol * self.vol * (self.T2 - self.T1)
                         - self.vol * np.sqrt(self.T2 - self.T1) * returns[:, 1])
            S2a = S1 * R2a
            paya = np.maximum(0, S2a - self.K)
            X, Y = S1, 0.5 * (pay + paya)
            Z1 = np.where(S2  > self.K, R2,  0.0).reshape((-1, 1))
            Z2 = np.where(S2a > self.K, R2a, 0.0).reshape((-1, 1))
            Z = 0.5 * (Z1 + Z2)
        else:
            X, Y = S1, pay
            Z = np.where(S2 > self.K, R2, 0.0).reshape((-1, 1))

        return X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)

    def testSet(self, lower=0.35, upper=1.65, num=100, seed=None):
        """Returns spots, x-axis (= spots), exact prices, deltas and vegas on a grid."""
        spots  = np.linspace(lower, upper, num).reshape((-1, 1))
        prices = bs_price(spots, self.K, self.vol, self.T2 - self.T1).reshape((-1, 1))
        deltas = bs_delta(spots, self.K, self.vol, self.T2 - self.T1).reshape((-1, 1))
        vegas  = bs_vega (spots, self.K, self.vol, self.T2 - self.T1).reshape((-1, 1))
        return spots, spots, prices, deltas, vegas


def genCorrel(n, rng):
    """Random correlation matrix."""
    randoms = rng.uniform(low=-1.0, high=1.0, size=(2 * n, n))
    cov = randoms.T @ randoms
    invvols = np.diag(1.0 / np.sqrt(np.diagonal(cov)))
    return np.linalg.multi_dot([invvols, cov, invvols])


class Bachelier:
    """Basket call on n correlated Bachelier assets.

    Note: weights, correlation and vols are drawn in trainingSet(), so testSet() must be
    called after it (same behaviour as in the authors' notebook).
    """

    def __init__(self, n, T1=1, T2=2, K=1.10, volMult=1.5):
        self.n = n
        self.T1, self.T2, self.K, self.volMult = T1, T2, K, volMult

    def trainingSet(self, m, anti=True, seed=None, bktVol=0.2):
        """Returns S1 (m,n), payoff (m,1), dPayoff/dS1 (m,n)."""
        rng = np.random.default_rng(seed)

        self.S0 = np.repeat(1.0, self.n)
        self.corr = genCorrel(self.n, rng)

        self.a = rng.uniform(low=1.0, high=10.0, size=self.n)
        self.a /= np.sum(self.a)
        vols = rng.uniform(low=5.0, high=50.0, size=self.n)

        avols = (self.a * vols).reshape((-1, 1))
        v = np.sqrt(np.linalg.multi_dot([avols.T, self.corr, avols]).reshape(1))
        self.vols = vols * bktVol / v
        self.bktVol = bktVol

        diagv = np.diag(self.vols)
        self.cov = np.linalg.multi_dot([diagv, self.corr, diagv])
        self.chol  = np.linalg.cholesky(self.cov) * np.sqrt(self.T2 - self.T1)
        self.chol0 = self.chol * self.volMult * np.sqrt(self.T1 / (self.T2 - self.T1))

        normals = rng.normal(size=[2, m, self.n])
        inc0 = normals[0] @ self.chol0.T
        inc1 = normals[1] @ self.chol.T

        S1 = self.S0 + inc0; S2 = S1 + inc1
        bkt2 = S2 @ self.a
        pay  = np.maximum(0, bkt2 - self.K)

        if anti:
            S2a = S1 - inc1
            bkt2a = S2a @ self.a
            paya = np.maximum(0, bkt2a - self.K)
            X, Y = S1, 0.5 * (pay + paya)
            Z1 = np.where(bkt2  > self.K, 1.0, 0.0).reshape((-1, 1)) * self.a.reshape((1, -1))
            Z2 = np.where(bkt2a > self.K, 1.0, 0.0).reshape((-1, 1)) * self.a.reshape((1, -1))
            Z = 0.5 * (Z1 + Z2)
        else:
            X, Y = S1, pay
            Z = np.where(bkt2 > self.K, 1.0, 0.0).reshape((-1, 1)) * self.a.reshape((1, -1))

        return X, Y.reshape(-1, 1), Z

    def testSet(self, lower=0.5, upper=1.50, num=4096, seed=None):
        """Returns spots, baskets, exact prices, deltas (num,n) and vegas."""
        rng = np.random.default_rng(seed)
        adj = 1 + 0.5 * np.sqrt((self.n - 1) * (upper - lower) / 12)
        adj_lower = 1.0 - (1.0 - lower) * adj
        adj_upper = 1.0 + (upper - 1.0) * adj
        spots = rng.uniform(low=adj_lower, high=adj_upper, size=(num, self.n))

        baskets = (spots @ self.a).reshape((-1, 1))
        prices  = bachelier_price(baskets, self.K, self.bktVol, self.T2 - self.T1).reshape((-1, 1))
        deltas  = bachelier_delta(baskets, self.K, self.bktVol, self.T2 - self.T1) @ self.a.reshape((1, -1))
        vegas   = bachelier_vega (baskets, self.K, self.bktVol, self.T2 - self.T1)
        return spots, baskets, prices, deltas, vegas
