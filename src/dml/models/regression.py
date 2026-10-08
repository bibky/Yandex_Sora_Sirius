"""Polynomial regression: classic (OLS), ridge and differential (paper Appx 3).

Compared with DifferentialRegression.ipynb:
* sklearn >= 1.2 removed `normalize=True`. The exact replacement is a StandardScaler step
  with the penalty multiplied by the number of samples; `PolynomialRidge*` do this, so their
  `alpha` is on the same scale as in the original notebook.
* Derivatives of the polynomial basis are computed exactly from the exponents
  (the notebook divided by x, which breaks near x = 0).
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

DEFAULT_ALPHAS = np.exp(np.linspace(np.log(1e-5), np.log(1e2), 100))  # grid of the original notebook


# ---------------------------------------------------------------- polynomial basis

class PolynomialBasis:
    """All monomials of total degree <= `degree` (constant included) and their gradients."""

    def __init__(self, degree: int = 5):
        self.degree = degree
        self._poly = PolynomialFeatures(degree=degree, include_bias=True)

    def fit(self, x):
        self._poly.fit(x)
        self.powers_ = self._poly.powers_          # (p, n) exponents
        return self

    def transform(self, x):
        return self._poly.transform(x)

    def gradient(self, x):
        """d phi_k / d x_j, shape (m, p, n)."""
        x = np.asarray(x, dtype=float)
        m, n = x.shape
        out = np.empty((m, self.powers_.shape[0], n))
        for j in range(n):
            pw = self.powers_.copy()
            coef = pw[:, j].astype(float)
            pw[:, j] = np.maximum(pw[:, j] - 1, 0)
            # product over inputs one at a time: temporary arrays stay (m, p), not (m, p, n)
            prod = np.ones((m, pw.shape[0]))
            for i in range(n):
                prod *= x[:, i:i + 1] ** pw[None, :, i]
            out[:, :, j] = coef * prod
        return out


# ---------------------------------------------------------------- baselines

def make_polynomial(degree: int = 5, scale: bool = True):
    """Classic polynomial regression. `scale=False` = raw features (numerically fragile)."""
    steps = [PolynomialFeatures(degree=degree, include_bias=False)]
    if scale:
        steps.append(StandardScaler())
    return make_pipeline(*steps, LinearRegression())


class PolynomialRidge:
    """Ridge on standardized polynomial features; `alpha` on the original `normalize=True` scale."""

    def __init__(self, degree: int = 5, alpha: float = 1.0):
        self.degree, self.alpha = degree, alpha

    def fit(self, x, y):
        self.model_ = make_pipeline(PolynomialFeatures(self.degree, include_bias=False),
                                    StandardScaler(), Ridge(alpha=self.alpha * len(x)))
        self.model_.fit(x, y)
        return self

    def predict(self, x):
        return self.model_.predict(x).reshape(-1, 1)


class PolynomialRidgeCV:
    """RidgeCV (leave-one-out) on standardized polynomial features.

    `alpha_` is reported on the original `normalize=True` scale.
    `scale=False` reproduces a naive port without normalization (alpha then hits the grid edge).
    """

    def __init__(self, degree: int = 5, alphas=DEFAULT_ALPHAS, scale: bool = True):
        self.degree, self.alphas, self.scale = degree, np.asarray(alphas), scale

    def fit(self, x, y):
        if self.scale:
            n = len(x)
            self.model_ = make_pipeline(PolynomialFeatures(self.degree, include_bias=False),
                                        StandardScaler(), RidgeCV(alphas=self.alphas * n))
            self.model_.fit(x, y)
            self.alpha_ = self.model_[-1].alpha_ / n
        else:
            self.model_ = make_pipeline(PolynomialFeatures(self.degree), RidgeCV(alphas=self.alphas))
            self.model_.fit(x, y)
            self.alpha_ = self.model_[-1].alpha_
        return self

    def predict(self, x):
        return self.model_.predict(x).reshape(-1, 1)


# ---------------------------------------------------------------- differential regression

class DifferentialRegression:
    """Polynomial regression fitted on values and pathwise derivatives (Appx 3).

    Minimizes ||y - phi b||^2 + alpha * sum_j lambda_j ||z_j - phi_j b||^2 with
    lambda_j = mean(y^2) / mean(z_j^2), solved by the modified normal equation with an
    SVD-based pseudo-inverse. Defaults reproduce the original notebook.

    standardize: scale inputs to zero mean / unit std before building the basis
    (better conditioning; derivatives are converted back by the chain rule).
    """

    def __init__(self, degree: int = 5, alpha: float = 1.0, standardize: bool = False):
        self.degree, self.alpha, self.standardize = degree, alpha, standardize

    def _scale(self, x):
        return (x - self.x_mean_) / self.x_std_ if self.standardize else x

    def fit(self, x, y, z):
        x, y, z = (np.asarray(a, dtype=float) for a in (x, y, z))
        y = y.reshape(-1, 1)
        if self.standardize:
            self.x_mean_, self.x_std_ = x.mean(axis=0), x.std(axis=0)
            z = z * self.x_std_                 # dy/dx_scaled = dy/dx * std
        xs = self._scale(x)

        self.basis_ = PolynomialBasis(self.degree).fit(xs)
        phi = self.basis_.transform(xs)                       # (m, p)
        dphi = self.basis_.gradient(xs)                       # (m, p, n)

        self.lambda_j_ = (y**2).mean() / (z**2).mean(axis=0)  # (n,)
        dphi_w = dphi * self.lambda_j_.reshape(1, 1, -1)
        lhs = phi.T @ phi + self.alpha * np.tensordot(dphi_w, dphi, axes=([0, 2], [0, 2]))
        rhs = phi.T @ y + self.alpha * np.tensordot(dphi_w, z, axes=([0, 2], [0, 1])).reshape(-1, 1)
        self.coef_ = np.linalg.pinv(lhs, hermitian=True) @ rhs
        return self

    def predict(self, x, predict_derivs: bool = False):
        xs = self._scale(np.asarray(x, dtype=float))
        y = self.basis_.transform(xs) @ self.coef_
        if not predict_derivs:
            return y
        # in chunks: the basis gradient is (rows, p, n) and gets large for big test sets
        coef = self.coef_.ravel()
        z = np.concatenate([np.tensordot(self.basis_.gradient(xs[i:i + 1000]), coef, axes=(1, 0))
                            for i in range(0, len(xs), 1000)])
        if self.standardize:
            z = z / self.x_std_
        return y, z


# ---------------------------------------------------------------- derivatives of any model

def numerical_derivative(predict, x, rel_step: float = 1e-4):
    """Central finite differences of `predict` w.r.t. each input, shape (m, n).

    Used to get deltas from the baseline models (exact up to O(h^2) for polynomials).
    """
    x = np.asarray(x, dtype=float)
    h = rel_step * np.maximum(np.abs(x).mean(axis=0), 1.0)
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        up, dn = x.copy(), x.copy()
        up[:, j] += h[j]
        dn[:, j] -= h[j]
        out[:, j] = (np.ravel(predict(up)) - np.ravel(predict(dn))) / (2 * h[j])
    return out