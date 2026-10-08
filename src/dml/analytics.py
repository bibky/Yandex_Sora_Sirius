"""Closed-form prices and Greeks (zero rates), used as ground truth on test sets.

All functions are vectorized over `spot`.
"""

import numpy as np
from scipy.stats import norm


def _bs_d1(spot, strike, vol, T):
    return (np.log(spot / strike) + 0.5 * vol * vol * T) / (vol * np.sqrt(T))


def bs_price(spot, strike, vol, T):
    """Black-Scholes call price."""
    d1 = _bs_d1(spot, strike, vol, T)
    return spot * norm.cdf(d1) - strike * norm.cdf(d1 - vol * np.sqrt(T))


def bs_delta(spot, strike, vol, T):
    """Black-Scholes call delta."""
    return norm.cdf(_bs_d1(spot, strike, vol, T))


def bs_vega(spot, strike, vol, T):
    """Black-Scholes call vega."""
    return spot * np.sqrt(T) * norm.pdf(_bs_d1(spot, strike, vol, T))


def _bach_d(spot, strike, vol, T):
    return (spot - strike) / (vol * np.sqrt(T))


def bachelier_price(spot, strike, vol, T):
    """Bachelier (normal model) call price; `vol` is an absolute (normal) volatility."""
    d = _bach_d(spot, strike, vol, T)
    return vol * np.sqrt(T) * (d * norm.cdf(d) + norm.pdf(d))


def bachelier_delta(spot, strike, vol, T):
    """Bachelier call delta."""
    return norm.cdf(_bach_d(spot, strike, vol, T))


def bachelier_vega(spot, strike, vol, T):
    """Bachelier call vega."""
    return np.sqrt(T) * norm.pdf(_bach_d(spot, strike, vol, T))
