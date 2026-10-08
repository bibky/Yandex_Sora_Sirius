"""Error metrics and a small evaluation helper shared by the experiments."""

import numpy as np


def rmse(pred, target, axis=0):
    """Root mean squared error (along `axis`)."""
    return np.sqrt(np.mean((np.asarray(pred) - np.asarray(target)) ** 2, axis=axis))


def run_over_seeds(experiment, seeds):
    """Call `experiment(seed)` for every seed and collect the results in a list."""
    return [experiment(seed) for seed in seeds]


def evaluate_regressors(methods: dict, x, y, z, x_test, price_test, delta_test):
    """Fit every model and return {name: (price RMSE, delta RMSE averaged over inputs)}.

    methods: {name: factory}. Models with a `fit(x, y, z)` / `predict(x, predict_derivs=True)`
    interface (differential) are trained on derivatives; others are fitted on (x, y) and their
    deltas are obtained by finite differences.
    """
    from .models.regression import DifferentialRegression, numerical_derivative

    out = {}
    for name, make in methods.items():
        model = make()
        if isinstance(model, DifferentialRegression):
            model.fit(x, y, z)
            pred, dpred = model.predict(x_test, predict_derivs=True)
        else:
            model.fit(x, y)
            pred = model.predict(x_test)
            dpred = numerical_derivative(model.predict, x_test)
        out[name] = (float(rmse(np.ravel(pred), np.ravel(price_test))),
                     float(np.mean(rmse(dpred, delta_test))))
    return out
