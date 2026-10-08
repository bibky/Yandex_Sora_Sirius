"""Plots in the style of the original notebooks."""

import matplotlib.pyplot as plt
import numpy as np


def plot_methods(x_axis, y_true, preds: dict, samples=None, xlim=None, ylim=None,
                 ylabel="value", save_path=None):
    """One panel per method: predicted vs exact against `x_axis` (spot or basket value).

    preds: {method_name: predictions}; samples: optional (x_axis_train, y_train) shown as circles.
    """
    order = np.argsort(np.ravel(x_axis))
    fig, axs = plt.subplots(1, len(preds), figsize=(5 * len(preds), 4.5), squeeze=False)
    for ax, (name, pred) in zip(axs[0], preds.items()):
        if samples is not None:
            ax.plot(*samples, "o", color="tab:cyan", markersize=4, markerfacecolor="white", label="samples")
        ax.plot(np.ravel(x_axis)[order], np.ravel(pred)[order], ".", color="tab:blue", markersize=2, label="predicted")
        ax.plot(np.ravel(x_axis)[order], np.ravel(y_true)[order], "-", color="tab:red", lw=1.2, label="exact")
        ax.set_title(name)
        ax.set_ylabel(ylabel)
        if xlim: ax.set_xlim(*xlim)
        if ylim: ax.set_ylim(*ylim)
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
