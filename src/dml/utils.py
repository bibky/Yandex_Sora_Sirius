"""Small helpers shared by all experiments."""

import random

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Seed Python and NumPy and return a NumPy Generator.

    Experiments loop over several seeds, so seeds are passed as arguments
    rather than stored in a config file. The framework seed (e.g. torch) is
    set inside the corresponding model module.
    """
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
