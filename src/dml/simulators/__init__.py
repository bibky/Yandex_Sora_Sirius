"""Option models that produce training labels (payoffs + pathwise derivatives) and exact test values.

Design: a model knows how to simulate one payoff from a given initial state and how to
price exactly; *where* the initial states come from (grid, uniform, simulated horizon)
is decided by the experiment, using the samplers in `dml.simulators.sampling`.
"""

from .bachelier import BachelierBasket
from .black_scholes import BlackScholesCall
from .sampling import grid, uniform

__all__ = ["BlackScholesCall", "BachelierBasket", "grid", "uniform"]
