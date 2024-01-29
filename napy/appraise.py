import numpy as np
from numpy.typing import ArrayLike


class NAAppariser:
    def __init__(
        self, initial_ensemble: ArrayLike, n_resample: int, n_walkers: int = 1
    ):
        self.initial_ensemble = initial_ensemble
        self.Ne = len(initial_ensemble)
        self.Nr = n_resample
        self.j = n_walkers if n_walkers >= 1 else 1
        self.nr = self.Nr // self.j
