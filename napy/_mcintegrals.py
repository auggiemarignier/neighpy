from __future__ import annotations  # needed for type hint of self in accumulate
import numpy as np
from numpy.typing import ArrayLike
from typing import Union


class MCIntegrals:
    """
    Class for accumulating samples to calculate MC integrals in a single loop
    """

    def __init__(self, nd: int, save_samples: bool = False):
        self.mi = np.zeros(nd)
        self.mi2 = np.zeros(nd)
        self.mimj = np.zeros((nd, nd))
        self.mi2mj = np.zeros((nd, nd))
        self.mimj2 = np.zeros((nd, nd))
        self.mi2mj2 = np.zeros((nd, nd))
        self.N = 0
        self.samples = [] if save_samples else None

    def accumulate(self, x: Union[ArrayLike, MCIntegrals]):
        if isinstance(x, np.ndarray):
            # usual case
            self._accumulate_arraylike(x)
        else:
            # e.g. accumulating multiple parallel walkers each with their own accumulator
            self._accumulate_mcintegrals(x)

    def _accumulate_arraylike(self, x: ArrayLike):
        self.mi += x
        self.mi2 += x**2
        self.mimj += np.outer(x, x)
        self.mi2mj += np.outer(x**2, x)
        self.mimj2 += np.outer(x, x**2)
        self.mi2mj2 += np.outer(x**2, x**2)
        self.N += 1
        if self.samples is not None:
            self.samples.append(x.copy())

    def _accumulate_mcintegrals(self, x: MCIntegrals):
        self.mi += x.mi
        self.mi2 += x.mi2
        self.mimj += x.mimj
        self.mi2mj += x.mi2mj
        self.mimj2 += x.mimj2
        self.mi2mj2 += x.mi2mj2
        self.N += x.N
        if self.samples is not None and x.samples is not None:
            self.samples.extend(x.samples.copy())

    def mean(self):
        return self.mi / self.N

    def covariance(self):
        mean = self.mean()
        return self.mimj / self.N - np.outer(mean, mean)
