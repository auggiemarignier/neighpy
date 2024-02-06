import numpy as np
from numpy.typing import ArrayLike


class MCIntegrals:
    """
    Class for accumulating samples to calculate MC integrals in a single loop
    """

    def __init__(self, nd: int, save_sammples: bool = False):
        self.mi = np.zeros(nd)
        self.mi2 = np.zeros(nd)
        self.mimj = np.zeros((nd, nd))
        self.mi2mj = np.zeros((nd, nd))
        self.mimj2 = np.zeros((nd, nd))
        self.mi2mj2 = np.zeros((nd, nd))
        self.N = 0
        self.samples = [] if save_sammples else None

    def accumulate(self, x: ArrayLike):
        self.mi += x
        self.mi2 += x**2
        self.mimj += np.outer(x, x)
        self.mi2mj += np.outer(x**2, x)
        self.mimj2 += np.outer(x, x**2)
        self.mi2mj2 += np.outer(x**2, x**2)
        self.N += 1
        if self.samples is not None:
            self.samples.append(x)

    def mean(self):
        return self.mi / self.N

    def covariance(self):
        mean = self.mean()
        return self.mimj / self.N - np.outer(mean, mean)
