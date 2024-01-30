import numpy as np
from numpy.typing import ArrayLike


class NAAppariser:
    def __init__(
        self,
        initial_ensemble: ArrayLike,
        objectives: ArrayLike,
        bounds: tuple[tuple[float, float], ...],
        n_resample: int,
        n_walkers: int = 1,
    ):
        self.initial_ensemble = initial_ensemble
        self.objectives = objectives
        self.bounds = bounds
        self.nd = len(bounds)
        self.lower = np.array([b[0] for b in bounds])
        self.upper = np.array([b[1] for b in bounds])

        self.Ne = len(initial_ensemble)
        self.Nr = n_resample
        self.j = n_walkers if n_walkers >= 1 else 1
        self.nr = self.Nr // self.j

    def appraise(self):
        """
        Perform the appraisal stage of the Neighbourhood Algorithm.
        """
        start_point = self.initial_ensemble[-1]  # This will change at some point
        for _ in range(self.Nr):
            vk = start_point
            for i, xji in enumerate(self._voronoi._cell_axis_intersections(vk, -1)):
                vk[i] = self._random_step(xji)

    def _random_step(self, xji):
        xpi = np.random.uniform(self.lower, self.upper)  # fix the limits
        r = np.random.uniform(0, 1)
        Pxpi = 0  # fix this
        Pmax = 1  # fix this
        accepted = False
        while not accepted:
            if np.log(r) < np.log(Pxpi) - np.log(Pmax):
                accepted = True
                return xpi

    def axis_intersections(self, axis: int, vk: ArrayLike, k: int):
        """
        Calculate the intersections of an axis passing through point vk in the kth cell
        with the boundaries of all cells

        Yields the intersection points.
        """

        # eqn (19) Sambridge 1999
        d2 = np.sum(self.weights * (vk - self.initial_ensemble) ** 2, axis=1)
        vji = self.initial_ensemble[:, axis]
        vki = vk[axis]
        a = d2[k] - d2
        b = vki - vji
        xji = 0.5 * (vki + vji + np.divide(a, b, out=np.zeros_like(a), where=b != 0))
