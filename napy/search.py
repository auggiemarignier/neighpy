import numpy as np
from numpy.typing import ArrayLike
from typing import Callable

from ._voronoi import _VoronoiDiagram


class NASearcher:
    def __init__(
        self,
        objective: Callable[[ArrayLike], float],
        ns: int,
        nr: int,
        ni: int,
        n: int,
        bounds: tuple[tuple[float, float], ...],
    ) -> None:
        """
        Initialize a new instance of the NASearcher class.

        Args:
            objective (Callable[[ArrayLike], float]): The objective function to minimize.
                This function should take a single argument of type ArrayLike and return a float.
            ns (int): The number of samples generated at each iteration.
            nr (int): The number of cells to resample.
            ni (int): The number of samples from initial random search.
            n (int): The number of iterations.
            bounds (tuple[tuple[float, float], ...]): A tuple of tuples representing the bounds of the search space.
                Each inner tuple represents the lower and upper bounds for a specific dimension.

        Returns:
            None
        """

        self.objective = objective

        self.ns = ns  # number of samples generated at each iteration
        self.nr = nr  # number of cells to resample
        self.nspnr = ns // nr  # number of samples per cell to generate
        self.ni = ni  # number of samples from initial random search
        self.n = n  # number of iterations
        self.nt = ni + n * ns  # total number of samples
        self.np = 0  # running total of number of samples

        self.bounds = bounds  # bounds of the search space
        self.nd = len(bounds)  # number of dimensions
        self.lower = np.array([b[0] for b in bounds])
        self.upper = np.array([b[1] for b in bounds])
        self.Cm = (
            1 / (self.upper - self.lower) ** 2
        )  # (diagonal) prior covariance matrix

        self.samples = np.zeros((self.nt, self.nd))
        self.objectives = np.full(
            self.nt, np.inf
        )  # start with inf since we want to minimize
        self._current_best_ind = 0

        self._voronoi = _VoronoiDiagram(self.samples, self.Cm)

    def run(self) -> None:
        # initial random search
        new_samples = self._initial_random_search()
        self._update_ensemble(new_samples)

        # main optimisation loop
        for _ in range(self.n):
            inds = self._get_best_indices()
            self._current_best_ind = inds[0]
            cells_to_resample = self.samples[inds]

            for k, cell in zip(inds, cells_to_resample):
                new_samples = self._random_walk_in_voronoi(cell, k)
                self._update_ensemble(new_samples)

    def _initial_random_search(self) -> ArrayLike:
        return np.random.uniform(
            low=self.lower,
            high=self.upper,
            size=(self.ni, self.nd),
        )

    def _random_walk_in_voronoi(self, vk: ArrayLike, k: int) -> ArrayLike:
        # FOLLOWING https://github.com/underworldcode/pyNA/blob/30d1cb7955d6b1389eae885127389ed993fa6940/pyNA/sampler.py#L85

        # vk is the current voronoi cell
        # k is the index of the current voronoi cell

        new_samples = np.empty((self.nspnr, self.nd))
        walk_length = self.nspnr
        if k == self._current_best_ind:
            # best model so walk a bit further
            walk_length += self.ns % self.nr

        for _step in range(walk_length):
            xA = vk.copy()  # start of walk at cell centre
            for i, xji in enumerate(self._voronoi._cell_axis_intersections(vk, k)):
                # generator yields intersection points with axis i

                # eqns (20, 21) Sambridge 1999
                li = np.nanmax(np.hstack((self.lower[i], xji[xji < xA[i]])))
                ui = np.nanmin(np.hstack((self.upper[i], xji[xji > xA[i]])))
                xA[i] = np.random.uniform(li, ui)

            new_samples[_step] = xA

        return new_samples

    def _get_best_indices(self) -> ArrayLike:
        # there may be a faster way to do this using np.argpartition
        return np.argsort(self.objectives)[: self.nr]

    def _update_ensemble(self, new_samples: ArrayLike):
        n = new_samples.shape[0]
        self.samples[self.np : self.np + n] = new_samples
        self.objectives[self.np : self.np + n] = np.apply_along_axis(
            self.objective, 1, new_samples
        )
        self.np += n
