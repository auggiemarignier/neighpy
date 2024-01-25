import numpy as np
from numpy.typing import ArrayLike
from typing import Callable


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
        self.nspnr = (
            ns // nr
        )  # number of samples per cell to generate TODO: sort out what happens with the remainder
        self.ni = ni  # number of samples from initial random search
        self.n = n  # number of iterations
        self.nt = ni + n * (nr * ns)  # total number of samples
        self.np = 0  # running total of number of samples

        self.bounds = bounds  # bounds of the search space
        self.nd = len(bounds)  # number of dimensions
        self.lower = np.array([b[0] for b in bounds])
        self.upper = np.array([b[1] for b in bounds])

        self.samples = np.zeros((self.nt, self.nd))
        self.objectives = np.full(
            self.nt, np.inf
        )  # start with inf since we want to minimize

    def run(self) -> None:
        # initial random search
        new_samples = self._initial_random_search()
        self._update_ensemble(new_samples)

        # main optimisation loop
        for i in range(1, self.n):
            inds = self._get_best_indices()
            cells_to_resample = self.samples[inds]
            for cell in cells_to_resample:
                new_samples = self._resample_cell(cell, inds)
                self._update_ensemble(new_samples)

    def _initial_random_search(self) -> None:
        return np.random.uniform(
            low=self.lower,
            high=self.upper,
            size=(self.ni, self.nd),
        )

    def _get_best_indices(self) -> np.ndarray:
        # there may be a faster way to do this using np.argpartition
        return np.argsort(self.objectives)[: self.nr]

    def _resample_cell(self, cell: ArrayLike, inds: np.ndarray) -> ArrayLike:
        new_samples = np.empty((self.nspnr, self.nd))

        # IMPLEMENT RANDOM WALK HERE

        return new_samples

    def _update_ensemble(self, new_samples):
        n = new_samples.shape[0]
        self.samples[self.np : self.np + n] = new_samples
        self.objectives[self.np : self.np + n] = np.apply_along_axis(
            self.objective, 1, new_samples
        )
        self.np += n

    def _random_walk_in_voronoi(self, vk: ArrayLike, k: int) -> ArrayLike:
        # FOLLOWING https://github.com/underworldcode/pyNA/blob/30d1cb7955d6b1389eae885127389ed993fa6940/pyNA/sampler.py#L85

        # vk is the current voronoi cell
        # k is the index of the current voronoi cell

        new_samples = np.empty((self.nspnr, self.nd))
        old_samples = self.samples[: self.np]
        walk_length = self.nspnr

        # find cell boundaries along each dimension
        d2 = np.sum((vk - old_samples) ** 2, axis=1)  # distance to all other cells

        d2_previous_axis = 0  # distance to previous axis

        for _step in range(walk_length):
            xA = vk.copy()  # start of walk at cell centre
            for i in range(self.nd):  # step along each axis
                d2_current_axis = (xA[i] - old_samples[:, i]) ** 2
                d2 += d2_previous_axis - d2_current_axis
                dk2 = d2[k]  # disctance of cell centre to axis

                # eqn (19) Sambridge 1999
                vji = old_samples[:, i]
                vki = vk[i]
                xji = 0.5 * (vki + vji + (dk2 - d2) / (vki - vji))

                # eqns (20, 21) Sambridge 1999
                li = np.nanmax(np.hstack((self.lower[i], xji[xji < xA[i]])))
                ui = np.nanmin(np.hstack((self.upper[i], xji[xji > xA[i]])))
                xA[i] = np.random.uniform(li, ui)

                d2_previous_axis = d2_current_axis

            new_samples[_step] = xA

        return new_samples
