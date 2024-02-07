import numpy as np
from numpy.typing import ArrayLike
from typing import Callable
from joblib import Parallel, delayed
from time import time


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

        self._check_objective()

    def run(self) -> None:
        # initial random search
        new_samples = self._initial_random_search()
        self._update_ensemble(new_samples)

        # main optimisation loop
        for _ in range(self.n):
            inds = self._get_best_indices()
            self._current_best_ind = inds[0]
            cells_to_resample = self.samples[inds]

            new_samples = Parallel(n_jobs=self.nr)(
                delayed(self._random_walk_in_voronoi)(cell, k)
                for k, cell in zip(inds, cells_to_resample)
            )
            self._update_ensemble(np.concatenate(new_samples))

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
        old_samples = self.samples[: self.np]
        walk_length = self.nspnr
        if k == self._current_best_ind:
            # best model so walk a bit further
            walk_length += self.ns % self.nr

        # find cell boundaries along each dimension
        d2 = np.sum(
            self.Cm * (vk - old_samples) ** 2, axis=1
        )  # distance to all other cells

        d2_previous_axis = 0  # distance to previous axis

        for _step in range(walk_length):
            xA = vk.copy()  # start of walk at cell centre
            for i in range(self.nd):  # step along each axis
                d2_current_axis = self.Cm[i] * (xA[i] - old_samples[:, i]) ** 2
                d2 += d2_previous_axis - d2_current_axis
                dk2 = d2[k]  # disctance of cell centre to axis

                # eqn (19) Sambridge 1999
                vji = old_samples[:, i]
                vki = vk[i]
                a = dk2 - d2
                b = vki - vji
                xji = 0.5 * (
                    vki + vji + np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                )

                # eqns (20, 21) Sambridge 1999
                li = np.nanmax(np.hstack((self.lower[i], xji[xji < xA[i]])))
                ui = np.nanmin(np.hstack((self.upper[i], xji[xji > xA[i]])))
                xA[i] = np.random.uniform(li, ui)

                d2_previous_axis = d2_current_axis

            new_samples[_step] = xA

        return new_samples

    def _get_best_indices(self) -> ArrayLike:
        # there may be a faster way to do this using np.argpartition
        return np.argsort(self.objectives)[: self.nr]

    def _update_ensemble(self, new_samples: ArrayLike):
        n = new_samples.shape[0]
        self.samples[self.np : self.np + n] = new_samples
        self.objectives[self.np : self.np + n] = self._apply_objective(new_samples)
        self.np += n

    def _check_objective(self):
        """
        Choose the fastest way to apply the objective function to a set of samples.
        """
        # check if the objective function is vectorised
        samples = np.random.randn(self.ns, self.nd)
        if ~np.array_equal(
            self._apply_objective_vectorised(samples),
            self._apply_objective_along_axis(samples),
        ):  # not vectorised
            self._apply_objective = self._apply_objective_along_axis
        else:  # vectorised
            self._apply_objective = self._apply_objective_vectorised

        # check if parallel is faster
        times_ = []
        for _ in range(100):
            t0 = time()
            self._apply_objective(samples)
            times_.append(time() - t0)
            self.np = 0

        times_2 = []
        for _ in range(100):
            t0 = time()
            self._apply_objective_parallel(samples)
            times_2.append(time() - t0)
            self.np = 0

        if np.mean(times_) > np.mean(times_2):
            self._apply_objective = self._apply_objective_parallel

        self.np = 0  # this is here just to be sure

    def _apply_objective_along_axis(self, new_samples: ArrayLike):
        """
        For if the objective function is not vectorised
            e.g. np.sum(x) where x is a 2D array of samples
        """
        return np.apply_along_axis(self.objective, 1, new_samples)

    def _apply_objective_vectorised(self, new_samples: ArrayLike):
        """
        For if the objective function is vectorised
            e.g. np.sum(x, axis=1) where x is a 2D array of samples
        """
        return self.objective(new_samples)

    def _apply_objective_parallel(self, new_samples: ArrayLike):
        """
        For speed!  Simple testing showed that if t(objective) > 0.006s then parallel is faster
        """
        return Parallel(n_jobs=new_samples.shape[0])(
            delayed(self.objective)(x) for x in new_samples
        )
