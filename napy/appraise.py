import numpy as np
from numpy.typing import ArrayLike
import warnings
from joblib import Parallel, delayed


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
        self.Cm = 1 / (self.upper - self.lower) ** 2

        self.Ne = len(initial_ensemble)
        self.Nr = n_resample
        self.j = n_walkers if n_walkers >= 1 else 1
        self.nr = self.Nr // self.j

    def appraise(self, save: bool = False):
        """
        Perform the appraisal stage of the Neighbourhood Algorithm.

        Calculates a few basic MC integrals
        """

        if save:
            samples = np.zeros((self.j, self.nr, self.nd))
        mean = np.zeros(self.nd)
        cov_crossterm = np.zeros((self.nd, self.nd))

        with Parallel(n_jobs=self.j) as parallel:
            # select start points for the random walks
            # ensure that at least one walker starts at the best cell
            start_points = np.random.choice(self.Ne, self.j, replace=False)
            start_points[0] = np.argmin(self.objectives)

            # run the walkers in parallel
            results = parallel(
                delayed(self._appraise)(save, start) for start in start_points
            )

        # combine the results
        for j, j_results in enumerate(results):
            j_mean, j_cov_crossterm = j_results[:-1] if save else j_results

            mean += j_mean / self.nr
            cov_crossterm += j_cov_crossterm / self.nr
            if save:
                j_samples = j_results[-1]
                samples[j] = j_samples

        mean /= self.j
        covariance = cov_crossterm / self.j - np.outer(mean, mean)

        results = {
            "mean": mean,
            "covariance": covariance,
        }
        if save:
            results["samples"] = samples.reshape(-1, self.nd)
        return results

    def _appraise(self, save: bool = False, start_k: int = 0):
        if save:
            j_samples = np.zeros((self.nr, self.nd))
            _i = 0
        j_mean = np.zeros(self.nd)
        j_cov_crossterm = np.zeros((self.nd, self.nd))

        for x in self.random_walk_through_parameter_space(start_k):
            j_mean += NAAppariser.g_mean(x)
            j_cov_crossterm += NAAppariser.g_covariance_cross(x)
            if save:
                j_samples[_i] = x.copy()
                _i += 1

        results = (j_mean, j_cov_crossterm)
        if save:
            results += (j_samples,)
        return results

    def random_walk_through_parameter_space(self, start_k: int = 0):
        """
        Perform the random walk through parameter space.
        Yields a new sample at each iteration to be used for calculating summary statistics.
        """
        xA = self.initial_ensemble[start_k].copy()  # This will change at some point
        for _ in range(self.nr):
            for i in range(self.nd):
                intersections, cells = self.axis_intersections(i, xA)
                xpi = self.random_step(i, intersections, cells)
                xA[i] = xpi
            yield xA

    def axis_intersections(
        self, axis: int, xA: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Calculate the intersections of an axis passing through point vk in the kth cell
        with the boundaries of all cells

        Returns the intersection points and the cells the axis passes through
        """

        # The perpendicular distance to the axis from the current point in the walk
        # is constant as we traverse along the axis, so calculate before recursion
        d = (
            xA - self.initial_ensemble
        ) ** 2  # component-wise squared distance to all other cells
        d2 = np.sum(d * self.Cm, axis=1)  # total scaled distance to all other cells
        k = np.argmin(d2)  # index of the nearest cell
        dk2 = np.sum(
            np.delete(d, axis, 1) * np.delete(self.Cm, axis), axis=1
        )  # perpendicular distance to axis

        # Travel down the axis
        down_intersections, down_cells = self._get_axis_intersections(
            axis, k, dk2, down=True
        )
        # reverse the order of the down intersections and cells
        # so that the order of the intersections is from lowest to highest
        down_intersections = down_intersections[::-1]
        down_cells = down_cells[::-1]

        # Travel up the axis
        up_intersections, up_cells = self._get_axis_intersections(axis, k, dk2, up=True)

        return np.array(down_intersections + up_intersections), np.array(
            down_cells + [k] + up_cells
        )

    def random_step(self, axis, intersections, cells):
        """
        intersections are the points where the axis intersects the boundaries of the cells
        """
        while True:
            xpi = np.random.uniform(self.lower[axis], self.upper[axis])  # proposed step
            k = self._identify_cell(xpi, intersections, cells)  # cell containing xpi

            r = np.random.uniform(0, 1)
            Pxpi = self.objectives[k]
            Pmax = np.max(self.objectives[cells])
            if np.log(r) < np.log(Pxpi) - np.log(Pmax):  # eqn (24) Sambridge 1999(II)
                return xpi

    def _get_axis_intersections(
        self, axis: int, k: int, di2: ArrayLike, down: bool = False, up: bool = False
    ):
        """
        axis: int - the axis to travel along
        k: int - the index of the current cell
        di2: ArrayLike - the perpendicular distance to the axis from current point in walk
        down: bool - whether to travel down the axis
        up: bool - whether to travel up the axis

        Returns:
            intersections: ArrayLike - the intersection points
            cells: ArrayLike - the cells the axis passes through
        """
        assert not (down and up)

        intersections = []
        cells = []

        # eqn (19) Sambridge 1999
        vk = self.initial_ensemble[k]
        vki = vk[axis]
        vji = self.initial_ensemble[:, axis]
        a = di2[k] - di2
        b = self.Cm[axis] * (vki - vji)
        with warnings.catch_warnings(action="ignore"):
            xji = 0.5 * (vki + vji + a / b)

        if down:
            # isfinite check handles previous divide by zero
            mask = (vki <= vji) | ~np.isfinite(xji)
            closest = np.argmax
        else:
            mask = (vki >= vji) | ~np.isfinite(xji)
            closest = np.argmin

        xji = np.ma.array(xji, mask=mask)
        if xji.count() > 0:  # valid intersections found
            k_new = closest(xji)  # closest to vk
            intersections += [xji[k_new]]
            cells += [k_new]

            new_intersections, new_cells = self._get_axis_intersections(
                axis, k_new, di2, down, up
            )
            return intersections + new_intersections, cells + new_cells

        return intersections, cells

    def _identify_cell(
        self, xp: float, intersections: ArrayLike, cells: ArrayLike
    ) -> int:
        """
        Given a set of intersections and the cells they pass through,
        identify the cell that contains the point xp.
        """
        closest_intersection = np.argmin(np.abs(intersections - xp))
        cell_id = (
            closest_intersection
            if xp < intersections[closest_intersection]
            else closest_intersection + 1
        )
        return cells[cell_id]

    @staticmethod
    def g_mean(x):
        return x

    @staticmethod
    def g_covariance_cross(x):
        return np.outer(x, x)


class MCIntegrals:
    """
    Class for accumulating samples to calculate MC integrals in a single loop
    """

    def __init__(self, nd: int):
        self.mi = np.zeros(nd)
        self.mi2 = np.zeros(nd)
        self.mimj = np.zeros((nd, nd))
        self.mi2mj = np.zeros((nd, nd))
        self.mimj2 = np.zeros((nd, nd))
        self.mi2mj2 = np.zeros((nd, nd))
        self.N = 0

    def accumulate(self, x: ArrayLike):
        self.mi += x
        self.mi2 += x**2
        self.mimj += np.outer(x, x)
        self.mi2mj += np.outer(x**2, x)
        self.mimj2 += np.outer(x, x**2)
        self.mi2mj2 += np.outer(x**2, x**2)
        self.N += 1

    def mean(self):
        return self.mi / self.N

    def covariance(self):
        mean = self.mean()
        return self.mimj / self.N - np.outer(mean, mean)
