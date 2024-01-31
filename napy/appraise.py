import numpy as np
from numpy.typing import ArrayLike
import warnings


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

    def appraise(self):
        """
        Perform the appraisal stage of the Neighbourhood Algorithm.

        Calculates a few basic MC integrals
        """

        def g_mean(x):
            return x

        mean = np.zeros(self.nd)
        for x in self.random_walk_through_parameter_space():
            mean += g_mean(x)

        return {
            "mean": mean / self.Nr,
        }

    def random_walk_through_parameter_space(self):
        """
        Perform the random walk through parameter space.
        Yields a new sample at each iteration to be used for calculating summary statistics.
        """
        k = (
            self.Ne - 1
        )  # start at the last cell for now. ultimately this will be random
        vk = self.initial_ensemble[k]  # This will change at some point
        for _ in range(self.Nr):
            for i in range(self.nd):
                intersections, cells = self.axis_intersections(i, k)
                xpi = self._random_step(i, intersections, cells)
                vk[i] = xpi
            yield vk

    def _random_step(self, axis, intersections, cells):
        """
        intersections are the points where the axis intersects the boundaries of the cells
        """
        accepted = False
        while not accepted:
            xpi = np.random.uniform(self.lower[axis], self.upper[axis])  # proposed step
            k = self._identify_cell(xpi, intersections, cells)  # cell containing xpi

            r = np.random.uniform(0, 1)
            Pxpi = self.objectives[k]
            Pmax = np.max(self.objectives[cells])
            if np.log(r) < np.log(Pxpi) - np.log(Pmax):  # eqn (24) Sambridge 1999(II)
                accepted = True
                return xpi

    def axis_intersections(self, axis: int, k: int) -> tuple[ArrayLike, ArrayLike]:
        """
        Calculate the intersections of an axis passing through point vk in the kth cell
        with the boundaries of all cells

        Returns the intersection points and the cells the axis passes through
        """

        down_intersections, down_cells = self._travel_along_axis(axis, k, down=True)
        # reverse the order of the down intersections and cells
        # so that the order of the intersections is from lowest to highest
        down_intersections = down_intersections[::-1]
        down_cells = down_cells[::-1]

        up_intersections, up_cells = self._travel_along_axis(axis, k, up=True)

        return np.array(down_intersections + up_intersections), np.array(
            down_cells + [k] + up_cells
        )

    def _travel_along_axis(
        self, axis: int, k: int, down: bool = False, up: bool = False
    ) -> list:
        intersections = []
        cells_traversed = []
        next_cell = k
        bound_reached = False
        while not bound_reached:
            _next_cell, intersection, bound_reached = self._get_axis_intersection(
                axis, next_cell, down=down, up=up
            )
            if intersection is not None and _next_cell is not None:
                intersections.append(intersection)
                cells_traversed.append(_next_cell)
                next_cell = _next_cell
            else:
                break
        return intersections, cells_traversed

    def _get_axis_intersection(
        self, axis: int, k: int, down: bool = False, up: bool = False
    ):
        """
        Returns the index of the next cell along the axis that shares a boundary with the kth cell,
        the intersection point, and whether the bound is reached.
        """
        assert not (down and up)

        # eqn (19) Sambridge 1999
        vk = self.initial_ensemble[k]
        vki = vk[axis]
        vji = self.initial_ensemble[:, axis]
        d2i = (
            np.sum(self.Cm * (vk - self.initial_ensemble) ** 2, axis=1)
            - self.Cm[axis] * (vki - vji) ** 2
        )  # perpendicular distance to current axis
        a = d2i[k] - d2i
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
            return k_new, xji[k_new], False
        else:
            return None, None, True

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
