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

    def axis_intersections(self, axis: int, k: int) -> tuple[ArrayLike, ArrayLike]:
        """
        Calculate the intersections of an axis passing through point vk in the kth cell
        with the boundaries of all cells

        Returns the intersection points and the cells the axis passes through
        """

        down_intersections, down_cells = self._travel_along_axis(axis, k, down=True)
        up_intersections, up_cells = self._travel_along_axis(axis, k, up=True)

        return np.array(down_intersections + up_intersections), np.array(
            down_cells + up_cells + [k]
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
