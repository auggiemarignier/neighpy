import numpy as np
from numpy.typing import ArrayLike


class _VoronoiDiagram:
    """
    A class to manage the 'voronoi diagram' aspects of both
    parts of the Neighbourhood Algorithm.
    The full Voronoi diagram is never computed.
    The main function of this class is to calculate distances
    between points and cell intersections with axes.
    """

    def __init__(self, samples: ArrayLike, bounds: ArrayLike):
        self.cell_centres = samples
        self.bounds = bounds
        self.nd = len(bounds)
        self.lower = np.array([b[0] for b in bounds])
        self.upper = np.array([b[1] for b in bounds])


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
