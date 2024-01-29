import numpy as np
from numpy.typing import ArrayLike
from typing import Union


class _VoronoiDiagram:
    """
    A class to manage the 'voronoi diagram' aspects of both
    parts of the Neighbourhood Algorithm.
    The full Voronoi diagram is never computed.
    The main function of this class is to calculate distances
    between points and cell intersections with axes.
    """

    def __init__(
        self,
        samples: Union[ArrayLike, None],
        weights: Union[ArrayLike, None] = None,
        nd: Union[int, None] = None,
    ):
        if samples is not None:
            self.nd = samples.shape[1]
        else:
            assert nd is not None
            self.nd = nd

        self.cell_centres = samples
        self.weights = weights if weights is not None else np.ones(self.nd)

    def _cell_axis_intersections(self, vk: ArrayLike, k: int):
        """
        Calculate the intersection of the voronoi cell vk with the axis i.

        Yields the intersection points vk.
        """
        # FOLLOWING https://github.com/underworldcode/pyNA/blob/30d1cb7955d6b1389eae885127389ed993fa6940/pyNA/sampler.py#L85

        # vk is the current voronoi cell
        # k is the index of the current voronoi cell

        d2 = _VoronoiDiagram._distance(vk, self.cell_centres, self.weights)
        d2 = np.sum(d2, axis=1)
        d2_previous_axis = 0  # distance to previous axis

        for j in range(self.nd):
            d2_current_axis = _VoronoiDiagram._distance(
                vk[j], self.cell_centres[:, j], self.weights[j]
            )
            d2 += d2_previous_axis - d2_current_axis
            d2_previous_axis = d2_current_axis
            dk2 = d2[k]  # disctance of cell centre to axis

            # eqn (19) Sambridge 1999
            vji = self.cell_centres[:, j]
            vki = vk[j]
            a = dk2 - d2
            b = vki - vji
            xji = 0.5 * (
                vki + vji + np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            )
            d2_previous_axis = d2_current_axis

            yield xji

    def _update_diagram(self, samples):
        self.cell_centres = samples

    @staticmethod
    def _distance(x1, x2, w):
        return w * (x1 - x2) ** 2
