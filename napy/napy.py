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
            ns (int): The number of samples in each cell.
            nr (int): The number of cells to resample.
            ni (int): The number of samples from initial random search.
            n (int): The number of iterations.
            bounds (tuple[tuple[float, float], ...]): A tuple of tuples representing the bounds of the search space.
                Each inner tuple represents the lower and upper bounds for a specific dimension.

        Returns:
            None
        """

        self.objective = objective

        self.ns = ns  # number of samples in each cell
        self.nr = nr  # number of cells to resample
        self.ni = ni  # number of samples from initial random search
        self.n = n  # number of iterations
        self.nt = ni + n * (nr * ns)  # total number of samples

        self.bounds = bounds  # bounds of the search space
        self.nd = len(bounds)  # number of dimensions

        self.samples = np.zeros((self.nt, self.nd))
        self.objectives = np.zeros(self.nt)

    def _initial_random_search(self) -> None:
        self.X = np.random.uniform(
            low=[b[0] for b in self.bounds],
            high=[b[1] for b in self.bounds],
            size=(self.ni, self.nd),
        )
        self.samples[: self.ni] = self.X
        self.objectives[: self.ni] = np.apply_along_axis(self.objective, 1, self.X)

    def _sample_voronoi_cell(self, i: int) -> None:
        """
        Args:
            i (int): The iteration number.
        """
        pass

    def _sample_index(self, i: int) -> int:
        """
        Args:
            i (int): The iteration number.
        Returns:
            int: The index of the sample given the iteration number and tuning parameters ns and nr.
        """
        pass

    def run(self) -> None:
        self._initial_random_search()
        for i in range(1, self.n):
            self._sample_voronoi_cell()
