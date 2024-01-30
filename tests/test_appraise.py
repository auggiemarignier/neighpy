import numpy as np
from numpy.typing import ArrayLike
import pytest
import matplotlib.pyplot as plt

from napy import NAAppariser


def objective(x: ArrayLike) -> float:
    return -np.sum(x)


@pytest.fixture
def NAA():
    lower = -1.0
    upper = 1.0
    initial_ensemble = np.meshgrid(  # regular grid
        np.linspace(lower, upper, 5),
        np.linspace(lower, upper, 5),
    )
    initial_ensemble = np.array(
        [
            [x, y]
            for x, y in zip(
                initial_ensemble[0].flatten(), initial_ensemble[1].flatten()
            )
        ]
    )
    objectives = np.apply_along_axis(objective, 1, initial_ensemble)
    bounds = ((lower, upper), (lower, upper))
    n_resample = 10
    n_walkers = 1
    return NAAppariser(initial_ensemble, objectives, bounds, n_resample, n_walkers)


def test_axis_intersections(NAA):
    # starting at point (0,0) on an evenly spaced grid of cells
    # the axis intersections with cell boundaries should be
    # halfway between the cell centres and prior bounds
    _plot = False

    vk = np.array([0.0, 0.0])
    k = np.argmin(np.sum((NAA.initial_ensemble - vk) ** 2, axis=1))

    if _plot:
        plt.scatter(*NAA.initial_ensemble.T)
        plt.scatter(*vk, c="r")

    for i in range(NAA.nd):
        intersections = NAA.axis_intersections(i, k)
        vk_row = NAA.initial_ensemble.reshape(5, 5, 2)[2, :, i]
        true_intersections = vk_row[:-1] + (vk_row[1:] - vk_row[:-1]) / 2

        if _plot:
            for l in true_intersections:
                plt.axvline(l, c="k", ls="--")
            for l in intersections:
                plt.axvline(l, c="b", ls="-")

        assert intersections == pytest.approx(true_intersections)
        break  # only testing one axis for now

    plt.show()
