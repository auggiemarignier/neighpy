import numpy as np
from numpy.typing import ArrayLike
import pytest
import matplotlib.pyplot as plt

from napy import NAAppariser


def objective(x: ArrayLike) -> float:
    return -np.sum(x)


@pytest.fixture
def NAA():
    lower = [-1.0, 0.0]
    upper = (1.0, 10)
    initial_ensemble = np.meshgrid(  # regular grid
        np.linspace(lower[0], upper[0], 5),
        np.linspace(lower[1], upper[1], 5),
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
    bounds = ((lower[0], upper[0]), (lower[1], upper[1]))
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

    axis = 0  # hoizontal axis
    h_intersections = NAA.axis_intersections(axis, k)
    vk_row = NAA.initial_ensemble.reshape(5, 5, 2)[2, :, axis]
    true_h_intersections = vk_row[:-1] + (vk_row[1:] - vk_row[:-1]) / 2
    assert h_intersections == pytest.approx(true_h_intersections)

    axis = 1  # vertical axis
    v_intersections = NAA.axis_intersections(axis, k)
    vk_col = NAA.initial_ensemble.reshape(5, 5, 2)[:, 2, axis]
    true_v_intersections = vk_col[:-1] + (vk_col[1:] - vk_col[:-1]) / 2
    assert v_intersections == pytest.approx(true_v_intersections)

    if _plot:
        plt.scatter(*NAA.initial_ensemble.T)
        plt.scatter(*vk, c="r")
        for l in true_h_intersections:
            plt.axvline(l, c="g")
        for l in h_intersections:
            plt.axvline(l, c="b", ls=(0, (3, 10, 1, 10)))
        for l in true_v_intersections:
            plt.axhline(l, c="g")
        for l in v_intersections:
            plt.axhline(l, c="b", ls=(0, (3, 10, 1, 10)))

    plt.show()
