import numpy as np
from numpy.typing import ArrayLike
import pytest
import matplotlib.pyplot as plt

from napy import NAAppariser


def objective(x: ArrayLike) -> float:
    return np.exp(-np.sum(x))


@pytest.fixture
def NAA():
    lower = [-1.0, 0.0]
    upper = (1.0, 10)
    initial_ensemble = np.meshgrid(  # regular grid
        np.sort(np.random.uniform(lower[0], upper[0], 5)),
        np.sort(np.random.uniform(lower[1], upper[1], 5)),
    )  # sorting needed to calculate true intersections in test_axis_intersections
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
    n_walkers = 2
    return NAAppariser(initial_ensemble, objectives, bounds, n_resample, n_walkers)


def test_axis_intersections(NAA):
    # the axis intersections with cell boundaries should be
    # halfway between the cell centres
    _plot = False

    # Choose a random cell from the ensemble
    k = np.random.randint(0, NAA.Ne)
    vk = NAA.initial_ensemble[k]

    # Determine 2D indices of cell
    i, j = np.unravel_index(k, (5, 5))

    axis = 0  # hoizontal axis
    h_intersections, _ = NAA.axis_intersections(axis, vk)
    vk_row = NAA.initial_ensemble.reshape(5, 5, 2)[i, :, axis]
    true_h_intersections = vk_row[:-1] + (vk_row[1:] - vk_row[:-1]) / 2
    assert np.sort(h_intersections) == pytest.approx(true_h_intersections)

    axis = 1  # vertical axis
    v_intersections, _ = NAA.axis_intersections(axis, vk)
    vk_col = NAA.initial_ensemble.reshape(5, 5, 2)[:, j, axis]
    true_v_intersections = vk_col[:-1] + (vk_col[1:] - vk_col[:-1]) / 2
    assert np.sort(v_intersections) == pytest.approx(true_v_intersections)

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


def test_axis_intersections_cells(NAA):
    # the axis intersections with cell boundaries should be
    # halfway between the cell centres
    _plot = False

    # Choose a random cell from the ensemble
    k = np.random.randint(0, NAA.Ne)
    vk = NAA.initial_ensemble[k]

    # Determine 2D indices of cell
    i, j = np.unravel_index(k, (5, 5))

    axis = 0  # hoizontal axis
    _, h_cells = NAA.axis_intersections(axis, vk)
    true_h_cells = np.arange(i * 5, (i + 1) * 5)
    assert np.sort(h_cells) == pytest.approx(true_h_cells)

    axis = 1  # vertical axis
    _, v_cells = NAA.axis_intersections(axis, vk)
    true_v_cells = np.arange(j, 25, 5)
    assert np.sort(v_cells) == pytest.approx(true_v_cells)

    if _plot:
        plt.scatter(*NAA.initial_ensemble.T)
        plt.scatter(*NAA.initial_ensemble[true_h_cells].T, c="g", marker="x")
        plt.scatter(*NAA.initial_ensemble[h_cells].T, c="b")
        plt.scatter(*NAA.initial_ensemble[true_v_cells].T, c="g", marker="x")
        plt.scatter(*NAA.initial_ensemble[v_cells].T, c="b")
        plt.scatter(*vk, c="r")

    plt.show()


@pytest.mark.parametrize("true_cell", [0, 1, 2, 3, 4])
def test__identify_cell(NAA, true_cell):
    intersections = np.sort(np.random.uniform(size=4))
    cells = np.arange(5)

    # cell bounds
    cell_low = intersections[true_cell - 1] if true_cell != 0 else 0
    cell_high = intersections[true_cell] if true_cell != 4 else 1

    # random point in cell
    xp = np.random.uniform(cell_low, cell_high)

    # identify cell
    cell = NAA._identify_cell(xp, intersections, cells)

    assert cell == true_cell


def test_appraise(NAA):
    results = NAA.appraise()
    assert results["mean"].shape == (2,)
    assert results["covariance"].shape == (2, 2)


def test_MC_integrals(NAA):
    results = NAA.appraise(save=True)
    mean = np.mean(results["samples"], axis=0)
    covariance = np.cov(results["samples"], rowvar=False, bias=True)

    assert results["mean"] == pytest.approx(mean)
    assert results["covariance"] == pytest.approx(covariance)
    assert results["samples"].shape == (NAA.nr * NAA.j, 2)
