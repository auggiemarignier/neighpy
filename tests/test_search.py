import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.spatial import Voronoi
from shapely.geometry import LineString, Point
from shapely.ops import polygonize
import matplotlib.pyplot as plt

from neighpy import NASearcher


def objective(x: NDArray) -> float:
    return -np.sum(x)


@pytest.fixture
def NAS():
    ns = 10
    nr = 5
    ni = 10
    n = 20
    bounds = ((-1.0, 1.0), (0.0, 10.0))
    seed = 42
    return NASearcher(objective, ns, nr, ni, n, bounds, seed=seed)


def test__initial_random_search(NAS):
    new_samples = NAS._initial_random_search()
    assert new_samples.shape == (NAS.ni, len(NAS.bounds))

    NAS._update_ensemble(new_samples)
    assert not np.array_equal(
        NAS.samples[: NAS.ni], np.zeros((NAS.ni, len(NAS.bounds)))
    )
    assert not np.array_equal(NAS.objectives[: NAS.ni], np.zeros(NAS.ni))

    assert np.array_equal(
        NAS.samples[NAS.ni :], np.zeros((NAS.nt - NAS.ni, len(NAS.bounds)))
    )
    assert np.array_equal(NAS.objectives[NAS.ni :], np.full(NAS.nt - NAS.ni, np.inf))


def test__get_best_indices(NAS):
    new_samples = NAS._initial_random_search()
    NAS._update_ensemble(new_samples)

    inds = NAS._get_best_indices()

    assert len(inds) == NAS.nr
    assert np.all(inds < NAS.ni)  # all indices should be from initial random search
    assert not np.array_equal(
        NAS.samples[inds], np.zeros((NAS.nr, len(NAS.bounds)))
    )  # all samples should be non-zero
    assert not np.array_equal(
        NAS.objectives[inds], np.zeros(NAS.nr)
    )  # all objectives should be non-zero


def test__get_best_indices_with_neg_inf(NAS):
    new_samples = NAS._initial_random_search()
    NAS._update_ensemble(new_samples)

    # artificially set some objectives to -inf
    NAS.objectives[0] = -np.inf
    NAS.objectives[1] = -np.inf

    inds = NAS._get_best_indices()

    assert len(inds) == NAS.nr
    assert 0 not in inds  # index with -inf objective should not be selected
    assert 1 not in inds  # index with -inf objective should not be selected
    assert NAS.objectives[0] == np.inf  # -inf should be converted to inf
    assert NAS.objectives[1] == np.inf  # -inf should be converted to


@pytest.mark.flaky(reruns=5, only_rerun="assert")
def test_random_walk_in_voronoi(NAS):
    _plot = False

    old_samples = np.meshgrid(
        np.random.uniform(NAS.lower[0], NAS.upper[0], 5),
        np.random.uniform(NAS.lower[1], NAS.upper[1], 5),
    )
    points = np.array(
        [[x, y] for x, y in zip(old_samples[0].flatten(), old_samples[1].flatten())]
    )
    points += np.random.randn(*points.shape) * 0.05
    for axis in range(NAS.nd):
        points[:, axis] = np.clip(points[:, axis], NAS.lower[axis], NAS.upper[axis])
    NAS._update_ensemble(points)

    vor = Voronoi(points)
    lines = [
        LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line
    ]

    polys = list(polygonize(lines))

    for k, vk in enumerate(vor.points):
        # find if point has a polygon
        for poly in polys:
            if poly.contains(Point(vk)):
                # perform random walk
                new_samples = NAS._random_walk_in_voronoi(
                    vk, k, np.random.default_rng(42)
                )
                assert new_samples.shape == (NAS.nspnr, NAS.nd)
                if _plot:
                    plt.plot(new_samples[:, 0], new_samples[:, 1], "o-", color="blue")
                for sample in new_samples:
                    # continue
                    assert poly.contains(Point(sample))
                    break

    if _plot:
        for poly in polys:
            plt.plot(*poly.exterior.xy, color="black")
        plt.scatter(*points.T, color="red")
        plt.axvline(NAS.lower[0], color="black", ls="--")
        plt.axvline(NAS.upper[0], color="black", ls="--")
        plt.axhline(NAS.lower[1], color="black", ls="--")
        plt.axhline(NAS.upper[1], color="black", ls="--")

        eps = 0.05 * np.sqrt(1 / NAS.Cm)
        plt.xlim(NAS.lower[0] - eps[0], NAS.upper[0] + eps[0])
        plt.ylim(NAS.lower[1] - eps[1], NAS.upper[1] + eps[1])
        plt.show()


def test_run(NAS):
    NAS.run()
    assert np.all(NAS.objectives != np.inf)
    assert np.all(NAS.samples != 0)
    assert np.all(NAS.samples != np.inf)
    assert np.all(NAS.samples != -np.inf)
    assert np.all(NAS.samples != np.nan)
    assert np.all(NAS.samples != -np.nan)
    assert np.all(NAS.samples >= NAS.lower)
    assert np.all(NAS.samples <= NAS.upper)
    assert NAS.np == NAS.nt


def test_objective_args():
    def objective(x: NDArray, a: int, b: float) -> float:
        return (-np.sum(x) * a) / b

    a = 2
    b = 3.0
    args = (a, b)
    NAS = NASearcher(objective, 10, 5, 5, 20, ((-1.0, 1.0), (0.0, 10.0)), args=args)

    x = np.array([1, 2, 3])
    assert NAS.objective(x) == -4.0


def test_seed():
    NAS1 = NASearcher(objective, 10, 5, 5, 20, ((-1.0, 1.0), (0.0, 10.0)), seed=42)
    NAS2 = NASearcher(objective, 10, 5, 5, 20, ((-1.0, 1.0), (0.0, 10.0)), seed=42)

    NAS1.run()
    NAS2.run()

    assert np.all(NAS1.samples == NAS2.samples)
    assert np.all(NAS1.objectives == NAS2.objectives)
