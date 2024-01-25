import numpy as np
from numpy.typing import ArrayLike
import pytest
from scipy.spatial import Voronoi
from shapely.geometry import LineString, Point
from shapely.ops import polygonize
import matplotlib.pyplot as plt

from napy import NASearcher


def objective(x: ArrayLike) -> float:
    return -np.sum(x)


@pytest.fixture
def NAS():
    ns = 10
    nr = 5
    ni = 5
    n = 20
    bounds = ((-1.0, 1.0), (0.0, 10.0))
    return NASearcher(objective, ns, nr, ni, n, bounds)


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


def test_random_walk_in_voronoi(NAS):
    old_samples = np.meshgrid(
        np.random.uniform(NAS.lower[0], NAS.upper[0], 5),
        np.random.uniform(NAS.lower[1], NAS.upper[1], 5),
    )
    points = np.array(
        [[x, y] for x, y in zip(old_samples[0].flatten(), old_samples[1].flatten())]
    )
    points += np.random.randn(*points.shape) * 0.1
    NAS._update_ensemble(points)

    vor = Voronoi(points)

    # ripped from scipy.spatial.voronoi_plot_2d to get the voronoi polygons
    center = vor.points.mean(axis=0)
    ptp_bound = np.ptp(vor.points, axis=0)
    finite_segments = []
    infinite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if vor.furthest_site:
                direction = -direction
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            infinite_segments.append(np.stack([vor.vertices[i], far_point]))
    # end of rip

    lines = [LineString(s) for s in finite_segments + infinite_segments]
    polys = list(polygonize(lines))
    for poly in polys:
        plt.plot(*poly.exterior.xy, color="black")
    plt.scatter(*points.T, color="red")

    for k, vk in enumerate(vor.points):
        # find if point has a polygon
        for poly in polys:
            if poly.contains(Point(vk)):
                # perform random walk
                new_samples = NAS._random_walk_in_voronoi(vk, k)
                assert new_samples.shape == (NAS.nspnr, NAS.nd)
                plt.plot(new_samples[:, 0], new_samples[:, 1], "o-", color="blue")
                for sample in new_samples:
                    # continue
                    assert poly.contains(Point(sample))
                    break
    # plt.axvline(NAS.lower[0], color="black", ls="--")
    # plt.axvline(NAS.upper[0], color="black", ls="--")
    # plt.axhline(NAS.lower[1], color="black", ls="--")
    # plt.axhline(NAS.upper[1], color="black", ls="--")

    # eps = 0.05 * np.sqrt(1 / NAS.Cm)
    # plt.xlim(NAS.lower[0] - eps[0], NAS.upper[0] + eps[0])
    # plt.ylim(NAS.lower[1] - eps[1], NAS.upper[1] + eps[1])
    plt.show()
