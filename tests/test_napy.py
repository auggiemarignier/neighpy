import numpy as np
from numpy.typing import ArrayLike
import pytest

from napy import NASearcher


def objective(x: ArrayLike) -> float:
    return np.sum(x)


@pytest.fixture
def nas_fixture():
    ns = 10
    nr = 5
    ni = 5
    n = 20
    bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    return NASearcher(objective, ns, nr, ni, n, bounds)


def test__initial_random_search(nas_fixture):
    nas_fixture._initial_random_search()

    assert not np.array_equal(
        nas_fixture.samples[: nas_fixture.ni], np.zeros((nas_fixture.ni, 3))
    )
    assert not np.array_equal(
        nas_fixture.objectives[: nas_fixture.ni], np.zeros(nas_fixture.ni)
    )

    assert np.array_equal(
        nas_fixture.samples[nas_fixture.ni :], np.zeros((nas_fixture.nt - nas_fixture.ni, 3))
    )
    assert np.array_equal(
        nas_fixture.objectives[nas_fixture.ni :], np.zeros(nas_fixture.nt - nas_fixture.ni)
    )


def test__get_best_indices(nas_fixture):
    nas_fixture._initial_random_search()
    inds = nas_fixture._get_best_indices()

    assert len(inds) == nas_fixture.nr
    assert np.all(inds < nas_fixture.ni)  # all indices should be from initial random search
    assert not np.array_equal(
        nas_fixture.samples[inds], np.zeros((nas_fixture.nr, 3))
    )  # all samples should be non-zero
    assert not np.array_equal(
        nas_fixture.objectives[inds], np.zeros(nas_fixture.nr)
    )  # all objectives should be non-zero