import numpy as np
from numpy.typing import ArrayLike
import pytest

from napy import NASearcher


def objective(x: ArrayLike) -> float:
    return -np.sum(x)


@pytest.fixture
def NAS():
    ns = 10
    nr = 5
    ni = 5
    n = 20
    bounds = ((0.0, 1.0), (0.0, 1.0))
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
