import numpy as np
import pytest

from napy._mcintegrals import MCIntegrals


@pytest.fixture
def MCIntegrals_obj():
    nd = 2
    return MCIntegrals(nd, save_samples=True)


@pytest.fixture
def samples():
    return np.random.rand(10, 2)


@pytest.fixture
def accumulator(MCIntegrals_obj, samples):
    for x in samples:
        MCIntegrals_obj.accumulate(x)
    return MCIntegrals_obj


def test_accumulate(MCIntegrals_obj):
    x = np.array([1.0, 2.0])
    MCIntegrals_obj.accumulate(x)
    assert MCIntegrals_obj.N == 1
    assert np.allclose(MCIntegrals_obj.mi, x)
    assert np.allclose(MCIntegrals_obj.mi2, x**2)
    assert np.allclose(MCIntegrals_obj.mimj, np.outer(x, x))
    assert np.allclose(MCIntegrals_obj.mi2mj, np.outer(x**2, x))
    assert np.allclose(MCIntegrals_obj.mimj2, np.outer(x, x**2))
    assert np.allclose(MCIntegrals_obj.mi2mj2, np.outer(x**2, x**2))

    x = np.array([3.0, 4.0])
    MCIntegrals_obj.accumulate(x)
    assert MCIntegrals_obj.N == 2
    assert np.allclose(MCIntegrals_obj.mi, [4.0, 6.0])
    assert np.allclose(MCIntegrals_obj.mi2, [10.0, 20.0])
    assert np.allclose(MCIntegrals_obj.mimj, [[10.0, 14.0], [14.0, 20.0]])
    assert np.allclose(MCIntegrals_obj.mi2mj, [[28.0, 38.0], [52.0, 72.0]])
    assert np.allclose(MCIntegrals_obj.mimj2, [[28.0, 52.0], [38.0, 72.0]])
    assert np.allclose(MCIntegrals_obj.mi2mj2, [[82.0, 148.0], [148.0, 272.0]])


def test_accumulate_objects(MCIntegrals_obj):
    x = np.array([1.0, 2.0])
    MCIntegrals_obj.accumulate(x)
    x = np.array([3.0, 4.0])
    MCIntegrals_obj2 = MCIntegrals(2)
    MCIntegrals_obj2.accumulate(x)
    MCIntegrals_obj.accumulate(MCIntegrals_obj2)
    assert MCIntegrals_obj.N == 2
    assert np.allclose(MCIntegrals_obj.mi, [4.0, 6.0])
    assert np.allclose(MCIntegrals_obj.mi2, [10.0, 20.0])
    assert np.allclose(MCIntegrals_obj.mimj, [[10.0, 14.0], [14.0, 20.0]])
    assert np.allclose(MCIntegrals_obj.mi2mj, [[28.0, 38.0], [52.0, 72.0]])
    assert np.allclose(MCIntegrals_obj.mimj2, [[28.0, 52.0], [38.0, 72.0]])
    assert np.allclose(MCIntegrals_obj.mi2mj2, [[82.0, 148.0], [148.0, 272.0]])


def test_mean(accumulator, samples):
    assert np.allclose(accumulator.mean(), samples.mean(axis=0))


def test_covariance(accumulator, samples):
    assert np.allclose(
        accumulator.covariance(), np.cov(samples, rowvar=False, bias=True)
    )


def test_sample_mean_error(accumulator, samples):
    # sample mean error is sigma/sqrt(N)
    cov = np.cov(samples, rowvar=False, bias=True)
    var = np.diag(cov)
    assert np.allclose(accumulator.sample_mean_error(), np.sqrt(var / accumulator.N))


def test_sample_covariance_error(accumulator, samples):
    # For the diagonal terms of the covariance matrix, the sample covariance error is
    # related to the 4th central moment, similar to how the sample mean error is related
    # to the 2nd central moment.
    #
    #   eps_ii = (1/sqrt(N)) * sqrt(4th central moment - variance^2)
    #
    from scipy.stats import moment

    var = np.diag(np.cov(samples, rowvar=False, bias=True))
    var2 = var**2
    fourth_moment = moment(samples, moment=4, axis=0)
    eps_ii = np.sqrt(fourth_moment - var2) / np.sqrt(accumulator.N)
    assert np.allclose(np.diag(accumulator.sample_covariance_error()), eps_ii)
