import numpy as np
import pytest

from napy._mcintegrals import MCIntegrals


@pytest.fixture
def MCIntegrals_obj():
    nd = 2
    return MCIntegrals(nd)


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


def test_mean(MCIntegrals_obj):
    samples = np.random.rand(10, 2)
    for x in samples:
        MCIntegrals_obj.accumulate(x)

    assert np.allclose(MCIntegrals_obj.mean(), samples.mean(axis=0))


def test_covariance(MCIntegrals_obj):
    samples = np.random.rand(10, 2)
    for x in samples:
        MCIntegrals_obj.accumulate(x)

    assert np.allclose(
        MCIntegrals_obj.covariance(), np.cov(samples, rowvar=False, bias=True)
    )
