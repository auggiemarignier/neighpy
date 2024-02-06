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
    assert MCIntegrals_obj.mi == pytest.approx(x)
    assert MCIntegrals_obj.mi2 == pytest.approx(x**2)
    assert MCIntegrals_obj.mimj == pytest.approx(np.outer(x, x))
    assert MCIntegrals_obj.mi2mj == pytest.approx(np.outer(x**2, x))
    assert MCIntegrals_obj.mimj2 == pytest.approx(np.outer(x, x**2))
    assert MCIntegrals_obj.mi2mj2 == pytest.approx(np.outer(x**2, x**2))

    x = np.array([3.0, 4.0])
    MCIntegrals_obj.accumulate(x)
    assert MCIntegrals_obj.N == 2
    assert MCIntegrals_obj.mi == pytest.approx([4.0, 6.0])
    assert MCIntegrals_obj.mi2 == pytest.approx([10.0, 20.0])
    assert MCIntegrals_obj.mimj == pytest.approx([[3.0, 4.0], [6.0, 8.0]])
    assert MCIntegrals_obj.mi2mj == pytest.approx([[9.0, 12.0], [12.0, 16.0]])
    assert MCIntegrals_obj.mimj2 == pytest.approx([[3.0, 8.0], [12.0, 32.0]])
    assert MCIntegrals_obj.mi2mj2 == pytest.approx([[9.0, 24.0], [24.0, 64.0]])


def test_mean(MCIntegrals_obj):
    x = np.array([1.0, 2.0])
    MCIntegrals_obj.accumulate(x)
    x = np.array([3.0, 4.0])
    MCIntegrals_obj.accumulate(x)
    assert MCIntegrals_obj.mean() == pytest.approx([2.0, 3.0])


def test_covariance(MCIntegrals_obj):
    x = np.array([1.0, 2.0])
    MCIntegrals_obj.accumulate(x)
    x = np.array([3.0, 4.0])
    MCIntegrals_obj.accumulate(x)
    mean = MCIntegrals_obj.mean()
    assert MCIntegrals_obj.covariance() == pytest.approx(
        [[1.0, 1.0], [1.0, 1.0]]
    ) - np.outer(mean, mean)
