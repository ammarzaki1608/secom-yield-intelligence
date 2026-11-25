import numpy as np
import pytest

from secom.spc import fit_limits, flag_outside


def test_fit_limits_known_answer():
    limits = fit_limits(np.array([9, 11, 9, 11]))

    assert limits["centre"] == 10
    assert limits["lower"] == pytest.approx(4.681, abs=0.001)
    assert limits["upper"] == pytest.approx(15.319, abs=0.001)


def test_shift_is_flagged():
    x = np.array([30, 31, 29, 30, 32, 28, 30, 31, 29, 30,
                50, 51, 49, 50, 52, 48, 50, 51, 49, 50])
    limits = fit_limits(x[:10])         # fit: learn from the first 10 only
    flagged = flag_outside(x, limits)      # apply: check all 20

    assert flagged.sum() == 10
    assert flagged[:10].sum() == 0 