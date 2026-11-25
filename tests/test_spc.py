import numpy as np
import pytest

from secom.spc import fit_limits, flag_outside, flag_runs


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


def test_flag_runs_known_answer():
    values = np.array([11, 12, 13, 9, 11, 9, 8, 7, 6])
    limits = {"centre": 10}

    flagged = flag_runs(values, limits, run_length=3)

    assert list(flagged) == [False, False, True, False, False, False, False, True, True]


def test_point_on_centre_resets_run():
    values = np.array([11, 12, 10, 13, 14, 15])
    limits = {"centre": 10}

    flagged = flag_runs(values, limits, run_length=3)

    assert list(flagged) == [False, False, False, False, False, True]


def test_default_run_length_is_nine():
    values = np.array([11] * 10)          # ten 11s in a row
    limits = {"centre": 10}

    flagged = flag_runs(values, limits)   # no run_length given

    assert list(flagged) == [False] * 8 + [True] * 2