"""Sanity tests for src/spc.py using data whose true answer is known.

Run from the project root:  python -m pytest tests -q   (or: python tests/test_spc.py)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.spc import cp_cpk, imr_limits, mr_sigma, rule1_outside_limits, rule2_run_one_side

RNG = np.random.default_rng(0)


def test_mr_sigma_recovers_true_sigma():
    x = RNG.normal(0, 1.0, 20000)
    assert abs(mr_sigma(x) - 1.0) < 0.03


def test_cp_cpk_centered_and_shifted():
    centered = RNG.normal(0, 1.0, 20000)
    r = cp_cpk(centered, lsl=-3, usl=3)
    assert abs(r["Cp"] - 1.0) < 0.05 and abs(r["Cpk"] - 1.0) < 0.05
    shifted = RNG.normal(1.0, 1.0, 20000)  # mean moved 1 sigma toward USL
    r = cp_cpk(shifted, lsl=-3, usl=3)
    assert abs(r["Cp"] - 1.0) < 0.05          # spread unchanged -> Cp unchanged
    assert abs(r["Cpk"] - (3 - 1.0) / 3) < 0.05  # centering penalty shows up in Cpk only


def test_ppk_below_cpk_when_process_drifts():
    n = 2000
    x = RNG.normal(0, 1.0, n) + np.linspace(0, 6, n)  # slow drift, tight short-term noise
    r = cp_cpk(x, lsl=-5, usl=12)
    assert r["Ppk"] < r["Cpk"] * 0.7


def test_baseline_limits_expose_unstable_early_regime():
    early = RNG.normal(0, 3.0, 300)    # unstable start-up period
    late = RNG.normal(0, 1.0, 1200)    # stable running period
    x = np.concatenate([early, late])
    whole = imr_limits(x)
    based = imr_limits(x, baseline=slice(300, None))
    flags_whole = rule1_outside_limits(x, whole).sum()
    flags_based = rule1_outside_limits(x, based)
    # Expected ~48 flags (whole-series limits) vs ~98 (baseline limits): ~2x, min 1.55x over
    # 200 seeds. Whole-series limits are contaminated by the unstable regime, so they miss ~half.
    assert flags_based.sum() > 1.3 * max(flags_whole, 1)
    assert flags_based[:300].sum() > flags_based[300:].sum()  # and the flags land in the early regime


def test_rule2_flags_a_run_of_nine():
    x = np.array([1, -1, 1, -1] + [1] * 9 + [-1, 1, -1], dtype=float)
    lim = {"center": 0.0}
    flags = rule2_run_one_side(x, lim, n=9)
    assert flags.sum() == 9 and flags[4:13].all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
