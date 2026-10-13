"""SPC and process-capability helpers for single-measurement-per-run data.

SECOM has ONE measurement per sensor per run (no rational subgroups), so the
right chart is an Individuals / Moving-Range (I-MR) chart, not X-bar/R.
Within-process sigma therefore comes from the average moving range, not from a
subgroup range.
"""
import numpy as np
import pandas as pd

D2_SPAN2 = 1.128  # bias-correction constant for moving range of span 2


def mr_sigma(x):
    """Short-term (within) sigma = mean(|x[i] - x[i-1]|) / 1.128.

    Missing values are dropped first, so a pair may straddle a gap. Always
    pass raw (non-imputed) data: median-filled runs create zero moving ranges
    and shrink this estimate.
    """
    v = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(v) < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(v))) / D2_SPAN2)


def imr_limits(x, baseline=None, k=3.0):
    """Individuals-chart limits.

    baseline: None (use the whole series) or a slice selecting the Phase I
    reference window. Limits built from a baseline are then applied to the
    FULL series (Phase II monitoring). Building limits from data that contains
    the instability you want to detect inflates sigma and hides it.
    """
    s = pd.Series(x).reset_index(drop=True)
    base = s if baseline is None else s.iloc[baseline]
    center = float(base.mean())
    sigma = mr_sigma(base)
    return {"center": center, "sigma": sigma,
            "ucl": center + k * sigma, "lcl": center - k * sigma}


def rule1_outside_limits(x, limits):
    """Nelson rule 1: a point beyond UCL or LCL."""
    s = pd.Series(x).reset_index(drop=True)
    return (s > limits["ucl"]) | (s < limits["lcl"])


def rule2_run_one_side(x, limits, n=9):
    """Nelson rule 2: n or more consecutive points on the same side of center.

    Catches small sustained shifts that rule 1 misses.
    """
    s = pd.Series(x).reset_index(drop=True)
    side = np.sign(s - limits["center"])
    grp = (side != side.shift()).cumsum()
    run_len = side.groupby(grp).transform("size")
    return (run_len >= n) & (side != 0)


def cp_cpk(x, lsl, usl):
    """Capability indices against GIVEN spec limits.

    Cp/Cpk use within sigma (moving range); Pp/Ppk use overall sigma. Ppk far
    below Cpk means the process drifts or shifts between short windows.
    """
    v = pd.Series(x).dropna()
    mean = float(v.mean())
    s_within = mr_sigma(v)
    s_overall = float(v.std(ddof=1))
    half_gap = min(usl - mean, mean - lsl)
    return {
        "mean": mean, "sigma_within": s_within, "sigma_overall": s_overall,
        "Cp": (usl - lsl) / (6 * s_within), "Cpk": half_gap / (3 * s_within),
        "Pp": (usl - lsl) / (6 * s_overall), "Ppk": half_gap / (3 * s_overall),
    }
