import numpy as np

D2 = 1.128  # converts the average moving range of 2 readings into sigma


def fit_limits(baseline):
    """Learn the centre line and control limits from a stable baseline."""
    centre = baseline.mean()
    moving_ranges = np.abs(np.diff(baseline))
    sigma = moving_ranges.mean() / D2

    return {
        "centre": centre,
        "lower": centre - 3 * sigma,
        "upper": centre + 3 * sigma,
    }


def flag_outside(values, limits):
    """Return True for each value outside the control limits."""
    return (values < limits["lower"]) | (values > limits["upper"])