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


def flag_runs(values, limits, run_length=9):
    """Return True where run_length or more points in a row are on one side of the centre."""
    flags = []
    count = 0
    previous_side = 0

    for value in values:
        side = np.sign(value - limits["centre"])

        if side == 0:                    # exactly on the centre line: no side
            count = 0
        elif side == previous_side:               # same side as the last point
            count = count + 1
        else:                            # changed side: a new run starts
            count = 1

        flags.append(count >= run_length)
        previous_side = side             # remember for the next point

    return np.array(flags)