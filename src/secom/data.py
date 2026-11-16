from pathlib import Path

import pandas as pd


def load_secom(raw_dir="data/raw"):
    """
    Load the SECOM sensor and label files into one table, sorted by time.

    The two raw files share no ID, so they are linked only by row order.
    They are joined first and sorted afterwards, so every label stays
    attached to its own unit.

    Args:
        raw_dir: folder that contains secom.data and secom_labels.data.
            It is relative to where the code runs, not to this file.

    Returns:
        DataFrame with columns sensor_0 ... sensor_589, label and timestamp,
        sorted by timestamp, with a fresh 0..n-1 index.
    """
    raw_dir = Path(raw_dir)
    labels = pd.read_csv(raw_dir / "secom_labels.data", sep=r"\s+", header=None)
    sensors = pd.read_csv(raw_dir / "secom.data", sep=r"\s+", header=None)

    # The join below is by row number. With unequal lengths pandas would
    # silently fill NaN, so fail loudly instead.
    if len(sensors) != len(labels):
        raise ValueError(
            f"Row count mismatch: {len(sensors)} sensor rows vs {len(labels)} label rows."
        )

    df = sensors.copy()
    df.columns = [f"sensor_{i}" for i in range(df.shape[1])]
    df["label"] = labels[0]
    df["timestamp"] = pd.to_datetime(labels[1], format="%d/%m/%Y %H:%M:%S")
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def split_by_time(df, train_frac=0.7):
    """
    Split a time-sorted table into a train period and a later test period.

    The split is by row position, never random, so the test set is always
    later in time than the train set (no future information in training).

    Args:
        df: table sorted by timestamp (for example the output of load_secom).
        train_frac: share of the earliest rows used for training, between 0 and 1.

    Returns:
        (train, test), both copies of the original rows.

    Raises:
        ValueError: if train_frac is not between 0 and 1, if df is not sorted
            by timestamp, if the split leaves an empty side, or if the cut
            falls between two rows with the same timestamp.
    """
    if not 0 < train_frac < 1:
        raise ValueError(f"train_frac must be between 0 and 1, got {train_frac}.")

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("DataFrame is not in chronological order.")

    split_index = int(len(df) * train_frac)

    if split_index == 0 or split_index >= len(df):
        raise ValueError("Dataset is too small for the requested split ratio.")

    last_train_time = df["timestamp"].iloc[split_index - 1]
    first_test_time = df["timestamp"].iloc[split_index]

    if last_train_time == first_test_time:
        raise ValueError(
            f"Split boundary falls inside duplicate timestamp: {last_train_time}. "
            "Adjust the split ratio or move the cutoff."
        )

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    return train, test
