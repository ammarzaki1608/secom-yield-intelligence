import pandas as pd

def fit_cleaning(train, missing_threshold=0.2):
    """
    Learn the cleaning rules from the training period only.

    Sensors are dropped if more than missing_threshold of their values
    are missing, or if they have at most one distinct value. The median
    of every remaining sensor is stored for filling gaps later.

    Args:
        train: training table from split_by_time.
        missing_threshold: largest share of missing values a sensor
            may have and still be kept (0.2 means 20%).

    Returns:
        rules: dict with two keys:
            "keep_cols": list of sensor names to keep.
            "medians": dict of sensor name -> train median.
    """
    # Only sensors: label and timestamp are never dropped or filled
    sensor_cols = [c for c in train.columns if c.startswith("sensor_")]

    # Step 1: keep sensors whose missing share is NOT over the threshold
    missing_pct = train[sensor_cols].isna().mean()
    keep = missing_pct[missing_pct <= missing_threshold].index

    # Step 2: of those, keep sensors with more than one distinct value
    n_distinct = train[keep].nunique()
    keep = n_distinct[n_distinct > 1].index.tolist()

    # Step 3: record the train median of each kept sensor
    medians = train[keep].median().to_dict()

    return {"keep_cols": keep, "medians": medians}


def apply_cleaning(df, rules):
    """
    Apply cleaning rules to any table

    Keeps only the sensors in rules["keep_cols"] (plus label and
    timestamp), then fills missing values with the train medians
    in rules["medians"]. Nothing is recomputed from df. 

    Args:
        df: any table. train or test
        rules: dict with two keys from the fit_cleaning

    Returns:
        A cleaned copy of df with the kept sensors, label and
        timestamp, and no missing values in the sensors.

    Raises:
        ValueError: if df is missing any column listed in rules.
    """
    keep = rules["keep_cols"]

    # Guard: fail loudly if a sensor the rules need is not in df
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    # Keep the rule sensors plus label and timestamp (if present)
    other_cols = [c for c in ["label", "timestamp"] if c in df.columns]
    cleaned = df[keep + other_cols].copy()

    # Fill gaps with the TRAIN medians stored in rules
    cleaned[keep] = cleaned[keep].fillna(rules["medians"])

    return cleaned