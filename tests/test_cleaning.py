import numpy as np
import pandas as pd
import pytest

from secom.cleaning import fit_cleaning, apply_cleaning

def make_train():
    """Tiny train table where each sensor tests one cleaning rule."""
    return pd.DataFrame({
        "sensor_0": [1, np.nan, np.nan, 4, 5],    # 40% missing -> drop
        "sensor_1": [7, 7, 7, 7, 7],              # constant -> drop
        "sensor_2": [3, np.nan, 5, 9, 1],         # 20% missing -> keep (boundary)
        "sensor_3": [10, 20, 30, 40, 50],         # clean -> keep
        "label": [-1, -1, 1, -1, -1],
        "timestamp": pd.date_range("2008-01-01", periods=5, freq="h"),
    })


def test_drops_high_missing_sensor():
    train = make_train()
    rules = fit_cleaning(train)

    assert "sensor_0" not in rules["keep_cols"]


def test_drops_constant_sensor():
    train = make_train()
    rules = fit_cleaning(train)

    assert "sensor_1" not in rules["keep_cols"]

    
def test_keeps_good_sensors():
    train = make_train()
    rules = fit_cleaning(train)

    assert "sensor_2" in rules["keep_cols"]
    assert "sensor_3" in rules["keep_cols"]


def test_stores_train_medians():
    train = make_train()
    rules = fit_cleaning(train)

    assert rules["medians"]["sensor_2"] == 4.0
    assert rules["medians"]["sensor_3"] == 30.0


def make_test():
    """Tiny test table with values deliberately different from the train table."""
    return pd.DataFrame({
        "sensor_0": [1, 2, 3],           # no missing here, but dropped in train, so must be dropped here too
        "sensor_1": [7, 8, 9],           # same
        "sensor_2": [100, np.nan, 200],  # NaN must become 4.0 (train median), not 150 (test median)
        "sensor_3": [np.nan, 60, 70],    # NaN must become 30.0 (train median), not 65 (test median)
        "label": [-1, -1, -1],
        "timestamp": pd.date_range("2008-01-01", periods=3, freq="h"),
    })


def test_apply_keeps_only_rule_columns():
    train = make_train()
    rules = fit_cleaning(train)
    df = make_test()
    cleaned = apply_cleaning(df, rules)
    
    assert set(cleaned.columns) == {"sensor_2", "sensor_3", "label", "timestamp"}


def test_apply_fills_with_train_median():
    train = make_train()
    rules = fit_cleaning(train)
    df = make_test()
    cleaned = apply_cleaning(df, rules)
    
    assert cleaned["sensor_2"].iloc[1] == 4.0    # row 1 was NaN
    assert cleaned["sensor_3"].iloc[0] == 30.0   # row 0 was NaN


def test_apply_raises_if_column_missing():
    train = make_train()
    rules = fit_cleaning(train) 
    df = make_test().drop(columns=["sensor_2"])

    with pytest.raises(ValueError):
        apply_cleaning(df, rules)