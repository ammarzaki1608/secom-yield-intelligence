import pandas as pd
import pytest

from secom.data import split_by_time

def make_df(timestamps):
    """A tiny fake table: just the columns split_by_time needs."""
    return pd.DataFrame({"timestamp": pd.to_datetime(timestamps), "label": -1})


def test_split_sizes():
    df = make_df(pd.date_range("2008-01-01", periods=10, freq="h"))
    train, test = split_by_time(df, train_frac=0.7)

    assert len(train) == 7
    assert len(test) == 3


def test_shuffled_table_raises():
    df = make_df(pd.date_range("2008-01-01", periods=10, freq="h"))   # arrange: sorted...
    shuffled = df.sample(frac=1, random_state=0)                      # ...then shuffled

    with pytest.raises(ValueError):                                   # act + assert
        split_by_time(shuffled, 0.7)


def test_tie_at_cut_raises():
    ts = list(pd.date_range("2008-01-01", periods=10, freq="h"))   # arrange
    ts[7] = ts[6]          # rows 6 and 7 now share a timestamp
    df = make_df(ts)

    with pytest.raises(ValueError):                                # act + assert
        split_by_time(df, 0.7)


def test_bad_train_frac_raises():
    df = make_df(pd.date_range("2008-01-01", periods=10, freq="h"))   

    with pytest.raises(ValueError):                                   
        split_by_time(df, train_frac=0)


def test_train_is_earlier_than_test():
    df = make_df(pd.date_range("2008-01-01", periods=10, freq="h"))
    train, test = split_by_time(df, 0.7)
    assert train["timestamp"].iloc[-1] < test["timestamp"].iloc[0]


