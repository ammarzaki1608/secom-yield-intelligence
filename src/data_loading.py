import pandas as pd
from pathlib import Path

def load_secom(raw_dir="data/raw"):
    raw_dir = Path(raw_dir)
    
    data = pd.read_csv(raw_dir / "secom.data", sep=r"\s+", header=None)
    labels = pd.read_csv(
        raw_dir / "secom_labels.data", sep=r"\s+", header=None,
        names=["label", "timestamp"]
    )
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], format="%d/%m/%Y %H:%M:%S")

    df = data.copy()
    df.columns = [f"sensor_{i}" for i in range(df.shape[1])]
    df["label"] = labels["label"]
    df["timestamp"] = labels["timestamp"]
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df

def clean_secom(df, missing_threshold=0.20):
    feature_cols = df.drop(columns=["label", "timestamp"]).columns
    missing_pct = df[feature_cols].isna().mean()
    high_missing = missing_pct[missing_pct > missing_threshold].index.tolist()
    df = df.drop(columns=high_missing)

    feature_cols = df.drop(columns=["label", "timestamp"]).columns
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    variances = df[feature_cols].var()
    zero_var = variances[variances == 0].index.tolist()
    df = df.drop(columns=zero_var)

    return df, {"dropped_high_missing": len(high_missing), "dropped_zero_variance": len(zero_var)}