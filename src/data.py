from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .config import CFG
from .utils import ensure_dir

def _generate_sample(start="2020-02-02", end="2020-07-18") -> pd.DataFrame:
    idx = pd.date_range(start=start, end=end, freq="D")
    n = len(idx)
    rng = np.random.default_rng(CFG.RANDOM_SEED)
    trend = np.linspace(0, 120, n)
    weekly = 60 * np.sin(2 * np.pi * (idx.dayofweek) / 7.0)
    noise = rng.normal(0, 25, n)
    base = 650 + trend + weekly + noise
    users = np.clip(base, 500, 1000).round().astype(int)
    return pd.DataFrame({"Date": idx, "Users": users})

def load_or_create_data(path: str = CFG.DATA_PATH) -> pd.DataFrame:
    ensure_dir(Path(path).parent.as_posix())
    p = Path(path)
    if not p.exists():
        df = _generate_sample()
        df.to_csv(p, index=False)
    else:
        df = pd.read_csv(p)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").dropna()
    df = df[["Date", "Users"]]
    df["Users"] = pd.to_numeric(df["Users"], errors="coerce")
    df = df.dropna()
    return df.reset_index(drop=True)

def train_test_split_series(df: pd.DataFrame, train_ratio: float = CFG.TRAIN_RATIO):
    n = len(df)
    train_end = int(n * train_ratio)
    train = df.iloc[:train_end].copy()
    test = df.iloc[train_end:].copy()
    return train, test
