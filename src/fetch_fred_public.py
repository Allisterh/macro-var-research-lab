"""
fetch_fred_public.py
Pulls FRED series via the public fredgraph.csv endpoint (no API key required).

For production work you should use fredapi with a free key, but for this run
the public endpoint is sufficient and reproducible.

Usage:
    python fetch_fred_public.py
"""

import time
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROC.mkdir(parents=True, exist_ok=True)

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def fetch_series(sid: str, retries: int = 3) -> pd.Series:
    cache_path = DATA_RAW / f"{sid}.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=[0])
    else:
        for attempt in range(retries):
            try:
                r = requests.get(FRED_URL.format(sid=sid), timeout=30)
                r.raise_for_status()
                df = pd.read_csv(StringIO(r.text), parse_dates=[0])
                df.to_csv(cache_path, index=False)
                time.sleep(0.3)
                break
            except Exception as e:
                if attempt == retries - 1:
                    raise
                time.sleep(2)
    df = df.set_index(df.columns[0])
    df.index.name = "date"
    s = df.iloc[:, 0]
    s.name = sid
    s = pd.to_numeric(s, errors="coerce")
    return s


# Stationarity transform codes (McCracken-Ng)
def transform(s: pd.Series, code: int) -> pd.Series:
    if code == 1:
        return s
    if code == 2:
        return s.diff()
    if code == 4:
        return np.log(s)
    if code == 5:
        return np.log(s).diff()
    if code == 6:
        return np.log(s).diff().diff()
    raise ValueError(code)


def to_quarterly(s: pd.Series, method: str = "mean") -> pd.Series:
    freq = pd.infer_freq(s.dropna().index[:30]) or ""
    if freq.startswith("Q"):
        # Align to QE
        return s.resample("QE").last()
    if method == "mean":
        return s.resample("QE").mean()
    if method == "sum":
        return s.resample("QE").sum()
    if method == "last":
        return s.resample("QE").last()
    raise ValueError(method)


# Variable definitions per dataset
DATASETS = {
    "small": {
        "GDPC1":    {"code": 5, "agg": "last"},
        "CPILFESL": {"code": 5, "agg": "mean"},
        "FEDFUNDS": {"code": 1, "agg": "mean"},
        "UNRATE":   {"code": 1, "agg": "mean"},
    },
    "medium": {
        "GDPC1":     {"code": 5, "agg": "last"},
        "CPILFESL":  {"code": 5, "agg": "mean"},
        "FEDFUNDS":  {"code": 1, "agg": "mean"},
        "UNRATE":    {"code": 1, "agg": "mean"},
        "T10Y3M":    {"code": 1, "agg": "mean"},
        "BAA10Y":    {"code": 1, "agg": "mean"},
        "DCOILWTICO":{"code": 5, "agg": "mean"},
    },
    "large": {
        "GDPC1":     {"code": 5, "agg": "last"},
        "CPILFESL":  {"code": 5, "agg": "mean"},
        "FEDFUNDS":  {"code": 1, "agg": "mean"},
        "UNRATE":    {"code": 1, "agg": "mean"},
        "T10Y3M":    {"code": 1, "agg": "mean"},
        "BAA10Y":    {"code": 1, "agg": "mean"},
        "DCOILWTICO":{"code": 5, "agg": "mean"},
        "INDPRO":    {"code": 5, "agg": "mean"},
        "HOUST":     {"code": 5, "agg": "mean"},
        "PAYEMS":    {"code": 5, "agg": "mean"},
        "PCEPILFE":  {"code": 5, "agg": "mean"},
        "M2SL":      {"code": 5, "agg": "mean"},
        "EMRATIO":   {"code": 1, "agg": "mean"},
        "AHETPI":    {"code": 5, "agg": "mean"},
        "UMCSENT":   {"code": 1, "agg": "mean"},
    },
}


def build(name: str, start="1985-01-01"):
    spec = DATASETS[name]
    frames = []
    for sid, cfg in spec.items():
        s = fetch_series(sid)
        q = to_quarterly(s, method=cfg["agg"])
        tx = transform(q, cfg["code"])
        tx.name = sid
        frames.append(tx)
    panel = pd.concat(frames, axis=1)
    panel = panel.loc[start:].dropna(how="any")
    out = DATA_PROC / f"{name}.parquet"
    panel.to_parquet(out)
    print(f"[{name}] shape={panel.shape}  range=[{panel.index.min().date()}, {panel.index.max().date()}]")
    return panel


if __name__ == "__main__":
    for name in ["small", "medium", "large"]:
        build(name)
    print("\nDone. Data in", DATA_PROC)
