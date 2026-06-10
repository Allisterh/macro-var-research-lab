"""
fetch_fred.py
Pull all FRED series defined in configs/ and build four datasets
(small, medium, large, wide). Caches raw downloads and writes
stationary, aligned panels to data/processed/.

Usage:
    export FRED_API_KEY="your_key_here"
    python src/fetch_fred.py --config configs/small.yaml
    python src/fetch_fred.py --all

Requirements:
    pip install pandas-datareader fredapi pyyaml pandas numpy
"""

from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from fredapi import Fred

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("fetch_fred")

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROC.mkdir(parents=True, exist_ok=True)


# ----- Stationarity transforms (McCracken-Ng codes) -----
# 1 = level
# 2 = first difference
# 4 = log
# 5 = log first difference (growth rate)
# 6 = log second difference

def transform(series: pd.Series, code: int) -> pd.Series:
    if code == 1:
        return series
    if code == 2:
        return series.diff()
    if code == 4:
        return np.log(series)
    if code == 5:
        return np.log(series).diff()
    if code == 6:
        return np.log(series).diff().diff()
    raise ValueError(f"Unknown transform code: {code}")


def fetch_one(fred: Fred, series_id: str, start: str = "1980-01-01") -> pd.Series:
    """Fetch a single FRED series, with caching."""
    cache_path = DATA_RAW / f"{series_id}.parquet"
    if cache_path.exists():
        log.debug("Using cached %s", series_id)
        return pd.read_parquet(cache_path).iloc[:, 0]
    log.info("Downloading %s from FRED", series_id)
    s = fred.get_series(series_id, observation_start=start)
    s.name = series_id
    s.to_frame().to_parquet(cache_path)
    return s


def to_quarterly(series: pd.Series, method: str = "mean") -> pd.Series:
    """Aggregate monthly/daily series to quarterly."""
    if series.empty:
        return series
    freq_chr = pd.infer_freq(series.index[:10]) or ""
    if freq_chr.startswith("Q"):
        return series
    if method == "mean":
        return series.resample("QE").mean()
    if method == "sum":
        return series.resample("QE").sum()
    if method == "last":
        return series.resample("QE").last()
    raise ValueError(f"Unknown method: {method}")


def build_dataset(config_path: Path, fred: Fred) -> pd.DataFrame:
    """Build one stationary, quarterly panel from a YAML config."""
    cfg = yaml.safe_load(config_path.read_text())
    name = cfg["name"]
    log.info("Building dataset: %s", name)

    frames = []
    for var in cfg["variables"]:
        sid = var["fred_id"]
        code = var["transform_code"]
        agg = var.get("agg", "mean")
        try:
            raw = fetch_one(fred, sid, start=cfg.get("start", "1980-01-01"))
        except Exception as e:
            log.warning("Skip %s: %s", sid, e)
            continue
        q = to_quarterly(raw, method=agg)
        tx = transform(q, code)
        tx.name = sid
        frames.append(tx)

    panel = pd.concat(frames, axis=1).dropna(how="all")
    panel = panel.loc[cfg["start"]: cfg.get("end")]
    # Drop rows where any target series is missing
    targets = cfg.get("targets", [])
    if targets:
        panel = panel.dropna(subset=targets)

    out = DATA_PROC / f"{name}.parquet"
    panel.to_parquet(out)
    log.info("Wrote %s | shape=%s | range=[%s, %s]",
             out.name, panel.shape, panel.index.min().date(), panel.index.max().date())
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="Single YAML config")
    parser.add_argument("--all", action="store_true", help="Build all configs/*.yaml")
    args = parser.parse_args()

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise SystemExit("Set FRED_API_KEY env variable. Get one free at https://fred.stlouisfed.org/docs/api/api_key.html")
    fred = Fred(api_key=api_key)

    configs_dir = ROOT / "configs"
    if args.all:
        configs = sorted(configs_dir.glob("*.yaml"))
    else:
        configs = [args.config]

    for c in configs:
        build_dataset(c, fred)


if __name__ == "__main__":
    main()
