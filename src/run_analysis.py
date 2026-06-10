"""
run_analysis.py - Recursive OOS horserace on real US FRED data (optimized).
"""

import sys
import warnings
import time
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from models import (
    RandomWalk, ARp, OLSVar, BVARMinnesota,
    XGBoostForecaster, ElasticNetForecaster,
)
from evaluate import build_metrics_table, diebold_mariano

DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

HORIZONS = [1, 4, 8]
TARGETS = ["GDPC1", "CPILFESL", "FEDFUNDS", "UNRATE"]
FIRST_OOS = pd.Timestamp("2000-01-01")
MIN_TRAIN = 50
VINTAGE_STRIDE = 2  # forecast every 2nd quarter


def make_models():
    return [
        ("RandomWalk", lambda: RandomWalk()),
        ("AR(BIC)",    lambda: ARp(max_lags=8)),
        ("VAR-BIC",    lambda: OLSVar(ic="bic", max_lags=8)),
        ("VAR-AIC",    lambda: OLSVar(ic="aic", max_lags=8)),
        ("BVAR-Tight", lambda: BVARMinnesota(p=4, lam1=0.1)),
        ("BVAR-Loose", lambda: BVARMinnesota(p=4, lam1=0.5)),
        ("XGBoost",    lambda: XGBoostForecaster(p=4, max_horizon=max(HORIZONS), n_estimators=100, max_depth=3)),
        ("ElasticNet", lambda: ElasticNetForecaster(p=4, max_horizon=max(HORIZONS))),
    ]


def safe_forecast(model, h, target):
    fc = model.forecast(h, target=target)
    if isinstance(fc, pd.DataFrame):
        fc = fc[target].values
    elif isinstance(fc, pd.Series):
        fc = fc.values
    return np.asarray(fc).flatten()


def run_dataset(name: str) -> pd.DataFrame:
    path = DATA_PROC / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path).dropna()
    eligible = df.index[df.index >= FIRST_OOS][::VINTAGE_STRIDE]
    print(f"\n=== {name.upper()}: shape={df.shape}  vintages={len(eligible)}", flush=True)

    records = []
    t0 = time.time()
    models = make_models()

    for vi, t in enumerate(eligible):
        train = df.loc[:t]
        if len(train) < MIN_TRAIN:
            continue
        fits = {}
        for mn, factory in models:
            try:
                m = factory()
                m.fit(train, target=TARGETS[0])
                fits[mn] = (m, factory)
            except Exception:
                continue

        for target in TARGETS:
            for mn, (m, factory) in fits.items():
                try:
                    if mn in ("AR(BIC)", "RandomWalk"):
                        uni = factory()
                        uni.fit(train, target=target)
                        fc = safe_forecast(uni, max(HORIZONS), target)
                    else:
                        fc = safe_forecast(m, max(HORIZONS), target)
                    for h in HORIZONS:
                        target_date = t + pd.tseries.offsets.QuarterEnd(h)
                        if target_date not in df.index:
                            continue
                        records.append({
                            "dataset": name, "model": mn, "target": target, "horizon": h,
                            "vintage": t, "y_true": df.loc[target_date, target],
                            "y_pred": fc[h - 1] if h - 1 < len(fc) else np.nan,
                        })
                except Exception:
                    continue
        if (vi + 1) % 5 == 0:
            print(f"  v{vi+1}/{len(eligible)} ({t.date()})  {time.time()-t0:.0f}s", flush=True)

    print(f"  done: {len(records)} recs in {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(records)


def main():
    all_recs = []
    for name in ["small", "medium", "large"]:
        recs = run_dataset(name)
        if not recs.empty:
            all_recs.append(recs)
    forecasts = pd.concat(all_recs, ignore_index=True)
    forecasts.to_parquet(RESULTS / "forecasts.parquet")
    print(f"\nSaved {len(forecasts)} records")

    metrics = build_metrics_table(forecasts, benchmark_name="RandomWalk")
    metrics.to_csv(RESULTS / "metrics.csv", index=False)

    dm_records = []
    for (ds, tgt, h), grp in forecasts.groupby(["dataset", "target", "horizon"]):
        rw = grp[grp["model"] == "RandomWalk"]
        for mn, mgrp in grp.groupby("model"):
            if mn == "RandomWalk":
                continue
            joined = pd.merge(
                rw[["vintage", "y_true", "y_pred"]].rename(columns={"y_pred": "rw_pred"}),
                mgrp[["vintage", "y_pred"]].rename(columns={"y_pred": "mdl_pred"}),
                on="vintage").dropna()
            if len(joined) < 10:
                continue
            stat, p = diebold_mariano(joined["y_true"].values, joined["rw_pred"].values,
                                       joined["mdl_pred"].values, h=h, loss="squared")
            dm_records.append({"dataset": ds, "model": mn, "target": tgt, "horizon": h,
                               "dm_stat": stat, "p_value": p, "n": len(joined)})
    pd.DataFrame(dm_records).to_csv(RESULTS / "dm_tests.csv", index=False)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)
    pivot = metrics.pivot_table(index=["dataset", "model"], columns=["target", "horizon"], values="rmse_ratio_vs_rw")
    print("\n=== RMSE ratio vs Random Walk ===")
    print(pivot.round(3))


if __name__ == "__main__":
    main()
