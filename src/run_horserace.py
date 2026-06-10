"""
run_horserace.py
Main recursive OOS forecasting loop.

For each (dataset, model, target, horizon):
  for each vintage t in [first_oos_date, last_date - h]:
    fit model on data[:t]
    forecast h steps ahead
    record y_pred vs y_true(t+h)

Outputs:
  results/forecasts.parquet   (long-format: every (ds, model, tgt, h, t, y_true, y_pred))
  results/metrics.csv         (pivot table of RMSE ratios)
  results/dm_tests.csv        (pairwise Diebold-Mariano)

Usage:
  python src/run_horserace.py --datasets small medium large
  python src/run_horserace.py --quick  # one vintage, smoke test
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from models import (
    RandomWalk, ARp, OLSVar,
    BVARMinnesota, FAVAR,
    XGBoostForecaster, ElasticNetForecaster,
)
from evaluate import build_metrics_table, diebold_mariano

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("horserace")

ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


HORIZONS = [1, 2, 4, 8, 12]
TARGETS = ["GDPC1", "CPILFESL", "FEDFUNDS", "UNRATE"]
FIRST_OOS = "2000-01-01"


def make_models():
    """Define the model roster."""
    return [
        RandomWalk(),
        ARp(max_lags=8),
        OLSVar(ic="bic", max_lags=8),
        OLSVar(ic="aic", max_lags=8),
        BVARMinnesota(p=4, lam1=0.2, hierarchical=False),
        BVARMinnesota(p=4, hierarchical=True),
        # FAVAR is added only for wide dataset
        XGBoostForecaster(p=4, max_horizon=12),
        ElasticNetForecaster(p=4, max_horizon=12),
    ]


def run_single(model, data: pd.DataFrame, target: str, horizon: int) -> np.ndarray:
    """Fit model on data and return h-step forecast for target."""
    m = model.fit(data, target=target)
    fc = m.forecast(horizon, target=target)
    if isinstance(fc, pd.DataFrame):
        return fc[target].values
    return np.asarray(fc).flatten()


def run_dataset(dataset_name: str, data: pd.DataFrame, quick: bool = False) -> pd.DataFrame:
    """Run the full recursive OOS loop for one dataset."""
    log.info("=== Dataset: %s ===", dataset_name)
    log.info("Shape: %s | Range: [%s, %s]",
             data.shape, data.index.min().date(), data.index.max().date())

    data = data.dropna()
    dates = data.index
    first_oos_date = pd.Timestamp(FIRST_OOS)
    eligible_dates = dates[dates >= first_oos_date]

    if quick:
        eligible_dates = eligible_dates[:3]
        log.info("QUICK MODE: only %d vintages", len(eligible_dates))

    records = []
    models = make_models()
    if dataset_name == "wide":
        # FAVAR shines on wide datasets
        models.append(FAVAR(n_factors=3))

    for t_idx, t in enumerate(eligible_dates):
        train = data.loc[:t]
        if len(train) < 60:
            continue
        for model in models:
            for target in TARGETS:
                if target not in train.columns:
                    continue
                try:
                    # Refit per vintage (per model)
                    fresh = model.__class__(**{k: v for k, v in model.__dict__.items()
                                                if not k.startswith("_") and k not in ("name", "kwargs")}) \
                        if hasattr(model, "__class__") else model
                    # For simplicity, re-instantiate by class for now
                    cls = type(model)
                    if cls is XGBoostForecaster:
                        instance = XGBoostForecaster(p=4, max_horizon=max(HORIZONS))
                    elif cls is ElasticNetForecaster:
                        instance = ElasticNetForecaster(p=4, max_horizon=max(HORIZONS))
                    elif cls is BVARMinnesota:
                        instance = BVARMinnesota(p=model.p, lam1=model.lam1, hierarchical=model.hierarchical)
                    elif cls is OLSVar:
                        instance = OLSVar(ic=model.ic, max_lags=model.max_lags)
                    elif cls is FAVAR:
                        instance = FAVAR(n_factors=model.n_factors)
                    elif cls is ARp:
                        instance = ARp(max_lags=model.max_lags)
                    else:
                        instance = cls()

                    fc_full = run_single(instance, train, target, max(HORIZONS))
                    for h in HORIZONS:
                        target_date = t + pd.tseries.offsets.QuarterEnd(h)
                        if target_date not in data.index:
                            continue
                        y_true = data.loc[target_date, target]
                        y_pred = fc_full[h - 1] if h - 1 < len(fc_full) else np.nan
                        records.append({
                            "dataset": dataset_name,
                            "model": instance.name,
                            "target": target,
                            "horizon": h,
                            "vintage": t,
                            "y_true": y_true,
                            "y_pred": y_pred,
                        })
                except Exception as e:
                    log.debug("Fail | %s | %s | %s | %s: %s",
                             dataset_name, model.name, target, t.date(), e)
                    continue

        if (t_idx + 1) % 10 == 0:
            log.info("  vintage %d/%d done (%s)", t_idx + 1, len(eligible_dates), t.date())

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["small", "medium", "large"])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    all_records = []
    for ds_name in args.datasets:
        path = DATA_PROC / f"{ds_name}.parquet"
        if not path.exists():
            log.warning("Skip %s, no processed data", ds_name)
            continue
        data = pd.read_parquet(path)
        recs = run_dataset(ds_name, data, quick=args.quick)
        all_records.append(recs)

    forecasts = pd.concat(all_records, ignore_index=True)
    out_path = RESULTS / "forecasts.parquet"
    forecasts.to_parquet(out_path)
    log.info("Wrote %s rows to %s", len(forecasts), out_path)

    metrics = build_metrics_table(forecasts, benchmark_name="RandomWalk")
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    log.info("Wrote metrics.csv")

    # Print summary
    print("\n=== RMSE RATIO vs RANDOM WALK (lower = better) ===\n")
    pivot = metrics.pivot_table(
        index=["dataset", "model"],
        columns=["target", "horizon"],
        values="rmse_ratio_vs_rw",
    )
    print(pivot.round(3))


if __name__ == "__main__":
    main()
