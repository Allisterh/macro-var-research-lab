"""
Evaluation: RMSE, MAE, MAPE, Diebold-Mariano test, OOS R-squared.
All metrics report relative to a benchmark series (typically random walk).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred)) & (y_true != 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def oos_r2(y_true: np.ndarray, y_pred: np.ndarray, y_bench: np.ndarray) -> float:
    """Campbell-Thompson 2008 OOS R-squared vs a benchmark forecast.
    Values > 0 mean the model beats the benchmark on MSE."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_bench))
    if mask.sum() == 0:
        return np.nan
    num = np.sum((y_true[mask] - y_pred[mask]) ** 2)
    den = np.sum((y_true[mask] - y_bench[mask]) ** 2)
    return 1.0 - num / den if den > 0 else np.nan


def diebold_mariano(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    h: int = 1,
    loss: str = "squared",
) -> tuple[float, float]:
    """
    Diebold-Mariano (1995) test of equal predictive accuracy.
    Positive DM stat means model 2 is more accurate (lower loss) than model 1.
    Uses Newey-West HAC variance estimator with lag h-1.
    Returns (DM statistic, two-sided p-value).
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred1) | np.isnan(y_pred2))
    e1 = y_true[mask] - y_pred1[mask]
    e2 = y_true[mask] - y_pred2[mask]
    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "abs":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(loss)

    n = len(d)
    if n < 10:
        return np.nan, np.nan

    d_bar = d.mean()
    # Newey-West variance with lag = h - 1
    L = max(h - 1, 0)
    gamma0 = np.var(d, ddof=1)
    var_d = gamma0
    for k in range(1, L + 1):
        gamma_k = np.cov(d[:-k], d[k:], ddof=1)[0, 1]
        var_d += 2 * (1 - k / (L + 1)) * gamma_k

    if var_d <= 0:
        return np.nan, np.nan

    dm_stat = d_bar / np.sqrt(var_d / n)
    # Harvey-Leybourne-Newbold (1997) small-sample correction
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat *= correction
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1))
    return float(dm_stat), float(p_value)


def build_metrics_table(forecasts: pd.DataFrame, benchmark_name: str = "RandomWalk") -> pd.DataFrame:
    """
    Input: forecasts DataFrame with columns:
        ['dataset', 'model', 'target', 'horizon', 'date', 'y_true', 'y_pred']
    Output: pivot table of RMSE ratios vs benchmark.
    """
    out = []
    for (ds, tgt, h), grp in forecasts.groupby(["dataset", "target", "horizon"]):
        bench = grp[grp["model"] == benchmark_name]
        if bench.empty:
            continue
        bench_rmse = rmse(bench["y_true"].values, bench["y_pred"].values)
        for model, mgrp in grp.groupby("model"):
            r = rmse(mgrp["y_true"].values, mgrp["y_pred"].values)
            row = {
                "dataset": ds,
                "model": model,
                "target": tgt,
                "horizon": h,
                "rmse": r,
                "rmse_ratio_vs_rw": r / bench_rmse if bench_rmse > 0 else np.nan,
            }
            out.append(row)
    return pd.DataFrame(out)
