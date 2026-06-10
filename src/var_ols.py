"""
OLS VAR wrapper around statsmodels.
This is the EViews-equivalent: lag length by information criterion, then OLS.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


class OLSVar:
    """OLS VAR with IC-selected lag length. Same as EViews 'lag length criteria'."""

    def __init__(self, ic: str = "bic", max_lags: int = 8):
        if ic not in ("aic", "bic", "hqic", "fpe"):
            raise ValueError(f"ic must be one of aic/bic/hqic/fpe, got {ic}")
        self.ic = ic
        self.max_lags = max_lags
        self.name = f"VAR-{ic.upper()}"
        self._fit = None
        self._cols = None
        self._chosen_lag = None

    def fit(self, df: pd.DataFrame, target: str) -> "OLSVar":
        """target is unused at fit time (multivariate); used at forecast unpacking."""
        d = df.dropna()
        self._cols = list(d.columns)
        model = VAR(d)
        # Cap maxlags at floor(T/3) to avoid degenerate sample
        T = len(d)
        max_lags = min(self.max_lags, max(1, T // 3 - 1))
        sel = model.select_order(maxlags=max_lags)
        chosen = getattr(sel, self.ic)
        # Statsmodels returns 0 if no lag is preferred; force at least 1
        self._chosen_lag = max(1, int(chosen))
        self._fit = model.fit(self._chosen_lag)
        return self

    def forecast(self, h: int, target: str = None) -> np.ndarray:
        """Returns h-step forecast for `target` column (or all columns if target=None)."""
        y_last = self._fit.endog[-self._fit.k_ar:]
        fc = self._fit.forecast(y=y_last, steps=h)  # shape (h, k)
        fc_df = pd.DataFrame(fc, columns=self._cols)
        if target:
            return fc_df[target].values
        return fc_df
