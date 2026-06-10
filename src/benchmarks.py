"""
Benchmark models: random walk and AR(p).
Every fancy model must beat these. Meese-Rogoff (1983) humility check.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg


class RandomWalk:
    """y(t+h) = y(t). The floor benchmark. Equivalent to 'no-change' forecast."""

    name = "RandomWalk"

    def __init__(self, with_drift: bool = False):
        self.with_drift = with_drift
        self._last = None
        self._drift = 0.0

    def fit(self, df: pd.DataFrame, target: str) -> "RandomWalk":
        self._last = df[target].iloc[-1]
        if self.with_drift:
            self._drift = df[target].diff().mean()
        return self

    def forecast(self, h: int) -> np.ndarray:
        if self.with_drift:
            return np.array([self._last + (i + 1) * self._drift for i in range(h)])
        return np.full(h, self._last)


class ARp:
    """Univariate AR(p) with BIC-selected lag length. Statsmodels under the hood."""

    name = "AR(BIC)"

    def __init__(self, max_lags: int = 8):
        self.max_lags = max_lags
        self._model = None
        self._res = None
        self._best_p = None

    def fit(self, df: pd.DataFrame, target: str) -> "ARp":
        y = df[target].dropna()
        # BIC lag selection
        bics = {}
        for p in range(1, self.max_lags + 1):
            try:
                res = AutoReg(y, lags=p, old_names=False).fit()
                bics[p] = res.bic
            except Exception:
                continue
        if not bics:
            self._best_p = 1
        else:
            self._best_p = min(bics, key=bics.get)
        self._res = AutoReg(y, lags=self._best_p, old_names=False).fit()
        return self

    def forecast(self, h: int) -> np.ndarray:
        fc = self._res.forecast(steps=h)
        return np.asarray(fc)
