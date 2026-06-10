"""
FAVAR: Factor-Augmented VAR (Bernanke, Boivin, Eliasz 2005).
Extract PCA factors from wide panel, put in VAR with observed targets.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR


class FAVAR:
    """
    FAVAR: extract k_factors PCA factors from full panel, put them in a VAR
    together with the observed targets.

    Trained on wide panel (~50-100 variables), forecasts the named targets.
    """

    def __init__(self, n_factors: int = 3, ic: str = "bic", max_lags: int = 6):
        self.n_factors = n_factors
        self.ic = ic
        self.max_lags = max_lags
        self.name = f"FAVAR({n_factors}f)"
        self._scaler = None
        self._pca = None
        self._var_fit = None
        self._var_cols = None
        self._targets = None

    def fit(self, df: pd.DataFrame, target: str = None, observed_cols: list = None) -> "FAVAR":
        """
        df: full wide panel including target columns.
        observed_cols: which columns are 'observed' (kept directly in VAR, not factorised).
            Defaults to [target].
        """
        d = df.dropna()
        if observed_cols is None:
            observed_cols = [target] if target else []
        self._targets = observed_cols
        factor_cols = [c for c in d.columns if c not in observed_cols]

        # Standardise the factor-candidate panel
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(d[factor_cols].values)
        self._pca = PCA(n_components=self.n_factors)
        F = self._pca.fit_transform(X)  # T x n_factors
        F_df = pd.DataFrame(F, index=d.index,
                            columns=[f"F{i+1}" for i in range(self.n_factors)])

        # Combine factors and observed targets into the VAR variable set
        var_data = pd.concat([F_df, d[observed_cols]], axis=1)
        self._var_cols = list(var_data.columns)

        model = VAR(var_data)
        T = len(var_data)
        max_lags = min(self.max_lags, max(1, T // 4))
        try:
            sel = model.select_order(maxlags=max_lags)
            chosen = max(1, int(getattr(sel, self.ic)))
        except Exception:
            chosen = 2
        self._var_fit = model.fit(chosen)
        return self

    def forecast(self, h: int, target: str = None):
        y_last = self._var_fit.endog[-self._var_fit.k_ar:]
        fc = self._var_fit.forecast(y=y_last, steps=h)
        fc_df = pd.DataFrame(fc, columns=self._var_cols)
        if target:
            return fc_df[target].values
        return fc_df
