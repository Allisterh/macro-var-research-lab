"""
ML benchmarks: XGBoost and ElasticNet with direct multistep forecasting.

Direct multistep: separate model per horizon h. More robust than recursive
for non-linear methods. Each model learns to predict y(t+h) directly from
features at t.

Features = lags 1..p of all variables in the panel.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def _make_features(df: pd.DataFrame, p: int) -> pd.DataFrame:
    """Build lag features: y_{t-1}, ..., y_{t-p} for every column."""
    feats = {}
    for col in df.columns:
        for L in range(1, p + 1):
            feats[f"{col}_lag{L}"] = df[col].shift(L)
    return pd.DataFrame(feats, index=df.index)


class _DirectMultistep:
    """Base class for direct-multistep forecasting (one model per horizon)."""

    def __init__(self, p: int, max_horizon: int):
        self.p = p
        self.max_horizon = max_horizon
        self._models = {}  # horizon -> fitted model
        self._target = None
        self._last_features = None

    def _build_xy(self, df: pd.DataFrame, target: str, h: int):
        X = _make_features(df, self.p)
        y = df[target].shift(-h + 1) if h > 0 else df[target]
        Z = pd.concat([X, y.rename("__target__")], axis=1).dropna()
        return Z.drop(columns="__target__").values, Z["__target__"].values

    def _build_last_features(self, df: pd.DataFrame):
        X = _make_features(df, self.p).iloc[[-1]]
        return X.values

    def _new_model(self):
        raise NotImplementedError

    def fit(self, df: pd.DataFrame, target: str) -> "_DirectMultistep":
        d = df.dropna()
        self._target = target
        self._last_features = self._build_last_features(d)
        for h in range(1, self.max_horizon + 1):
            X, y = self._build_xy(d, target, h)
            if len(y) < 20:
                continue
            model = self._new_model()
            model.fit(X, y)
            self._models[h] = model
        return self

    def forecast(self, h: int, target: str = None):
        out = []
        for hh in range(1, h + 1):
            mdl = self._models.get(hh)
            if mdl is None:
                # Fall back to last fitted horizon
                fallback = max(self._models.keys()) if self._models else None
                if fallback is None:
                    out.append(np.nan)
                    continue
                mdl = self._models[fallback]
            pred = mdl.predict(self._last_features)[0]
            out.append(pred)
        return np.array(out)


class XGBoostForecaster(_DirectMultistep):

    name = "XGBoost"

    def __init__(self, p: int = 4, max_horizon: int = 12,
                 n_estimators: int = 500, max_depth: int = 4,
                 learning_rate: float = 0.05, random_state: int = 42):
        super().__init__(p, max_horizon)
        if not HAS_XGB:
            raise ImportError("Install xgboost: pip install xgboost")
        self.kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            objective="reg:squarederror",
            verbosity=0,
        )

    def _new_model(self):
        return XGBRegressor(**self.kwargs)


class ElasticNetForecaster(_DirectMultistep):

    name = "ElasticNet"

    def __init__(self, p: int = 4, max_horizon: int = 12, cv: int = 5,
                 random_state: int = 42):
        super().__init__(p, max_horizon)
        self.cv = cv
        self.random_state = random_state
        self._scaler = None

    def _new_model(self):
        # Wrap in a small pipeline-like object to standardise
        return _ScaledEN(cv=self.cv, random_state=self.random_state)


class _ScaledEN:
    """ElasticNet with internal standardisation. Avoids leakage at forecast time."""

    def __init__(self, cv: int = 5, random_state: int = 42):
        self.cv = cv
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._en = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            cv=cv,
            random_state=random_state,
            max_iter=10000,
        )

    def fit(self, X, y):
        Xs = self._scaler.fit_transform(X)
        self._en.fit(Xs, y)
        return self

    def predict(self, X):
        Xs = self._scaler.transform(X)
        return self._en.predict(Xs)
