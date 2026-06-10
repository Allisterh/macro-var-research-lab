"""
BVAR with Minnesota prior.
Pure-Python implementation following Litterman (1986) and Banbura-Giannone-Reichlin (2010).
Uses dummy-observation approach (Sims-Zha) so we can use OLS on augmented data.

Hyperparameters:
  lambda1: overall tightness (smaller = more shrinkage; 0 = univariate AR, infinity = OLS VAR)
  lambda2: cross-equation tightness (relative tightness of off-diagonal lags)
  lambda3: lag decay (rate at which prior tightens for distant lags)
  lambda4: intercept tightness (large = uninformative on intercept)

Defaults follow Banbura-Giannone-Reichlin defaults for medium VARs.
Hierarchical version (Giannone-Lenza-Primiceri 2015) chooses lambda1 via marginal likelihood.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def _build_dummy_observations(
    y: np.ndarray,        # T x k matrix
    p: int,               # lag order
    lam1: float,
    lam2: float,
    lam3: float,
    lam4: float,
    sigma: np.ndarray,    # k-vector of AR(1) residual stds (per-variable scale)
):
    """
    Build Sims-Zha dummy observations such that running OLS on
    augmented (y*, X*) gives posterior mean under Minnesota prior.

    Returns Y_star (Td x k), X_star (Td x kp+1).
    """
    k = y.shape[1]
    # Block 1: prior on own first lag = 1, others = 0 (random walk centring for levels)
    # For stationary (log-diff) series user should pass in pre-differenced data;
    # alternatively change rw_prior to 0.
    rw_prior = np.ones(k)  # 1 for level series, set to 0 if you pre-difference

    Y_blocks = []
    X_blocks = []

    # Block 1: coefficients on own and cross lags
    for L in range(1, p + 1):
        # Y block
        diag_block = np.diag(sigma) * (rw_prior if L == 1 else np.zeros(k))
        Y_blocks.append(diag_block / lam1 * (L ** lam3))
        # X block: identity on the L-th lag block, zeros elsewhere
        X_row = np.zeros((k, k * p + 1))
        X_row[:, (L - 1) * k:L * k] = np.diag(sigma) / lam1 * (L ** lam3)
        X_blocks.append(X_row)

    # Block 2: prior on the residual covariance (sum-of-coefficients dummy)
    # Banbura et al use sum-of-coefficients and dummy-initial-observation. Skipping for brevity.

    # Block 3: prior on intercept (loose)
    intercept_prior_Y = np.zeros((1, k))
    intercept_prior_X = np.zeros((1, k * p + 1))
    intercept_prior_X[0, -1] = 1.0 / lam4
    Y_blocks.append(intercept_prior_Y)
    X_blocks.append(intercept_prior_X)

    return np.vstack(Y_blocks), np.vstack(X_blocks)


def _build_lag_matrix(y: np.ndarray, p: int):
    """Build [y_{t-1}, y_{t-2}, ..., y_{t-p}, 1] design matrix and y_t outcome."""
    T, k = y.shape
    rows = []
    for t in range(p, T):
        row = []
        for L in range(1, p + 1):
            row.extend(y[t - L])
        row.append(1.0)
        rows.append(row)
    X = np.array(rows)
    Y = y[p:]
    return Y, X


def _ar1_sigmas(y: np.ndarray) -> np.ndarray:
    """Per-variable AR(1) residual std, used to scale the prior."""
    k = y.shape[1]
    sigmas = np.zeros(k)
    for j in range(k):
        s = y[:, j]
        s = s[~np.isnan(s)]
        # OLS AR(1)
        y_dep = s[1:]
        y_lag = s[:-1]
        A = np.vstack([y_lag, np.ones_like(y_lag)]).T
        beta, _, _, _ = np.linalg.lstsq(A, y_dep, rcond=None)
        resid = y_dep - A @ beta
        sigmas[j] = np.std(resid)
    return sigmas


class BVARMinnesota:
    """
    BVAR with Minnesota prior using dummy-observation OLS estimation.
    """

    def __init__(
        self,
        p: int = 4,
        lam1: float = 0.2,    # overall tightness (Banbura et al default for medium)
        lam2: float = 0.5,    # cross-equation tightness
        lam3: float = 1.0,    # lag decay
        lam4: float = 100.0,  # intercept tightness (loose)
        hierarchical: bool = False,
    ):
        self.p = p
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        self.lam4 = lam4
        self.hierarchical = hierarchical
        self.name = "BVAR-Minnesota-Hier" if hierarchical else "BVAR-Minnesota"
        self._cols = None
        self._beta = None  # (kp+1) x k
        self._sigma = None

    # ---- core fit / forecast ----

    def _posterior_mean(self, y_arr: np.ndarray, lam1: float):
        T, k = y_arr.shape
        sigmas = _ar1_sigmas(y_arr)
        Y_d, X_d = _build_dummy_observations(
            y_arr, self.p, lam1, self.lam2, self.lam3, self.lam4, sigmas
        )
        Y, X = _build_lag_matrix(y_arr, self.p)
        Y_aug = np.vstack([Y_d, Y])
        X_aug = np.vstack([X_d, X])
        beta, _, _, _ = np.linalg.lstsq(X_aug, Y_aug, rcond=None)  # (kp+1) x k
        resid = Y - X @ beta
        sigma = (resid.T @ resid) / max(T - self.p, 1)
        return beta, sigma, sigmas

    def _neg_log_marg_like(self, log_lam1: float, y_arr: np.ndarray) -> float:
        """Approximate log marginal likelihood for hierarchical lambda1 selection.
        Following Giannone-Lenza-Primiceri 2015, we maximise this."""
        lam1 = np.exp(log_lam1)
        try:
            beta, sigma, _ = self._posterior_mean(y_arr, lam1)
            Y, X = _build_lag_matrix(y_arr, self.p)
            resid = Y - X @ beta
            T_eff = resid.shape[0]
            # Crude: -0.5 * T * log(det(Sigma)) (drop constants)
            sign, logdet = np.linalg.slogdet(sigma + 1e-8 * np.eye(sigma.shape[0]))
            return T_eff * logdet
        except Exception:
            return 1e10

    def fit(self, df: pd.DataFrame, target: str = None) -> "BVARMinnesota":
        d = df.dropna()
        self._cols = list(d.columns)
        y_arr = d.values

        if self.hierarchical:
            # Grid search log lambda from log(0.01) to log(1.0)
            best = minimize_scalar(
                self._neg_log_marg_like,
                args=(y_arr,),
                bounds=(np.log(0.01), np.log(1.0)),
                method="bounded",
                options={"xatol": 0.01},
            )
            self.lam1 = float(np.exp(best.x))

        self._beta, self._sigma, _ = self._posterior_mean(y_arr, self.lam1)
        self._last_y = y_arr[-self.p:]
        return self

    def forecast(self, h: int, target: str = None):
        """Iterated forecast h steps ahead. Returns DataFrame (h x k) or array if target given."""
        beta = self._beta
        p = self.p
        k = len(self._cols)
        history = list(self._last_y)
        forecasts = []
        for _ in range(h):
            x = []
            for L in range(1, p + 1):
                x.extend(history[-L])
            x.append(1.0)
            x = np.array(x)
            y_next = x @ beta  # shape (k,)
            forecasts.append(y_next)
            history.append(y_next)
        fc_df = pd.DataFrame(forecasts, columns=self._cols)
        if target:
            return fc_df[target].values
        return fc_df
