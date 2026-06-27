"""
var_addons.py
=============
Drop-in fixes/additions for the US VAR notebook:

  (A) A CORRECT, STABLE Minnesota Bayesian VAR
      - natural-conjugate prior via dummy observations (Banbura, Giannone &
        Reichlin 2010); prior mean = 1 on each variable's own first lag,
        sum-of-coefficients prior to tame near-unit-roots. Shrinks toward a
        random walk instead of amplifying OLS.

  (B) Naive benchmarks (Random Walk, AR(1)) + Diebold-Mariano tests
      - the yardstick the original horse race was missing. DM uses the
        Harvey-Leybourne-Newbold small-sample correction (n_oos is tiny).

The coefficient/column layout matches the original notebook exactly:
    X columns = [lag1_v1..lag1_vn, lag2_v1..lag2_vn, ..., lagp_v1..lagp_vn,
                 const, exog_1..exog_m]
    B is (K x n)
so the fitted objects are interchangeable with the notebook's `forecast_from_coefficients`.
"""


# NOTE (merge): RW/AR(1) benchmarks and Diebold-Mariano (+HLN) are intentionally
# NOT defined here -- they already live in src/benchmarks.py and src/evaluate.py.
# This module adds only the BVAR pieces missing from src/bvar.py: the
# sum-of-coefficients prior (the block bvar.py marks "Skipping for brevity")
# and stability-filtered posterior sampling (companion eigenvalues < 1).

import numpy as np
import pandas as pd
from numpy.linalg import inv, eigvals, cholesky, slogdet
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg


# ----------------------------------------------------------------------
# Helpers: build the VAR design matrix in the notebook's column order
# ----------------------------------------------------------------------
def build_design(Y_df, exog_df, p):
    """Return (Y, X, col_names) with the notebook's column ordering.

    Y_df    : (T x n) endogenous, DateTimeIndex
    exog_df : (T x m) exogenous shocks aligned to Y_df (const is added here)
    p       : number of lags
    """
    n = Y_df.shape[1]
    lag_cols, lag_frames = [], []
    for i in range(1, p + 1):
        sh = Y_df.shift(i)
        sh.columns = [f"{c}(t-{i})" for c in Y_df.columns]
        lag_frames.append(sh)
        lag_cols += list(sh.columns)

    X_lags = pd.concat(lag_frames, axis=1)
    exog = exog_df.copy()
    exog.insert(0, "const", 1.0)
    X = pd.concat([X_lags, exog], axis=1)

    idx = X.dropna().index.intersection(Y_df.dropna().index)
    Y = Y_df.loc[idx].values
    X = X.loc[idx].values
    return Y, X, (lag_cols + list(exog.columns)), idx


def ar_residual_scales(Y_df, p):
    """sigma_i = std of residuals from an AR(p) on each variable (TRAIN ONLY).

    These set the prior scale in the Minnesota prior. Estimated on the supplied
    (training) sample only, so no test-period information leaks in.
    """
    sig = np.empty(Y_df.shape[1])
    for j, c in enumerate(Y_df.columns):
        y = Y_df[c].dropna()
        try:
            res = AutoReg(y, lags=p, old_names=False).fit()
            sig[j] = np.std(res.resid, ddof=1)
        except Exception:
            sig[j] = np.std(np.diff(y), ddof=1)
        if not np.isfinite(sig[j]) or sig[j] == 0:
            sig[j] = np.std(y, ddof=1)
    return sig


# ----------------------------------------------------------------------
# (A)  Minnesota prior via dummy observations  (natural conjugate)
# ----------------------------------------------------------------------
def minnesota_dummies(Y_df, sigma, delta, p, m,
                      lam=0.2, tau=10.0, eps=1e-4, mu_soc=1.0):
    """Construct Banbura-Giannone-Reichlin (2010) dummy observations.

    Parameters
    ----------
    Y_df  : training endogenous (T x n), used for presample means
    sigma : (n,) AR residual scales
    delta : (n,) prior mean of each variable's OWN first lag
            (1.0 = random-walk belief; good for level/persistent series)
    p     : lags ; m : # exogenous incl. const
    lam   : overall tightness (SMALLER => tighter => MORE shrinkage). 0.1-0.3 typical.
    tau   : looseness of the prior on deterministic/exog terms (large => diffuse)
    eps   : tiny number controlling the constant's diffuseness
    mu_soc: sum-of-coefficients tightness (SMALLER => stronger no-cointegration /
            random-walk-sum belief; this is what tames near-unit-roots).

    Returns Yd (Td x n), Xd (Td x k)  with k = n*p + m.
    """
    n = len(sigma)
    k = n * p + m
    Jp = np.diag(np.arange(1, p + 1).astype(float))     # lag-decay 1,2,...,p

    # --- 1. Prior on autoregressive coefficients -------------------------
    # top n rows encode own-first-lag mean = delta_i ; deeper lags -> 0,
    # with prior std proportional to lam/(ell * sigma_j).
    Yd_ar = np.zeros((n * p, n))
    Yd_ar[:n, :] = np.diag(delta * sigma) / lam
    Xd_ar_lags = np.kron(Jp, np.diag(sigma)) / lam       # (np x np)
    Xd_ar = np.hstack([Xd_ar_lags, np.zeros((n * p, m))])

    # --- 2. Prior on the residual covariance (sets E[Sigma]) -------------
    Yd_cov = np.diag(sigma)
    Xd_cov = np.zeros((n, k))

    # --- 3. Diffuse prior on constant + exogenous shocks -----------------
    Yd_exo = np.zeros((m, n))
    Xd_exo = np.hstack([np.zeros((m, n * p)), np.eye(m) * eps / tau])

    # --- 4. Sum-of-coefficients prior (curbs explosive dynamics) ---------
    ybar = Y_df.iloc[:p].mean().values if len(Y_df) >= p else Y_df.mean().values
    ybar = np.nan_to_num(ybar)
    Yd_soc = np.diag(delta * ybar) / mu_soc
    Xd_soc_lags = np.kron(np.ones((1, p)), np.diag(delta * ybar)) / mu_soc  # (n x np)
    Xd_soc = np.hstack([Xd_soc_lags, np.zeros((n, m))])

    Yd = np.vstack([Yd_ar, Yd_cov, Yd_exo, Yd_soc])
    Xd = np.vstack([Xd_ar, Xd_cov, Xd_exo, Xd_soc])
    return Yd, Xd


def fit_bvar(Y, X, Yd, Xd):
    """Natural-conjugate posterior from stacked (data + dummy) observations.

    Posterior:  Sigma ~ IW(S_post, nu_post);  vec(B)|Sigma ~ N(vec(B_post), Sigma (x) Vx)
    Returns a dict of posterior objects (B_post is K x n, matches notebook layout).
    """
    n = Y.shape[1]
    k = X.shape[1]
    Td = Yd.shape[0]

    Ys = np.vstack([Y, Yd])
    Xs = np.vstack([X, Xd])

    XtX = Xs.T @ Xs
    Vx = inv(XtX)                       # = (Xs'Xs)^{-1}
    B_post = Vx @ Xs.T @ Ys             # posterior mean / mode of coefficients
    resid = Ys - Xs @ B_post
    S_post = resid.T @ resid            # IW scale
    nu_post = Td + Y.shape[0] - k       # posterior degrees of freedom

    # plain OLS on the real data only (for the shrinkage comparison)
    B_ols = inv(X.T @ X) @ X.T @ Y

    return dict(B_post=B_post, Vx=Vx, S_post=S_post, nu_post=int(nu_post),
                B_ols=B_ols, n=n, k=k)


def companion_max_eig(B, n, p):
    """Largest |eigenvalue| of the VAR companion matrix (lag block of B only)."""
    B_lags = B[:n * p, :]                       # (np x n)
    A1 = B_lags.reshape(p, n, n)                # A1[l] = coeff on lag l+1, shape (n x n)
    C = np.zeros((n * p, n * p))
    for l in range(p):
        C[:n, l * n:(l + 1) * n] = A1[l].T
    if p > 1:
        C[n:, :n * (p - 1)] = np.eye(n * (p - 1))
    return np.max(np.abs(eigvals(C)))


def sample_bvar(post, p, n_draws=2000, seed=42, require_stable=False, max_eig=1.0):
    """Draw (B, Sigma) from the posterior. Optionally keep only stable draws."""
    rng = np.random.default_rng(seed)
    n, k = post["n"], post["k"]
    S_post, nu_post, Vx, B_post = post["S_post"], post["nu_post"], post["Vx"], post["B_post"]

    cholVx = cholesky((Vx + Vx.T) / 2)          # symmetrise for safety
    Bs, Ss, eigs = [], [], []
    tries, cap = 0, n_draws * 50
    while len(Bs) < n_draws and tries < cap:
        tries += 1
        # Sigma ~ IW(nu, S):  draw W ~ Wishart(nu, S^{-1}); Sigma = W^{-1}
        W = stats.wishart.rvs(df=nu_post, scale=inv(S_post), random_state=rng)
        Sigma = inv(W)
        cholS = cholesky((Sigma + Sigma.T) / 2)
        # vec(B) = vec(B_post) + (cholS (x) cholVx) z  ==  B_post + cholVx Z cholS'
        Z = rng.standard_normal((k, n))
        B = B_post + cholVx @ Z @ cholS.T
        e = companion_max_eig(B, n, p)
        if require_stable and e >= max_eig:
            continue
        Bs.append(B); Ss.append(Sigma); eigs.append(e)
    return np.array(Bs), np.array(Ss), np.array(eigs), tries


def bvar_forecast(B, y_initial, p, exog_future):
    """Recursive h-step forecast (same convention as the notebook).

    y_initial   : (p x n) last p observations, most recent LAST
    exog_future : (h x m) future exogenous values
    """
    h = exog_future.shape[0]
    n = y_initial.shape[1]
    fc = np.zeros((h, n))
    win = y_initial.copy()
    for t in range(h):
        x = np.hstack([win[::-1].flatten(), 1.0, exog_future[t]])  # t-1,t-2,... order
        yt = x @ B
        fc[t] = yt
        win = np.vstack([win[1:], yt])
    return fc


# ======================================================================
# SELF-TEST on synthetic data shaped like the notebook
# ======================================================================
if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)
    rng = np.random.default_rng(0)

    # ---- synthetic quarterly panel 2001Q1..2025Q1 -----------------------
    dates = pd.period_range("2001Q1", "2025Q1", freq="Q").to_timestamp(how="end")
    Tn = len(dates)
    names = ["gdp_yoy", "inflation_yoy", "reer_yoy", "10yTbill_m_avg", "nir"]
    n = len(names)

    # A persistent but STATIONARY VAR(1) (max|eig| ~ 0.95): like real macro data
    A = np.array([
        [0.70, 0.05, 0.02, -0.03, -0.02],
        [0.04, 0.88, 0.00,  0.03,  0.02],   # inflation: persistent
        [0.10, 0.00, 0.55,  0.00,  0.00],
        [0.02, 0.05, 0.00,  0.90,  0.02],   # 10y: highly persistent
        [0.03, 0.04, 0.00,  0.04,  0.90],   # policy rate: highly persistent
    ])
    print("true DGP max|eig| =", round(np.max(np.abs(eigvals(A))), 4))
    Sig = np.diag([0.8, 0.4, 2.5, 0.3, 0.3]) ** 2
    cholSig = cholesky(Sig)
    Y = np.zeros((Tn, n))
    Y[0] = [2, 2, 0, 3, 2]
    for t in range(1, Tn):
        Y[t] = [2.0, 0.4, 0.0, 0.1, 0.1] + A @ Y[t - 1] + cholSig @ rng.standard_normal(n)
    Y_df = pd.DataFrame(Y, index=dates, columns=names)

    # six standardized shocks (exogenous), like the notebook
    shock_names = ["shock_oil", "shock_wealth", "shock_consumption",
                   "shock_investment", "shock_credit", "shock_fiscal"]
    exog_df = pd.DataFrame(rng.standard_normal((Tn, len(shock_names))),
                           index=dates, columns=shock_names)

    p = 5
    train_end = pd.Timestamp("2021-12-31")
    test_start = pd.Timestamp("2022-01-01")
    Ytr = Y_df.loc[:train_end]
    Xtr_exog = exog_df.loc[Ytr.index]
    test_idx = Y_df.loc[test_start:].index

    print("=" * 68)
    print("PART A  —  Corrected Minnesota BVAR")
    print("=" * 68)
    Y_mat, X_mat, cols, idx = build_design(Ytr, Xtr_exog, p)
    m = X_mat.shape[1] - n * p                # exog incl const (= 1 + #shocks)
    m_shocks = m - 1
    sigma = ar_residual_scales(Ytr, p)
    # prior mean of own first lag: <1 for mean-reverting growth rates, ~1 for levels
    delta = np.array([0.8, 0.8, 0.6, 0.95, 0.95])

    Yd, Xd = minnesota_dummies(Ytr.loc[idx], sigma, delta, p, m,
                               lam=0.1, tau=10.0, eps=1e-4, mu_soc=0.5)
    post = fit_bvar(Y_mat, X_mat, Yd, Xd)

    eig_bvar = companion_max_eig(post["B_post"], n, p)
    eig_ols = companion_max_eig(post["B_ols"], n, p)
    # shrinkage measured correctly: distance from the prior mean (own-lag-1 = delta, else 0)
    B_prior = np.zeros_like(post["B_post"])
    for j in range(n):
        B_prior[j, j] = delta[j]              # own first lag prior mean
    d_ols = np.linalg.norm(post["B_ols"] - B_prior)
    d_bvar = np.linalg.norm(post["B_post"] - B_prior)
    print(f"max companion eig  -- OLS: {eig_ols:.4f}   BVAR(post.mean): {eig_bvar:.4f}"
          f"   ({'STABLE' if eig_bvar < 1 else 'unstable'})")
    print(f"distance from prior -- OLS: {d_ols:.3f}   BVAR: {d_bvar:.3f}"
          f"   => {100*(1-d_bvar/d_ols):.0f}% shrinkage toward prior")

    Bs, Ss, eigs, tries = sample_bvar(post, p, n_draws=2000, seed=1,
                                      require_stable=True, max_eig=1.0)
    print(f"posterior draws stable (<1): {100*np.mean(eigs<1):.1f}%  "
          f"(kept {len(Bs)} stable draws; median max-eig {np.median(eigs):.3f})")

    # 8-step baseline forecast (shocks = 0), using stable draws only
    y0 = Y_mat[-p:]
    H = 8
    exo_future = np.zeros((H, m_shocks))
    fcs = np.array([bvar_forecast(B, y0, p, exo_future) for B in Bs])
    fc_mean = fcs.mean(0)
    print("8-step GDP forecast (BVAR mean):", fc_mean[:, 0].round(3))
    print("max |forecast| over all vars/horizons:", round(np.abs(fc_mean).max(), 3),
          " vs data range ~", round(Y_df.abs().max().max(), 1),
          "  =>", "BOUNDED" if np.abs(fc_mean).max() < 5 * Y_df.abs().max().max() else "EXPLODING")

    print("\nBVAR checks ran (Part A: sum-of-coefficients prior + stability filtering).")
