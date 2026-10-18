"""
Study 4: economic value of the forecasts (see ../PREREG.md).

    python studies/economic_value/src/run_study.py --bbg PATH/bbg_tr_monthly.csv

At each decision (end of quarter t), from data published by then:
  forecasts   CORE+OIL(x) and a univariate AR(BIC) per core variable, refit on the
              real-time panel ending at t-1, one quarter ahead (quarter t)
  regression  next-quarter excess return on the four core forecasts, expanding, pairs
              (forecast made at s, return in s+1) completed by t, 40-quarter minimum
  benchmark   the historical mean excess return to t
  investor    mean-variance, risk aversion 3, weight in [0, 1.5], variance of the trailing
              40 quarters of excess returns
Scored on the realised excess return in quarter t+1.

The Bloomberg file holds monthly total-return levels for SPXT, LUATTRUU and LUACTRUU.
It is read, never copied: only aggregate results are written to the repository.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.api import VAR

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import realtime as rt  # noqa: E402

STUDY = HERE.parent
RES, FIG, DATA = STUDY / "results", STUDY / "figures", STUDY / "data"
ASSETS = {"SPXT": "S&P 500", "LUATTRUU": "US Treasuries", "LUACTRUU": "US corporates"}
CORE = ["GDP", "INF", "POLICY", "FX"]
MIN_Q, GAMMA, W_MAX = 40, 3.0, 1.5
START = pd.Timestamp("2000-09-30")            # first decision: core PCE's first vintage covers 2000Q2


# ---------------------------------------------------------------- forecasts (Study 1)

def ar_bic_path(y, h, maxlag=8):
    """AR(p) with BIC-selected p (manual OLS), iterated h steps. Copied from src/var_horserace.py."""
    y = np.asarray(y, float); n = len(y)
    best = (np.inf, 1, None)
    for p in range(1, maxlag + 1):
        if n - p < p + 5: break
        Y = y[p:]
        X = np.column_stack([np.ones(n - p)] + [y[p - 1 - j:n - 1 - j] for j in range(p)])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta; s2 = max(resid @ resid / len(Y), 1e-12)
        bic = len(Y) * np.log(s2) + (p + 1) * np.log(len(Y))
        if bic < best[0]: best = (bic, p, beta)
    _, p, beta = best
    hist = list(y[-p:]); out = []
    for _ in range(h):
        x = np.array([1.0] + [hist[-1 - j] for j in range(p)])
        yh = float(x @ beta); out.append(yh); hist.append(yh)
    return np.array(out)


def fit_var(endog_arr, exog_arr=None):
    """Copied from src/var_horserace.py: lag by BIC up to 4, constant."""
    m = VAR(endog_arr, exog=exog_arr)
    res = m.fit(maxlags=4, ic="bic", trend="c")
    if res.k_ar == 0:
        res = m.fit(1, trend="c")
    return res


def forecasts_at(t, finals):
    """One-quarter-ahead forecasts of the four core variables made at the end of quarter t."""
    p, vdate = rt.panel_at(t, finals)
    ar = [ar_bic_path(p[v].values, 1)[0] for v in CORE]
    oil = ar_bic_path(p["OIL"].values, 1)                    # exogenous: AR-projected, as in Study 1
    res = fit_var(p[CORE].values, p[["OIL"]].values)
    var = res.forecast(p[CORE].values[-res.k_ar:], steps=1, exog_future=oil.reshape(-1, 1))[0]
    return {"decision": t, "vintage": vdate, **{f"VAR_{v}": x for v, x in zip(CORE, var)},
            **{f"AR_{v}": x for v, x in zip(CORE, ar)}}


# ---------------------------------------------------------------- returns

def excess_returns(bbg_path):
    """Quarterly excess returns over T-bills, quarter-end stamps. Stops on a missing series."""
    if not Path(bbg_path).exists():
        raise FileNotFoundError(f"Bloomberg file not found: {bbg_path}")
    raw = pd.read_csv(bbg_path)
    date = next(c for c in raw.columns if c.lower() in ("date", "dates"))
    lv = raw.set_index(pd.to_datetime(raw[date])).drop(columns=date)
    missing = [a for a in ASSETS if a not in lv.columns]
    if missing:
        raise RuntimeError(f"{bbg_path} has no column for {missing}")
    lv = lv[list(ASSETS)].apply(pd.to_numeric, errors="coerce")
    lv.index = lv.index.to_period("M").to_timestamp("M")
    q = lv.groupby(lv.index.to_period("Q")).last()
    q.index = q.index.to_timestamp("Q")
    ret = q.pct_change(fill_method=None)
    tb = rt.fred_final("TB3MS")
    rf = (tb.groupby(tb.index.to_period("Q")).mean() / 400)
    rf.index = rf.index.to_timestamp("Q")
    for a in ASSETS:
        print(f"  {a}: total-return levels from {lv[a].first_valid_index():%Y-%m}")
    return ret.sub(rf.reindex(ret.index), axis=0), rf


# ---------------------------------------------------------------- evaluation at one decision

def predict_at(t, fc, ex):
    """Predictions for quarter t+1 made at t, from forecasts and returns known at t."""
    f, r = fc[fc.index <= t], ex[ex.index <= t]
    out = {}
    for a in ASSETS:
        y = r[a].dropna()
        out[f"HM_{a}"] = y.mean()
        out[f"var40_{a}"] = y.iloc[-MIN_Q:].var() if len(y) >= MIN_Q else np.nan
        for m in ("VAR", "AR"):
            X = f[[f"{m}_{v}" for v in CORE]]
            # pairs: forecast made at s, return in s+1, both known at t
            pairs = pd.concat([X.shift(1), y], axis=1).dropna()
            pairs = pairs[pairs.index <= t]
            if len(pairs) < MIN_Q:
                out[f"{m}_{a}"] = np.nan
                continue
            b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(pairs)), pairs.iloc[:, :4].values]),
                                    pairs.iloc[:, 4].values, rcond=None)
            out[f"{m}_{a}"] = float(np.r_[1.0, X.loc[t].values] @ b)
    return out


def weight(mu, var):
    return float(np.clip(mu / (GAMMA * var), 0.0, W_MAX))


def clark_west(e_bench, e_model, f_bench, f_model):
    """Clark-West (2007) MSPE-adjusted test that the model beats the nested benchmark; one-sided p."""
    adj = e_bench ** 2 - (e_model ** 2 - (f_bench - f_model) ** 2)
    m = sm.OLS(adj, np.ones(len(adj))).fit()
    return float(m.tvalues[0]), float(1 - stats.norm.cdf(m.tvalues[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbg", required=True, help="Bloomberg monthly total-return levels (not committed)")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

    ex, rf = excess_returns(a.bbg)
    finals = rt.load_finals()
    decisions = [d for d in ex.index if d >= START]
    fc = pd.DataFrame([forecasts_at(t, finals) for t in decisions]).set_index("decision")
    print(f"  forecasts at {len(fc)} decisions, {fc.index[0]:%Y-%m} to {fc.index[-1]:%Y-%m}, "
          f"vintages {fc.vintage.iloc[0]} to {fc.vintage.iloc[-1]}")

    rows = []
    for t in fc.index:
        nxt = t + pd.offsets.QuarterEnd(1)
        if nxt not in ex.index:
            continue
        p = predict_at(t, fc, ex)
        if all(np.isnan(p[f"VAR_{x}"]) for x in ASSETS):
            continue
        rows.append({"decision": t, "target": nxt, **p, **{f"r_{x}": ex.loc[nxt, x] for x in ASSETS},
                     "rf": rf.get(nxt, np.nan)})
    oos = pd.DataFrame(rows).set_index("target")
    oos = oos.dropna(subset=[f"VAR_{x}" for x in ASSETS] + [f"AR_{x}" for x in ASSETS] + [f"var40_{x}" for x in ASSETS])
    print(f"  out of sample: {len(oos)} quarters, {oos.index[0]:%Y-%m} to {oos.index[-1]:%Y-%m}")

    table, cum = [], {}
    for x, label in ASSETS.items():
        r, hm = oos[f"r_{x}"], oos[f"HM_{x}"]
        e_hm = r - hm
        w_hm = np.array([weight(m, v) for m, v in zip(hm, oos[f"var40_{x}"])])
        port_hm = oos["rf"] + w_hm * r
        ce_hm = port_hm.mean() - GAMMA / 2 * port_hm.var()
        errs = {}
        for m in ("VAR", "AR"):
            fhat = oos[f"{m}_{x}"]
            e = r - fhat
            errs[m] = e
            t_cw, p_cw = clark_west(e_hm.values, e.values, hm.values, fhat.values)
            w = np.array([weight(mu, v) for mu, v in zip(fhat, oos[f"var40_{x}"])])
            port = oos["rf"] + w * r
            ce = port.mean() - GAMMA / 2 * port.var()
            table.append({"asset": x, "label": label, "model": "CORE+OIL(x)" if m == "VAR" else "AR",
                          "quarters": len(r), "oos_r2": 1 - (e ** 2).sum() / (e_hm ** 2).sum(),
                          "cw_t": t_cw, "cw_p": p_cw, "ce_gain_ann_pct": 400 * (ce - ce_hm),
                          "mean_weight": w.mean()})
            cum[(x, m)] = (e_hm ** 2 - e ** 2).cumsum()
        d = errs["VAR"] ** 2 - errs["AR"] ** 2                       # Diebold-Mariano, h = 1
        dm = sm.OLS(d.values, np.ones(len(d))).fit()
        table[-2].update({"dm_vs_ar_t": float(dm.tvalues[0]),
                          "dm_vs_ar_p": float(2 * (1 - stats.norm.cdf(abs(dm.tvalues[0]))))})
    tab = pd.DataFrame(table)
    tab.round(4).to_csv(RES / "economic_value.csv", index=False)
    print(tab.round(3).to_string(index=False))
    print(f"  tests: {int(tab.cw_p.notna().sum())} Clark-West + {int(tab.dm_vs_ar_p.notna().sum())} "
          f"Diebold-Mariano; Bonferroni p < {0.05 / 9:.4f}")

    # look-ahead check: predictions at four past decisions from inputs cut at that date
    checks = []
    for t in oos["decision"].iloc[[0, len(oos) // 3, 2 * len(oos) // 3, -1]]:
        cut_fin = {k: v[v.index <= t] for k, v in finals.items()}
        fc_cut = pd.DataFrame([forecasts_at(s, cut_fin) for s in fc.index[fc.index <= t]]).set_index("decision")
        pc = predict_at(t, fc_cut, ex[ex.index <= t])
        full = oos[oos["decision"] == t].iloc[0]
        keys = [k for k in pc if k.split("_")[0] in ("VAR", "AR", "HM", "var40")]
        diff = max(abs(pc[k] - full[k]) for k in keys)
        checks.append({"decision": t.date(), "max_abs_diff": diff})
    chk = pd.DataFrame(checks)
    chk.to_csv(RES / "lookahead_check.csv", index=False)
    print(chk.to_string(index=False))
    if (chk.max_abs_diff > 1e-12).any():
        raise RuntimeError("look-ahead check failed")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for ax, (x, label) in zip(axes, ASSETS.items()):
        for m, c in (("VAR", "C3"), ("AR", "C0")):
            ax.plot(cum[(x, m)].index, cum[(x, m)].values, color=c, lw=1.3, label="CORE+OIL(x)" if m == "VAR" else "AR")
        ax.axhline(0, color="k", lw=0.6); ax.set_title(label, fontsize=10); ax.grid(alpha=0.3)
    axes[0].set_ylabel("cumulative SSE(mean) - SSE(model)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Out-of-sample excess-return forecasts against the historical mean (rising = model better)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "welch_goyal.png", dpi=150)


if __name__ == "__main__":
    main()
