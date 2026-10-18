"""
Real-time macro inputs for Study 4 (see ../PREREG.md and its amendment).

GDP and core PCE come from ALFRED vintages through Study 3's vintage code and cache
(studies/varx_permutation_study/src/varx_realtime.py). The fed funds rate and WTI oil
(market prices, not revised) and the BIS real broad dollar (lightly revised) are taken
at final values from FRED, cut at quarter t-1 like the vintage series. TB3MS, also
final, is the risk-free rate.
"""
import datetime as dt
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "studies" / "varx_permutation_study" / "src"))
import varx_realtime as vr  # noqa: E402

VINTAGE = {"GDP": "GDPC1", "INF": "PCEPILFE"}
FINAL = {"POLICY": "FEDFUNDS", "FX": "RBUSBIS", "OIL": "WTISPLC"}
STUDY_DATA = Path(__file__).resolve().parents[1] / "data"


def fred_final(sid):
    """Final-vintage FRED series, keyless CSV endpoint, cached in the study's data folder."""
    STUDY_DATA.mkdir(exist_ok=True)
    path = STUDY_DATA / f"{sid}.csv"
    if not path.exists():
        r = subprocess.run(["curl", "-s", "-f", "--max-time", "60", "-o", str(path),
                            f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"])
        if r.returncode != 0 or not path.read_bytes().startswith(b"observation_date"):
            path.unlink(missing_ok=True)
            raise RuntimeError(f"FRED download of {sid} failed")
    df = pd.read_csv(path)
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values, index=pd.to_datetime(df.iloc[:, 0]))
    return s.dropna()


def first_vintage(sid, lo=dt.date(1985, 1, 1), hi=dt.date(2026, 10, 1)):
    """Earliest ALFRED vintage of `sid`, by bisection on whether a vintage exists by that date."""
    if vr.fetch_vintage(sid, hi.isoformat()).empty:
        raise RuntimeError(f"{sid}: no vintage at all")
    while (hi - lo).days > 1:
        mid = lo + (hi - lo) / 2
        if vr.fetch_vintage(sid, mid.isoformat()).empty:
            lo = mid
        else:
            hi = mid
    return hi


def load_finals():
    return {name: fred_final(sid) for name, sid in FINAL.items()}


def panel_at(t, finals):
    """Stationary macro panel known at the end of quarter t, ending at quarter t-1.
    Vintage series at the vintage dated 45 days after t-1 closed (Study 3's standpoint).
    Study 1's transforms: 400 x dlog for GDP, core PCE and the dollar, the fed funds
    level, 100 x dlog oil. Quarter-start stamps. Stops if any input ends before t-1."""
    prev_end = (t - pd.offsets.QuarterEnd(1)).date()
    vdate = vr._vintage_date(prev_end)
    last_q = pd.Timestamp(prev_end).to_period("Q").to_timestamp()
    lv = {}
    for name, sid in VINTAGE.items():
        s = vr.fetch_vintage(sid, vdate)
        if s.empty:
            raise RuntimeError(f"{sid} has no vintage on {vdate}")
        lv[name] = s.resample("QS").mean()
    for name, s in finals.items():
        lv[name] = s[s.index < last_q + pd.DateOffset(months=3)].resample("QS").mean()   # months up to t-1
    lv = pd.DataFrame(lv)
    p = pd.DataFrame({"GDP": 400 * np.log(lv.GDP).diff(), "INF": 400 * np.log(lv.INF).diff(),
                      "POLICY": lv.POLICY, "FX": 400 * np.log(lv.FX).diff(),
                      "OIL": 100 * np.log(lv.OIL).diff()})
    p = p.loc["1994-04-01":last_q].dropna()
    if p.index[-1] != last_q:
        raise RuntimeError(f"vintage {vdate}: complete data end at {p.index[-1]:%Y-%m}, not at quarter t-1")
    return p, vdate


if __name__ == "__main__":
    for name, sid in {**VINTAGE, **FINAL}.items():
        print(f"{name:6s} {sid:9s} first ALFRED vintage {first_vintage(sid)}"
              + ("" if name in VINTAGE else "  (used at final values)"))
