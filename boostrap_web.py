#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap EUR & USD curves and basic FX, resilient to ECB fetch failures.

Data sources :
- FRED fredgraph CSV: USD Treasuries (DGS*), FX DEXUSEU, EURIBOR 3M (EUR3MTD156N), USD 3M (DGS3MO/TB3MS),
  Euro Area 10Y gov yield from OECD via FRED (IRLTLT01EZM156N).
- ECB quickview (optional, best-quality EUR curve). If ECB is unavailable, we synthesize the EUR curve
  by adding tenor-dependent EUR–USD spreads (anchored at 3M & 10Y) to the USD curve fetched from FRED.

Outputs :
  data/usd_curve.csv   -> columns: tenor_years, yield_dec
  data/eur_curve.csv   -> columns: tenor_years, yield_dec
  data/fx_meta.json    -> pair, atm_iv, r_dom_guess, r_for_guess, spot, spot_date, timestamp, sources
  data/cds_proxy.json  -> cds_spread_bps, lgd
"""

import os, json, math, datetime as dt, urllib.parse
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

# -------------------- CONFIG --------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "/data")
PAIR = "EUR/USD"
ASSUMED_ATM_IV = 0.10          # set to None if you prefer not to pass IV
DEFAULT_CDS_BPS = 150.0        # simple proxy

USD_SERIES = {
    1/12: "DGS1MO",
    0.25: "DGS3MO",
    0.5:  "DGS6MO",
    1.0:  "DGS1",
    2.0:  "DGS2",
    3.0:  "DGS3",
    5.0:  "DGS5",
    7.0:  "DGS7",
    10.0: "DGS10",
    20.0: "DGS20",
    30.0: "DGS30",
}

ECB_TENOR_MAP = {  # only used if ECB endpoint is up
    1/12: "1M", 0.25: "3M", 0.5: "6M", 1.0: "1Y",
    2.0: "2Y", 3.0: "3Y", 5.0: "5Y", 7.0: "7Y",
    10.0:"10Y", 15.0:"15Y", 20.0:"20Y", 30.0:"30Y",
}

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def _fred_csv(series_id: str) -> pd.DataFrame:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + urllib.parse.quote(series_id)
    return pd.read_csv(url)

def _last_num(s: pd.Series) -> Optional[float]:
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    return None if s2.empty else float(s2.iloc[-1])

def _latest_date(df: pd.DataFrame) -> Optional[dt.date]:
    if "DATE" in df.columns:
        d = pd.to_datetime(df["DATE"], errors="coerce").dropna()
        if not d.empty:
            return d.iloc[-1].date()
    return None

# -------------------- USD curve (FRED) --------------------
def fetch_usd_curve_from_fred() -> Dict[str, Tuple[float, float]]:
    out = {}
    for ten, code in USD_SERIES.items():
        try:
            df = _fred_csv(code)
            v = _last_num(df[code])
            if v is not None and math.isfinite(v):
                out[f"USD_{code}"] = (float(ten), v/100.0)  # % -> decimal
        except Exception:
            continue
    if not out:
        raise RuntimeError("Failed to fetch USD curve from FRED.")
    return out

# -------------------- ECB EUR curve (best quality, optional) --------------------
def fetch_eur_curve_from_ecb() -> Dict[str, Tuple[float, float]]:
    out = {}
    for ty, code in ECB_TENOR_MAP.items():
        url = (
            "https://sdw.ecb.europa.eu/quickviewexport.do"
            f"?trans=N&downloadFileName=yc&series=YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{code}&type=csv"
        )
        try:
            df = pd.read_csv(url)
            # pick the last numeric column (OBS_VALUE in most exports)
            val_col = None
            for c in df.columns[::-1]:
                if pd.api.types.is_numeric_dtype(df[c]):
                    val_col = c; break
            if val_col is None:
                for c in ["OBS_VALUE","VALUE"]: 
                    if c in df.columns: val_col = c; break
            if val_col is None:
                continue
            v = _last_num(df[val_col])
            if v is None or not math.isfinite(v): 
                continue
            out[f"EUR_ECB_{code}"] = (float(ty), v/100.0)
        except Exception:
            continue
    return out  # may be empty if ECB is down

# -------------------- EUR via FRED spreads (Plan B) --------------------
def fetch_eur_us_spread_anchors() -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Returns (eur3m, usd3m, eur10y, usd10y) in decimal yields if available.
    EUR 3M: FRED EUR3MTD156N (3-Month Euribor)
    USD 3M: FRED DGS3MO (fallback TB3MS)
    EUR 10Y: FRED IRLTLT01EZM156N (Euro Area long-term, ~10Y, OECD)
    USD 10Y: FRED DGS10
    """
    eur3m = usd3m = eur10y = usd10y = None
    # EUR 3M
    try:
        df = _fred_csv("EUR3MTD156N")
        eur3m = _last_num(df["EUR3MTD156N"])
        if eur3m is not None: eur3m /= 100.0
    except Exception:
        pass
    # USD 3M
    for code in ["DGS3MO","TB3MS"]:
        if usd3m is None:
            try:
                df = _fred_csv(code)
                v = _last_num(df[code])
                if v is not None: usd3m = v/100.0
            except Exception:
                continue
    # EUR 10Y (OECD via FRED)
    try:
        df = _fred_csv("IRLTLT01EZM156N")
        v = _last_num(df["IRLTLT01EZM156N"])
        if v is not None: eur10y = v/100.0
    except Exception:
        pass
    # USD 10Y
    try:
        df = _fred_csv("DGS10")
        v = _last_num(df["DGS10"])
        if v is not None: usd10y = v/100.0
    except Exception:
        pass
    return eur3m, usd3m, eur10y, usd10y

def synthesize_eur_curve_from_usd_spreads(usd_curve: Dict[str, Tuple[float,float]]) -> Dict[str, Tuple[float,float]]:
    """
    Use EUR–USD spreads at 3M and 10Y, interpolate spread by tenor, add to USD curve.
    If 10Y spread is unavailable, use flat spread = EUR3M-USD3M.
    """
    eur3m, usd3m, eur10y, usd10y = fetch_eur_us_spread_anchors()
    tenors = sorted([ten for (_k,(ten,_)) in usd_curve.items()])

    # determine spreads
    s3m = (eur3m - usd3m) if (eur3m is not None and usd3m is not None) else None
    s10 = (eur10y - usd10y) if (eur10y is not None and usd10y is not None) else None

    out = {}
    for lbl, (ten, y_usd) in usd_curve.items():
        if s3m is None and s10 is None:
            # no info -> leave unchanged (worst-case)
            y_eur = y_usd
        elif s10 is None:
            # flat spread from 3M
            y_eur = y_usd + s3m
        else:
            # linearly blend spread between 3M (0.25y) and 10Y
            t_lo, t_hi = 0.25, 10.0
            if ten <= t_lo:
                spread = s3m if s3m is not None else s10
            elif ten >= t_hi:
                spread = s10
            else:
                # if s3m missing, use s10 flat; else interpolate
                if s3m is None:
                    spread = s10
                else:
                    w = (ten - t_lo) / (t_hi - t_lo)
                    spread = (1-w)*s3m + w*s10
            y_eur = y_usd + spread
        out[f"EUR_SYN_{ten:g}y"] = (ten, float(y_eur))
    return out

# -------------------- FX meta (FRED) --------------------
def fetch_fx_meta():
    spot = None; spot_date = None
    try:
        df = _fred_csv("DEXUSEU")  # USD per EUR
        s = pd.to_numeric(df["DEXUSEU"], errors="coerce").dropna()
        if not s.empty:
            spot = float(s.iloc[-1])
            spot_date = _latest_date(df)
    except Exception:
        pass
    return spot, spot_date

# -------------------- Utilities --------------------
def _save_curve_csv(curve: Dict[str, Tuple[float,float]], path: str):
    df = pd.DataFrame([{"tenor_years":ten, "yield_dec":y} for (_k,(ten,y)) in curve.items()])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sort_values("tenor_years")
    df.to_csv(path, index=False)

def _interp_rate(curve: Dict[str, Tuple[float,float]], target: float) -> float:
    pts = sorted([(ten,y) for (_k,(ten,y)) in curve.items()])
    if not pts:
        return 0.02
    # exact match?
    for ten,y in pts:
        if abs(ten - target) < 1e-12:
            return float(y)
    # linear interpolation across tenor
    lo = max([t for (t,_) in pts if t <= target], default=pts[0][0])
    hi = min([t for (t,_) in pts if t >= target], default=pts[-1][0])
    ylo = dict(pts)[lo]; yhi = dict(pts)[hi]
    if lo == hi:
        return float(ylo)
    w = (target - lo) / (hi - lo)
    return float((1-w)*ylo + w*yhi)

# -------------------- Main --------------------
def main():
    ensure_data_dir()

    # 1) USD curve (from FRED)
    usd_curve = fetch_usd_curve_from_fred()

    # 2) EUR curve: try ECB; if empty, synthesize from USD using FRED spreads
    eur_curve = fetch_eur_curve_from_ecb()
    used_source = "ECB quickview Svensson"
    if not eur_curve:
        eur_curve = synthesize_eur_curve_from_usd_spreads(usd_curve)
        used_source = "FRED EUR proxies (EUR3MTD156N & IRLTLT01EZM156N) spread over USD curve"

    # 3) Save both curves
    usd_path = os.path.join(DATA_DIR, "usd_curve.csv")
    eur_path = os.path.join(DATA_DIR, "eur_curve.csv")
    _save_curve_csv(usd_curve, usd_path)
    _save_curve_csv(eur_curve, eur_path)

    # 4) FX meta
    spot, spot_date = fetch_fx_meta()
    r_usd_1y = _interp_rate(usd_curve, 1.0)
    r_eur_1y = _interp_rate(eur_curve, 1.0)
    meta = {
        "pair": PAIR,
        "atm_iv": ASSUMED_ATM_IV,
        "r_dom_guess": float(r_usd_1y),
        "r_for_guess": float(r_eur_1y),
        "spot": spot,
        "spot_date": spot_date.isoformat() if isinstance(spot_date, dt.date) else None,
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "sources": {
            "usd_curve": "FRED Treasury (DGS*) fredgraph.csv",
            "eur_curve": used_source,
            "fx_spot": "FRED DEXUSEU fredgraph.csv"
        }
    }
    with open(os.path.join(DATA_DIR, "fx_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 5) CDS proxy
    with open(os.path.join(DATA_DIR, "cds_proxy.json"), "w") as f:
        json.dump({"cds_spread_bps": float(DEFAULT_CDS_BPS), "lgd": 0.60}, f, indent=2)

    print("Saved:")
    print(" -", usd_path)
    print(" -", eur_path)
    print(" -", os.path.join(DATA_DIR, "fx_meta.json"))
    print(" -", os.path.join(DATA_DIR, "cds_proxy.json"))
    print(f"EUR curve source: {used_source}")

if __name__ == "__main__":
    main()
