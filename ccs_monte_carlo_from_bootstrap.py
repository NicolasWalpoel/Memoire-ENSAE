# -*- coding: utf-8 -*-
"""
------------
- Exposes:
    * interp_zero_rate(curve_df, t_years): linear interpolation on a zero curve DF
    * load_inputs_via_adapter(as_of=None, prefer_files=True): returns the minimal
      dict of inputs required by the slim/full CCS pricers

Inputs it can read
-------------------------------
If present in the current working directory, these files will be used:
- usd_zero_curve.csv : columns ['tenor_years','zero_rate']
- eur_zero_curve.csv : columns ['tenor_years','zero_rate']
- corr_matrix.csv    : 3x3 numeric CSV for the order [FX, USD, EUR]
- spot_fx.txt        : a single number (domestic per foreign), e.g., 1.08

If any are missing, we use:
- Flat USD curve @ 3%, EUR curve @ 2%, tenors 0..30y
- spot_fx = 1.08
- vols: fx=12%, usd_rate=1%, eur_rate=1%
- corr_matrix default built from typical signs:
    [ [1,      0.20, -0.10],   # FX with USD/EUR rates
      [0.20,   1,    -0.10],
      [-0.10, -0.10,  1   ] ]

------------------
'spot_fx' (float)
'usd_zero_curve' (pd.DataFrame: tenor_years, zero_rate)
'eur_zero_curve' (pd.DataFrame: tenor_years, zero_rate)
'basis_curve' (pd.DataFrame: tenor_years, basis_bp)  # filled with zeros; optional in pricer
'fx_vol', 'usd_rate_vol', 'eur_rate_vol' (floats)
'corr_matrix' (np.ndarray 3x3) ordered as [FX, USD, EUR]
"""

from pathlib import Path
import numpy as np
import pandas as pd

def _maybe_load_curve(csv_name: str, fallback_rate: float) -> pd.DataFrame:
    p = Path(csv_name)
    if p.exists():
        df = pd.read_csv(p)
        # normalize column names
        cols = {c.lower(): c for c in df.columns}
        # try to map to expected names
        if 'tenor_years' not in cols and 'tenor' in cols:
            df.rename(columns={cols['tenor']: 'tenor_years'}, inplace=True)
        if 'zero_rate' not in cols and 'rate' in cols:
            df.rename(columns={cols['rate']: 'zero_rate'}, inplace=True)
        if 'tenor_years' not in df.columns or 'zero_rate' not in df.columns:
            raise ValueError(f"{csv_name} must have columns tenor_years, zero_rate")
        df = df[['tenor_years','zero_rate']].astype(float).sort_values('tenor_years')
        return df.reset_index(drop=True)
    # fallback: flat curve 0..30y
    tenors = np.array([0, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)
    rates = np.full_like(tenors, float(fallback_rate), dtype=float)
    return pd.DataFrame({'tenor_years': tenors, 'zero_rate': rates})

def _maybe_load_spot_fx(txt_name: str, fallback: float) -> float:
    p = Path(txt_name)
    if p.exists():
        try:
            return float(p.read_text().strip())
        except Exception:
            pass
    return float(fallback)

def _maybe_load_corr(csv_name: str, fallback: np.ndarray) -> np.ndarray:
    p = Path(csv_name)
    if p.exists():
        arr = np.loadtxt(p, delimiter=',')
        arr = np.array(arr, dtype=float)
        if arr.shape != (3,3):
            raise ValueError("corr_matrix.csv must be 3x3 for [FX, USD, EUR]")
        return arr
    return fallback

def interp_zero_rate(curve_df: pd.DataFrame, t_years: float) -> float:
    """
    Simple linear interpolation on 'tenor_years' for the 'zero_rate' column.
    Extrapolates flat beyond boundaries.
    """
    x = curve_df['tenor_years'].values.astype(float)
    y = curve_df['zero_rate'].values.astype(float)
    if t_years <= x.min():
        return float(y[0])
    if t_years >= x.max():
        return float(y[-1])
    return float(np.interp(t_years, x, y))

def load_inputs_via_adapter(as_of=None, prefer_files: bool = True) -> dict:
    # 1) curves
    usd_curve = _maybe_load_curve('usd_zero_curve.csv', fallback_rate=0.03)
    eur_curve = _maybe_load_curve('eur_zero_curve.csv', fallback_rate=0.02)

    # 2) spot FX
    spot_fx = _maybe_load_spot_fx('spot_fx.txt', fallback=1.08)

    # 3) vols
    fx_vol = 0.12
    usd_rate_vol = 0.01
    eur_rate_vol = 0.01

    # 4) corr matrix [FX, USD, EUR]
    corr_fallback = np.array([[1.0,  0.20, -0.10],
                              [0.20, 1.0,  -0.10],
                              [-0.10,-0.10, 1.0 ]], dtype=float)
    corr_matrix = _maybe_load_corr('corr_matrix.csv', corr_fallback)

    # 5) basis curve (optional, not used by slim pricer) — keep zeros for shape
    # basis_curve = usd_curve.copy()
    # basis_curve.rename(columns={'zero_rate': 'basis_bp'}, inplace=True)
    # basis_curve['basis_bp'] = 0.0

    return {
        'spot_fx': spot_fx,
        'usd_zero_curve': usd_curve,
        'eur_zero_curve': eur_curve,
        'basis_curve': basis_curve,
        'fx_vol': fx_vol,
        'usd_rate_vol': usd_rate_vol,
        'eur_rate_vol': eur_rate_vol,
        'corr_matrix': corr_matrix,
    }

if __name__ == "__main__":
    d = load_inputs_via_adapter()
    print("spot_fx:", d['spot_fx'])
    print("usd_zero_curve head:\n", d['usd_zero_curve'].head())
    print("eur_zero_curve head:\n", d['eur_zero_curve'].head())
    print("corr_matrix:\n", d['corr_matrix'])
