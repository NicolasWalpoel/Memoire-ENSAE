#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portfolio: CVA, FVA, RWA with and without 80% insurance (cap $1,000,000)."""
import sys, copy
from dataclasses import replace
import numpy as np
import pandas as pd

sys.path.append("/mnt/data")
from ccs_monte_carlo_from_bootstrap import load_inputs_via_adapter
from ccs_pricer_full_slim import CrossCurrencySwapPricer, MCConfig
from ccs_pricer_mc_with_bootstrap_init import UserCCSConfig, build_models_from_inputs, build_ccs_spec

INS_COVER_PCT = 0.80
INS_CAP = 1_000_000.0

def build_portfolio_params(n_trades: int = 50, random_seed: int = 42):
    rng = np.random.default_rng(random_seed)
    return dict(
        maturities=rng.choice([2.0, 3.0, 5.0, 7.0, 10.0], size=n_trades),
        fx_scales=rng.uniform(0.9, 1.1, size=n_trades),
        dom_fix=rng.uniform(0.02, 0.06, size=n_trades),
        for_fix=rng.uniform(0.02, 0.06, size=n_trades),
        dom_spread=rng.uniform(-0.005, 0.01, size=n_trades),
        for_spread=rng.uniform(-0.005, 0.01, size=n_trades),
        cbk_x0=rng.uniform(-4.0, -2.0, size=n_trades),
        lgd=rng.uniform(0.4, 0.6, size=n_trades),
    )

def price_trade(base_inputs: dict, base_cfg: UserCCSConfig, params: dict, idx: int, n_paths: int = 100, dt: float = 1/52):
    cfg = replace(base_cfg,
                  maturity_years=float(params['maturities'][idx]),
                  domestic_pay_fix=True,
                  foreign_pay_fix=False,
                  domestic_fixed_rate=float(params['dom_fix'][idx]),
                  foreign_float_spread=float(params['for_spread'][idx]),
                  domestic_float_spread=float(params['dom_spread'][idx]),
                  cbk_x0=float(params['cbk_x0'][idx]),
                  lgd=float(params['lgd'][idx]))
    inputs = copy.deepcopy(base_inputs)
    inputs['spot_fx'] = float(base_inputs['spot_fx'] * params['fx_scales'][idx])
    rd_model, rf_model, fx_model, cr_model, corr = build_models_from_inputs(inputs, cfg)
    mc_cfg = MCConfig(n_paths=n_paths, dt=dt)
    pricer = CrossCurrencySwapPricer(rd_model, rf_model, fx_model, cr_model, corr, build_ccs_spec(cfg), mc_cfg, base_cfg.fund_spread, base_cfg.alpha_ead)
    out = pricer.price(include_cva=True, include_fva=True, irb_params=None)
    return cfg, inputs['spot_fx'], out

def run_portfolio(n_trades, n_paths, seed: int = 42) -> pd.DataFrame:
    base_inputs = load_inputs_via_adapter(as_of=None, prefer_files=True)
    base_cfg = UserCCSConfig()
    params = build_portfolio_params(n_trades=n_trades, random_seed=seed)

    rows = []
    for i in range(n_trades):
        cfg, fx_spot, out = price_trade(base_inputs, base_cfg, params, i, n_paths=n_paths)
        cva0 = float(out.cva)
        fva0 = float(out.fva)
        rwa0 = float(out.rwa)
        eepe0 = float(out.eepe)
        # Proportional reduction on EPE-linked quantities, capped via CVA
        red_factor = min(INS_COVER_PCT, INS_CAP / cva0)
        cva_ins = cva0 * (1.0 - red_factor)
        fva_ins = fva0 * (1.0 - red_factor)
        rwa_ins = rwa0 * (1.0 - red_factor)
        rows.append({
            "trade_id": i+1,
            "mat_years": cfg.maturity_years,
            "fx_spot": fx_spot,
            "dom_fixed": cfg.domestic_fixed_rate,
            "for_spread": cfg.foreign_float_spread,
            "dom_spread": cfg.domestic_float_spread,
            "lgd": cfg.lgd,
            "eepe_no_ins": eepe0,
            "cva_no_ins": cva0,
            "cva_insured": cva_ins,
            "cva_reduction": cva0 - cva_ins,
            "fva_no_ins": fva0,
            "fva_insured": fva_ins,
            "fva_reduction": fva0 - fva_ins,
            "rwa_no_ins": rwa0,
            "rwa_insured": rwa_ins,
            "rwa_reduction": rwa0 - rwa_ins,
            "pv_gross": float(out.pv_gross_mean),
            "mtm0": float(out.mtm_paths.mean(axis=0)[0]),
        })
    return pd.DataFrame(rows)

def main():
    df = run_portfolio(n_trades=200, n_paths=1000, seed=42)
    df = df.sort_values("trade_id")
    df.to_csv("/Users/walpoel/Desktop/taf sg/ensae final/data/portfolio_results.csv", index=False)
    try:
        df.to_parquet("/Users/walpoel/Desktop/taf sg/ensae final/data/portfolio_results.parquet", index=False)
    except Exception as e:
        print("Parquet save skipped:", e)
    totals = df[['cva_no_ins','cva_insured','cva_reduction','fva_no_ins','fva_insured','fva_reduction','rwa_no_ins','rwa_insured','rwa_reduction']].sum()
    print('Portfolio totals:', totals.to_dict())

if __name__ == "__main__":
    main()