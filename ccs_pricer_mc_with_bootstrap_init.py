"""
Purpose
-------
- Initialize all model inputs from your existing "adapter" (the file that scrapes/
  bootstraps market data). By default, this script tries to import
  `load_inputs_via_adapter` from `ccs_monte_carlo_from_bootstrap.py`, which in turn
  attempts to read your modules (e.g. boostrap_yahoo_direct, etc.).

  • simulate joint FX + domestic/foreign short rates (Hull-White 1F + GK FX)
  • price ALL cashflows leg-by-leg along each simulated path and discount them
    consistently at each step
  • aggregate PVs and compute EE / ENE / EPE / EEPE, CVA, FVA
  • plot example diffusions, exposures, and valuation charges

How to run
----------
This will:
- Load market inputs from your adapter (with safe fallbacks if missing).
- Build the pricer objects from `ccs_pricer_full_slim.py`.
- Run a Monte Carlo and print the PV summary + plot the requested series.

Notes
-----
- You can edit the "USER CONFIG" section below (notional, fixed rates, spreads,
  maturities, MC paths/step, correlations, credit params).
- If your adapter already contains better Hull–White params (a, sigma, r0), you
  can change how they are set in `build_models_from_inputs`.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

# ---- Try to import the adapter loader (provided in your uploaded file) ----
try:
    from ccs_monte_carlo_from_bootstrap import load_inputs_via_adapter, interp_zero_rate
except Exception as e:
    raise ImportError(
        "Cannot import load_inputs_via_adapter from ccs_monte_carlo_from_bootstrap.py. "
        "Make sure this file is available next to this script."
    ) from e

# ---- Import the full CCS pricer & helpers ----
from ccs_pricer_full_slim import (
    HullWhite1F, GarmanKohlhagenFX, CreditBK,
    CCSLegSpec, CCSpec, MCConfig, CrossCurrencySwapPricer,
    plot_diffusions, plot_exposures, plot_mtm_mean, plot_valuation_charges
)

# =====================
# USER CONFIG
# =====================

@dataclass
class UserCCSConfig:
    # Instrument
    maturity_years: float = 5.0
    notional_domestic: float = 117_000_000.0
    notional_foreign: float = 100_000_000.0
    # Leg definitions
    # Domestic: pay fixed? (True => fixed leg; False => floating leg)
    domestic_pay_fix: bool = False
    domestic_fixed_rate: float = 0.03   # used only if pay_fix=True
    domestic_float_spread: float = 0.00 # used only if pay_fix=False (added to floating index)
    domestic_exchange: str = "maturity" # "none" | "maturity" | "both"
    domestic_daycount: str = "ACT/360"  # matches ccs_pricer_full_slim DayCountBasis literals

    # Foreign leg
    foreign_pay_fix: bool = False
    foreign_fixed_rate: float = 0.025
    foreign_float_spread: float = 0.019
    foreign_exchange: str = "maturity"
    foreign_daycount: str = "ACT/360"

    # Optional "leverage/spread" add-ons supported by ccs_pricer_full_slim
    domestic_rate_multiplier: float = 1.0
    domestic_rate_spread_fixed: float = 0.0
    foreign_rate_multiplier: float = 1.0
    foreign_rate_spread_fixed: float = 0.0

    # Monte Carlo
    n_paths: int = 100
    dt: float = 1/252
    seed: int = 42

    # Models (fallback values; a/sigma/r0 will usually be inferred below)
    hw_a_domestic: float = 0.03
    hw_sigma_domestic: float = 0.01
    hw_a_foreign: float = 0.03
    hw_sigma_foreign: float = 0.01
    fx_vol_fallback: float = 0.12

    # Correlations (names as used by CrossCurrencySwapPricer: rd, rf, fx, cr)
    rho_rd_rf: float = 0.20
    rho_rd_fx: float = -0.10
    rho_rf_fx: float = -0.10

    # Credit BK (proxy params if you want CVA/FVA)
    include_credit: bool = True
    cbk_kappa: float = 1.0
    cbk_vol: float = 0.6
    cbk_x0: float = -3.0     # exp(x0) = initial intensity
    cbk_mu: float = -3.0
    lgd: float = 0.6

    # Funding
    include_fva: bool = True
    fund_spread: float = 0.01
    alpha_ead: float = 1.4

    # IRB proxy (for RWA via EEPE -> EAD)
    irb_PD: float = 0.01
    irb_LGD: float = 0.45
    irb_M: float = 2.5


def map_corr_from_adapter(corr_matrix_3x3: np.ndarray) -> Dict[Tuple[str, str], float]:
    """
    Adapter returns a 3x3 correlation between [FX, USD rate, EUR rate].
    We need a dict on names ('rd','rf','fx') for the pricer.
    """
    # Index: 0=FX, 1=USD (domestic), 2=EUR (foreign). Adjust if your domestic/foreign reversed.
    rho_fx_rd = float(corr_matrix_3x3[0, 1])
    rho_fx_rf = float(corr_matrix_3x3[0, 2])
    rho_rd_rf = float(corr_matrix_3x3[1, 2])
    rho = {
        ("rd", "rf"): rho_rd_rf,
        ("rd", "fx"): rho_fx_rd,
        ("rf", "fx"): rho_fx_rf,
    }
    return rho


def build_models_from_inputs(inputs: dict, cfg: UserCCSConfig):
    """
    Convert adapter outputs to model objects used by ccs_pricer_full_slim.
    Expected `inputs` keys:
      - 'spot_fx' (float)
      - 'usd_zero_curve' (DataFrame: tenor_years, zero_rate)
      - 'eur_zero_curve' (DataFrame: tenor_years, zero_rate)
      - 'basis_curve' (DataFrame: tenor_years, basis_bp)
      - 'fx_vol', 'usd_rate_vol', 'eur_rate_vol' (floats)
      - 'corr_matrix' (3x3 ndarray for [FX, USD, EUR])
    """
    usd_curve = inputs["usd_zero_curve"]
    eur_curve = inputs["eur_zero_curve"]
    spot_fx = float(inputs["spot_fx"])  # domestic per foreign

    # r0 as short end of zero curve (or 1m/3m linear interp); tweak as you see fit.
    r0_dom = float(interp_zero_rate(usd_curve, 0.04))
    r0_for = float(interp_zero_rate(eur_curve, 0.04))

    hw_dom = HullWhite1F(a=cfg.hw_a_domestic,
                         sigma=float(inputs.get("usd_rate_vol", cfg.hw_sigma_domestic)),
                         r0=r0_dom)
    hw_for = HullWhite1F(a=cfg.hw_a_foreign,
                         sigma=float(inputs.get("eur_rate_vol", cfg.hw_sigma_foreign)),
                         r0=r0_for)

    fx_vol = float(inputs.get("fx_vol", cfg.fx_vol_fallback))
    fx = GarmanKohlhagenFX(spot0=spot_fx, vol=fx_vol)

    cbk = None
    if cfg.include_credit:
        cbk = CreditBK(kappa=cfg.cbk_kappa, vol=cfg.cbk_vol,
                       x0=cfg.cbk_x0, mu=cfg.cbk_mu, lgd=cfg.lgd)

    # Correlations
    if "corr_matrix" in inputs and isinstance(inputs["corr_matrix"], np.ndarray):
        rho = map_corr_from_adapter(inputs["corr_matrix"])
    else:
        rho = {
            ("rd", "rf"): cfg.rho_rd_rf,
            ("rd", "fx"): cfg.rho_rd_fx,
            ("rf", "fx"): cfg.rho_rf_fx,
        }

    return hw_dom, hw_for, fx, cbk, rho


def build_ccs_spec(cfg: UserCCSConfig) -> CCSpec:
    legD = CCSLegSpec(
        pay_fix=cfg.domestic_pay_fix,
        currency="domestic",
        notional=cfg.notional_domestic,
        rate=cfg.domestic_fixed_rate,
        tenor_years=cfg.maturity_years,
        daycount=cfg.domestic_daycount,
        spread=cfg.domestic_float_spread,
        exchange=cfg.domestic_exchange,
        rate_multiplier=cfg.domestic_rate_multiplier,
        rate_spread_fixed=cfg.domestic_rate_spread_fixed,
    )
    legF = CCSLegSpec(
        pay_fix=cfg.foreign_pay_fix,
        currency="foreign",
        notional=cfg.notional_foreign,
        rate=cfg.foreign_fixed_rate,
        tenor_years=cfg.maturity_years,
        daycount=cfg.foreign_daycount,
        spread=cfg.foreign_float_spread,
        exchange=cfg.foreign_exchange,
        rate_multiplier=cfg.foreign_rate_multiplier,
        rate_spread_fixed=cfg.foreign_rate_spread_fixed,
    )
    return CCSpec(val_date=0.0, maturity=cfg.maturity_years,
                  leg_domestic=legD, leg_foreign=legF)


def main():
    # 1) Load market inputs from your adapter (with robust fallbacks inside)
    inputs = load_inputs_via_adapter(as_of=None)

    # 2) Build models (HW1F, GK FX, optional CreditBK) + correlations
    cfg = UserCCSConfig()
    hw_dom, hw_for, fx, cbk, rho = build_models_from_inputs(inputs, cfg)

    # 3) CCS spec
    cc_spec = build_ccs_spec(cfg)

    # 4) MC config
    mc = MCConfig(n_paths=cfg.n_paths, dt=cfg.dt, seed=cfg.seed)

    # 5) Instantiate pricer (uses full path-by-path cashflow generation and discounting)
    pricer = CrossCurrencySwapPricer(
        hw_dom=hw_dom, hw_for=hw_for, fx=fx, credit=cbk,
        rho=rho, cc_spec=cc_spec, mc=mc,
        fund_spread=cfg.fund_spread, alpha_ead=cfg.alpha_ead
    )

    # 6) Run pricing + exposures/charges
    outputs = pricer.price(
        include_cva=cfg.include_credit,
        include_fva=cfg.include_fva,
        irb_params={"PD": cfg.irb_PD, "LGD": cfg.irb_LGD, "M": cfg.irb_M}
    )

    plot_diffusions(outputs.series)
    plot_exposures(outputs.series)
    plot_mtm_mean(outputs.series)            # MTM moyen(t)
    plot_valuation_charges(outputs.series)   # CVA/FVA(t) + cumul

    # 7) Print summary
    print("=== CCS Monte Carlo (full) with bootstrap init ===")
    print(f"Domestic leg PV (mean): {outputs.pv_domestic_leg_mean:,.2f}")
    print(f"Foreign  leg PV (mean): {outputs.pv_foreign_leg_mean:,.2f}")
    print(f"Gross PV (mean):        {outputs.pv_gross_mean:,.2f}")
    print(f"CVA: {outputs.cva:,.2f} | FVA: {outputs.fva:,.2f} | EEPE: {outputs.eepe:,.2f}")
    print(f"EAD (alpha*EEPE): {outputs.ead_alpha:,.2f} | RWA (IRB proxy): {outputs.rwa:,.2f}")

    # 8) Plot requested paths
    #    (Each helper produces its own figure with matplotlib, no styling/colors set.)
    try:
        plot_diffusions(outputs.series)
        plot_exposures(outputs.series)
        plot_valuation_charges(outputs.series)
    except Exception as e:
        # Plotting is optional—skip if running in a non-GUI environment.
        print("Plotting skipped:", e)

if __name__ == "__main__":
    main()
