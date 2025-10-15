from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, sqrt
from scipy.stats import norm  # if scipy not available in your env, swap by an approximation

# =====================
# Model building blocks
# =====================

@dataclass
class HullWhite1F:
    a: float
    sigma: float
    r0: float

    def simulate(self, n_paths: int, times: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        dt = np.diff(times, prepend=times[0])
        r = np.empty((n_paths, len(times)), dtype=float)
        r[:, 0] = self.r0
        for i in range(1, len(times)):
            dW = rng.normal(0.0, np.sqrt(dt[i]), size=n_paths)
            r[:, i] = r[:, i-1] + self.a * (0.0 - r[:, i-1]) * dt[i] + self.sigma * dW
        return r


@dataclass
class GarmanKohlhagenFX:
    spot0: float
    vol: float


@dataclass
class CreditBK:
    kappa: float = 1.0
    vol: float = 0.0
    x0: float = -3.0
    mu: float = -3.0
    lgd: float = 0.6

    def hazard(self) -> float:
        return float(np.exp(self.x0))




# Follows the diffusions for booth Garman and CBK, for running time purposes Ihave used a simpler method
# @dataclass
# class GarmanKohlhagenFX:
#     spot0: float
#     vol: float
#     def simulate(self, dt: float, n_steps: int, n_paths: int, z: np.ndarray,
#                  rd_path: np.ndarray, rf_path: np.ndarray) -> np.ndarray:
#         S = np.empty((n_paths, n_steps + 1))
#         S[:, 0] = self.spot0
#         v = self.vol
#         sqdt = np.sqrt(dt)
#         for t in range(n_steps):
#             drift = (rd_path[:, t] - rf_path[:, t] - 0.5 * v * v) * dt
#             diff = v * sqdt * z[:, t]
#             S[:, t+1] = S[:, t] * np.exp(drift + diff)
#         return S

# @dataclass
# class CreditBK:
#     kappa: float
#     vol: float
#     x0: float
#     mu: float
#     lgd: float
#     def simulate(self, dt: float, n_steps: int, n_paths: int, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         k, v = self.kappa, self.vol
#         x = np.empty((n_paths, n_steps + 1))
#         lam = np.empty_like(x)
#         H = np.zeros_like(x)
#         x[:, 0] = self.x0
#         exp_kdt = np.exp(-k * dt)
#         var = (v**2) * (1 - np.exp(-2*k*dt)) / (2*k) if k > 1e-12 else (v**2) * dt
#         std = np.sqrt(var)
#         for t in range(n_steps):
#             mean = x[:, t] * exp_kdt + self.mu * (1 - exp_kdt)
#             x[:, t+1] = mean + std * z[:, t]
#             lam[:, t] = np.exp(x[:, t])
#             if t > 0:
#                 H[:, t] = H[:, t-1] + 0.5 * dt * (lam[:, t-1] + lam[:, t])
#         lam[:, n_steps] = np.exp(x[:, n_steps])
#         H[:, n_steps] = H[:, n_steps-1] + 0.5 * dt * (lam[:, n_steps-1] + lam[:, n_steps])
#         return lam, H



# ===============
# CCS definitions
# ===============

@dataclass
class CCSLegSpec:
    pay_fix: bool
    currency: str           # "domestic" or "foreign"
    notional: float
    rate: float
    tenor_years: float
    daycount: str = "ACT/360"
    spread: float = 0.0
    exchange: str = "maturity"  # "none" | "maturity" | "both"
    rate_multiplier: float = 1.0
    rate_spread_fixed: float = 0.0


@dataclass
class CCSpec:
    val_date: float
    maturity: float
    leg_domestic: CCSLegSpec
    leg_foreign: CCSLegSpec


@dataclass
class MCConfig:
    n_paths: int
    dt: float
    seed: int = 42


@dataclass
class PricerOutputs:
    times: np.ndarray
    rd_paths: np.ndarray
    rf_paths: np.ndarray
    fx_paths: np.ndarray

    mtm_paths: np.ndarray
    ee_t: np.ndarray
    ene_t: np.ndarray
    epe_t: np.ndarray
    eepe: float

    pv_domestic_leg_mean: float
    pv_foreign_leg_mean: float
    pv_gross_mean: float

    cva_t: np.ndarray
    fva_t: np.ndarray
    cva: float
    fva: float

    # new
    ead_alpha: float
    rwa: float

    series: Dict[str, np.ndarray]


# ====================
# Utility/helper funcs
# ====================

def _build_time_grid(maturity: float, dt: float) -> np.ndarray:
    n = int(np.ceil(maturity / dt)) + 1
    return np.linspace(0.0, maturity, n)


def _coupon_schedule(tenor_years: float, freq_per_year: int = 4) -> np.ndarray:
    n_cpn = int(np.round(tenor_years * freq_per_year))
    return np.arange(1, n_cpn + 1, dtype=float) / float(freq_per_year)


def _daycount_frac(freq_per_year: int = 4, basis: str = "ACT/360") -> float:
    if basis.upper() in ("ACT/360", "30/360"):
        return 90.0 / 360.0
    return 90.0 / 365.0


def _df_from_rates(r_paths: np.ndarray, times: np.ndarray) -> np.ndarray:
    dt = np.diff(times, prepend=times[0])
    cum_int = np.cumsum(r_paths * dt[None, :], axis=1)
    return np.exp(-cum_int)


def _present_value_leg_at_time(t_idx: int, times: np.ndarray, r_paths: np.ndarray,
                               leg: CCSLegSpec, fx_paths: Optional[np.ndarray],
                               is_domestic: bool) -> np.ndarray:
    n_paths, n_times = r_paths.shape
    pay_times = _coupon_schedule(leg.tenor_years, freq_per_year=4)
    accrual = _daycount_frac(4, leg.daycount)
    pv = np.zeros(n_paths, dtype=float)
    t_now = times[t_idx]
    dt = np.diff(times, prepend=times[0])
    cum_int = np.cumsum(r_paths * dt[None, :], axis=1)

    for tp in pay_times:
        if tp <= t_now:
            continue
        j = np.searchsorted(times, tp, side='right') - 1
        j = max(min(j, n_times - 1), t_idx + 1)
        if leg.pay_fix:
            cpn = (leg.rate * leg.rate_multiplier + leg.rate_spread_fixed) * accrual * leg.notional
        else:
            rfwd = r_paths[:, j-1] * leg.rate_multiplier + leg.spread + leg.rate_spread_fixed
            cpn = rfwd * accrual * leg.notional
        exch = leg.notional if (leg.exchange in ("both", "maturity") and np.isclose(tp, pay_times[-1])) else 0.0
        df_0_tp = np.exp(-cum_int[:, j]); df_0_tn = np.exp(-cum_int[:, t_idx])
        df_tn_tp = df_0_tp / df_0_tn
        pv += (cpn + exch) * df_tn_tp

    if not is_domestic and fx_paths is not None:
        S_t = fx_paths[:, t_idx]
        pv = pv * S_t
    return pv


def _irb_corporate_K(PD: float, LGD: float, M: float) -> float:
    """Basel IRB corporate capital function K (as fraction of EAD)."""
    PD = min(max(PD, 1e-6), 0.999)  # clip for stability
    R = 0.12 * (1 - np.exp(-50*PD)) / (1 - np.exp(-50)) + \
        0.24 * (1 - (1 - np.exp(-50*PD)) / (1 - np.exp(-50)))
    b = (0.11852 - 0.05478 * log(PD))**2
    scaling = (1 + (M - 2.5)*b) / (1 - 1.5*b)
    GPD = norm.ppf(PD)
    G999 = norm.ppf(0.999)
    term = (1.0/np.sqrt(1.0 - R)) * GPD + np.sqrt(R/(1.0 - R)) * G999
    K = LGD * norm.cdf(term) - PD * LGD
    return float(max(K * scaling, 0.0))


# ===================
# Main pricer object
# ===================

class CrossCurrencySwapPricer:
    def __init__(self,
                 hw_dom: HullWhite1F,
                 hw_for: HullWhite1F,
                 fx: GarmanKohlhagenFX,
                 credit: Optional[CreditBK],
                 rho: Dict[Tuple[str, str], float],
                 cc_spec: CCSpec,
                 mc: MCConfig,
                 fund_spread: float = 0.01,
                 alpha_ead: float = 1.4):
        self.hw_dom = hw_dom
        self.hw_for = hw_for
        self.fx = fx
        self.credit = credit
        self.rho = rho
        self.cc = cc_spec
        self.mc = mc
        self.fund_spread = fund_spread
        self.alpha_ead = alpha_ead

    def _simulate_joint(self, times: np.ndarray, n_paths: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.mc.seed)
        # short rates
        rd = self.hw_dom.simulate(n_paths, times, rng)
        rf = self.hw_for.simulate(n_paths, times, rng)
        # correlate FX with (rd, rf)
        rho_rd_rf = self.rho.get(("rd", "rf"), 0.2)
        rho_rd_fx = self.rho.get(("rd", "fx"), -0.1)
        rho_rf_fx = self.rho.get(("rf", "fx"), -0.1)
        Corr = np.array([[1.0,       rho_rd_rf, rho_rd_fx],
                         [rho_rd_rf, 1.0,       rho_rf_fx],
                         [rho_rd_fx, rho_rf_fx, 1.0      ]], dtype=float)
        L = np.linalg.cholesky(Corr + 1e-12*np.eye(3))
        dt = np.diff(times, prepend=times[0])
        S = np.empty((n_paths, len(times)), dtype=float)
        S[:, 0] = self.fx.spot0
        for i in range(1, len(times)):
            Z = rng.normal(size=(n_paths, 3))
            dW = (Z @ L.T) * np.sqrt(dt[i])
            mu = (rd[:, i-1] - rf[:, i-1]) * dt[i]
            S[:, i] = S[:, i-1] * np.exp((mu - 0.5 * self.fx.vol**2 * dt[i]) + self.fx.vol * dW[:, 2])
        return rd, rf, S

    def price(self,
              include_cva: bool = True,
              include_fva: bool = True,
              irb_params: Optional[Dict[str, float]] = None) -> PricerOutputs:
        T = self.cc.maturity
        times = _build_time_grid(T, self.mc.dt)
        n_paths = self.mc.n_paths

        rd, rf, S = self._simulate_joint(times, n_paths)

        DFd_0_t = _df_from_rates(rd, times)

        legD = self.cc.leg_domestic
        legF = self.cc.leg_foreign

        mtm = np.empty((n_paths, len(times)), dtype=float)
        for i in range(len(times)):
            pvD_t = _present_value_leg_at_time(i, times, rd, legD, fx_paths=None, is_domestic=True)
            pvF_t_dom = _present_value_leg_at_time(i, times, rf, legF, fx_paths=S, is_domestic=False)
            mtm[:, i] = pvD_t - pvF_t_dom

        epe_t = np.mean(np.maximum(mtm, 0.0), axis=0)
        ene_t = np.mean(np.minimum(mtm, 0.0), axis=0)
        ee_t = ene_t + epe_t
        eepe = float(np.mean(epe_t))

        pv_dom_mean = float(np.mean(_present_value_leg_at_time(0, times, rd, legD, fx_paths=None, is_domestic=True)))
        pv_for_mean = float(np.mean(_present_value_leg_at_time(0, times, rf, legF, fx_paths=S, is_domestic=False)))
        pv_gross_mean = pv_dom_mean - pv_for_mean

        dt_arr = np.diff(times, prepend=times[0])
        DF_mean = np.mean(DFd_0_t, axis=0)

        hazard = self.credit.hazard() if (include_cva and self.credit is not None) else 0.0
        surv = np.exp(-hazard * times)
        dPD = -np.diff(surv, prepend=1.0)

        cva_t = self.credit.lgd * ee_t * dPD * DF_mean if include_cva and self.credit is not None else np.zeros_like(times)
        fva_t = (self.alpha_ead * self.fund_spread) * ee_t * DF_mean * dt_arr if include_fva else np.zeros_like(times)

        cva = float(np.sum(cva_t))
        fva = float(np.sum(fva_t))

        # EAD & RWA proxy
        ead_alpha = self.alpha_ead * eepe
        
        # IRB proxy (for RWA via EEPE -> EAD)
        irb_PD: float = 0.01
        irb_LGD: float = 0.45
        irb_M: float = 2.5

        PD = float(irb_PD)
        LGD = float(irb_LGD)
        M = float(irb_M)
        K = _irb_corporate_K(PD, LGD, M)
        rwa = 12.5 * K * ead_alpha  # proxy
            
        series = {
            "times": times,
            "rd_sample": rd[:15],
            "rf_sample": rf[:15],
            "fx_sample": S[:15],
            "mtm_mean": np.mean(mtm, axis=0),
            "ee_t": ee_t,
            "ene_t": ene_t,
            "epe_t": epe_t,
            "cva_t": cva_t,
            "fva_t": fva_t,
            "cva_cum": np.cumsum(cva_t),
            "fva_cum": np.cumsum(fva_t),
        }

        return PricerOutputs(times=times, rd_paths=rd, rf_paths=rf, fx_paths=S,
                             mtm_paths=mtm, ee_t=ee_t, ene_t=ene_t, epe_t=epe_t, eepe=eepe,
                             pv_domestic_leg_mean=pv_dom_mean,
                             pv_foreign_leg_mean=pv_for_mean,
                             pv_gross_mean=pv_gross_mean,
                             cva_t=cva_t, fva_t=fva_t, cva=cva, fva=fva,
                             ead_alpha=ead_alpha, rwa=rwa,
                             series=series)


# ==============
# Plotting
# ==============

def plot_diffusions(series: Dict[str, np.ndarray]) -> None:
    t = series["times"]
    plt.figure()
    for path in series["rd_sample"]:
        plt.plot(t, path)
    plt.title("Domestic short rate (samples)")
    plt.xlabel("Time (y)"); plt.ylabel("r_d")
    plt.tight_layout()

    plt.figure()
    for path in series["rf_sample"]:
        plt.plot(t, path)
    plt.title("Foreign short rate (samples)")
    plt.xlabel("Time (y)"); plt.ylabel("r_f")
    plt.tight_layout()

    plt.figure()
    for path in series["fx_sample"]:
        plt.plot(t, path)
    plt.title("FX (samples)")
    plt.xlabel("Time (y)"); plt.ylabel("S_t")
    plt.tight_layout()


def plot_exposures(series: Dict[str, np.ndarray]) -> None:
    t = series["times"]
    plt.figure()
    plt.plot(t, series["ee_t"], label="EE")
    plt.plot(t, series["ene_t"], label="ENE")
    plt.plot(t, series["epe_t"], label="EPE (running mean)")
    plt.title("Exposures (mean over ALL paths)")
    plt.xlabel("Time (y)"); plt.ylabel("Exposure")
    plt.legend()
    plt.tight_layout()


def plot_mtm_mean(series: Dict[str, np.ndarray]) -> None:
    t = series["times"]
    plt.figure()
    plt.plot(t, series["mtm_mean"])
    plt.title("Mean MTM over time (ALL paths)")
    plt.xlabel("Time (y)"); plt.ylabel("Mean MTM (domestic)")
    plt.tight_layout()


def plot_valuation_charges(series: Dict[str, np.ndarray]) -> None:
    t = series["times"]
    plt.figure()
    plt.plot(t, series["cva_t"], label="CVA(t) incremental")
    plt.plot(t, series["fva_t"], label="FVA(t) incremental")
    plt.title("Valuation charges over time (ALL paths)")
    plt.xlabel("Time (y)"); plt.ylabel("Charge per step (PV)")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(t, series["cva_cum"], label="CVA cumulative")
    plt.plot(t, series["fva_cum"], label="FVA cumulative")
    plt.title("Cumulative CVA / FVA (ALL paths)")
    plt.xlabel("Time (y)"); plt.ylabel("Cumulative PV")
    plt.legend()
    plt.tight_layout()
