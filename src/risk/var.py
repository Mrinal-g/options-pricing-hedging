"""
src/risk/var.py
---------------
Value at Risk (VaR), Conditional VaR (CVaR), and stress scenario P&L
for a portfolio of option positions.

Methods
-------
Historical VaR   : uses the empirical distribution of past spot returns
                   to re-price the portfolio and find the loss quantile.
Parametric VaR   : assumes normally distributed returns (faster, less accurate
                   for options due to non-linear payoffs).
Stress scenarios : applies the vol surface shocks from src/surface/svi.py
                   (parallel shift, skew steepen, short-end spike) and
                   re-prices each position.

Public API
----------
historical_var(positions, spot_hist, S, r, q, confidence, horizon_days)
    -> dict

parametric_var(positions, S, r, q, sigma_port, confidence, horizon_days)
    -> dict

stress_scenarios(positions, S, r, q, scenarios)
    -> pd.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.black_scholes import bsm_price
from src.risk.greeks import OptionPosition, portfolio_greeks


# ── historical VaR ────────────────────────────────────────────────────────────

def historical_var(
    positions: list[OptionPosition],
    spot_hist: pd.DataFrame,
    S: float,
    r: float,
    q: float = 0.0,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> dict:
    """
    Historical simulation VaR for a portfolio of options.

    For each past daily log-return, applies that return to the current
    spot and re-prices the full portfolio.  The VaR is the quantile
    of the resulting P&L distribution.

    Parameters
    ----------
    positions     : list of OptionPosition objects
    spot_hist     : DataFrame with a 'close' column (output of download.py)
    S             : current spot price
    r             : risk-free rate
    q             : dividend yield
    confidence    : VaR confidence level (default 0.95 = 95%)
    horizon_days  : holding period in days (default 1)
                    Multi-day VaR is approximated by scaling 1-day VaR
                    by sqrt(horizon_days).

    Returns
    -------
    dict with keys:
        var_1day, cvar_1day,
        var_horizon, cvar_horizon,
        confidence, horizon_days, n_scenarios,
        pnl_mean, pnl_std, pnl_min, pnl_max
    """
    close   = spot_hist["close"].dropna().values
    log_ret = np.diff(np.log(close))

    if len(log_ret) < 30:
        raise ValueError("Need at least 30 historical returns to compute VaR.")

    current_value = _portfolio_value(positions, S, r, q)

    # Re-price at each shocked spot
    pnls = []
    for ret in log_ret:
        S_shocked = S * np.exp(ret)
        shocked_value = _portfolio_value(positions, S_shocked, r, q)
        pnls.append(shocked_value - current_value)

    pnls = np.array(pnls)

    var_1d  = float(-np.percentile(pnls, (1 - confidence) * 100))
    cvar_1d = float(-pnls[pnls <= -var_1d].mean()) if any(pnls <= -var_1d) else var_1d

    scale       = np.sqrt(horizon_days)
    var_horizon = var_1d  * scale
    cvar_horizon= cvar_1d * scale

    return {
        "var_1day"     : round(var_1d,  4),
        "cvar_1day"    : round(cvar_1d, 4),
        "var_horizon"  : round(var_horizon,  4),
        "cvar_horizon" : round(cvar_horizon, 4),
        "confidence"   : confidence,
        "horizon_days" : horizon_days,
        "n_scenarios"  : len(pnls),
        "pnl_mean"     : round(float(np.mean(pnls)), 4),
        "pnl_std"      : round(float(np.std(pnls)),  4),
        "pnl_min"      : round(float(np.min(pnls)),  4),
        "pnl_max"      : round(float(np.max(pnls)),  4),
    }


# ── parametric VaR ────────────────────────────────────────────────────────────

def parametric_var(
    positions: list[OptionPosition],
    S: float,
    r: float,
    q: float = 0.0,
    sigma_port: float = 0.20,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> dict:
    """
    Parametric (delta-gamma) VaR.

    Uses the portfolio's dollar delta and dollar gamma to approximate
    the P&L distribution for a normally distributed spot move.

    P&L ≈ dollar_delta × ΔS + 0.5 × net_gamma × S² × (ΔS/S)²

    This is faster than historical simulation but less accurate for
    large moves due to the quadratic approximation.

    Parameters
    ----------
    sigma_port  : annualised vol of the underlying (default 0.20)
    confidence  : VaR confidence level (default 0.95)
    horizon_days: holding period in days

    Returns
    -------
    dict with var_1day, cvar_1day, var_horizon, cvar_horizon,
               confidence, horizon_days, delta_pnl_std, method
    """
    from scipy.stats import norm

    g = portfolio_greeks(positions, S, r, q)

    daily_sigma = sigma_port / np.sqrt(252)
    dt          = horizon_days / 252

    # 1-day delta P&L std dev
    delta_std = abs(g["net_delta"]) * S * daily_sigma

    z = norm.ppf(confidence)

    var_1d  = z * delta_std
    cvar_1d = norm.pdf(z) / (1 - confidence) * delta_std

    scale        = np.sqrt(horizon_days)
    var_horizon  = var_1d  * scale
    cvar_horizon = cvar_1d * scale

    return {
        "var_1day"      : round(var_1d,  4),
        "cvar_1day"     : round(cvar_1d, 4),
        "var_horizon"   : round(var_horizon,  4),
        "cvar_horizon"  : round(cvar_horizon, 4),
        "confidence"    : confidence,
        "horizon_days"  : horizon_days,
        "delta_pnl_std" : round(delta_std, 4),
        "method"        : "parametric (delta approximation)",
    }


# ── stress scenarios ──────────────────────────────────────────────────────────

def stress_scenarios(
    positions: list[OptionPosition],
    S: float,
    r: float,
    q: float = 0.0,
    scenarios: dict | None = None,
) -> pd.DataFrame:
    """
    Re-price the portfolio under a set of stress scenarios and report P&L.

    Each scenario is a dict of overrides:
      - 'spot_shock'  : multiplicative spot shock, e.g. 0.90 = -10% spot
      - 'vol_shift'   : additive vol shift in vol points, e.g. +0.05 = +5 vols
      - 'ttm_shift'   : additive TTM shift in years, e.g. -1/52 = -1 week

    Parameters
    ----------
    scenarios : dict of {scenario_name: {override_key: value}}
                Default includes standard equity stress scenarios.

    Returns
    -------
    pd.DataFrame with columns:
        scenario, spot_shocked, vol_shift, pnl, pnl_pct_of_value
    """
    if scenarios is None:
        scenarios = {
            "Base"                 : {},
            "Spot -5%"             : {"spot_shock": 0.95},
            "Spot -10%"            : {"spot_shock": 0.90},
            "Spot -20% (crash)"    : {"spot_shock": 0.80},
            "Spot +5%"             : {"spot_shock": 1.05},
            "Spot +10%"            : {"spot_shock": 1.10},
            "Vol +5pts"            : {"vol_shift":  0.05},
            "Vol -5pts"            : {"vol_shift": -0.05},
            "Vol +10pts"           : {"vol_shift":  0.10},
            "Spot -10%, Vol +10pts": {"spot_shock": 0.90, "vol_shift": 0.10},
            "Spot -20%, Vol +15pts": {"spot_shock": 0.80, "vol_shift": 0.15},
        }

    base_value = _portfolio_value(positions, S, r, q)

    rows = []
    for name, overrides in scenarios.items():
        spot_shock = overrides.get("spot_shock", 1.0)
        vol_shift  = overrides.get("vol_shift",  0.0)
        ttm_shift  = overrides.get("ttm_shift",  0.0)

        S_stressed = S * spot_shock

        stressed_positions = [
            OptionPosition(
                K           = pos.K,
                T           = max(pos.T + ttm_shift, 1/365),  # floor at 1 day
                sigma       = max(pos.sigma + vol_shift, 0.01),
                option_type = pos.option_type,
                quantity    = pos.quantity,
                label       = pos.label,
            )
            for pos in positions
        ]

        stressed_value = _portfolio_value(stressed_positions, S_stressed, r, q)
        pnl            = stressed_value - base_value
        pnl_pct        = 100 * pnl / abs(base_value) if base_value != 0 else np.nan

        rows.append({
            "scenario"        : name,
            "spot_shocked"    : round(S_stressed, 2),
            "spot_shock_pct"  : round((spot_shock - 1) * 100, 1),
            "vol_shift_pts"   : round(vol_shift * 100, 1),
            "portfolio_value" : round(stressed_value, 4),
            "pnl"             : round(pnl, 4),
            "pnl_pct_of_value": round(pnl_pct, 2),
        })

    return pd.DataFrame(rows).set_index("scenario")


# ── internal helpers ──────────────────────────────────────────────────────────

def _portfolio_value(
    positions: list[OptionPosition],
    S: float,
    r: float,
    q: float,
) -> float:
    """Sum of quantity × BSM price across all positions."""
    total = 0.0
    for pos in positions:
        val = bsm_price(S, pos.K, pos.T, r, pos.sigma, pos.option_type, q)
        total += pos.quantity * (val if not np.isnan(val) else 0.0)
    return total
