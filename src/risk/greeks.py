"""
src/risk/greeks.py
------------------
Portfolio-level Greek aggregation across multiple option positions.

Real desks never care about a single option's Greeks in isolation.
They care about the net exposure of the entire book:
  - Is the book net long or short gamma?
  - What is the total vega exposure to a 1 vol-point move?
  - What is the daily theta bleed?
  - What dollar delta do we need to hedge?

Public API
----------
OptionPosition  : dataclass for a single position
portfolio_greeks(positions, S, r, q) -> dict
greeks_by_expiry(positions, S, r, q) -> pd.DataFrame
dollar_greeks(greeks_dict, S)        -> dict
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.models.black_scholes import bsm_greeks, bsm_price


# ── position dataclass ────────────────────────────────────────────────────────

@dataclass
class OptionPosition:
    """
    A single option position in a portfolio.

    Attributes
    ----------
    K           : strike price
    T           : time to maturity in years
    sigma       : implied volatility for this position
    option_type : 'call' or 'put'
    quantity    : number of contracts.  Positive = long, negative = short.
                  One contract = one option on one share (no multiplier applied;
                  add a multiplier field if modelling exchange-listed contracts).
    label       : optional human-readable label (e.g. 'Mar-25 put K=160')
    """
    K          : float
    T          : float
    sigma      : float
    option_type: str   = "put"
    quantity   : float = 1.0
    label      : str   = ""

    def __post_init__(self):
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {self.option_type!r}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")


# ── portfolio aggregation ─────────────────────────────────────────────────────

def portfolio_greeks(
    positions: list[OptionPosition],
    S: float,
    r: float,
    q: float = 0.0,
) -> dict:
    """
    Compute net Greeks for a portfolio of option positions.

    Each Greek is summed across positions, weighted by quantity.
    Positive quantity = long; negative = short.

    Parameters
    ----------
    positions : list of OptionPosition objects
    S         : current spot price
    r         : continuously compounded risk-free rate
    q         : continuous dividend yield

    Returns
    -------
    dict with keys:
        n_positions, total_value,
        net_delta, net_gamma, net_vega, net_theta, net_rho,
        dollar_delta, dollar_gamma, dollar_vega (dollar Greeks)
    """
    if not positions:
        return {k: 0.0 for k in [
            "n_positions", "total_value",
            "net_delta", "net_gamma", "net_vega", "net_theta", "net_rho",
            "dollar_delta", "dollar_gamma", "dollar_vega",
        ]}

    net = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    total_value = 0.0

    for pos in positions:
        g   = bsm_greeks(S, pos.K, pos.T, r, pos.sigma, pos.option_type, q)
        val = bsm_price(S, pos.K, pos.T, r, pos.sigma, pos.option_type, q)

        for key in net:
            net[key] += pos.quantity * (g[key] if not np.isnan(g[key]) else 0.0)
        total_value += pos.quantity * (val if not np.isnan(val) else 0.0)

    # Dollar Greeks — the actual hedge sizes a desk would use
    # Dollar delta  = net_delta * S       (shares to trade for delta-neutral)
    # Dollar gamma  = 0.5 * net_gamma * S² (P&L for a 1% S move, ×100)
    # Dollar vega   = net_vega / 100      (P&L for a 1 vol-point move)
    return {
        "n_positions"  : len(positions),
        "total_value"  : round(total_value, 4),
        "net_delta"    : round(net["delta"], 6),
        "net_gamma"    : round(net["gamma"], 6),
        "net_vega"     : round(net["vega"],  6),
        "net_theta"    : round(net["theta"], 6),
        "net_rho"      : round(net["rho"],   6),
        "dollar_delta" : round(net["delta"] * S, 4),
        "dollar_gamma" : round(0.5 * net["gamma"] * S ** 2 * 0.01, 4),
        "dollar_vega"  : round(net["vega"] / 100, 4),
    }


def greeks_by_expiry(
    positions: list[OptionPosition],
    S: float,
    r: float,
    q: float = 0.0,
) -> pd.DataFrame:
    """
    Greeks broken down by expiry bucket, then aggregated.

    Useful for seeing where the book's risk concentrates across the
    term structure.

    Returns
    -------
    pd.DataFrame indexed by (T_bucket, option_type) with Greek columns
    """
    rows = []
    for pos in positions:
        g   = bsm_greeks(S, pos.K, pos.T, r, pos.sigma, pos.option_type, q)
        val = bsm_price(S, pos.K, pos.T, r, pos.sigma, pos.option_type, q)

        # Round TTM to nearest tenth for bucketing
        t_bucket = round(pos.T, 2)

        rows.append({
            "T_bucket"   : t_bucket,
            "option_type": pos.option_type,
            "K"          : pos.K,
            "quantity"   : pos.quantity,
            "value"      : pos.quantity * (val if not np.isnan(val) else 0.0),
            "delta"      : pos.quantity * (g["delta"] if not np.isnan(g["delta"]) else 0.0),
            "gamma"      : pos.quantity * (g["gamma"] if not np.isnan(g["gamma"]) else 0.0),
            "vega"       : pos.quantity * (g["vega"]  if not np.isnan(g["vega"])  else 0.0),
            "theta"      : pos.quantity * (g["theta"] if not np.isnan(g["theta"]) else 0.0),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return (
        df.groupby(["T_bucket", "option_type"])
        [["value", "delta", "gamma", "vega", "theta"]]
        .sum()
        .round(6)
    )


def dollar_greeks(greeks_dict: dict, S: float) -> dict:
    """
    Convert unit Greeks to dollar Greeks for a given spot price.

    Dollar delta  = net_delta × S              ($-worth of shares to trade)
    Dollar gamma  = ½ × net_gamma × S² × 0.01 (P&L per 1% spot move)
    Dollar vega   = net_vega ÷ 100             (P&L per 1 vol point)
    Daily theta   = net_theta                  (already in $/day from BSM)

    Parameters
    ----------
    greeks_dict : output of portfolio_greeks() or a dict with net_* keys
    S           : current spot price

    Returns
    -------
    dict with dollar-denominated Greek values
    """
    nd = greeks_dict.get("net_delta", 0.0)
    ng = greeks_dict.get("net_gamma", 0.0)
    nv = greeks_dict.get("net_vega",  0.0)
    nt = greeks_dict.get("net_theta", 0.0)

    return {
        "dollar_delta"      : round(nd * S, 4),
        "dollar_gamma_1pct" : round(0.5 * ng * (S * 0.01) ** 2 * 2, 4),
        "dollar_vega_1pt"   : round(nv / 100, 4),
        "daily_theta"       : round(nt, 4),
    }
