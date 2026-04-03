"""
src/surface/iv_solver.py
------------------------
Implied volatility solvers for both European and American options.

The key distinction:
  European IV  : inverts Black-Scholes formula — fast, exact for European options.
                 Systematically inflates IV for American puts because it must
                 absorb the early exercise premium into the vol parameter.

  American IV  : inverts the CRR binomial tree — gives the TRUE implied vol
                 the market is pricing. CRR(sigma_A) = market_mid exactly.
                 sigma_A < sigma_E for puts (less inflation needed).

Why this matters:
  - iv_model built from sigma_A -> SVI surface on true vols
  - CRR(sigma_A from surface) ~= market_mid -> hit rate improves materially
  - BSM(sigma_A) < market_mid for puts — expected, European cannot capture EEP

For calls with q=0, American = European, so both solvers return identical
results. The speed difference only affects puts.

Public API
----------
implied_volatility(market_price, S, K, T, r, option_type, q,
                   sigma_lower, sigma_upper, american) -> float
european_implied_volatility(...) -> float   [fast, BSM-based]
american_implied_volatility(...) -> float   [accurate, CRR-based]
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.models.black_scholes import bsm_price
from src.models.binomial import crr_price

# CRR steps used inside the IV solver.
# N=50 is fast and accurate enough for root finding —
# the SVI surface smooths residual discretisation noise.
_CRR_N_IV = 50


# ── lower bounds ──────────────────────────────────────────────────────────────

def _am_lb(S, K, T, r, q, option_type):
    if option_type == "call":
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), S - K, 0.0)
    return max(K * np.exp(-r * T) - S * np.exp(-q * T), K - S, 0.0)


def _eu_lb(S, K, T, r, q, option_type):
    if option_type == "call":
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)


# ── European IV solver ────────────────────────────────────────────────────────

def european_implied_volatility(
    market_price: float, S: float, K: float,
    T: float, r: float, option_type: str = "call",
    q: float = 0.0, sigma_lower: float = 1e-4,
    sigma_upper: float = 5.0,
) -> float:
    """
    European Black-Scholes implied volatility via Brent's method.

    Fast (analytic BSM) but inflates IV for puts by absorbing early
    exercise premium. Use for calls with q=0 or speed-critical paths.
    """
    if pd.isna(market_price) or market_price <= 0:
        return np.nan
    if T <= 0 or S <= 0 or K <= 0:
        return np.nan
    if option_type not in ("call", "put"):
        return np.nan
    if market_price < _eu_lb(S, K, T, r, q, option_type) - 1e-8:
        return np.nan

    def obj(sigma):
        return bsm_price(S, K, T, r, sigma, option_type, q) - market_price

    try:
        f_lo, f_hi = obj(sigma_lower), obj(sigma_upper)
        if np.isnan(f_lo) or np.isnan(f_hi) or f_lo * f_hi > 0:
            return np.nan
        return float(brentq(obj, sigma_lower, sigma_upper, maxiter=200))
    except Exception:
        return np.nan


# ── American IV solver ────────────────────────────────────────────────────────

def american_implied_volatility(
    market_price: float, S: float, K: float,
    T: float, r: float, option_type: str = "call",
    q: float = 0.0, sigma_lower: float = 1e-4,
    sigma_upper: float = 5.0,
    n_tree: int = _CRR_N_IV,
) -> float:
    """
    American option implied volatility via CRR tree inversion.

    Inverts CRR(american=True) to find sigma such that
    CRR(S, K, T, r, sigma) = market_price.

    For puts, extracts lower sigma than European solver because it does
    not inflate vol for early exercise premium. CRR priced with this
    sigma recovers market_price exactly (within tree discretisation).

    Parameters
    ----------
    n_tree : CRR steps for the solver (default 50 — fast, sufficient)
    """
    if pd.isna(market_price) or market_price <= 0:
        return np.nan
    if T <= 0 or S <= 0 or K <= 0:
        return np.nan
    if option_type not in ("call", "put"):
        return np.nan
    if market_price < _am_lb(S, K, T, r, q, option_type) - 1e-8:
        return np.nan

    def obj(sigma):
        p = crr_price(S, K, T, r, sigma, option_type, q,
                      N=n_tree, american=True)
        return (p if not np.isnan(p) else 1e10) - market_price

    try:
        f_lo, f_hi = obj(sigma_lower), obj(sigma_upper)
        if np.isnan(f_lo) or np.isnan(f_hi):
            return np.nan
        # Widen lower bound if bracket fails at very low sigma
        if f_lo * f_hi > 0:
            f_lo2 = obj(0.01)
            if f_lo2 * f_hi < 0:
                sigma_lower, f_lo = 0.01, f_lo2
            else:
                return np.nan
        return float(brentq(obj, sigma_lower, sigma_upper,
                             maxiter=200, xtol=1e-6))
    except Exception:
        return np.nan


# ── smart dispatcher (used everywhere in the pipeline) ────────────────────────

def implied_volatility(
    market_price: float, S: float, K: float,
    T: float, r: float, option_type: str = "call",
    q: float = 0.0, sigma_lower: float = 1e-4,
    sigma_upper: float = 5.0,
    american: bool = True,
) -> float:
    """
    Implied volatility with smart dispatch to European or American solver.

    american=True (default, recommended for real market data):
      - calls with q=0  -> european_implied_volatility  (same result, faster)
      - puts            -> american_implied_volatility   (correct, no EEP inflation)
      - calls with q>0  -> american_implied_volatility   (EEP exists for calls too)

    american=False:
      - always uses european_implied_volatility

    Returns np.nan if the solver fails for any reason.
    """
    if american:
        if option_type == "call" and q == 0.0:
            return european_implied_volatility(
                market_price, S, K, T, r, option_type, q,
                sigma_lower, sigma_upper)
        return american_implied_volatility(
            market_price, S, K, T, r, option_type, q,
            sigma_lower, sigma_upper)
    return european_implied_volatility(
        market_price, S, K, T, r, option_type, q,
        sigma_lower, sigma_upper)
