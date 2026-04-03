"""
src/models/black_scholes.py
---------------------------
Black-Scholes-Merton European option pricing and Greeks.

All functions accept scalar inputs and return scalars.
They guard against invalid inputs (T<=0, sigma<=0, etc.) by returning np.nan.

Public API
----------
bsm_price(S, K, T, r, sigma, option_type, q) -> float
bsm_delta(S, K, T, r, sigma, option_type, q) -> float
bsm_gamma(S, K, T, r, sigma, q)              -> float
bsm_vega(S, K, T, r, sigma, q)               -> float
bsm_theta(S, K, T, r, sigma, option_type, q) -> float
bsm_rho(S, K, T, r, sigma, option_type, q)   -> float
bsm_greeks(S, K, T, r, sigma, option_type, q) -> dict
"""

import numpy as np
from scipy.stats import norm


# ── helpers ───────────────────────────────────────────────────────────────────

def _d1_d2(S: float, K: float, T: float, r: float,
           sigma: float, q: float = 0.0):
    """Compute d1 and d2 for the BSM formula."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def _valid(S, K, T, sigma) -> bool:
    return T > 0 and sigma > 0 and S > 0 and K > 0


# ── pricing ───────────────────────────────────────────────────────────────────

def bsm_price(S: float, K: float, T: float, r: float,
              sigma: float, option_type: str = "call",
              q: float = 0.0) -> float:
    """
    Black-Scholes-Merton European option price with continuous dividend yield.

    Parameters
    ----------
    S           : spot price
    K           : strike price
    T           : time to maturity in years
    r           : continuously compounded risk-free rate
    sigma       : implied volatility
    option_type : 'call' or 'put'
    q           : continuous dividend yield (default 0.0)

    Returns
    -------
    float : option price, or np.nan if inputs are invalid
    """
    if not _valid(S, K, T, sigma):
        return np.nan

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)

    if option_type == "call":
        return (S * np.exp(-q * T) * norm.cdf(d1)
                - K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == "put":
        return (K * np.exp(-r * T) * norm.cdf(-d2)
                - S * np.exp(-q * T) * norm.cdf(-d1))
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


# ── Greeks ────────────────────────────────────────────────────────────────────

def bsm_delta(S: float, K: float, T: float, r: float,
              sigma: float, option_type: str = "call",
              q: float = 0.0) -> float:
    """
    BSM delta: dPrice/dS.

    Positive for calls (0 to 1), negative for puts (-1 to 0).
    """
    if not _valid(S, K, T, sigma):
        return np.nan

    d1, _ = _d1_d2(S, K, T, r, sigma, q)

    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    elif option_type == "put":
        return np.exp(-q * T) * (norm.cdf(d1) - 1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bsm_gamma(S: float, K: float, T: float, r: float,
              sigma: float, q: float = 0.0) -> float:
    """
    BSM gamma: d²Price/dS².  Identical for calls and puts.

    High gamma near ATM and at short maturities — delta changes quickly,
    requiring more frequent hedge rebalancing.
    """
    if not _valid(S, K, T, sigma):
        return np.nan

    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bsm_vega(S: float, K: float, T: float, r: float,
             sigma: float, q: float = 0.0) -> float:
    """
    BSM vega: dPrice/dSigma.  Identical for calls and puts.

    Returned in dollar terms per unit of volatility (not per vol point).
    Divide by 100 to get vega per 1 vol point.
    """
    if not _valid(S, K, T, sigma):
        return np.nan

    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bsm_theta(S: float, K: float, T: float, r: float,
              sigma: float, option_type: str = "call",
              q: float = 0.0) -> float:
    """
    BSM theta: dPrice/dt, expressed as decay per calendar day.

    Negative for long options — the position loses value as time passes.
    """
    if not _valid(S, K, T, sigma):
        return np.nan

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)

    term1 = -(S * np.exp(-q * T) * pdf_d1 * sigma) / (2 * np.sqrt(T))

    if option_type == "call":
        return (term1
                - r * K * np.exp(-r * T) * norm.cdf(d2)
                + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365
    elif option_type == "put":
        return (term1
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
                - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bsm_rho(S: float, K: float, T: float, r: float,
            sigma: float, option_type: str = "call",
            q: float = 0.0) -> float:
    """
    BSM rho: dPrice/dr.

    Positive for calls (higher rates → calls worth more),
    negative for puts.
    """
    if not _valid(S, K, T, sigma):
        return np.nan

    _, d2 = _d1_d2(S, K, T, r, sigma, q)

    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bsm_greeks(S: float, K: float, T: float, r: float,
               sigma: float, option_type: str = "call",
               q: float = 0.0) -> dict:
    """
    Return all five BSM Greeks in a single dict.

    Keys: delta, gamma, vega, theta, rho
    Theta is per calendar day; vega is per unit volatility.
    All values are np.nan if inputs are invalid.
    """
    keys = ["delta", "gamma", "vega", "theta", "rho"]
    if not _valid(S, K, T, sigma):
        return {k: np.nan for k in keys}

    return {
        "delta": bsm_delta(S, K, T, r, sigma, option_type, q),
        "gamma": bsm_gamma(S, K, T, r, sigma, q),
        "vega":  bsm_vega(S, K, T, r, sigma, q),
        "theta": bsm_theta(S, K, T, r, sigma, option_type, q),
        "rho":   bsm_rho(S, K, T, r, sigma, option_type, q),
    }
