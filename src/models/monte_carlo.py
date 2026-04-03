"""
src/models/monte_carlo.py
-------------------------
Longstaff-Schwartz (2001) least-squares Monte Carlo for American options.

The implementation uses:
- Antithetic variates (M/2 base paths + M/2 mirror paths) to halve variance
- Polynomial basis [1, X, X²] where X = S_t/K (scaled for numerical stability)
- OLS regression restricted to in-the-money paths at each exercise date

Public API
----------
lsm_price(S, K, T, r, sigma, option_type, q, M, n, seed) -> float
lsm_price_with_stderr(S, K, T, r, sigma, option_type, q, M, n, seed)
    -> (price, stderr, ci_lower_95, ci_upper_95)
"""

import numpy as np


def lsm_price(S: float, K: float, T: float, r: float,
              sigma: float, option_type: str = "put",
              q: float = 0.0, M: int = 10_000,
              n: int = 252, seed: int = 42) -> float:
    """
    Longstaff-Schwartz least-squares Monte Carlo American option pricer.

    Parameters
    ----------
    S           : spot price
    K           : strike price
    T           : time to maturity in years
    r           : continuously compounded risk-free rate
    sigma       : volatility
    option_type : 'call' or 'put'
    q           : continuous dividend yield (default 0.0)
    M           : total simulation paths (antithetic: M/2 base + M/2 mirror)
    n           : time steps per path — exercise opportunities (default 252)
    seed        : random seed for reproducibility

    Returns
    -------
    float : option price estimate, or np.nan if inputs are invalid

    Notes
    -----
    At M=10,000 the 95% CI width is typically < $0.10 for near-ATM options
    with T <= 3 months.  Increase M for LEAPS or when higher precision is
    needed.  Computation scales linearly with M.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0 or M < 2:
        return np.nan

    rng  = np.random.default_rng(seed)
    dt   = T / n
    disc = np.exp(-r * dt)

    # ── Step 1: Simulate GBM paths with antithetic variates ──────────────────
    half      = M // 2
    Z         = rng.standard_normal((half, n))
    Z         = np.concatenate([Z, -Z], axis=0)          # (M, n)
    drift     = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_S     = np.log(S) + np.cumsum(drift + diffusion, axis=1)
    S_paths   = np.exp(log_S)                            # (M, n) — excludes t=0

    # ── Step 2: Terminal payoffs ──────────────────────────────────────────────
    S_T      = S_paths[:, -1]
    cashflow = (np.maximum(S_T - K, 0.0) if option_type == "call"
                else np.maximum(K - S_T, 0.0))

    # ── Step 3: Backward induction (Longstaff-Schwartz regression) ───────────
    for t in range(n - 2, -1, -1):
        S_t = S_paths[:, t]

        intrinsic = (np.maximum(S_t - K, 0.0) if option_type == "call"
                     else np.maximum(K - S_t, 0.0))

        itm = intrinsic > 0
        if itm.sum() < 5:
            cashflow *= disc
            continue

        Y     = cashflow[itm] * disc        # discounted future cashflow
        X     = S_t[itm] / K                # scaled spot (numerical stability)
        basis = np.column_stack([np.ones_like(X), X, X ** 2])

        coef, _, _, _ = np.linalg.lstsq(basis, Y, rcond=None)
        continuation  = basis @ coef

        exercise            = itm.copy()
        exercise[itm]       = intrinsic[itm] > continuation
        cashflow[exercise]  = intrinsic[exercise]
        cashflow[~exercise] *= disc

    # ── Step 4: Price = mean discounted cashflow ──────────────────────────────
    return float(np.mean(cashflow) * disc)


def lsm_price_with_stderr(S: float, K: float, T: float, r: float,
                          sigma: float, option_type: str = "put",
                          q: float = 0.0, M: int = 10_000,
                          n: int = 252, seed: int = 42):
    """
    LSM price with a 95% confidence interval estimate.

    Runs two independent batches of M/2 paths each and uses the
    half-range as a proxy for the standard error.

    Returns
    -------
    tuple : (price, stderr, ci_lower_95, ci_upper_95)
            All np.nan if either batch fails.
    """
    half = M // 2
    p1   = lsm_price(S, K, T, r, sigma, option_type, q, M=half, n=n, seed=seed)
    p2   = lsm_price(S, K, T, r, sigma, option_type, q, M=half, n=n, seed=seed + 1)

    if np.isnan(p1) or np.isnan(p2):
        return np.nan, np.nan, np.nan, np.nan

    price  = (p1 + p2) / 2
    stderr = abs(p1 - p2) / 2
    return price, stderr, price - 1.96 * stderr, price + 1.96 * stderr
