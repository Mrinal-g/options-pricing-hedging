"""
src/models/binomial.py
----------------------
Cox-Ross-Rubinstein (1979) recombining binomial tree for American and
European option pricing.

Public API
----------
crr_price(S, K, T, r, sigma, option_type, q, N, american) -> float
crr_delta(S, K, T, r, sigma, option_type, q, N, american, dS_frac) -> float
crr_early_exercise_boundary(S, K, T, r, sigma, q, N) -> (times, boundary)
"""

import numpy as np


def crr_price(S: float, K: float, T: float, r: float,
              sigma: float, option_type: str = "put",
              q: float = 0.0, N: int = 200,
              american: bool = True) -> float:
    """
    Cox-Ross-Rubinstein binomial tree option pricer.

    Parameters
    ----------
    S           : spot price
    K           : strike price
    T           : time to maturity in years
    r           : continuously compounded risk-free rate
    sigma       : volatility
    option_type : 'call' or 'put'
    q           : continuous dividend yield (default 0.0)
    N           : number of time steps (default 200; converged at N>=150)
    american    : if True, check for early exercise at each node

    Returns
    -------
    float : option price, or np.nan if inputs are invalid

    Notes
    -----
    For q=0 calls the American price equals the European BSM price —
    early exercise of a call on a non-dividend-paying stock is never optimal.
    The early exercise value is non-trivial only for puts (and calls with q>0).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0 or N < 1:
        return np.nan

    dt   = T / N
    u    = np.exp(sigma * np.sqrt(dt))
    d    = 1.0 / u
    disc = np.exp(-r * dt)
    p    = (np.exp((r - q) * dt) - d) / (u - d)

    if not (0 < p < 1):
        return np.nan

    # Terminal asset prices: node j has j up-moves and (N-j) down-moves
    j   = np.arange(N + 1)
    S_T = S * (u ** j) * (d ** (N - j))

    # Terminal payoffs
    V = np.maximum(S_T - K, 0.0) if option_type == "call" \
        else np.maximum(K - S_T, 0.0)

    # Backward induction
    for i in range(N - 1, -1, -1):
        j_i = np.arange(i + 1)
        S_i = S * (u ** j_i) * (d ** (i - j_i))

        V = disc * (p * V[1:i + 2] + (1 - p) * V[0:i + 1])

        if american:
            intrinsic = (np.maximum(S_i - K, 0.0) if option_type == "call"
                         else np.maximum(K - S_i, 0.0))
            V = np.maximum(V, intrinsic)

    return float(V[0])


def crr_delta(S: float, K: float, T: float, r: float,
              sigma: float, option_type: str = "put",
              q: float = 0.0, N: int = 200,
              american: bool = True,
              dS_frac: float = 0.01) -> float:
    """
    CRR delta via central finite difference: (V(S+dS) - V(S-dS)) / (2·dS).

    Parameters
    ----------
    dS_frac : fractional bump size as a fraction of S (default 0.01 = 1%)
    """
    dS     = S * dS_frac
    V_up   = crr_price(S + dS, K, T, r, sigma, option_type, q, N, american)
    V_down = crr_price(S - dS, K, T, r, sigma, option_type, q, N, american)

    if np.isnan(V_up) or np.isnan(V_down):
        return np.nan

    return (V_up - V_down) / (2 * dS)


def crr_early_exercise_boundary(S: float, K: float, T: float,
                                r: float, sigma: float,
                                q: float = 0.0,
                                N: int = 200):
    """
    Compute the early exercise boundary S*(t) for an American put via CRR.

    At each time step, S*(t) is the highest spot price at which immediate
    exercise dominates holding the option.  Above S*(t) the holder waits;
    at or below S*(t) the holder exercises immediately.

    Parameters
    ----------
    S, K, T, r, sigma, q : standard option parameters
    N : number of tree steps (default 200)

    Returns
    -------
    times    : np.ndarray of shape (N+1,) — time in years from today
    boundary : np.ndarray of shape (N+1,) — S*(t) at each node,
               np.nan where no early exercise is optimal
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0 or N < 1:
        return None, None

    dt   = T / N
    u    = np.exp(sigma * np.sqrt(dt))
    d    = 1.0 / u
    disc = np.exp(-r * dt)
    p    = (np.exp((r - q) * dt) - d) / (u - d)

    if not (0 < p < 1):
        return None, None

    # Full asset price tree: asset[i, j] = S * u^j * d^(i-j)
    asset = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            asset[i, j] = S * (u ** j) * (d ** (i - j))

    # Terminal payoffs (puts only — calls have no early exercise for q=0)
    V = np.maximum(K - asset[N, :N + 1], 0.0)

    boundary     = np.full(N + 1, np.nan)
    boundary[N]  = K  # at expiry, exercise boundary is the strike

    for i in range(N - 1, -1, -1):
        V_cont    = disc * (p * V[1:i + 2] + (1 - p) * V[0:i + 1])
        S_i       = asset[i, :i + 1]
        intrinsic = np.maximum(K - S_i, 0.0)
        V         = np.maximum(V_cont, intrinsic)

        exercise = intrinsic > V_cont
        if exercise.any():
            boundary[i] = S_i[exercise].max()

    times = np.linspace(0, T, N + 1)
    return times, boundary
