"""
src/risk/analytics.py
---------------------
Sensitivity analysis, P&L attribution, Greeks ladder, and smile
visualisation data — functions that a desk uses every morning.

Public API
----------
greeks_ladder(options_df, S, r, q)         -> pd.DataFrame
pnl_attribution(sim_df, S0, r, sigma, q)   -> pd.DataFrame
smile_data(options_df)                     -> pd.DataFrame
spot_ladder(positions, S, r, q, shocks)    -> pd.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.black_scholes import bsm_greeks, bsm_price, bsm_delta


# ── Greeks ladder ─────────────────────────────────────────────────────────────

def greeks_ladder(
    options_df: pd.DataFrame,
    S: float,
    r: float,
    q: float = 0.0,
) -> pd.DataFrame:
    """
    Greeks ladder: compute BSM Greeks for every option in the dataset
    and return a structured table organised by expiry and strike.

    This is what a desk looks at every morning — not individual options
    but the full cross-section of delta, gamma, vega, theta across the
    entire option surface.

    Parameters
    ----------
    options_df : priced options DataFrame (output of Stage 4).
                 Must have columns: expiration, option_type, strike,
                 ttm, iv_engine, mid, bid, ask
    S          : current spot price
    r          : risk-free rate
    q          : dividend yield

    Returns
    -------
    pd.DataFrame with columns:
        expiration, option_type, strike, ttm, moneyness,
        iv_engine, mid, delta, gamma, vega, theta, rho,
        dollar_delta, dollar_gamma_1pct, dollar_vega_1pt,
        daily_theta
    """
    rows = []
    for _, row in options_df.iterrows():
        sigma = row.get("iv_engine", np.nan)
        if pd.isna(sigma) or sigma <= 0:
            continue

        ttm  = float(row["ttm"])
        K    = float(row["strike"])
        otype = str(row["option_type"])

        g = bsm_greeks(S, K, ttm, r, sigma, otype, q)

        rows.append({
            "expiration"        : row["expiration"],
            "option_type"       : otype,
            "strike"            : K,
            "ttm"               : round(ttm, 4),
            "log_moneyness"     : round(float(row.get("log_moneyness", np.log(K / S))), 4),
            "iv_engine"         : round(float(sigma), 4),
            "mid"               : round(float(row.get("mid", np.nan)), 4),
            "delta"             : round(float(g["delta"]), 4),
            "gamma"             : round(float(g["gamma"]), 6),
            "vega"              : round(float(g["vega"]),  4),
            "theta"             : round(float(g["theta"]), 4),
            "rho"               : round(float(g["rho"]),   4),
            # Dollar Greeks — what matters for hedging
            "dollar_delta"      : round(float(g["delta"]) * S, 2),
            "dollar_gamma_1pct" : round(0.5 * float(g["gamma"]) * (S * 0.01) ** 2 * 200, 4),
            "dollar_vega_1pt"   : round(float(g["vega"]) / 100, 4),
            "daily_theta"       : round(float(g["theta"]), 4),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["expiration"] = pd.to_datetime(df["expiration"])
    return df.sort_values(
        ["expiration", "option_type", "strike"]
    ).reset_index(drop=True)


# ── Greeks by expiry (aggregated) ─────────────────────────────────────────────

def greeks_by_expiry_from_df(
    options_df: pd.DataFrame,
    S: float,
    r: float,
    q: float = 0.0,
    quantity: float = -1.0,
) -> pd.DataFrame:
    """
    Aggregate Greeks by expiry across the entire options dataset.

    Assumes the entire dataset represents a short book (quantity = -1 per
    option by default, i.e. sold all options). Adjust quantity for your
    actual book composition.

    Returns
    -------
    pd.DataFrame indexed by expiration with columns:
        n_options, total_value, net_delta, net_gamma, net_vega, net_theta,
        dollar_delta, dollar_vega_1pt, daily_theta
    """
    ladder = greeks_ladder(options_df, S, r, q)
    if ladder.empty:
        return pd.DataFrame()

    # Apply position sign (default short = -1)
    for col in ["delta", "gamma", "vega", "theta", "rho",
                "dollar_delta", "dollar_gamma_1pct", "dollar_vega_1pt"]:
        if col in ladder.columns:
            ladder[col] = ladder[col] * quantity

    agg = (
        ladder.groupby("expiration")
        .agg(
            n_options       =("strike",          "count"),
            net_delta       =("delta",            "sum"),
            net_gamma       =("gamma",            "sum"),
            net_vega        =("vega",             "sum"),
            net_theta       =("theta",            "sum"),
            dollar_delta    =("dollar_delta",     "sum"),
            dollar_vega_1pt =("dollar_vega_1pt",  "sum"),
            daily_theta     =("daily_theta",      "sum"),
        )
        .round(4)
    )
    return agg


# ── P&L attribution ───────────────────────────────────────────────────────────

def pnl_attribution(
    sim_df: pd.DataFrame,
    S0: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> pd.DataFrame:
    """
    Break down daily P&L into delta, gamma, theta, and residual components.

    The Taylor expansion of option price change over one step:
        ΔV ≈ delta × ΔS  +  0.5 × gamma × ΔS²  +  theta × Δt  +  residual

    This tells you WHY you made or lost money each day:
      - Delta P&L : linear exposure to spot move (eliminated by delta hedge)
      - Gamma P&L : quadratic exposure (unavoidable without gamma hedge)
      - Theta P&L : time decay earned (positive for short options)
      - Residual  : everything else (vol moves, higher-order terms)

    Parameters
    ----------
    sim_df : output of simulate_delta_hedge() — one row per time step
    S0     : initial spot price
    r      : risk-free rate
    sigma  : implied volatility used for the hedge
    q      : dividend yield

    Returns
    -------
    pd.DataFrame with columns:
        step, t, S, dS, pnl_daily, delta_pnl, gamma_pnl,
        theta_pnl, residual_pnl, cumulative_delta,
        cumulative_gamma, cumulative_theta, cumulative_residual
    """
    df = sim_df.copy()
    df["dS"] = df["S"].diff().fillna(0.0)
    dt = float(df["t"].diff().fillna(df["t"].iloc[1] if len(df) > 1 else 1/252).iloc[1])

    rows = []
    for i, row in df.iterrows():
        t_rem = max(float(row.get("option_value", 0)) , 0)
        S_t   = float(row["S"])
        dS    = float(row["dS"])
        step  = int(row["step"])

        # Greeks at this step (use previous row's remaining TTM)
        if step == 0:
            rows.append({
                "step": 0, "t": 0.0, "S": S_t, "dS": 0.0,
                "pnl_daily": 0.0, "delta_pnl": 0.0,
                "gamma_pnl": 0.0, "theta_pnl": 0.0, "residual_pnl": 0.0,
            })
            continue

        # TTM at start of this step
        t_step = float(row["t"])
        t_prev = t_step - dt
        if t_prev <= 0:
            t_prev = 1e-6

        # Use the delta stored in the sim (already computed)
        delta_t = float(df.iloc[i - 1]["delta"]) if not np.isnan(df.iloc[i - 1]["delta"]) else 0.0
        gamma_t = float(df.iloc[i - 1].get("gamma", np.nan))
        theta_t = float(df.iloc[i - 1].get("theta", np.nan))

        if np.isnan(gamma_t):
            from src.models.black_scholes import bsm_gamma
            gamma_t = bsm_gamma(S_t, df.iloc[i-1]["S"],
                                 t_prev, r, sigma, q)
        if np.isnan(theta_t):
            from src.models.black_scholes import bsm_theta
            theta_t = bsm_theta(S_t, df.iloc[i-1]["S"],
                                 t_prev, r, sigma, "put", q)

        pnl       = float(row["pnl_daily"])
        delta_pnl = delta_t * dS
        gamma_pnl = 0.5 * gamma_t * dS ** 2
        theta_pnl = theta_t * dt * 365  # theta is per calendar day

        # For short option: delta hedge cancels delta P&L
        # Net daily P&L = -delta_pnl (from short option) + delta_pnl (from shares)
        #               + gamma_pnl + theta_pnl + residual
        residual  = pnl - gamma_pnl - theta_pnl

        rows.append({
            "step"       : step,
            "t"          : t_step,
            "S"          : S_t,
            "dS"         : dS,
            "pnl_daily"  : round(pnl, 6),
            "delta_pnl"  : round(delta_pnl, 6),
            "gamma_pnl"  : round(gamma_pnl, 6),
            "theta_pnl"  : round(theta_pnl, 6),
            "residual_pnl": round(residual, 6),
        })

    result = pd.DataFrame(rows)

    # Cumulative components
    for col in ["delta_pnl", "gamma_pnl", "theta_pnl", "residual_pnl", "pnl_daily"]:
        result[f"cum_{col}"] = result[col].cumsum()

    return result


# ── Smile data for plotting ───────────────────────────────────────────────────

def smile_data(options_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare smile plot data from the options dataset.

    Returns market IV points and SVI fitted smile curves per expiry,
    ready for plotting. This is the 2D slice view of the surface that
    the 3D chart cannot show — you can inspect individual expiry
    calibration quality here.

    Parameters
    ----------
    options_df : options DataFrame with columns:
                 expiration, log_moneyness, iv_model, iv_engine,
                 option_type, ttm, days_to_expiry

    Returns
    -------
    pd.DataFrame with columns:
        expiration, log_moneyness, iv_model (market),
        iv_engine (SVI surface), option_type, days_to_expiry
    """
    needed = ["expiration", "log_moneyness", "iv_model", "iv_engine", "option_type"]
    missing = [c for c in needed if c not in options_df.columns]
    if missing:
        return pd.DataFrame()

    df = options_df[needed + ["days_to_expiry"]].dropna(
        subset=["log_moneyness", "iv_model", "iv_engine"]
    ).copy()

    df["expiration"]   = pd.to_datetime(df["expiration"])
    df["log_moneyness"] = pd.to_numeric(df["log_moneyness"], errors="coerce")
    df["iv_model"]     = pd.to_numeric(df["iv_model"],      errors="coerce")
    df["iv_engine"]    = pd.to_numeric(df["iv_engine"],     errors="coerce")

    return df.sort_values(["expiration", "log_moneyness"]).reset_index(drop=True)


# ── Spot ladder (what-if across spot levels) ──────────────────────────────────

def spot_ladder(
    options_df: pd.DataFrame,
    S: float,
    r: float,
    q: float = 0.0,
    shocks: list[float] | None = None,
) -> pd.DataFrame:
    """
    Re-price the entire option book at a range of spot levels.

    Shows how total portfolio value, net delta, net gamma, and net vega
    change as spot moves. This is the 'what-if' sensitivity table a
    trader looks at to understand their exposure before hedging.

    Parameters
    ----------
    options_df : options DataFrame with iv_engine column
    S          : current spot price
    r          : risk-free rate
    q          : dividend yield
    shocks     : list of multiplicative spot shocks
                 (default: -20% to +20% in 5% steps)

    Returns
    -------
    pd.DataFrame indexed by shocked_spot with columns:
        spot_shock_pct, portfolio_value, value_change,
        net_delta, net_gamma, net_vega, net_theta
    """
    if shocks is None:
        shocks = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]

    # Build current book as OptionPosition list
    from src.risk.greeks import OptionPosition, portfolio_greeks

    def _build_positions(df, sigma_col="iv_engine"):
        positions = []
        for _, row in df.iterrows():
            sig = row.get(sigma_col, np.nan)
            if pd.isna(sig) or sig <= 0 or row["ttm"] <= 0:
                continue
            positions.append(OptionPosition(
                K=float(row["strike"]),
                T=float(row["ttm"]),
                sigma=float(sig),
                option_type=str(row["option_type"]),
                quantity=-1.0,   # short the whole book
            ))
        return positions

    positions    = _build_positions(options_df)
    base_greeks  = portfolio_greeks(positions, S, r, q)
    base_value   = base_greeks["total_value"]

    rows = []
    for shock in shocks:
        S_shocked = S * shock
        g = portfolio_greeks(positions, S_shocked, r, q)
        rows.append({
            "shocked_spot"   : round(S_shocked, 2),
            "spot_shock_pct" : round((shock - 1) * 100, 1),
            "portfolio_value": round(g["total_value"], 4),
            "value_change"   : round(g["total_value"] - base_value, 4),
            "net_delta"      : round(g["net_delta"],  4),
            "net_gamma"      : round(g["net_gamma"],  6),
            "net_vega"       : round(g["net_vega"],   4),
            "net_theta"      : round(g["net_theta"],  4),
            "dollar_delta"   : round(g["dollar_delta"], 2),
        })

    return pd.DataFrame(rows).set_index("shocked_spot")
