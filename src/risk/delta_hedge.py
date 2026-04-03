"""
src/risk/delta_hedge.py
-----------------------
Delta hedging simulator for a short option position.

Simulates the classic textbook P&L replication argument:
  - At t=0  : sell one option, delta-hedge by buying delta shares
  - Each day : rebalance the hedge to the new delta
  - At expiry: close all positions and compute net P&L

The cumulative P&L reveals how well the hedge worked and what the
main sources of error are:
  - Gamma P&L    : curvature drag from discrete rebalancing
  - Theta P&L    : time decay earned by the short option
  - Vol P&L      : difference between realised vol and implied vol

This is exactly the exercise junior quants run on option desks to
understand model risk and hedge effectiveness.

Public API
----------
simulate_delta_hedge(
    S0, K, T, r, sigma_implied, option_type, q,
    n_steps, transaction_cost_pct, seed
) -> pd.DataFrame

hedge_summary(sim_df, S0, K, T, r, sigma_implied, option_type, q)
    -> dict

run_frequency_analysis(
    S0, K, T, r, sigma_implied, option_type, q,
    frequencies, n_sims, seed
) -> pd.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.black_scholes import bsm_delta, bsm_greeks, bsm_price


# ── core simulator ────────────────────────────────────────────────────────────

def simulate_delta_hedge(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma_implied: float,
    option_type: str = "put",
    q: float = 0.0,
    n_steps: int = 252,
    sigma_realised: float | None = None,
    transaction_cost_pct: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a delta-hedged short option position over the option's life.

    Setup
    -----
    - At t=0: sell 1 option at BSM price (sigma_implied).
              Buy delta shares to delta-hedge.  Fund the net position with cash.
    - Each step: rebalance hedge to the new BSM delta.
                 Transaction costs are charged on the share turnover.
    - At expiry: exercise/expire option.  Close share position.
                 Final cash balance = total P&L.

    Parameters
    ----------
    S0                   : initial spot price
    K                    : strike price
    T                    : time to maturity in years
    r                    : continuously compounded risk-free rate
    sigma_implied        : IV used to price the option and compute deltas
    option_type          : 'call' or 'put'
    q                    : continuous dividend yield (default 0.0)
    n_steps              : number of hedging steps (default 252 = daily)
    sigma_realised       : vol used to simulate the underlying path.
                           If None, uses sigma_implied (perfect vol forecast).
                           Set sigma_realised != sigma_implied to study vol P&L.
    transaction_cost_pct : one-way transaction cost as % of trade value (e.g. 0.001)
    seed                 : random seed for reproducibility

    Returns
    -------
    pd.DataFrame with one row per time step, columns:
        step, t, S, delta, shares_held, cash, option_value,
        portfolio_value, pnl_cumulative, pnl_daily,
        gamma, theta, vega, transaction_cost
    """
    if sigma_realised is None:
        sigma_realised = sigma_implied

    rng = np.random.default_rng(seed)
    dt  = T / n_steps

    # ── Step 0: initialise at t=0 ─────────────────────────────────────────────
    S            = S0
    option_value = bsm_price(S, K, T, r, sigma_implied, option_type, q)
    delta        = bsm_delta(S, K, T, r, sigma_implied, option_type, q)

    # Sell the option: receive option_value in cash.
    # Buy delta shares to hedge: pay delta * S from cash.
    shares_held = delta
    cash        = option_value - delta * S   # net cash position (funded at risk-free)

    records = []
    greeks  = bsm_greeks(S, K, T, r, sigma_implied, option_type, q)

    records.append({
        "step"            : 0,
        "t"               : 0.0,
        "S"               : S,
        "delta"           : delta,
        "shares_held"     : shares_held,
        "cash"            : cash,
        "option_value"    : option_value,
        "portfolio_value" : shares_held * S + cash - option_value,
        "pnl_daily"       : 0.0,
        "pnl_cumulative"  : 0.0,
        "gamma"           : greeks["gamma"],
        "theta"           : greeks["theta"],
        "vega"            : greeks["vega"],
        "transaction_cost": 0.0,
    })

    prev_portfolio = shares_held * S + cash - option_value

    # ── Steps 1 … n_steps ─────────────────────────────────────────────────────
    for step in range(1, n_steps + 1):
        t_remaining = T - step * dt

        # Simulate one GBM step
        Z  = rng.standard_normal()
        dS = S * ((r - q - 0.5 * sigma_realised ** 2) * dt
                  + sigma_realised * np.sqrt(dt) * Z)
        S  = S + dS

        # Accrue risk-free interest on cash
        cash *= np.exp(r * dt)

        if step == n_steps:
            # ── Expiry: close all positions ────────────────────────────────────
            if option_type == "call":
                payoff = max(S - K, 0.0)
            else:
                payoff = max(K - S, 0.0)

            # Close share position
            proceeds       = shares_held * S
            tc             = transaction_cost_pct * abs(proceeds)
            cash          += proceeds - tc
            # Pay option payoff to the buyer
            cash          -= payoff

            new_delta      = np.nan
            new_option_val = payoff
            gamma = theta = vega = np.nan
        else:
            # ── Rebalance: compute new delta and trade shares ───────────────────
            new_delta      = bsm_delta(S, K, t_remaining, r, sigma_implied, option_type, q)
            new_option_val = bsm_price(S, K, t_remaining, r, sigma_implied, option_type, q)
            greeks         = bsm_greeks(S, K, t_remaining, r, sigma_implied, option_type, q)
            gamma          = greeks["gamma"]
            theta          = greeks["theta"]
            vega           = greeks["vega"]

            # Trade the difference in shares
            shares_trade   = new_delta - shares_held
            tc             = transaction_cost_pct * abs(shares_trade * S)
            cash          -= shares_trade * S + tc
            shares_held    = new_delta

        portfolio_value = (shares_held if not np.isnan(new_delta) else 0.0) * S \
                          + cash - (new_option_val if step < n_steps else 0.0)

        if step == n_steps:
            portfolio_value = cash  # all positions closed

        pnl_daily       = portfolio_value - prev_portfolio
        pnl_cumulative  = portfolio_value

        records.append({
            "step"            : step,
            "t"               : step * dt,
            "S"               : S,
            "delta"           : new_delta if step < n_steps else np.nan,
            "shares_held"     : shares_held if step < n_steps else 0.0,
            "cash"            : cash,
            "option_value"    : new_option_val,
            "portfolio_value" : portfolio_value,
            "pnl_daily"       : pnl_daily,
            "pnl_cumulative"  : pnl_cumulative,
            "gamma"           : gamma if step < n_steps else np.nan,
            "theta"           : theta if step < n_steps else np.nan,
            "vega"            : vega  if step < n_steps else np.nan,
            "transaction_cost": tc,
        })

        prev_portfolio = portfolio_value

    return pd.DataFrame(records)


# ── summary statistics ────────────────────────────────────────────────────────

def hedge_summary(
    sim_df: pd.DataFrame,
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma_implied: float,
    option_type: str = "put",
    q: float = 0.0,
) -> dict:
    """
    Compute summary statistics for a single hedge simulation.

    Returns
    -------
    dict with keys:
        initial_option_value  : BSM price at t=0
        final_pnl             : total P&L at expiry (positive = profit)
        pnl_as_pct_premium    : P&L as % of initial option premium
        total_transaction_cost: sum of all transaction costs paid
        pnl_net_of_tc         : final P&L after transaction costs
        max_drawdown          : worst intra-period cumulative P&L
        pnl_std               : daily P&L standard deviation
        n_steps               : number of hedging steps
        sigma_implied         : vol used to price and delta-hedge
        moneyness             : ln(S0/K)
    """
    initial_premium = bsm_price(S0, K, T, r, sigma_implied, option_type, q)

    final_pnl  = sim_df["pnl_cumulative"].iloc[-1]
    total_tc   = sim_df["transaction_cost"].sum()
    daily_pnls = sim_df["pnl_daily"].dropna()

    return {
        "initial_option_value" : round(initial_premium, 4),
        "final_pnl"            : round(final_pnl, 4),
        "pnl_as_pct_premium"   : round(100 * final_pnl / initial_premium, 2)
                                  if initial_premium > 0 else np.nan,
        "total_transaction_cost": round(total_tc, 4),
        "pnl_net_of_tc"        : round(final_pnl - total_tc, 4),
        "max_drawdown"         : round(sim_df["pnl_cumulative"].min(), 4),
        "pnl_std"              : round(float(daily_pnls.std()), 4),
        "n_steps"              : len(sim_df) - 1,
        "sigma_implied"        : sigma_implied,
        "moneyness"            : round(np.log(S0 / K), 4),
    }


# ── hedging frequency analysis ────────────────────────────────────────────────

def run_frequency_analysis(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma_implied: float,
    option_type: str = "put",
    q: float = 0.0,
    frequencies: list[int] | None = None,
    n_sims: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compare hedge P&L across different rebalancing frequencies.

    Runs n_sims independent GBM paths for each frequency and reports
    the distribution of final P&L.  Shows the classic tradeoff:
      - More frequent rebalancing → tighter hedge → lower P&L variance
      - But higher transaction costs eat into net P&L

    Parameters
    ----------
    frequencies : list of step counts per year to test
                  e.g. [4, 12, 52, 252] = quarterly/monthly/weekly/daily
                  Default: [4, 12, 52, 252]
    n_sims      : number of Monte Carlo paths per frequency (default 200)

    Returns
    -------
    pd.DataFrame indexed by frequency with columns:
        mean_pnl, std_pnl, pnl_5pct, pnl_95pct,
        mean_tc, mean_pnl_net_tc
    """
    if frequencies is None:
        frequencies = [4, 12, 52, 252]

    freq_labels = {4: "Quarterly", 12: "Monthly", 52: "Weekly", 252: "Daily"}
    rows = []

    for freq in frequencies:
        pnls    = []
        tcs     = []
        for sim_idx in range(n_sims):
            sim_df = simulate_delta_hedge(
                S0=S0, K=K, T=T, r=r,
                sigma_implied=sigma_implied,
                option_type=option_type,
                q=q,
                n_steps=freq,
                seed=seed + sim_idx,
            )
            pnls.append(sim_df["pnl_cumulative"].iloc[-1])
            tcs.append(sim_df["transaction_cost"].sum())

        pnls = np.array(pnls)
        tcs  = np.array(tcs)

        rows.append({
            "frequency"       : freq,
            "label"           : freq_labels.get(freq, f"{freq} steps/yr"),
            "mean_pnl"        : round(float(np.mean(pnls)), 4),
            "std_pnl"         : round(float(np.std(pnls)), 4),
            "pnl_5pct"        : round(float(np.percentile(pnls, 5)), 4),
            "pnl_95pct"       : round(float(np.percentile(pnls, 95)), 4),
            "mean_tc"         : round(float(np.mean(tcs)), 4),
            "mean_pnl_net_tc" : round(float(np.mean(pnls - tcs)), 4),
        })

    return pd.DataFrame(rows).set_index("frequency")


# ── vol mismatch analysis ─────────────────────────────────────────────────────

def run_vol_mismatch_analysis(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma_implied: float,
    option_type: str = "put",
    q: float = 0.0,
    realised_vols: list[float] | None = None,
    n_sims: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Show how P&L changes when realised vol differs from implied vol.

    This is the vol P&L (or "vega P&L"): if you delta-hedge with the wrong
    vol, you systematically over- or under-hedge and accumulate a drift.
    Specifically:

        E[P&L] ≈ 0.5 * Γ * S² * (σ_realised² - σ_implied²) * dt  per step

    So if σ_realised > σ_implied, the short gamma position bleeds money
    (you undercharged for the option).

    Parameters
    ----------
    realised_vols : list of realised vols to test
                    Default: [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    Returns
    -------
    pd.DataFrame indexed by realised_vol with columns:
        mean_pnl, std_pnl, pnl_5pct, pnl_95pct,
        vol_spread (realised - implied)
    """
    if realised_vols is None:
        realised_vols = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    rows = []
    for sig_r in realised_vols:
        pnls = []
        for sim_idx in range(n_sims):
            sim_df = simulate_delta_hedge(
                S0=S0, K=K, T=T, r=r,
                sigma_implied=sigma_implied,
                sigma_realised=sig_r,
                option_type=option_type,
                q=q,
                n_steps=252,
                seed=seed + sim_idx,
            )
            pnls.append(sim_df["pnl_cumulative"].iloc[-1])

        pnls = np.array(pnls)
        rows.append({
            "realised_vol" : round(sig_r, 4),
            "vol_spread"   : round(sig_r - sigma_implied, 4),
            "mean_pnl"     : round(float(np.mean(pnls)), 4),
            "std_pnl"      : round(float(np.std(pnls)), 4),
            "pnl_5pct"     : round(float(np.percentile(pnls, 5)), 4),
            "pnl_95pct"    : round(float(np.percentile(pnls, 95)), 4),
        })

    return pd.DataFrame(rows).set_index("realised_vol")


# ── historical path hedge simulator ──────────────────────────────────────────

def simulate_delta_hedge_historical(
    spot_hist: "pd.DataFrame",
    K: float,
    T_days: int,
    r: float,
    sigma_implied: float,
    option_type: str = "put",
    q: float = 0.0,
    transaction_cost_pct: float = 0.0,
    start_idx: int = None,
) -> "pd.DataFrame":
    """
    Replay a delta hedge using the ACTUAL historical spot price path.

    Unlike simulate_delta_hedge() which generates a synthetic GBM path,
    this function uses the real closing prices from spot_hist. This gives
    a genuine out-of-sample hedge P&L: what would have happened if you
    had sold this option on a past date and hedged it daily using the
    actual subsequent stock prices.

    The realised vol of the hedge is whatever GOOG actually did —
    not a model assumption. This is the honest test of hedge quality.

    Parameters
    ----------
    spot_hist   : DataFrame with columns ['date', 'close']
                  (output of src.data.download.download_spot_history)
    K           : strike price
    T_days      : option tenor in calendar days (e.g. 90)
    r           : risk-free rate
    sigma_implied: IV used to compute BSM deltas (from the SVI surface
                   calibrated on the day the option is sold)
    option_type : 'call' or 'put'
    q           : continuous dividend yield
    transaction_cost_pct : one-way cost as % of trade value
    start_idx   : row index in spot_hist where the option is SOLD.
                  If None, uses the row that is T_days before the end
                  of the history — i.e. the most recent complete period.

    Returns
    -------
    pd.DataFrame with one row per trading day, columns:
        date, step, S, delta, shares_held, cash, option_value,
        pnl_cumulative, pnl_daily, transaction_cost

    Notes
    -----
    This uses calendar days for TTM (T_days / 365) but advances through
    the spot_hist row by row (trading days only). Short months with fewer
    trading days than expected are handled gracefully — the simulation
    ends at the last available row or expiry, whichever comes first.
    """
    import pandas as pd

    prices = spot_hist["close"].dropna().values
    dates  = pd.to_datetime(spot_hist["date"]).values

    T_years = T_days / 365.0

    # Determine start index
    if start_idx is None:
        # Find a start point such that ~T_days calendar days of history remain
        end_date   = pd.Timestamp(dates[-1])
        start_date = end_date - pd.Timedelta(days=T_days)
        # Find closest row at or after start_date
        date_series = pd.Series(dates)
        candidates  = date_series[date_series >= start_date]
        if candidates.empty:
            raise ValueError(
                f"Not enough history for a {T_days}-day hedge. "
                f"Earliest available: {pd.Timestamp(dates[0]).date()}"
            )
        start_idx = candidates.index[0]

    n_avail = len(prices) - start_idx
    if n_avail < 5:
        raise ValueError("Too few historical rows remain after start_idx.")

    # ── t=0: sell option ──────────────────────────────────────────────────────
    S0           = float(prices[start_idx])
    option_value = bsm_price(S0, K, T_years, r, sigma_implied, option_type, q)
    delta        = bsm_delta(S0, K, T_years, r, sigma_implied, option_type, q)

    shares_held = delta
    cash        = option_value - delta * S0

    records = [{
        "date"            : pd.Timestamp(dates[start_idx]).date(),
        "step"            : 0,
        "S"               : S0,
        "delta"           : delta,
        "shares_held"     : shares_held,
        "cash"            : cash,
        "option_value"    : option_value,
        "pnl_cumulative"  : 0.0,
        "pnl_daily"       : 0.0,
        "transaction_cost": 0.0,
    }]
    prev_portfolio = shares_held * S0 + cash - option_value
    dt_rf = 1 / 252  # one trading day for risk-free accrual

    for step in range(1, n_avail):
        row_idx = start_idx + step
        if row_idx >= len(prices):
            break

        S   = float(prices[row_idx])
        dt_ = pd.Timestamp(dates[row_idx])

        # Calendar days elapsed since start — used for remaining TTM
        days_elapsed  = (dt_ - pd.Timestamp(dates[start_idx])).days
        t_remaining   = max(T_years - days_elapsed / 365.0, 0.0)

        cash *= np.exp(r * dt_rf)

        is_last = (step == n_avail - 1) or (t_remaining <= 0)

        if is_last:
            # Close all positions
            payoff       = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
            proceeds     = shares_held * S
            tc           = transaction_cost_pct * abs(proceeds)
            cash        += proceeds - tc - payoff
            new_delta    = np.nan
            new_opt_val  = payoff
        else:
            new_delta   = bsm_delta(S, K, t_remaining, r, sigma_implied, option_type, q)
            new_opt_val = bsm_price(S, K, t_remaining, r, sigma_implied, option_type, q)
            trade       = new_delta - shares_held
            tc          = transaction_cost_pct * abs(trade * S)
            cash       -= trade * S + tc
            shares_held = new_delta

        portfolio = cash if is_last else shares_held * S + cash - new_opt_val
        pnl_daily = portfolio - prev_portfolio

        records.append({
            "date"            : dt_.date(),
            "step"            : step,
            "S"               : S,
            "delta"           : new_delta,
            "shares_held"     : shares_held if not is_last else 0.0,
            "cash"            : cash,
            "option_value"    : new_opt_val,
            "pnl_cumulative"  : portfolio,
            "pnl_daily"       : pnl_daily,
            "transaction_cost": tc,
        })
        prev_portfolio = portfolio

        if is_last:
            break

    return pd.DataFrame(records)


# ── hedge effectiveness benchmark ─────────────────────────────────────────────

def compare_hedge_strategies(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma_implied: float,
    option_type: str = "put",
    q: float = 0.0,
    n_steps: int = 252,
    sigma_realised: float | None = None,
    transaction_cost_pct: float = 0.0,
    n_sims: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compare three strategies on the same set of random paths.

    This is the standard hedge effectiveness benchmark. Comparing only
    dynamic hedging in isolation is meaningless — you need to show how
    much variance it removes relative to doing nothing.

    Strategies
    ----------
    unhedged  : Sell option at t=0, collect premium, pay payoff at expiry.
                No stock position. P&L = premium - payoff.
                This is the worst case — full gamma and delta exposure.

    static    : Sell option, buy exactly delta_0 shares at t=0, never
                rebalance. Closes everything at expiry.
                Better than unhedged for large moves, worse near ATM.

    dynamic   : Full daily delta hedge — what we already have.
                Best variance reduction but highest transaction cost.

    Parameters
    ----------
    n_sims : number of independent GBM paths to run (default 200)
             Each path uses a different random seed so results are stable.

    Returns
    -------
    pd.DataFrame with columns:
        strategy, mean_pnl, std_pnl, pnl_5pct, pnl_95pct,
        pnl_range (95th-5th), variance_reduction_pct
        (variance_reduction_pct is relative to unhedged)
    """
    if sigma_realised is None:
        sigma_realised = sigma_implied

    premium    = bsm_price(S0, K, T, r, sigma_implied, option_type, q)
    delta_0    = bsm_delta(S0, K, T, r, sigma_implied, option_type, q)
    rng_master = np.random.default_rng(seed)

    unhedged_pnls = []
    static_pnls   = []
    dynamic_pnls  = []

    dt   = T / n_steps
    disc = np.exp(-r * dt)

    for sim_i in range(n_sims):
        sim_seed = int(rng_master.integers(0, 2**31))
        rng      = np.random.default_rng(sim_seed)

        # Simulate spot path (same path for all three strategies)
        Z      = rng.standard_normal(n_steps)
        drift  = (r - q - 0.5 * sigma_realised**2) * dt
        diff   = sigma_realised * np.sqrt(dt) * Z
        log_S  = np.log(S0) + np.cumsum(drift + diff)
        S_path = np.concatenate([[S0], np.exp(log_S)])   # shape (n_steps+1,)
        S_T    = S_path[-1]

        # Terminal payoff
        payoff = max(S_T - K, 0.0) if option_type == "call" else max(K - S_T, 0.0)

        # ── Strategy 1: Unhedged ──────────────────────────────────────────────
        # Sell option, invest premium at risk-free, pay payoff at expiry
        cash_rf  = premium * np.exp(r * T)   # premium grown at risk-free
        unhedged_pnls.append(cash_rf - payoff)

        # ── Strategy 2: Static hedge ──────────────────────────────────────────
        # Buy delta_0 shares at t=0, funded by borrowing
        # At expiry: close shares, repay borrowing + interest, pay/receive payoff
        tc_entry   = transaction_cost_pct * abs(delta_0 * S0)
        tc_exit    = transaction_cost_pct * abs(delta_0 * S_T)
        share_pnl  = delta_0 * (S_T - S0 * np.exp(r * T))  # shares gain minus cost of carry
        prem_rf    = premium * np.exp(r * T)
        static_pnls.append(prem_rf + share_pnl - payoff - tc_entry - tc_exit)

        # ── Strategy 3: Dynamic delta hedge ──────────────────────────────────
        sim_df = simulate_delta_hedge(
            S0=S0, K=K, T=T, r=r,
            sigma_implied=sigma_implied,
            option_type=option_type,
            q=q,
            n_steps=n_steps,
            sigma_realised=sigma_realised,
            transaction_cost_pct=transaction_cost_pct,
            seed=sim_seed,
        )
        dynamic_pnls.append(sim_df["pnl_cumulative"].iloc[-1])

    results   = {
        "Unhedged"       : np.array(unhedged_pnls),
        "Static hedge"   : np.array(static_pnls),
        "Dynamic hedge"  : np.array(dynamic_pnls),
    }
    var_unhedged = np.var(results["Unhedged"])

    rows = []
    for strategy, pnls in results.items():
        var_red = (1 - np.var(pnls) / var_unhedged) * 100 if var_unhedged > 0 else np.nan
        rows.append({
            "strategy"              : strategy,
            "mean_pnl"              : round(float(np.mean(pnls)), 4),
            "std_pnl"               : round(float(np.std(pnls)),  4),
            "pnl_5pct"              : round(float(np.percentile(pnls,  5)), 4),
            "pnl_95pct"             : round(float(np.percentile(pnls, 95)), 4),
            "pnl_range"             : round(float(np.percentile(pnls, 95) -
                                                   np.percentile(pnls,  5)), 4),
            "variance_reduction_pct": round(float(var_red), 1),
        })

    return pd.DataFrame(rows).set_index("strategy")
