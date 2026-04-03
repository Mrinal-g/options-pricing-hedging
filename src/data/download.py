"""
src/data/download.py
--------------------
Downloads spot price history and option chains from Yahoo Finance.

This module is the entry point for the data pipeline.  It wraps the
yfinance calls in structured functions with proper error handling so
the pipeline can be called from refresh.py or from a notebook.

Public API
----------
download_spot_history(ticker, start, end) -> pd.DataFrame
select_expiries(ticker_obj, valuation_date, target_days) -> pd.DataFrame
download_option_chains(ticker_obj, expiries, ticker, valuation_date, spot)
    -> pd.DataFrame
run_download(config) -> dict
"""

from __future__ import annotations

import warnings
from typing import List

import pandas as pd
import yfinance as yf

# Suppress noisy yfinance deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ── spot history ──────────────────────────────────────────────────────────────

def download_spot_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLC price history for `ticker` between `start` and `end`.

    Parameters
    ----------
    ticker : Yahoo Finance ticker symbol (e.g. 'GOOG')
    start  : start date string 'YYYY-MM-DD'
    end    : end date string 'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame with columns: date, open, high, low, close, volume, ...
    Raises ValueError if no data is returned.
    """
    ticker_obj = yf.Ticker(ticker)
    hist = ticker_obj.history(start=start, end=end, auto_adjust=False)

    if hist.empty:
        raise ValueError(
            f"No price history downloaded for {ticker}. "
            "Check the ticker symbol and date range in config.toml."
        )

    hist = hist.reset_index()
    hist.columns = [c.lower().replace(" ", "_") for c in hist.columns]

    # Strip timezone so date arithmetic with tz-naive option expirations works
    if "date" in hist.columns:
        hist["date"] = pd.to_datetime(hist["date"])
        if hist["date"].dt.tz is not None:
            hist["date"] = hist["date"].dt.tz_convert(None)

    return hist.copy()


# ── expiry selection ──────────────────────────────────────────────────────────

def select_expiries(
    ticker_obj: yf.Ticker,
    valuation_date: pd.Timestamp,
    target_days: List[int] = None,
) -> pd.DataFrame:
    """
    Select listed expiry dates closest to log-linearly spaced target maturities.

    Rather than taking the first N near-term expiries (which cluster short),
    we select expiries closest to [7, 30, 90, 180, 365] days.  This gives
    balanced term-structure coverage from weekly to one-year maturities.

    Parameters
    ----------
    ticker_obj     : yfinance Ticker object
    valuation_date : pricing date (timezone-naive Timestamp)
    target_days    : list of target days-to-expiry (default [7,30,90,180,365])

    Returns
    -------
    pd.DataFrame with columns: expiration, days_to_expiry
    Raises ValueError if no future expiries are found.
    """
    if target_days is None:
        target_days = [7, 30, 90, 180, 365]

    available = list(ticker_obj.options)
    if not available:
        raise ValueError(f"No option expiries found.")

    expiry_table = pd.DataFrame({"expiration": pd.to_datetime(available)})
    expiry_table["days_to_expiry"] = (
        expiry_table["expiration"] - valuation_date
    ).dt.days
    expiry_table = expiry_table[expiry_table["days_to_expiry"] > 0].copy()

    if expiry_table.empty:
        raise ValueError("No future expiries found.")

    selected = []
    for target in target_days:
        temp = expiry_table.copy()
        temp["dist"] = (temp["days_to_expiry"] - target).abs()
        selected.append(temp.sort_values("dist").iloc[0])

    df = pd.DataFrame(selected).drop_duplicates(subset=["expiration"])
    df = df.sort_values("expiration").reset_index(drop=True)
    return df[["expiration", "days_to_expiry"]]


# ── option chain download ─────────────────────────────────────────────────────

def download_option_chains(
    ticker_obj: yf.Ticker,
    selected_expiries: pd.DataFrame,
    ticker: str,
    valuation_date: pd.Timestamp,
    spot: float,
) -> pd.DataFrame:
    """
    Download calls and puts for each selected expiry.

    Failures on individual expiries are logged as warnings and skipped
    rather than aborting the entire download.

    Parameters
    ----------
    ticker_obj        : yfinance Ticker object
    selected_expiries : DataFrame with columns [expiration, days_to_expiry]
    ticker            : ticker symbol string (for tagging rows)
    valuation_date    : pricing date
    spot              : latest spot price

    Returns
    -------
    pd.DataFrame of all downloaded option rows with enriched columns added.
    Raises RuntimeError if ALL expiries fail.
    """
    all_options     = []
    failed_expiries = []
    chosen_expiries = selected_expiries["expiration"].dt.strftime("%Y-%m-%d").tolist()

    for expiry in chosen_expiries:
        try:
            chain = ticker_obj.option_chain(expiry)
        except Exception as exc:
            print(f"  WARNING: {expiry} failed — {exc}")
            failed_expiries.append(expiry)
            continue

        calls = chain.calls.copy()
        puts  = chain.puts.copy()
        calls["option_type"] = "call"
        puts["option_type"]  = "put"

        combined = pd.concat([calls, puts], ignore_index=True)
        combined["ticker"]         = ticker
        combined["expiration"]     = expiry
        combined["valuation_date"] = valuation_date
        all_options.append(combined)

    if not all_options:
        raise RuntimeError(
            "No option chains downloaded. "
            "Check ticker symbol and internet connection."
        )

    df = pd.concat(all_options, ignore_index=True)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return _enrich_raw(df, spot)


def _enrich_raw(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Add derived columns to the raw options DataFrame."""
    df["mid"]          = (df["bid"] + df["ask"]) / 2
    df["expiration"]   = pd.to_datetime(df["expiration"])
    df["valuation_date"] = pd.to_datetime(df["valuation_date"])

    df["days_to_expiry"] = (df["expiration"] - df["valuation_date"]).dt.days
    df["ttm"]            = df["days_to_expiry"] / 365.0
    df["spot"]           = spot
    df["moneyness"]      = df["strike"] / spot

    return df.copy()


# ── top-level pipeline function ───────────────────────────────────────────────

def _get_live_spot(ticker_obj: yf.Ticker, spot_hist: pd.DataFrame) -> float:
    """
    Return the most current available spot price.

    Tries yfinance fast_info (real-time or 15-min delayed) first.
    Falls back to the last close in the historical data if that fails.
    This keeps log_moneyness and the IV solver consistent with live
    option quotes rather than using a potentially stale historical close.
    """
    try:
        live = ticker_obj.fast_info.get("last_price") or                ticker_obj.fast_info.get("regularMarketPrice")
        if live and live > 0:
            return float(live)
    except Exception:
        pass
    # Fallback: last close from history
    return float(spot_hist["close"].dropna().iloc[-1])


def run_download(config: dict) -> dict:
    """
    Run the full data download pipeline from a config dict.

    Parameters
    ----------
    config : dict with keys matching config.toml:
             ticker, history_start, history_end, risk_free_rate, dividend_yield

    Returns
    -------
    dict with keys:
        spot_hist        : pd.DataFrame
        options_raw      : pd.DataFrame
        selected_expiries: pd.DataFrame
        metadata         : pd.DataFrame
        valuation_date   : pd.Timestamp
        latest_spot      : float
    """
    ticker         = config["ticker"]
    history_start  = config["history_start"]
    history_end    = config["history_end"]
    risk_free_rate = config["risk_free_rate"]
    dividend_yield = config["dividend_yield"]

    print(f"[download] Fetching spot history for {ticker}...")
    ticker_obj = yf.Ticker(ticker)
    spot_hist  = download_spot_history(ticker, history_start, history_end)

    # ── valuation date: always use today, not the last row of history ──────────
    # The spot history may end 1+ days before today (e.g. running on a weekday
    # morning before the previous day's close is in yfinance, or over a weekend).
    # Using today as valuation_date keeps TTM consistent with live option quotes,
    # which the market always prices relative to the current calendar date.
    valuation_date = pd.Timestamp.today().normalize()

    # ── spot: try live price first, fall back to last historical close ─────────
    # Live price keeps log_moneyness and IV solver consistent with live quotes.
    latest_spot = _get_live_spot(ticker_obj, spot_hist)

    print(f"[download] Valuation date: {valuation_date.date()}  Spot: {latest_spot:.2f}")

    print("[download] Selecting expiries...")
    selected = select_expiries(ticker_obj, valuation_date)
    chosen   = selected["expiration"].dt.strftime("%Y-%m-%d").tolist()
    print(f"[download] Selected {len(chosen)} expiries: {chosen}")

    print("[download] Downloading option chains...")
    options_raw = download_option_chains(
        ticker_obj, selected, ticker, valuation_date, latest_spot
    )
    print(f"[download] Downloaded {len(options_raw)} option rows.")

    metadata = pd.DataFrame({
        "ticker"            : [ticker],
        "valuation_date"    : [str(valuation_date.date())],
        "latest_spot"       : [latest_spot],
        "risk_free_rate"    : [risk_free_rate],
        "dividend_yield"    : [dividend_yield],
        "history_start"     : [history_start],
        "history_end"       : [history_end],
        "selected_expiries" : [", ".join(chosen)],
    })

    return {
        "spot_hist"        : spot_hist,
        "options_raw"      : options_raw,
        "selected_expiries": selected,
        "metadata"         : metadata,
        "valuation_date"   : valuation_date,
        "latest_spot"      : latest_spot,
    }
