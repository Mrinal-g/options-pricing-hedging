"""
src/data/cleaning.py
--------------------
Applies a 8-step filter pipeline to the raw option chain to remove
illiquid, stale, and unreliable quotes before IV estimation.

Each filter is auditable: an attrition table records exactly how many
rows each step removes.

Public API
----------
clean_options(options_raw, metadata, config) -> (pd.DataFrame, pd.DataFrame)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_options(
    options_raw: pd.DataFrame,
    metadata: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply a sequential filter pipeline to the raw option chain.

    Filters applied (in order):
      1. Drop rows with missing critical fields
      2. Minimum bid and mid price
      3. Ask > bid (no crossed/flat markets)
      4. Positive time to maturity
      5. Open interest > 0
      6. Strike within ±{multiplier}% of spot
      7. Bid-ask spread / mid ≤ max_spread_ratio
      8. Last trade staleness ≤ 5 calendar days

    Parameters
    ----------
    options_raw : raw options DataFrame from src.data.download
    metadata    : single-row metadata DataFrame (contains latest_spot)
    config      : config dict with cleaning thresholds:
                  min_bid, min_mid, max_spread_ratio,
                  strike_lower_multiplier, strike_upper_multiplier

    Returns
    -------
    options_clean : filtered and enriched DataFrame
    attrition_df  : table of rows remaining after each filter step
    """
    min_bid                 = config["min_bid"]
    min_mid                 = config["min_mid"]
    max_spread_ratio        = config["max_spread_ratio"]
    strike_lower_multiplier = config["strike_lower_multiplier"]
    strike_upper_multiplier = config["strike_upper_multiplier"]

    df = options_raw.copy()

    # ── parse types ──────────────────────────────────────────────────────────
    df["valuation_date"] = pd.to_datetime(df["valuation_date"])
    df["expiration"]     = pd.to_datetime(df["expiration"])

    numeric_cols = [
        "strike", "bid", "ask", "mid", "lastprice", "volume",
        "open_interest", "impliedvolatility", "days_to_expiry",
        "ttm", "spot", "moneyness",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    attrition: dict[str, int] = {"raw": len(df)}

    # ── 1. Missing critical fields ────────────────────────────────────────────
    critical = ["strike", "bid", "ask", "mid", "ttm", "spot", "option_type"]
    df = df.dropna(subset=critical).copy()
    attrition["missing_critical"] = len(df)

    # ── 2. Minimum bid and mid ────────────────────────────────────────────────
    df = df[(df["bid"] >= min_bid) & (df["mid"] >= min_mid)].copy()
    attrition["min_bid_mid"] = len(df)

    # ── 3. Ask > bid ──────────────────────────────────────────────────────────
    df = df[df["ask"] > df["bid"]].copy()
    attrition["ask_gt_bid"] = len(df)

    # ── 4. Positive TTM ───────────────────────────────────────────────────────
    df = df[df["ttm"] > 0].copy()
    attrition["positive_ttm"] = len(df)

    # ── 5. Open interest > 0 ──────────────────────────────────────────────────
    oi_col = "openinterest" if "openinterest" in df.columns else "open_interest"
    if oi_col in df.columns:
        df = df[df[oi_col] > 0].copy()
    attrition["open_interest"] = len(df)

    # ── 6. Strike within band ─────────────────────────────────────────────────
    spot = float(metadata["latest_spot"].iloc[0])
    lo   = spot * strike_lower_multiplier
    hi   = spot * strike_upper_multiplier
    df   = df[(df["strike"] >= lo) & (df["strike"] <= hi)].copy()
    attrition["strike_band"] = len(df)

    # ── 7. Spread ratio ───────────────────────────────────────────────────────
    df["spread"]          = df["ask"] - df["bid"]
    df["spread_over_mid"] = df["spread"] / df["mid"]
    df = df[df["spread_over_mid"] <= max_spread_ratio].copy()
    attrition["spread_ratio"] = len(df)

    # ── 8. Trade staleness ≤ 5 days ───────────────────────────────────────────
    if "lasttradedate" in df.columns:
        df["lasttradedate"] = pd.to_datetime(
            df["lasttradedate"], utc=True, errors="coerce"
        ).dt.tz_convert(None)
        ref = pd.Timestamp.today().normalize()
        df["trade_age_days"] = (ref - df["lasttradedate"].dt.normalize()).dt.days
        df = df[df["trade_age_days"] <= 5].copy()
    attrition["staleness"] = len(df)

    # ── enrich ────────────────────────────────────────────────────────────────
    df["log_moneyness"] = np.log(df["strike"] / df["spot"])

    conditions = [
        df["days_to_expiry"] <= 14,
        (df["days_to_expiry"] > 14)  & (df["days_to_expiry"] <= 60),
        (df["days_to_expiry"] > 60)  & (df["days_to_expiry"] <= 150),
        (df["days_to_expiry"] > 150) & (df["days_to_expiry"] <= 270),
        df["days_to_expiry"] > 270,
    ]
    labels = ["weekly", "short", "medium", "long", "one_year"]
    df["maturity_bucket"] = np.select(conditions, labels, default="other")

    df = df.sort_values(["expiration", "option_type", "strike"]).reset_index(drop=True)

    # ── attrition table ───────────────────────────────────────────────────────
    atr_df = pd.DataFrame({
        "stage"          : list(attrition.keys()),
        "rows_remaining" : list(attrition.values()),
    })
    atr_df["rows_removed"] = (
        atr_df["rows_remaining"].shift(1) - atr_df["rows_remaining"]
    ).fillna(0).astype(int)
    atr_df["pct_removed"] = (
        atr_df["rows_removed"] / atr_df["rows_remaining"].iloc[0] * 100
    ).round(2)

    return df, atr_df
