"""
src/validation/metrics.py
-------------------------
Model validation: price errors, bid-ask containment, and fit summaries.

These functions are used in both the validation notebook (NB05) and the
Streamlit dashboard to produce consistent metrics without code duplication.

Public API
----------
add_price_errors(options_df, models)              -> pd.DataFrame
add_containment_flags(options_df, models)         -> pd.DataFrame
fit_summary(options_df, models)                   -> pd.DataFrame
fit_by_type(options_df, models)                   -> pd.DataFrame
fit_by_moneyness(options_df, models, bucket_order) -> pd.DataFrame
fit_by_expiry(options_df, models)                 -> pd.DataFrame
moneyness_bucket(log_moneyness)                   -> str
"""

from __future__ import annotations

import numpy as np
import pandas as pd


BUCKET_ORDER = ["Deep ITM", "ITM", "ATM", "OTM", "Deep OTM"]

DEFAULT_MODELS = {
    "price_bsm": "BSM (European)",
    "price_crr": "CRR (American)",
    "price_lsm": "LSM (American)",
}


def moneyness_bucket(log_moneyness: float) -> str:
    """Classify a log-moneyness value into one of five buckets."""
    if log_moneyness < -0.10:
        return "Deep ITM"
    elif log_moneyness < -0.03:
        return "ITM"
    elif log_moneyness <= 0.03:
        return "ATM"
    elif log_moneyness <= 0.10:
        return "OTM"
    else:
        return "Deep OTM"


def add_price_errors(
    df: pd.DataFrame,
    models: dict = None,
) -> pd.DataFrame:
    """
    Add absolute error, percentage error, and APE columns for each model.

    For model column 'price_X', adds:
      err_X     = price_X − mid
      pct_err_X = 100 · err_X / mid
      ape_X     = |pct_err_X|

    Parameters
    ----------
    df     : options DataFrame with 'mid' and model price columns
    models : dict mapping price column name → label (default DEFAULT_MODELS)

    Returns
    -------
    df with error columns added (copy)
    """
    if models is None:
        models = DEFAULT_MODELS
    df = df.copy()
    for col in models:
        if col not in df.columns:
            continue
        base = col.replace("price_", "")
        df[f"err_{base}"]     = df[col] - df["mid"]
        df[f"pct_err_{base}"] = 100 * df[f"err_{base}"] / df["mid"]
        df[f"ape_{base}"]     = df[f"pct_err_{base}"].abs()
    return df


def add_containment_flags(
    df: pd.DataFrame,
    models: dict = None,
) -> pd.DataFrame:
    """
    Add boolean columns indicating whether each model price falls within
    the bid-ask spread.

    For model column 'price_X', adds:
      in_spread_X = (bid <= price_X <= ask)

    Parameters
    ----------
    df     : options DataFrame with bid, ask and model price columns
    models : dict mapping price column name → label (default DEFAULT_MODELS)

    Returns
    -------
    df with containment flag columns added (copy)
    """
    if models is None:
        models = DEFAULT_MODELS
    df = df.copy()
    for col in models:
        if col not in df.columns:
            continue
        base = col.replace("price_", "")
        df[f"in_spread_{base}"] = (
            (df[col] >= df["bid"]) & (df[col] <= df["ask"])
        )
    return df


def fit_summary(
    df: pd.DataFrame,
    models: dict = None,
) -> pd.DataFrame:
    """
    Overall fit statistics across all options for each model.

    Columns: mean_err, mae, rmse, median_abs_err, max_abs_err,
             mean_pct_err, mape, bid_ask_hit_%

    Returns
    -------
    pd.DataFrame indexed by model label
    """
    if models is None:
        models = DEFAULT_MODELS
    rows = []
    for col, label in models.items():
        base = col.replace("price_", "")
        if f"err_{base}" not in df.columns:
            continue
        errs = df[f"err_{base}"].dropna()
        pct  = df[f"pct_err_{base}"].dropna()
        ape  = df[f"ape_{base}"].dropna()
        hit  = df[f"in_spread_{base}"].mean() * 100

        rows.append({
            "model"         : label,
            "mean_err"      : round(errs.mean(), 4),
            "mae"           : round(errs.abs().mean(), 4),
            "rmse"          : round(float(np.sqrt((errs ** 2).mean())), 4),
            "median_abs_err": round(errs.abs().median(), 4),
            "max_abs_err"   : round(errs.abs().max(), 4),
            "mean_pct_err"  : round(pct.mean(), 2),
            "mape"          : round(ape.mean(), 2),
            "bid_ask_hit_%" : round(hit, 1),
        })
    return pd.DataFrame(rows).set_index("model")


def fit_by_type(
    df: pd.DataFrame,
    models: dict = None,
) -> pd.DataFrame:
    """
    Fit statistics broken down by option type (call / put / all).

    Returns
    -------
    pd.DataFrame indexed by (option_type, model)
    """
    if models is None:
        models = DEFAULT_MODELS
    rows = []
    for otype in ["call", "put", "all"]:
        sub = df if otype == "all" else df[df["option_type"] == otype]
        for col, label in models.items():
            base = col.replace("price_", "")
            if f"err_{base}" not in sub.columns:
                continue
            errs = sub[f"err_{base}"].dropna()
            rows.append({
                "option_type": otype,
                "model"      : label,
                "n"          : len(sub),
                "mean_err"   : round(errs.mean(), 4),
                "mae"        : round(errs.abs().mean(), 4),
                "rmse"       : round(float(np.sqrt((errs ** 2).mean())), 4),
                "mape"       : round(sub[f"ape_{base}"].mean(), 2),
                "hit_%"      : round(sub[f"in_spread_{base}"].mean() * 100, 1),
            })
    return pd.DataFrame(rows).set_index(["option_type", "model"])


def fit_by_moneyness(
    df: pd.DataFrame,
    models: dict = None,
    bucket_order: list = None,
) -> pd.DataFrame:
    """
    Fit statistics broken down by moneyness bucket.

    Returns
    -------
    pd.DataFrame indexed by (moneyness_bucket, model)
    """
    if models is None:
        models = DEFAULT_MODELS
    if bucket_order is None:
        bucket_order = BUCKET_ORDER

    if "moneyness_bucket" not in df.columns:
        df = df.copy()
        df["moneyness_bucket"] = df["log_moneyness"].apply(moneyness_bucket)

    rows = []
    for bucket in bucket_order:
        sub = df[df["moneyness_bucket"] == bucket]
        if len(sub) == 0:
            continue
        for col, label in models.items():
            base = col.replace("price_", "")
            if f"err_{base}" not in sub.columns:
                continue
            errs = sub[f"err_{base}"].dropna()
            rows.append({
                "moneyness_bucket": bucket,
                "model"           : label,
                "n"               : len(sub),
                "mean_err"        : round(errs.mean(), 4),
                "mae"             : round(errs.abs().mean(), 4),
                "mape"            : round(sub[f"ape_{base}"].mean(), 2),
                "hit_%"           : round(sub[f"in_spread_{base}"].mean() * 100, 1),
            })
    return pd.DataFrame(rows).set_index(["moneyness_bucket", "model"])


def fit_by_expiry(
    df: pd.DataFrame,
    models: dict = None,
) -> pd.DataFrame:
    """
    Fit statistics broken down by expiry date.

    Returns
    -------
    pd.DataFrame indexed by (expiry, model)
    """
    if models is None:
        models = DEFAULT_MODELS
    rows = []
    for expiry, grp in df.groupby("expiration"):
        for col, label in models.items():
            base = col.replace("price_", "")
            if f"err_{base}" not in grp.columns:
                continue
            errs = grp[f"err_{base}"].dropna()
            rows.append({
                "expiry" : expiry.date() if hasattr(expiry, "date") else expiry,
                "ttm"    : round(grp["ttm"].iloc[0], 4),
                "model"  : label,
                "n"      : len(grp),
                "mean_err": round(errs.mean(), 4),
                "mae"    : round(errs.abs().mean(), 4),
                "mape"   : round(grp[f"ape_{base}"].mean(), 2),
                "hit_%"  : round(grp[f"in_spread_{base}"].mean() * 100, 1),
            })
    return pd.DataFrame(rows).set_index(["expiry", "model"])


# ── realised vol vs implied vol (volatility risk premium) ─────────────────────

def vol_risk_premium(
    options_df: pd.DataFrame,
    spot_hist: pd.DataFrame,
    atm_band: float = 0.03,
) -> pd.DataFrame:
    """
    Compare near-ATM implied volatility against realised volatility for each
    expiry — the volatility risk premium (VRP).

    In equity markets the VRP is almost always positive: implied vol > realised
    vol because option sellers demand a premium for bearing gamma risk.
    A large positive VRP means options are "expensive" (good time to sell vol).
    Near zero or negative means options are "cheap".

    Parameters
    ----------
    options_df : priced options DataFrame (output of Stage 4)
                 Must have columns: expiration, log_moneyness, iv_model, days_to_expiry
    spot_hist  : daily spot history DataFrame with columns: date, close
    atm_band   : |log_moneyness| threshold for "near-ATM" (default 0.03)

    Returns
    -------
    pd.DataFrame indexed by expiration with columns:
        days_to_expiry, atm_iv, rv_window, rv_matched,
        vrp (= atm_iv - rv_matched), vrp_pct
    """
    spot_hist = spot_hist.copy()
    spot_hist["date"]  = pd.to_datetime(spot_hist["date"])
    spot_hist          = spot_hist.sort_values("date").reset_index(drop=True)
    spot_hist["close"] = pd.to_numeric(spot_hist["close"], errors="coerce")
    spot_hist          = spot_hist.dropna(subset=["close"])

    # Pre-compute rolling realised vols at standard windows (annualised)
    log_ret = np.log(spot_hist["close"] / spot_hist["close"].shift(1))
    for w in [5, 10, 21, 63, 126, 252]:
        spot_hist[f"rv_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

    latest_rv = spot_hist.dropna().iloc[-1]

    def _map_rv(days: int) -> tuple[float, str]:
        """Pick the realised vol window closest to the option's tenor."""
        windows = [(5, "5d"), (10, "10d"), (21, "21d"),
                   (63, "63d"), (126, "126d"), (252, "252d")]
        best_w, best_label = min(windows, key=lambda x: abs(x[0] - days))
        rv_col = f"rv_{best_w}d"
        return float(latest_rv.get(rv_col, np.nan)), f"{best_label} realised vol"

    # Near-ATM IV per expiry
    atm_df = options_df[options_df["log_moneyness"].abs() <= atm_band].copy()
    if atm_df.empty:
        return pd.DataFrame()

    expiry_summary = (
        atm_df
        .groupby("expiration", as_index=False)
        .agg(
            atm_iv        =("iv_model",       "mean"),
            days_to_expiry=("days_to_expiry", "first"),
        )
    )

    rows = []
    for _, row in expiry_summary.iterrows():
        rv_val, rv_label = _map_rv(int(row["days_to_expiry"]))
        vrp              = row["atm_iv"] - rv_val
        rows.append({
            "expiration"    : row["expiration"],
            "days_to_expiry": int(row["days_to_expiry"]),
            "atm_iv"        : round(row["atm_iv"], 4),
            "rv_window"     : rv_label,
            "rv_matched"    : round(rv_val, 4),
            "vrp"           : round(vrp, 4),
            "vrp_pct"       : round(100 * vrp / rv_val, 2) if rv_val > 0 else np.nan,
        })

    return pd.DataFrame(rows).set_index("expiration")
