#!/usr/bin/env python3
"""
refresh.py
----------
One-command pipeline that runs all five stages end-to-end for a given ticker:

    NB01 → download       : spot history + option chains from Yahoo Finance
    NB02 → clean          : filter illiquid / stale quotes
    NB03 → surface        : compute IVs, fit SVI smiles, build vol surface
    NB04 → price          : BSM, CRR, LSM prices for every clean option
    NB05 → validate       : price errors, bid-ask hit rates, early exercise

Usage
-----
    python refresh.py                         # uses ticker from config.toml
    python refresh.py --ticker AAPL           # override ticker
    python refresh.py --ticker SPY --no-lsm  # skip slow LSM (fast mode)
    python refresh.py --stages 1 2 3          # run only selected stages

Output
------
All CSVs and pickle files written to data/raw/, data/processed/, outputs/.
The dashboard (dashboard/app.py) reads these files automatically on next load.
"""

import argparse
import sys
import time
import tomllib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── ensure src is importable ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.cleaning import clean_options
from src.data.download import run_download
from src.models.binomial import crr_price, crr_delta
from src.models.black_scholes import bsm_price, bsm_delta
from src.models.monte_carlo import lsm_price
from src.surface.iv_solver import implied_volatility
from src.surface.svi import (
    build_surface,
    fit_svi,
    get_engine_iv,
    save_surface,
    svi_total_variance,
)
from src.validation.metrics import (
    add_containment_flags,
    add_price_errors,
    fit_summary,
    moneyness_bucket,
    vol_risk_premium,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _hdr(msg: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {msg}")
    print(f"{'─' * width}")


def _elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}m"


def load_config(ticker_override: str = None) -> dict:
    from datetime import date as _date

    cfg_path = ROOT / "config.toml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"config.toml not found at {cfg_path}. "
            "Please create it with ticker, history_start, history_end, etc."
        )
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)

    # "today" resolves to the actual current date at runtime so the pipeline
    # always fetches fresh data without manual config edits.
    if str(cfg.get("history_end", "")).lower() == "today":
        cfg["history_end"] = _date.today().isoformat()

    if ticker_override:
        cfg["ticker"] = ticker_override.upper()
    return cfg


# ── stage functions ───────────────────────────────────────────────────────────

def stage1_download(cfg: dict) -> dict:
    """Download spot history and option chains."""
    _hdr("Stage 1 — Download")
    t0 = time.time()

    result = run_download(cfg)
    ticker = cfg["ticker"]

    # Save
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    result["spot_hist"].to_csv("data/raw/spot_history.csv", index=False)
    result["options_raw"].to_csv("data/raw/options_raw.csv", index=False)
    result["selected_expiries"].to_csv("data/raw/selected_expiries.csv", index=False)
    result["metadata"].to_csv("data/raw/download_metadata.csv", index=False)

    n = len(result["options_raw"])
    e = result["selected_expiries"]["expiration"].nunique()
    print(f"[stage1] ✓  {n} option rows, {e} expiries  [{_elapsed(t0)}]")
    return result


def stage2_clean(cfg: dict, download_result: dict) -> dict:
    """Apply filter pipeline."""
    _hdr("Stage 2 — Clean")
    t0 = time.time()

    options_clean, attrition_df = clean_options(
        download_result["options_raw"],
        download_result["metadata"],
        cfg,
    )

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    options_clean.to_csv("data/processed/options_clean.csv", index=False)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    attrition_df.to_csv("outputs/tables/attrition.csv", index=False)

    n_raw   = len(download_result["options_raw"])
    n_clean = len(options_clean)
    print(f"[stage2] ✓  {n_clean}/{n_raw} rows kept "
          f"({100*(n_raw-n_clean)/n_raw:.1f}% removed)  [{_elapsed(t0)}]")
    return {"options_clean": options_clean, "attrition": attrition_df}


def stage3_surface(cfg: dict, clean_result: dict,
                   download_result: dict) -> dict:
    """Compute IVs, fit SVI smiles, build and save vol surface."""
    _hdr("Stage 3 — Vol Surface")
    t0 = time.time()

    r_f = cfg["risk_free_rate"]
    q   = cfg["dividend_yield"]
    options_clean = clean_result["options_clean"].copy()

    # ── compute raw implied vols ──────────────────────────────────────────────
    print("[stage3] Computing implied volatilities...")
    options_clean["iv_model"] = options_clean.apply(
        lambda row: implied_volatility(
            market_price=row["mid"],
            S=row["spot"], K=row["strike"], T=row["ttm"],
            r=r_f, option_type=row["option_type"], q=q,
        ), axis=1
    )
    options_iv = options_clean[
        options_clean["iv_model"].notna() &
        (options_clean["iv_model"] > 0.05) &
        (options_clean["iv_model"] < 0.80)
    ].copy()
    print(f"[stage3]   IV computed: {len(options_iv)} rows")

    # ── OTM smile construction ────────────────────────────────────────────────
    atm_band = 0.005
    def _flag(row):
        lm, ot = row["log_moneyness"], row["option_type"]
        if abs(lm) <= atm_band:
            return "atm"
        elif ot == "put" and lm < 0:
            return "otm"
        elif ot == "call" and lm > 0:
            return "otm"
        return "itm"

    options_iv["moneyness_flag"] = options_iv.apply(_flag, axis=1)
    smile_base = options_iv[options_iv["moneyness_flag"].isin(["otm", "atm"])].copy()
    smile_base = smile_base[smile_base["log_moneyness"].abs() <= 0.15].copy()

    # Preserve bid/ask so we can compute liquidity weights in the SVI fit.
    # Options with tighter spreads are more reliable and should dominate.
    smile_df = (
        smile_base
        .groupby(["expiration", "log_moneyness"], as_index=False)
        .agg(
            iv_model=("iv_model", "mean"),
            bid     =("bid",      "mean"),
            ask     =("ask",      "mean"),
        )
        .sort_values(["expiration", "log_moneyness"])
    )

    # ── attach TTM ────────────────────────────────────────────────────────────
    expiry_ttm = (
        options_iv[["expiration", "ttm", "days_to_expiry"]]
        .drop_duplicates(subset=["expiration"])
        .sort_values("expiration")
    )
    smile_df = smile_df.merge(expiry_ttm, on="expiration", how="left")

    # ── fit SVI per expiry ────────────────────────────────────────────────────
    print("[stage3] Fitting SVI smiles...")
    svi_smile_rows, svi_param_rows = [], []

    for expiry, grp in smile_df.groupby("expiration"):
        grp = grp.sort_values("log_moneyness").copy()
        ttm = float(grp["ttm"].iloc[0])
        if len(grp) < 8:
            continue

        k = grp["log_moneyness"].values
        v = grp["iv_model"].values
        w = v ** 2 * ttm

        # Liquidity weights: weight = 1 / bid-ask spread.
        # Tight-spread ATM options drive the fit.
        # Wide-spread illiquid wing options have low weight.
        # This prevents stressed-market outliers from distorting rho.
        if "bid" in grp.columns and "ask" in grp.columns:
            spreads = (grp["ask"] - grp["bid"]).clip(lower=0.05).values
            fit_weights = 1.0 / spreads

            # Floor: no option gets less than 20% of ATM weight
            # This preserves skew signal in stressed markets where
            # OTM spreads are wide and would otherwise be ignored
            _atm_idx    = np.argmin(np.abs(grp["log_moneyness"].values))
            _atm_w      = fit_weights[_atm_idx]
            fit_weights = np.maximum(fit_weights, 0.20 * _atm_w)
        else:
            fit_weights = None

        params = fit_svi(k, w, ttm, weights=fit_weights)
        if params is None:
            print(f"[stage3]   WARNING: SVI fit failed for {expiry}")
            continue

        a, b, rho, m, sigma_p = params
        k_grid  = np.linspace(k.min(), k.max(), 300)
        w_grid  = np.clip(svi_total_variance(k_grid, params), 1e-8, None)
        iv_grid = np.sqrt(w_grid / ttm)

        svi_smile_rows.append(pd.DataFrame({
            "expiration"   : expiry,
            "ttm"          : ttm,
            "log_moneyness": k_grid,
            "w_svi"        : w_grid,
            "iv_smooth"    : iv_grid,
        }))
        svi_param_rows.append({
            "expiration": expiry, "ttm": ttm,
            "a": a, "b": b, "rho": rho, "m": m, "sigma": sigma_p,
            "fit_rmse": float(np.sqrt(np.mean(
                (svi_total_variance(k, params) - w) ** 2
            ))),
        })

    if not svi_smile_rows:
        raise RuntimeError("[stage3] All SVI fits failed — cannot build surface. "
                           "Check that cleaned data has enough OTM points per expiry.")
    svi_smile_df  = pd.concat(svi_smile_rows, ignore_index=True)
    svi_params_df = pd.DataFrame(svi_param_rows)
    print(f"[stage3]   SVI fitted {len(svi_params_df)} expiries")

    # ── cross-sectional out-of-sample split ──────────────────────────────────
    # Calibration set : near-ATM options only (|log_moneyness| <= 0.05)
    # OOS test set    : the OTM wings (|log_moneyness| > 0.05) that were
    #                   never used to fit the SVI surface.
    # This gives an honest test of surface extrapolation quality.
    ATM_CALIB_BAND = 0.05

    options_iv["oos_flag"] = (options_iv["log_moneyness"].abs() > ATM_CALIB_BAND)
    options_iv["sample"]   = np.where(options_iv["oos_flag"], "oos", "calibration")

    n_calib = (~options_iv["oos_flag"]).sum()
    n_oos   = options_iv["oos_flag"].sum()
    print(f"[stage3]   Calibration (ATM) : {n_calib} options")
    print(f"[stage3]   OOS (wings)       : {n_oos} options")

    # ── build and save surface interpolators ──────────────────────────────────
    w_linear, w_nearest = build_surface(svi_smile_df)
    save_surface(w_linear, w_nearest, "data/processed")

    # ── attach iv_engine to each option ──────────────────────────────────────
    options_iv["iv_engine"] = options_iv.apply(
        lambda row: get_engine_iv(
            row["log_moneyness"], row["ttm"], w_linear, w_nearest
        ), axis=1
    )
    options_iv = options_iv[options_iv["iv_engine"].notna()].copy()

    # ── save ──────────────────────────────────────────────────────────────────
    options_iv.to_csv("data/processed/options_with_iv_engine.csv", index=False)
    svi_smile_df.to_csv("data/processed/smile_svi.csv", index=False)
    svi_params_df.to_csv("data/processed/svi_params.csv", index=False)
    svi_smile_df.to_csv("data/processed/vol_surface_points.csv", index=False)

    print(f"[stage3] ✓  Surface built  [{_elapsed(t0)}]")
    return {
        "options_iv"   : options_iv,
        "svi_smile_df" : svi_smile_df,
        "svi_params_df": svi_params_df,
        "w_linear"     : w_linear,
        "w_nearest"    : w_nearest,
    }


def stage4_price(cfg: dict, surface_result: dict,
                 run_lsm: bool = True) -> dict:
    """Apply all three pricing engines."""
    _hdr("Stage 4 — Price")
    t0 = time.time()

    r_f = cfg["risk_free_rate"]
    q   = cfg["dividend_yield"]
    options = surface_result["options_iv"].copy()

    CRR_STEPS = 200
    LSM_PATHS = 10_000
    LSM_STEPS = 252

    # BSM
    print("[stage4] BSM pricing...")
    options["price_bsm"] = options.apply(
        lambda row: bsm_price(row["spot"], row["strike"], row["ttm"],
                              r_f, row["iv_engine"], row["option_type"], q),
        axis=1
    )
    options["delta_bsm"] = options.apply(
        lambda row: bsm_delta(row["spot"], row["strike"], row["ttm"],
                              r_f, row["iv_engine"], row["option_type"], q),
        axis=1
    )

    # CRR
    print("[stage4] CRR pricing...")
    options["price_crr"] = options.apply(
        lambda row: crr_price(row["spot"], row["strike"], row["ttm"],
                              r_f, row["iv_engine"], row["option_type"], q,
                              N=CRR_STEPS, american=True),
        axis=1
    )
    options["delta_crr"] = options.apply(
        lambda row: crr_delta(row["spot"], row["strike"], row["ttm"],
                              r_f, row["iv_engine"], row["option_type"], q,
                              N=CRR_STEPS, american=True),
        axis=1
    )

    # LSM (optional — slow)
    if run_lsm:
        print(f"[stage4] LSM pricing ({len(options)} options, ~1–3 min)...")
        lsm_prices = []
        for i, (_, row) in enumerate(options.iterrows()):
            p = lsm_price(
                row["spot"], row["strike"], row["ttm"],
                r_f, row["iv_engine"], row["option_type"], q,
                M=LSM_PATHS, n=LSM_STEPS, seed=42,
            )
            lsm_prices.append(p)
            if (i + 1) % 25 == 0 or (i + 1) == len(options):
                print(f"[stage4]   LSM: {i+1}/{len(options)}")
        options["price_lsm"] = lsm_prices
    else:
        print("[stage4] LSM skipped (--no-lsm flag)")
        options["price_lsm"] = np.nan

    # Early exercise premiums
    options["eep_crr"] = options["price_crr"] - options["price_bsm"]
    options["eep_lsm"] = options["price_lsm"] - options["price_bsm"]

    options.to_csv("data/processed/options_with_prices.csv", index=False)
    print(f"[stage4] ✓  Pricing complete  [{_elapsed(t0)}]")
    return {"options_priced": options}


def stage5_validate(cfg: dict, price_result: dict) -> dict:
    """Compute validation metrics and save tables."""
    _hdr("Stage 5 — Validate")
    t0 = time.time()

    options = price_result["options_priced"].copy()
    options["moneyness_bucket"] = options["log_moneyness"].apply(moneyness_bucket)
    options = add_price_errors(options)
    options = add_containment_flags(options)

    summary = fit_summary(options)

    # ── in-sample vs out-of-sample breakdown ─────────────────────────────────
    # If stage3 added the oos_flag column, report separately.
    # Calibration (ATM) should have lower errors than OOS (wings) —
    # if OOS is worse, it reveals genuine extrapolation risk.
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    options.to_csv("data/processed/options_with_validation.csv", index=False)
    summary.to_csv("outputs/tables/fit_summary.csv")

    print("\n=== ALL OPTIONS ===")
    print(summary.to_string())

    if "sample" in options.columns:
        for sample_label in ["calibration", "oos"]:
            subset = options[options["sample"] == sample_label]
            if len(subset) == 0:
                continue
            sub_summary = fit_summary(subset)
            label = "CALIBRATION (ATM)" if sample_label == "calibration" else "OOS (WINGS)"
            print(f"\n=== {label}  (n={len(subset)}) ===")
            print(sub_summary.to_string())
            sub_summary.to_csv(f"outputs/tables/fit_summary_{sample_label}.csv")

    # ── volatility risk premium ──────────────────────────────────────────────
    # Load spot history to compare ATM IV against realised vol.
    # This is a genuine benchmark: are GOOG options expensive or cheap
    # relative to what the stock actually did?
    try:
        spot_hist_path = Path("data/raw/spot_history.csv")
        if spot_hist_path.exists():
            spot_hist = pd.read_csv(spot_hist_path)
            vrp_df    = vol_risk_premium(options, spot_hist)
            if not vrp_df.empty:
                vrp_df.to_csv("outputs/tables/vol_risk_premium.csv")
                print("\n=== VOLATILITY RISK PREMIUM (ATM IV minus Realised Vol) ===")
                print(vrp_df[["days_to_expiry","atm_iv","rv_matched","vrp","vrp_pct"]].to_string())
                print("  Positive VRP = options expensive (IV > RV). Typical in equity markets.")
    except Exception as e:
        print(f"[stage5]   VRP skipped: {e}")

    print(f"\n[stage5] ✓  Validation complete  [{_elapsed(t0)}]")
    return {"options_validated": options, "fit_summary": summary}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh the options pricing pipeline for a given ticker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ticker", type=str, default=None,
        help="Ticker symbol to price (default: read from config.toml)",
    )
    parser.add_argument(
        "--no-lsm", action="store_true",
        help="Skip LSM Monte Carlo pricing (faster, disables stage 4 LSM column)",
    )
    parser.add_argument(
        "--stages", nargs="+", type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run only the specified stages (default: all 1–5)",
    )
    args = parser.parse_args()

    run_stages = set(args.stages) if args.stages else {1, 2, 3, 4, 5}
    run_lsm    = not args.no_lsm

    total_start = time.time()
    cfg = load_config(args.ticker)

    print(f"\n{'═' * 60}")
    print(f"  Options Pricing Pipeline  —  {cfg['ticker']}")
    print(f"  Stages: {sorted(run_stages)}")
    print(f"  LSM:    {'enabled' if run_lsm else 'disabled (--no-lsm)'}")
    print(f"{'═' * 60}")

    # Accumulate stage results so each stage can consume the previous
    dl_result  = None
    cln_result = None
    srf_result = None
    prc_result = None

    # Stage 1
    if 1 in run_stages:
        dl_result = stage1_download(cfg)
    else:
        print("\n[skip] Stage 1 — loading from disk...")
        options_raw = pd.read_csv("data/raw/options_raw.csv")
        metadata    = pd.read_csv("data/raw/download_metadata.csv")
        selected    = pd.read_csv("data/raw/selected_expiries.csv")
        spot_hist   = pd.read_csv("data/raw/spot_history.csv")
        latest_spot = float(metadata["latest_spot"].iloc[0])
        dl_result   = {
            "options_raw"      : options_raw,
            "metadata"         : metadata,
            "selected_expiries": selected,
            "spot_hist"        : spot_hist,
            "latest_spot"      : latest_spot,
        }

    # Stage 2
    if 2 in run_stages:
        cln_result = stage2_clean(cfg, dl_result)
    else:
        print("[skip] Stage 2 — loading from disk...")
        cln_result = {"options_clean": pd.read_csv("data/processed/options_clean.csv")}

    # Stage 3
    if 3 in run_stages:
        srf_result = stage3_surface(cfg, cln_result, dl_result)
    else:
        print("[skip] Stage 3 — loading from disk...")
        w_linear, w_nearest = None, None
        try:
            import joblib
            w_linear  = joblib.load("data/processed/iv_surface_linear.pkl")
            w_nearest = joblib.load("data/processed/iv_surface_nearest.pkl")
        except FileNotFoundError:
            print("  WARNING: surface files not found")
        srf_result = {
            "options_iv": pd.read_csv("data/processed/options_with_iv_engine.csv"),
            "w_linear"  : w_linear,
            "w_nearest" : w_nearest,
        }

    # Stage 4
    if 4 in run_stages:
        prc_result = stage4_price(cfg, srf_result, run_lsm=run_lsm)
    else:
        print("[skip] Stage 4 — loading from disk...")
        prc_result = {"options_priced": pd.read_csv("data/processed/options_with_prices.csv")}

    # Stage 5
    if 5 in run_stages:
        stage5_validate(cfg, prc_result)

    total = _elapsed(total_start)
    print(f"\n{'═' * 60}")
    print(f"  ✓  Pipeline complete for {cfg['ticker']}  [{total}]")
    print(f"  Dashboard: streamlit run dashboard/app.py")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
