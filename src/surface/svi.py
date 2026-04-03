"""
src/surface/svi.py
------------------
SVI (Stochastic Volatility Inspired) smile fitting and volatility surface
construction via total-variance interpolation.

The SVI parametrisation (Gatheral 2004) models total implied variance
w(k) = IV(k)² · T as a function of log-moneyness k = ln(K/S):

    w(k) = a + b · (ρ·(k-m) + √((k-m)² + σ²))

Working in total variance space rather than IV space:
  - Preserves calendar no-arbitrage: w must be non-decreasing in T
  - Avoids artificial kinks when term-structure interpolation crosses tenors

Public API
----------
svi_total_variance(k, params) -> np.ndarray
fit_svi(k_points, w_points, ttm) -> np.ndarray | None
build_surface(svi_smile_df) -> (LinearNDInterpolator, NearestNDInterpolator)
get_engine_iv(log_moneyness, ttm, w_linear, w_nearest) -> float
load_surface(surface_dir) -> (LinearNDInterpolator, NearestNDInterpolator)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.optimize import minimize


# ── SVI parametrisation ───────────────────────────────────────────────────────

def svi_total_variance(k: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate the SVI total variance function at log-moneyness points k.

    Parameters
    ----------
    k      : log-moneyness array, shape (N,)
    params : (a, b, rho, m, sigma) — five SVI parameters

    Returns
    -------
    np.ndarray of total implied variance values w = IV² · T
    """
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def fit_svi(k_points: np.ndarray, w_points: np.ndarray,
            ttm: float,
            weights: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Fit SVI parameters to a single expiry slice.

    Uses weighted least squares so that liquid near-ATM options
    (tight bid-ask spreads) have more influence than illiquid
    deep-OTM options.  This is critical for stressed markets where
    outlier wing quotes would otherwise dominate the fit and pull
    rho to the wrong sign.

    Uses L-BFGS-B with multiple starting points to avoid local minima.
    rho is constrained to be negative — equity options always have
    left skew (puts more expensive than equidistant calls).

    Parameters
    ----------
    k_points : log-moneyness array for this expiry
    w_points : total implied variance array (IV² · T) for this expiry
    ttm      : time to maturity in years
    weights  : per-point weights (default: 1/spread, normalised to sum=1).
               Higher weight = more influence on the fit.
               If None, uses equal weights (unweighted, original behaviour).

    Returns
    -------
    np.ndarray : fitted (a, b, rho, m, sigma), or None if all fits fail
    """
    if weights is None:
        weights = np.ones(len(k_points))
    # Normalise weights so they sum to the number of points
    # (keeps the loss scale comparable to the unweighted case)
    weights = weights / weights.sum() * len(weights)

    def objective(params):
        w_model = svi_total_variance(k_points, params)
        if np.any(np.isnan(w_model)) or np.any(w_model < 0):
            return 1e10
        # Weighted sum of squared errors — liquid options matter more
        return float(np.sum(weights * (w_model - w_points) ** 2))

    atm_idx = np.argmin(np.abs(k_points))
    w_atm   = float(w_points[atm_idx])

    bounds = [
        (1e-6, float(np.max(w_points))),   # a : variance level
        (1e-6, 2.0),                        # b : slope/curvature
        (-0.999, -0.01),                    # rho : NEGATIVE ONLY — equity left skew
        (-0.5, 0.5),                        # m : smile minimum location
        (1e-4, 1.0),                        # sigma : ATM smoothness
    ]

    starting_points = [
        [w_atm * 0.5, 0.10, -0.5, 0.0, 0.10],
        [w_atm * 0.5, 0.15, -0.7, 0.0, 0.15],
        [w_atm * 0.5, 0.05, -0.3, 0.0, 0.05],
        [w_atm * 0.8, 0.10, -0.5, 0.0, 0.20],
    ]

    best_result = None
    for p0 in starting_points:
        result = minimize(
            objective, p0, bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    if best_result is not None and best_result.fun < 1e6:
        return best_result.x
    return None


def build_surface(
    svi_smile_df: pd.DataFrame,
) -> Tuple[LinearNDInterpolator, NearestNDInterpolator]:
    """
    Build 2D interpolators over (log_moneyness, ttm) → total variance w.

    The surface uses:
    - LinearNDInterpolator  — accurate within the convex hull of input points
    - NearestNDInterpolator — fallback for extrapolation outside the hull

    Parameters
    ----------
    svi_smile_df : DataFrame with columns [log_moneyness, ttm, w_svi]
                   (output of the SVI fitting loop in NB03)

    Returns
    -------
    (w_linear, w_nearest) interpolator pair
    """
    xy = svi_smile_df[["log_moneyness", "ttm"]].values
    w  = svi_smile_df["w_svi"].values

    w_linear  = LinearNDInterpolator(xy, w)
    w_nearest = NearestNDInterpolator(xy, w)
    return w_linear, w_nearest


def get_engine_iv(log_moneyness: float, ttm: float,
                  w_linear: LinearNDInterpolator,
                  w_nearest: NearestNDInterpolator) -> float:
    """
    Query the volatility surface at (log_moneyness, ttm) and return IV.

    Converts total variance w back to IV via IV = √(w / TTM).
    Falls back to the nearest-neighbour interpolator if the point lies
    outside the convex hull of the surface (i.e. if linear returns NaN).

    Parameters
    ----------
    log_moneyness : ln(K/S)
    ttm           : time to maturity in years
    w_linear      : LinearNDInterpolator over (log_moneyness, ttm) → w
    w_nearest     : NearestNDInterpolator  over (log_moneyness, ttm) → w

    Returns
    -------
    float : implied volatility, or np.nan if ttm <= 0
    """
    if ttm <= 0:
        return np.nan

    w_val = w_linear(log_moneyness, ttm)
    if np.ndim(w_val) > 0:
        w_val = w_val.item()

    if pd.isna(w_val):
        w_val = w_nearest(log_moneyness, ttm)
        if np.ndim(w_val) > 0:
            w_val = w_val.item()

    w_val = max(float(w_val), 1e-8)
    return float(np.sqrt(w_val / ttm))


# ── disk persistence ──────────────────────────────────────────────────────────

# Module-level cache so the surface is only loaded once per process
_cache: dict = {}


def save_surface(w_linear: LinearNDInterpolator,
                 w_nearest: NearestNDInterpolator,
                 surface_dir: str = "data/processed") -> None:
    """Persist surface interpolators to disk and clear the in-process cache.

    Clearing the cache is critical: if refresh.py saves a new surface and
    then calls load_surface() in the same process, without this clear it
    would return the previously cached (stale) interpolators.
    """
    Path(surface_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(w_linear,  Path(surface_dir) / "iv_surface_linear.pkl")
    joblib.dump(w_nearest, Path(surface_dir) / "iv_surface_nearest.pkl")
    # Clear cache so any subsequent load_surface() call reads from disk
    cache_key = str(Path(surface_dir).resolve())
    _cache.pop(cache_key, None)


def load_surface(
    surface_dir: str = "data/processed",
) -> Tuple[LinearNDInterpolator, NearestNDInterpolator]:
    """
    Load surface interpolators from disk, using a module-level cache.

    Parameters
    ----------
    surface_dir : directory containing iv_surface_linear.pkl and
                  iv_surface_nearest.pkl (produced by NB03 / refresh.py)

    Returns
    -------
    (w_linear, w_nearest) interpolator pair

    Raises
    ------
    FileNotFoundError if the .pkl files are not found.
    Run `python refresh.py` first to generate them.
    """
    cache_key = str(Path(surface_dir).resolve())

    if cache_key not in _cache:
        linear_path  = Path(surface_dir) / "iv_surface_linear.pkl"
        nearest_path = Path(surface_dir) / "iv_surface_nearest.pkl"

        if not linear_path.exists() or not nearest_path.exists():
            raise FileNotFoundError(
                f"Surface files not found in {surface_dir}. "
                "Run `python refresh.py --ticker TICKER` first."
            )

        _cache[cache_key] = (
            joblib.load(linear_path),
            joblib.load(nearest_path),
        )

    return _cache[cache_key]


# ── vol surface stress scenarios ──────────────────────────────────────────────

def iv_parallel_up(log_moneyness: float, ttm: float,
                   w_linear, w_nearest,
                   shift: float = 0.02) -> float:
    """Parallel +shift to the entire surface (e.g. +2 vol points)."""
    return max(get_engine_iv(log_moneyness, ttm, w_linear, w_nearest) + shift, 1e-4)


def iv_parallel_down(log_moneyness: float, ttm: float,
                     w_linear, w_nearest,
                     shift: float = 0.02) -> float:
    """Parallel -shift to the entire surface (e.g. -2 vol points)."""
    return max(get_engine_iv(log_moneyness, ttm, w_linear, w_nearest) - shift, 1e-4)


def iv_skew_steepen(log_moneyness: float, ttm: float,
                    w_linear, w_nearest,
                    factor: float = 0.10) -> float:
    """
    Skew steepening: OTM puts get more expensive, OTM calls cheaper.
    Shift = factor * (-log_moneyness); ATM is unaffected.
    """
    skew_shift = factor * (-log_moneyness)
    return max(get_engine_iv(log_moneyness, ttm, w_linear, w_nearest) + skew_shift, 1e-4)


def iv_shortend_shock(log_moneyness: float, ttm: float,
                      w_linear, w_nearest,
                      short_shift: float = 0.05,
                      cutoff_ttm: float = 0.25) -> float:
    """
    Short-end vol shock: near-dated options get a vol bump that decays
    linearly to zero at cutoff_ttm.  Long-dated options are unaffected.
    """
    blend = min(ttm / cutoff_ttm, 1.0)
    shift = short_shift * (1.0 - blend)
    return max(get_engine_iv(log_moneyness, ttm, w_linear, w_nearest) + shift, 1e-4)
