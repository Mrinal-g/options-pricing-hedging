"""
tests/test_svi.py
-----------------
Unit tests for src/surface/svi.py and src/surface/iv_solver.py working
together as a surface.

Key invariants tested:
- SVI total variance is always non-negative
- SVI fit reproduces ATM variance to tight tolerance
- Surface interpolation round-trips IV correctly
- Calendar no-arbitrage: w is non-decreasing across expiries at ATM
- get_engine_iv returns nan for ttm <= 0
- save/load cycle returns the same IV values (also tests cache clearing)
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from src.surface.svi import (
    svi_total_variance,
    fit_svi,
    build_surface,
    get_engine_iv,
    save_surface,
    load_surface,
    _cache,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_smile(atm_iv: float = 0.25, skew: float = -0.10,
                ttm: float = 0.25, n_points: int = 15):
    """Generate a synthetic smile with known SVI shape for testing."""
    k = np.linspace(-0.20, 0.20, n_points)
    # True SVI params: a, b, rho, m, sigma
    params_true = np.array([atm_iv**2 * ttm * 0.9, 0.10, -0.50, 0.0, 0.10])
    w = svi_total_variance(k, params_true)
    w = np.clip(w, 1e-6, None)
    iv = np.sqrt(w / ttm)
    return k, iv, w, ttm


def _make_surface_df(expiries_ttm: list = None):
    """Build a synthetic multi-expiry SVI surface DataFrame."""
    if expiries_ttm is None:
        expiries_ttm = [7/365, 30/365, 90/365, 180/365, 365/365]
    rows = []
    for ttm in expiries_ttm:
        k, iv, w, _ = _make_smile(ttm=ttm)
        for ki, wi, ivi in zip(k, w, iv):
            rows.append({
                "log_moneyness": ki,
                "ttm"          : ttm,
                "w_svi"        : wi,
                "iv_smooth"    : ivi,
            })
    return pd.DataFrame(rows)


# ── svi_total_variance ────────────────────────────────────────────────────────

class TestSviTotalVariance:
    def test_non_negative(self):
        """Total variance must be >= 0 everywhere."""
        params = np.array([0.04, 0.10, -0.50, 0.0, 0.10])
        k = np.linspace(-0.5, 0.5, 100)
        w = svi_total_variance(k, params)
        assert np.all(w >= 0)

    def test_atm_value(self):
        """At k=0, w = a + b*sigma (since rho*(0-m)+sqrt((0-m)^2+sigma^2) = sqrt(m^2+s^2))."""
        a, b, rho, m, sigma = 0.05, 0.10, -0.50, 0.0, 0.10
        params = np.array([a, b, rho, m, sigma])
        w_atm = svi_total_variance(np.array([0.0]), params)[0]
        expected = a + b * (rho * (0 - m) + np.sqrt(m**2 + sigma**2))
        assert abs(w_atm - expected) < 1e-12

    def test_skew_shape(self):
        """Negative rho → smile tilts left (lower strike has higher IV)."""
        params = np.array([0.04, 0.10, -0.70, 0.0, 0.10])
        w_otm_put  = svi_total_variance(np.array([-0.10]), params)[0]
        w_atm      = svi_total_variance(np.array([0.0]),   params)[0]
        w_otm_call = svi_total_variance(np.array([0.10]),  params)[0]
        assert w_otm_put > w_atm     # left wing higher
        assert w_otm_call < w_otm_put  # asymmetric skew

    def test_returns_array(self):
        params = np.array([0.04, 0.10, -0.5, 0.0, 0.10])
        k = np.linspace(-0.2, 0.2, 50)
        w = svi_total_variance(k, params)
        assert w.shape == (50,)


# ── fit_svi ───────────────────────────────────────────────────────────────────

class TestFitSvi:
    def test_fit_recovers_atm_variance(self):
        """Fitted SVI should reproduce ATM total variance to within 1%."""
        k, iv, w, ttm = _make_smile(atm_iv=0.25, ttm=0.25)
        params = fit_svi(k, w, ttm)
        assert params is not None
        w_fitted_atm = svi_total_variance(np.array([0.0]), params)[0]
        w_true_atm   = w[np.argmin(np.abs(k))]
        assert abs(w_fitted_atm - w_true_atm) / w_true_atm < 0.01

    def test_fit_returns_array_of_5(self):
        k, _, w, ttm = _make_smile()
        params = fit_svi(k, w, ttm)
        assert params is not None
        assert len(params) == 5

    def test_fit_produces_non_negative_variance(self):
        k, _, w, ttm = _make_smile()
        params = fit_svi(k, w, ttm)
        assert params is not None
        k_dense = np.linspace(k.min(), k.max(), 200)
        w_fit   = svi_total_variance(k_dense, params)
        assert np.all(w_fit >= -1e-8)

    def test_fit_fails_gracefully_with_too_few_points(self):
        """Fewer than 5 points — caller should skip, fit_svi returns None."""
        k = np.array([-0.05, 0.0, 0.05])
        w = np.array([0.06, 0.05, 0.06])
        # fit_svi will try but result quality is undefined;
        # important is it doesn't crash
        result = fit_svi(k, w, 0.25)
        # Either None or a 5-element array — never an exception
        assert result is None or len(result) == 5

    def test_rmse_is_small(self):
        """Fit RMSE should be small on clean synthetic data."""
        k, _, w, ttm = _make_smile()
        params  = fit_svi(k, w, ttm)
        assert params is not None
        w_fit   = svi_total_variance(k, params)
        rmse    = float(np.sqrt(np.mean((w_fit - w)**2)))
        assert rmse < 1e-4


# ── build_surface + get_engine_iv ─────────────────────────────────────────────

class TestSurface:
    def setup_method(self):
        self.surface_df   = _make_surface_df()
        self.w_lin, self.w_near = build_surface(self.surface_df)

    def test_atm_iv_reasonable(self):
        """ATM IV queried from surface should be in a sensible range."""
        iv = get_engine_iv(0.0, 90/365, self.w_lin, self.w_near)
        assert 0.05 < iv < 1.0

    def test_returns_float(self):
        iv = get_engine_iv(0.0, 90/365, self.w_lin, self.w_near)
        assert isinstance(iv, float)

    def test_nan_for_zero_ttm(self):
        assert np.isnan(get_engine_iv(0.0, 0.0, self.w_lin, self.w_near))

    def test_nan_for_negative_ttm(self):
        assert np.isnan(get_engine_iv(0.0, -0.1, self.w_lin, self.w_near))

    def test_iv_positive(self):
        for k in [-0.10, -0.05, 0.0, 0.05, 0.10]:
            iv = get_engine_iv(k, 90/365, self.w_lin, self.w_near)
            assert iv > 0, f"IV not positive at k={k}"

    def test_nearest_fallback_for_extrapolation(self):
        """Query outside convex hull should not return NaN (uses nearest fallback)."""
        iv = get_engine_iv(-0.50, 90/365, self.w_lin, self.w_near)
        assert not np.isnan(iv)
        assert iv > 0

    def test_calendar_no_arbitrage_atm(self):
        """ATM total variance must be non-decreasing across TTMs."""
        ttms = sorted(self.surface_df["ttm"].unique())
        atm_ws = []
        for t in ttms:
            w = float(self.w_lin(0.0, t))
            if np.isnan(w):
                w = float(self.w_near(0.0, t))
            atm_ws.append(w)

        diffs = np.diff(atm_ws)
        violations = np.sum(diffs < -1e-4)
        assert violations == 0, (
            f"Calendar arbitrage: w decreases at {violations} TTM transition(s). "
            f"ATM w values: {[round(w, 6) for w in atm_ws]}"
        )

    def test_smile_has_skew(self):
        """OTM put IV should be higher than OTM call IV (equity skew)."""
        iv_otm_put  = get_engine_iv(-0.10, 90/365, self.w_lin, self.w_near)
        iv_otm_call = get_engine_iv( 0.10, 90/365, self.w_lin, self.w_near)
        assert iv_otm_put > iv_otm_call


# ── save / load / cache ───────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        """Surface saved to disk then loaded should give the same IV values."""
        surface_df = _make_surface_df()
        w_lin, w_near = build_surface(surface_df)

        # Record some reference IV values before saving
        test_points = [(-0.10, 90/365), (0.0, 90/365), (0.05, 180/365)]
        iv_before = [get_engine_iv(k, t, w_lin, w_near) for k, t in test_points]

        # Save, then load from the same directory
        save_surface(w_lin, w_near, str(tmp_path))
        w_lin2, w_near2 = load_surface(str(tmp_path))

        iv_after = [get_engine_iv(k, t, w_lin2, w_near2) for k, t in test_points]

        for before, after, (k, t) in zip(iv_before, iv_after, test_points):
            assert abs(before - after) < 1e-6, (
                f"Round-trip IV mismatch at k={k}, t={t:.3f}: "
                f"before={before:.6f}, after={after:.6f}"
            )

    def test_save_clears_cache(self, tmp_path):
        """save_surface() must clear the in-process cache for that directory."""
        surface_df = _make_surface_df()
        w_lin, w_near = build_surface(surface_df)

        save_surface(w_lin, w_near, str(tmp_path))
        cache_key = str(tmp_path.resolve())
        # Cache should be cleared after save — next load reads from disk
        assert cache_key not in _cache

    def test_load_missing_files_raises(self, tmp_path):
        """load_surface() on an empty directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_surface(str(tmp_path))
