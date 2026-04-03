"""
tests/test_analytics.py
-----------------------
Unit tests for src/risk/analytics.py.

Covers:
- greeks_ladder       : correct columns, signs, dollar Greeks
- greeks_by_expiry_from_df : aggregation correctness
- smile_data          : returns correct columns, sorted correctly
- spot_ladder         : base case zero change, monotonicity
"""

import numpy as np
import pandas as pd
import pytest

from src.risk.analytics import (
    greeks_ladder,
    greeks_by_expiry_from_df,
    smile_data,
    spot_ladder,
)

# ── shared fixtures ────────────────────────────────────────────────────────────

S, r, q = 100.0, 0.045, 0.0


def _make_options_df(n=6):
    """Minimal options DataFrame with required columns for analytics functions."""
    strikes     = [90.0, 95.0, 100.0, 105.0, 110.0, 115.0][:n]
    option_types = ["put", "put", "put", "call", "call", "call"][:n]
    expiration   = pd.to_datetime("2026-06-18")
    return pd.DataFrame({
        "expiration"    : [expiration] * n,
        "option_type"   : option_types,
        "strike"        : strikes,
        "ttm"           : [0.25] * n,
        "log_moneyness" : [np.log(k / S) for k in strikes],
        "iv_engine"     : [0.32, 0.28, 0.25, 0.25, 0.27, 0.30],
        "iv_model"      : [0.33, 0.29, 0.26, 0.26, 0.28, 0.31],
        "mid"           : [10.5, 6.2, 3.8, 4.1, 2.9, 1.8],
        "days_to_expiry": [90] * n,
        "spot"          : [S] * n,
    })


# ── greeks_ladder ─────────────────────────────────────────────────────────────

class TestGreeksLadder:

    def test_returns_dataframe(self):
        df = _make_options_df()
        result = greeks_ladder(df, S, r, q)
        assert isinstance(result, pd.DataFrame)

    def test_correct_row_count(self):
        df = _make_options_df(6)
        result = greeks_ladder(df, S, r, q)
        assert len(result) == 6

    def test_required_columns(self):
        df = _make_options_df()
        result = greeks_ladder(df, S, r, q)
        required = ["option_type", "strike", "ttm", "iv_engine", "mid",
                    "delta", "gamma", "vega", "theta",
                    "dollar_delta", "dollar_gamma_1pct", "dollar_vega_1pt"]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_put_delta_negative(self):
        df = _make_options_df()
        result = greeks_ladder(df, S, r, q)
        puts = result[result["option_type"] == "put"]
        assert (puts["delta"] < 0).all(), "All put deltas should be negative"

    def test_call_delta_positive(self):
        df = _make_options_df()
        result = greeks_ladder(df, S, r, q)
        calls = result[result["option_type"] == "call"]
        assert (calls["delta"] > 0).all(), "All call deltas should be positive"

    def test_gamma_positive(self):
        df = _make_options_df()
        result = greeks_ladder(df, S, r, q)
        assert (result["gamma"] > 0).all(), "Gamma always positive for long options"

    def test_theta_negative(self):
        """Long options lose value over time — theta should be negative."""
        df = _make_options_df()
        result = greeks_ladder(df, S, r, q)
        assert (result["theta"] < 0).all(), "Theta should be negative for long options"

    def test_dollar_delta_sign_consistent(self):
        """Dollar delta sign should match unit delta sign."""
        df = _make_options_df()
        result = greeks_ladder(df, S, r, q)
        assert ((result["dollar_delta"] > 0) == (result["delta"] > 0)).all()

    def test_sorted_by_expiry_and_strike(self):
        df = _make_options_df()
        result = greeks_ladder(df, S, r, q)
        # Within same expiry and type, strikes should be sorted
        for (exp, otype), grp in result.groupby(["expiration", "option_type"]):
            assert list(grp["strike"]) == sorted(grp["strike"].tolist())

    def test_invalid_iv_skipped(self):
        """Options with iv_engine=NaN should be skipped."""
        df = _make_options_df()
        df.loc[0, "iv_engine"] = np.nan
        result = greeks_ladder(df, S, r, q)
        assert len(result) == 5  # one skipped

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=_make_options_df().columns)
        result = greeks_ladder(df, S, r, q)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ── greeks_by_expiry_from_df ──────────────────────────────────────────────────

class TestGreeksByExpiry:

    def test_returns_dataframe(self):
        df = _make_options_df()
        result = greeks_by_expiry_from_df(df, S, r, q)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_expiry(self):
        df = _make_options_df()
        result = greeks_by_expiry_from_df(df, S, r, q)
        assert len(result) == 1  # one expiry in fixture

    def test_short_book_negative_gamma(self):
        """Short book (quantity=-1) should have negative net gamma."""
        df = _make_options_df()
        result = greeks_by_expiry_from_df(df, S, r, q, quantity=-1.0)
        assert (result["net_gamma"] < 0).all()

    def test_short_book_positive_theta(self):
        """Short book should have positive theta (earn time decay)."""
        df = _make_options_df()
        result = greeks_by_expiry_from_df(df, S, r, q, quantity=-1.0)
        assert (result["net_theta"] > 0).all()

    def test_required_columns(self):
        df = _make_options_df()
        result = greeks_by_expiry_from_df(df, S, r, q)
        for col in ["n_options", "net_delta", "net_gamma", "net_vega", "net_theta"]:
            assert col in result.columns

    def test_n_options_correct(self):
        df = _make_options_df(6)
        result = greeks_by_expiry_from_df(df, S, r, q)
        assert result["n_options"].iloc[0] == 6


# ── smile_data ────────────────────────────────────────────────────────────────

class TestSmileData:

    def test_returns_dataframe(self):
        df = _make_options_df()
        result = smile_data(df)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        df = _make_options_df()
        result = smile_data(df)
        for col in ["expiration", "log_moneyness", "iv_model",
                    "iv_engine", "option_type", "days_to_expiry"]:
            assert col in result.columns

    def test_no_nan_in_key_columns(self):
        df = _make_options_df()
        result = smile_data(df)
        for col in ["log_moneyness", "iv_model", "iv_engine"]:
            assert not result[col].isna().any(), f"NaN found in {col}"

    def test_sorted_by_log_moneyness(self):
        df = _make_options_df()
        result = smile_data(df)
        for exp, grp in result.groupby("expiration"):
            assert list(grp["log_moneyness"]) == sorted(grp["log_moneyness"].tolist())

    def test_all_iv_positive(self):
        df = _make_options_df()
        result = smile_data(df)
        assert (result["iv_model"] > 0).all()
        assert (result["iv_engine"] > 0).all()

    def test_missing_columns_returns_empty(self):
        df = _make_options_df().drop(columns=["iv_model"])
        result = smile_data(df)
        assert len(result) == 0


# ── spot_ladder ───────────────────────────────────────────────────────────────

class TestSpotLadder:

    def test_returns_dataframe(self):
        df = _make_options_df()
        result = spot_ladder(df, S, r, q)
        assert isinstance(result, pd.DataFrame)

    def test_default_nine_shock_levels(self):
        """Default shocks: -20% to +20% in 5% steps = 9 levels."""
        df = _make_options_df()
        result = spot_ladder(df, S, r, q)
        assert len(result) == 9

    def test_base_case_zero_change(self):
        """At 0% shock (spot = S), value_change should be exactly 0."""
        df = _make_options_df()
        result = spot_ladder(df, S, r, q)
        assert abs(result.loc[S, "value_change"]) < 1e-6

    def test_required_columns(self):
        df = _make_options_df()
        result = spot_ladder(df, S, r, q)
        for col in ["spot_shock_pct", "portfolio_value", "value_change",
                    "net_delta", "net_gamma"]:
            assert col in result.columns

    def test_short_book_loses_on_down_move(self):
        """Short put book should lose money when spot falls (puts gain value)."""
        df = _make_options_df()
        # Make all positions puts (short = quantity=-1 in spot_ladder)
        df_puts = df[df["option_type"] == "put"].copy()
        result = spot_ladder(df_puts, S, r, q)
        # Spot -20%: short puts lose (portfolio value decreases = negative change)
        down_pnl = result[result["spot_shock_pct"] < 0]["value_change"]
        assert (down_pnl < 0).all(), "Short put book should lose when spot falls"

    def test_custom_shocks(self):
        df = _make_options_df()
        custom_shocks = [0.95, 1.00, 1.05]
        result = spot_ladder(df, S, r, q, shocks=custom_shocks)
        assert len(result) == 3

    def test_shock_pct_values_correct(self):
        df = _make_options_df()
        result = spot_ladder(df, S, r, q, shocks=[0.90, 1.00, 1.10])
        pcts = sorted(result["spot_shock_pct"].tolist())
        assert abs(pcts[0] - (-10.0)) < 0.01
        assert abs(pcts[1] - 0.0) < 0.01
        assert abs(pcts[2] - 10.0) < 0.01
