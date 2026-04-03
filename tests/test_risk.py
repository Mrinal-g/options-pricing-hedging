"""
tests/test_risk.py
------------------
Unit tests for src/risk/ modules:
  - delta_hedge.py : P&L invariants, frequency/vol-mismatch analysis
  - greeks.py      : portfolio Greek aggregation
  - var.py         : stress scenario P&L direction, VaR positivity
"""

import numpy as np
import pandas as pd
import pytest

from src.risk.delta_hedge import (
    simulate_delta_hedge,
    hedge_summary,
    run_frequency_analysis,
    run_vol_mismatch_analysis,
)
from src.risk.greeks import OptionPosition, portfolio_greeks, greeks_by_expiry
from src.risk.var import stress_scenarios, historical_var, parametric_var

# ── shared parameters ─────────────────────────────────────────────────────────
S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.20
q = 0.0


# ── delta hedging ─────────────────────────────────────────────────────────────

class TestDeltaHedge:
    def test_returns_dataframe(self):
        df = simulate_delta_hedge(S, K, T, r, sigma, n_steps=20, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self):
        """n_steps + 1 rows (including t=0)."""
        n = 30
        df = simulate_delta_hedge(S, K, T, r, sigma, n_steps=n, seed=0)
        assert len(df) == n + 1

    def test_required_columns_present(self):
        df = simulate_delta_hedge(S, K, T, r, sigma, n_steps=20, seed=0)
        required = ["step", "t", "S", "delta", "cash",
                    "pnl_cumulative", "pnl_daily", "transaction_cost"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_initial_pnl_is_zero(self):
        """P&L at t=0 (perfect hedge) should be zero."""
        df = simulate_delta_hedge(S, K, T, r, sigma, n_steps=50, seed=0)
        assert df["pnl_cumulative"].iloc[0] == 0.0

    def test_pnl_small_with_perfect_vol(self):
        """
        With sigma_realised == sigma_implied and daily hedging, final P&L
        should be small relative to the option premium (< 10% of premium).
        """
        from src.models.black_scholes import bsm_price
        premium = bsm_price(S, K, T, r, sigma, "put")
        df   = simulate_delta_hedge(S, K, T, r, sigma, n_steps=252, seed=42)
        pnl  = abs(df["pnl_cumulative"].iloc[-1])
        assert pnl < 0.20 * premium, (
            f"Final P&L {pnl:.4f} too large vs premium {premium:.4f}"
        )

    def test_transaction_costs_non_negative(self):
        df = simulate_delta_hedge(S, K, T, r, sigma, n_steps=52,
                                  transaction_cost_pct=0.001, seed=0)
        assert (df["transaction_cost"] >= 0).all()

    def test_tc_zero_without_costs(self):
        df = simulate_delta_hedge(S, K, T, r, sigma, n_steps=20,
                                  transaction_cost_pct=0.0, seed=0)
        # Step 0 always has 0 TC; subsequent steps may have small numerical noise
        assert df["transaction_cost"].iloc[0] == 0.0

    def test_reproducible(self):
        df1 = simulate_delta_hedge(S, K, T, r, sigma, n_steps=50, seed=7)
        df2 = simulate_delta_hedge(S, K, T, r, sigma, n_steps=50, seed=7)
        assert (df1["S"] == df2["S"]).all()
        assert (df1["pnl_cumulative"] == df2["pnl_cumulative"]).all()

    def test_different_seeds_different_paths(self):
        df1 = simulate_delta_hedge(S, K, T, r, sigma, n_steps=50, seed=1)
        df2 = simulate_delta_hedge(S, K, T, r, sigma, n_steps=50, seed=2)
        assert not (df1["S"] == df2["S"]).all()

    def test_call_and_put_both_work(self):
        for otype in ["call", "put"]:
            df = simulate_delta_hedge(S, K, T, r, sigma,
                                      option_type=otype, n_steps=20, seed=0)
            assert len(df) == 21
            assert not df["pnl_cumulative"].isna().any()

    def test_vol_mismatch_positive_when_realised_higher(self):
        """
        Short option position: if realised vol > implied vol, you systematically
        lose money (negative P&L) because you undercharged for gamma.
        Average over many paths — mean P&L should be negative.
        """
        pnls = []
        for seed in range(50):
            df = simulate_delta_hedge(
                S, K, T, r,
                sigma_implied=0.20, sigma_realised=0.35,
                option_type="put", n_steps=252, seed=seed,
            )
            pnls.append(df["pnl_cumulative"].iloc[-1])
        assert np.mean(pnls) < 0, (
            "Expected negative mean P&L when realised vol > implied vol"
        )


class TestHedgeSummary:
    def test_returns_dict(self):
        df = simulate_delta_hedge(S, K, T, r, sigma, n_steps=50, seed=0)
        result = hedge_summary(df, S, K, T, r, sigma, "put")
        assert isinstance(result, dict)

    def test_required_keys(self):
        df = simulate_delta_hedge(S, K, T, r, sigma, n_steps=50, seed=0)
        result = hedge_summary(df, S, K, T, r, sigma, "put")
        for key in ["initial_option_value", "final_pnl", "pnl_as_pct_premium",
                    "total_transaction_cost", "n_steps"]:
            assert key in result

    def test_n_steps_correct(self):
        n = 40
        df = simulate_delta_hedge(S, K, T, r, sigma, n_steps=n, seed=0)
        result = hedge_summary(df, S, K, T, r, sigma)
        assert result["n_steps"] == n

    def test_initial_value_positive(self):
        df = simulate_delta_hedge(S, K, T, r, sigma, n_steps=20, seed=0)
        result = hedge_summary(df, S, K, T, r, sigma)
        assert result["initial_option_value"] > 0


class TestFrequencyAnalysis:
    def test_returns_dataframe(self):
        df = run_frequency_analysis(S, K, T, r, sigma, frequencies=[4, 12],
                                    n_sims=10, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_correct_frequencies_indexed(self):
        freqs = [4, 52]
        df = run_frequency_analysis(S, K, T, r, sigma, frequencies=freqs,
                                    n_sims=10, seed=0)
        assert list(df.index) == freqs

    def test_higher_frequency_lower_pnl_std(self):
        """
        More frequent hedging → lower P&L variance.
        Test with enough sims that this should reliably hold.
        """
        df = run_frequency_analysis(S, K, T, r, sigma,
                                    frequencies=[4, 252],
                                    n_sims=100, seed=42)
        std_quarterly = df.loc[4,   "std_pnl"]
        std_daily     = df.loc[252, "std_pnl"]
        assert std_daily < std_quarterly, (
            f"Expected daily std ({std_daily:.4f}) < quarterly std ({std_quarterly:.4f})"
        )

    def test_required_columns(self):
        df = run_frequency_analysis(S, K, T, r, sigma, frequencies=[12],
                                    n_sims=5, seed=0)
        for col in ["mean_pnl", "std_pnl", "pnl_5pct", "pnl_95pct", "mean_tc"]:
            assert col in df.columns


# ── portfolio Greeks ──────────────────────────────────────────────────────────

class TestPortfolioGreeks:
    def _positions(self):
        return [
            OptionPosition(K=95.0,  T=0.25, sigma=0.22, option_type="put",  quantity=10),
            OptionPosition(K=105.0, T=0.25, sigma=0.20, option_type="call", quantity=-5),
            OptionPosition(K=100.0, T=0.50, sigma=0.21, option_type="put",  quantity=3),
        ]

    def test_returns_dict(self):
        g = portfolio_greeks(self._positions(), S, r)
        assert isinstance(g, dict)

    def test_required_keys(self):
        g = portfolio_greeks(self._positions(), S, r)
        for key in ["net_delta", "net_gamma", "net_vega", "net_theta",
                    "total_value", "dollar_delta", "n_positions"]:
            assert key in g

    def test_n_positions(self):
        pos = self._positions()
        g   = portfolio_greeks(pos, S, r)
        assert g["n_positions"] == len(pos)

    def test_long_call_positive_delta(self):
        pos = [OptionPosition(K=100, T=0.25, sigma=0.20, option_type="call", quantity=1)]
        g   = portfolio_greeks(pos, S, r)
        assert g["net_delta"] > 0

    def test_long_put_negative_delta(self):
        pos = [OptionPosition(K=100, T=0.25, sigma=0.20, option_type="put", quantity=1)]
        g   = portfolio_greeks(pos, S, r)
        assert g["net_delta"] < 0

    def test_long_call_positive_vega(self):
        pos = [OptionPosition(K=100, T=0.25, sigma=0.20, option_type="call", quantity=1)]
        g   = portfolio_greeks(pos, S, r)
        assert g["net_vega"] > 0

    def test_short_option_negative_gamma(self):
        pos = [OptionPosition(K=100, T=0.25, sigma=0.20, option_type="put", quantity=-1)]
        g   = portfolio_greeks(pos, S, r)
        assert g["net_gamma"] < 0

    def test_empty_positions(self):
        g = portfolio_greeks([], S, r)
        assert g["net_delta"] == 0.0
        assert g["total_value"] == 0.0

    def test_opposite_positions_cancel(self):
        """Long and short same option should give near-zero net Greeks."""
        pos = [
            OptionPosition(K=100, T=0.25, sigma=0.20, option_type="put", quantity= 1),
            OptionPosition(K=100, T=0.25, sigma=0.20, option_type="put", quantity=-1),
        ]
        g = portfolio_greeks(pos, S, r)
        assert abs(g["net_delta"]) < 1e-10
        assert abs(g["net_gamma"]) < 1e-10

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError):
            OptionPosition(K=100, T=0.25, sigma=0.20, option_type="future")

    def test_greeks_by_expiry_returns_dataframe(self):
        df = greeks_by_expiry(self._positions(), S, r)
        assert isinstance(df, pd.DataFrame)
        assert "delta" in df.columns


# ── VaR and stress ────────────────────────────────────────────────────────────

class TestStressScenarios:
    def _positions(self):
        return [
            OptionPosition(K=100, T=0.25, sigma=0.20, option_type="put", quantity=10),
        ]

    def _spot_hist(self):
        """50 days of synthetic spot history."""
        np.random.seed(0)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 60)))
        return pd.DataFrame({"close": prices})

    def test_returns_dataframe(self):
        df = stress_scenarios(self._positions(), S, r)
        assert isinstance(df, pd.DataFrame)

    def test_base_scenario_zero_pnl(self):
        """Base scenario (no shocks) should have P&L = 0."""
        df = stress_scenarios(self._positions(), S, r)
        assert abs(df.loc["Base", "pnl"]) < 1e-8

    def test_spot_down_increases_long_put_value(self):
        """Spot -10% → long put gains value → positive P&L."""
        df = stress_scenarios(self._positions(), S, r)
        assert df.loc["Spot -10%", "pnl"] > 0

    def test_spot_up_decreases_long_put_value(self):
        """Spot +10% → long put loses value → negative P&L."""
        df = stress_scenarios(self._positions(), S, r)
        assert df.loc["Spot +10%", "pnl"] < 0

    def test_vol_up_increases_long_option_value(self):
        """Vol +5pts → long put gains (positive vega)."""
        df = stress_scenarios(self._positions(), S, r)
        assert df.loc["Vol +5pts", "pnl"] > 0

    def test_crash_scenario_pnl_large_positive(self):
        """Spot -20%, Vol +15pts → long put should have large gain."""
        df = stress_scenarios(self._positions(), S, r)
        assert df.loc["Spot -20%, Vol +15pts", "pnl"] > df.loc["Spot -10%", "pnl"]

    def test_historical_var_positive(self):
        var = historical_var(self._positions(), self._spot_hist(), S, r)
        assert var["var_1day"] >= 0

    def test_cvar_ge_var(self):
        """CVaR (expected shortfall) must be >= VaR by definition."""
        var = historical_var(self._positions(), self._spot_hist(), S, r)
        assert var["cvar_1day"] >= var["var_1day"] - 1e-8

    def test_parametric_var_positive(self):
        var = parametric_var(self._positions(), S, r, sigma_port=0.20)
        assert var["var_1day"] >= 0

    def test_parametric_var_scales_with_horizon(self):
        """Horizon VaR = 1-day VaR * sqrt(horizon)."""
        var = parametric_var(self._positions(), S, r, sigma_port=0.20,
                             horizon_days=10)
        expected_scale = np.sqrt(10)
        ratio = var["var_horizon"] / var["var_1day"]
        assert abs(ratio - expected_scale) < 0.01
