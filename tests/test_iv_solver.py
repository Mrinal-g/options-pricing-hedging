"""
tests/test_iv_solver.py
-----------------------
Tests for both European and American IV solvers.

Key invariants:
- European round-trip: BSM(S, K, T, r, european_iv) == market_price
- American round-trip: CRR(S, K, T, r, american_iv) ≈ market_price
- American IV <= European IV for puts (no EEP inflation)
- American IV == European IV for calls with q=0
- Invalid inputs return NaN in all cases
"""

import numpy as np
import pytest

from src.models.black_scholes import bsm_price
from src.models.binomial import crr_price
from src.surface.iv_solver import (
    implied_volatility,
    european_implied_volatility,
    american_implied_volatility,
)

S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.30
q = 0.0


class TestEuropeanIV:
    def test_roundtrip_call(self):
        mp = bsm_price(S, K, T, r, sigma, "call")
        iv = european_implied_volatility(mp, S, K, T, r, "call")
        assert abs(iv - sigma) < 1e-6

    def test_roundtrip_put(self):
        mp = bsm_price(S, K, T, r, sigma, "put")
        iv = european_implied_volatility(mp, S, K, T, r, "put")
        assert abs(iv - sigma) < 1e-6

    def test_otm_call(self):
        mp = bsm_price(100, 115, 0.5, 0.04, 0.22, "call")
        iv = european_implied_volatility(mp, 100, 115, 0.5, 0.04, "call")
        assert abs(iv - 0.22) < 1e-5

    def test_otm_put(self):
        mp = bsm_price(100, 88, 0.25, 0.045, 0.28, "put")
        iv = european_implied_volatility(mp, 100, 88, 0.25, 0.045, "put")
        assert abs(iv - 0.28) < 1e-5

    def test_zero_price_nan(self):
        assert np.isnan(european_implied_volatility(0.0, S, K, T, r))

    def test_nan_price_nan(self):
        assert np.isnan(european_implied_volatility(np.nan, S, K, T, r))

    def test_zero_ttm_nan(self):
        assert np.isnan(european_implied_volatility(5.0, S, K, 0.0, r))

    def test_positive_result(self):
        mp = bsm_price(S, K, T, r, sigma, "put")
        iv = european_implied_volatility(mp, S, K, T, r, "put")
        assert iv > 0


class TestAmericanIV:
    def test_roundtrip_put_crr(self):
        """CRR(sigma_A) should recover the American market price."""
        am_price = crr_price(S, K, T, r, sigma, "put", q, N=100)
        iv_a = american_implied_volatility(am_price, S, K, T, r, "put", q)
        crr_check = crr_price(S, K, T, r, iv_a, "put", q, N=100)
        assert abs(crr_check - am_price) < 0.05   # within 5 cents

    def test_american_iv_le_european_iv_for_puts(self):
        """
        For puts: sigma_A <= sigma_E because American price > European price.
        The American solver extracts lower vol — it doesn't need to inflate
        vol to absorb early exercise premium.
        """
        am_price = crr_price(S, K, T, r, sigma, "put", q, N=100)
        iv_a = american_implied_volatility(am_price, S, K, T, r, "put", q)
        iv_e = european_implied_volatility(am_price, S, K, T, r, "put", q)
        assert iv_a <= iv_e + 1e-6, (
            f"Expected American IV <= European IV for puts. "
            f"Got sigma_A={iv_a:.4f}, sigma_E={iv_e:.4f}"
        )

    def test_american_equals_european_for_call_no_dividend(self):
        """
        For calls with q=0, American = European option (no early exercise).
        Both solvers should return the same IV.
        """
        eu_price = bsm_price(S, K, T, r, sigma, "call", q)
        iv_a = american_implied_volatility(eu_price, S, K, T, r, "call", q)
        iv_e = european_implied_volatility(eu_price, S, K, T, r, "call", q)
        assert abs(iv_a - iv_e) < 0.005, (
            f"Call IVs should be equal for q=0. "
            f"sigma_A={iv_a:.4f}, sigma_E={iv_e:.4f}"
        )

    def test_deep_itm_put(self):
        """Deep ITM puts have significant EEP — American IV should be meaningfully lower."""
        S_deep = 80.0
        am_price = crr_price(S_deep, K, T, r, sigma, "put", q, N=100)
        iv_a = american_implied_volatility(am_price, S_deep, K, T, r, "put", q)
        iv_e = european_implied_volatility(am_price, S_deep, K, T, r, "put", q)
        assert iv_a <= iv_e, "American IV should be <= European for deep ITM put"

    def test_zero_price_nan(self):
        assert np.isnan(american_implied_volatility(0.0, S, K, T, r))

    def test_nan_price_nan(self):
        assert np.isnan(american_implied_volatility(np.nan, S, K, T, r))

    def test_zero_ttm_nan(self):
        assert np.isnan(american_implied_volatility(5.0, S, K, 0.0, r))

    def test_positive_result(self):
        am_price = crr_price(S, K, T, r, sigma, "put", q, N=50)
        iv = american_implied_volatility(am_price, S, K, T, r, "put", q)
        assert iv > 0 and np.isfinite(iv)


class TestDispatcher:
    def test_default_is_american(self):
        """Default american=True should use American solver for puts."""
        am_price = crr_price(S, K, T, r, sigma, "put", q, N=100)
        iv_dispatch = implied_volatility(am_price, S, K, T, r, "put", q)
        iv_american = american_implied_volatility(am_price, S, K, T, r, "put", q)
        assert abs(iv_dispatch - iv_american) < 1e-8

    def test_calls_q0_uses_bsm_path(self):
        """Calls with q=0 go to European solver (faster, same result)."""
        eu_price = bsm_price(S, K, T, r, sigma, "call")
        iv = implied_volatility(eu_price, S, K, T, r, "call", q=0.0, american=True)
        assert abs(iv - sigma) < 1e-5

    def test_american_false_uses_european(self):
        eu_price = bsm_price(S, K, T, r, sigma, "put")
        iv = implied_volatility(eu_price, S, K, T, r, "put", american=False)
        assert abs(iv - sigma) < 1e-6

    def test_multiple_strikes_puts(self):
        """American IV round-trip across a range of put strikes."""
        for strike in [85, 90, 95, 100, 105]:
            am_p = crr_price(S, strike, T, r, sigma, "put", q, N=100)
            iv   = implied_volatility(am_p, S, strike, T, r, "put", q)
            if not np.isnan(iv):
                crr_check = crr_price(S, strike, T, r, iv, "put", q, N=100)
                assert abs(crr_check - am_p) < 0.10, (
                    f"Round-trip failed at K={strike}: "
                    f"market={am_p:.4f}, CRR(iv)={crr_check:.4f}"
                )
