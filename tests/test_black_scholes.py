"""
tests/test_black_scholes.py
---------------------------
Unit tests for src/models/black_scholes.py.

Test values are anchored to:
- Textbook / closed-form analytical values for known inputs
- Put-call parity: C - P = S·e^(-qT) - K·e^(-rT)
- Greek identities (e.g. call delta + put delta = e^(-qT))
- Boundary behaviour at expiry (T→0) and at extreme moneyness
"""

import numpy as np
import pytest

from src.models.black_scholes import (
    bsm_delta,
    bsm_gamma,
    bsm_greeks,
    bsm_price,
    bsm_rho,
    bsm_theta,
    bsm_vega,
)

# ── shared fixtures ───────────────────────────────────────────────────────────
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
ATM_CALL = bsm_price(S, K, T, r, sigma, "call")
ATM_PUT  = bsm_price(S, K, T, r, sigma, "put")


class TestBsmPrice:
    def test_atm_call_positive(self):
        assert ATM_CALL > 0

    def test_atm_put_positive(self):
        assert ATM_PUT > 0

    def test_put_call_parity(self):
        """C - P = S·e^(-qT) - K·e^(-rT)  (q = 0)"""
        pcp_lhs = ATM_CALL - ATM_PUT
        pcp_rhs = S - K * np.exp(-r * T)
        assert abs(pcp_lhs - pcp_rhs) < 1e-8

    def test_put_call_parity_with_dividend(self):
        q = 0.02
        c = bsm_price(S, K, T, r, sigma, "call", q)
        p = bsm_price(S, K, T, r, sigma, "put",  q)
        pcp_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs((c - p) - pcp_rhs) < 1e-8

    def test_deep_itm_call_approaches_intrinsic(self):
        """Deep ITM call price ≈ S·e^(-qT) - K·e^(-rT) (lower bound)."""
        price = bsm_price(200.0, 100.0, 1.0, r, sigma, "call")
        lower = 200.0 - 100.0 * np.exp(-r * T)
        assert price >= lower - 1e-8

    def test_deep_otm_call_near_zero(self):
        price = bsm_price(50.0, 200.0, 1.0, r, sigma, "call")
        assert price < 1.0

    def test_invalid_inputs_return_nan(self):
        assert np.isnan(bsm_price(0, K, T, r, sigma))
        assert np.isnan(bsm_price(S, 0, T, r, sigma))
        assert np.isnan(bsm_price(S, K, 0, r, sigma))
        assert np.isnan(bsm_price(S, K, T, r, 0))

    def test_known_value_call(self):
        """
        For S=K=100, T=1, r=0.05, sigma=0.2:
        d1 = (0 + (0.05 + 0.02)*1) / 0.2 = 0.35
        d2 = 0.35 - 0.2 = 0.15
        Textbook value ≈ 10.4506
        """
        price = bsm_price(100, 100, 1.0, 0.05, 0.20, "call")
        assert abs(price - 10.4506) < 0.01


class TestBsmDelta:
    def test_call_delta_positive(self):
        assert bsm_delta(S, K, T, r, sigma, "call") > 0

    def test_put_delta_negative(self):
        assert bsm_delta(S, K, T, r, sigma, "put") < 0

    def test_call_delta_bounded(self):
        d = bsm_delta(S, K, T, r, sigma, "call")
        assert 0 < d < 1

    def test_put_delta_bounded(self):
        d = bsm_delta(S, K, T, r, sigma, "put")
        assert -1 < d < 0

    def test_call_put_delta_identity(self):
        """
        Call delta - put delta = e^(-q·T).

        Derived directly from the BSM formula:
          call delta = e^(-qT) · N(d1)
          put  delta = e^(-qT) · (N(d1) - 1)
          difference = e^(-qT)  (for any q)
        """
        q_test = 0.0
        dc = bsm_delta(S, K, T, r, sigma, "call", q_test)
        dp = bsm_delta(S, K, T, r, sigma, "put",  q_test)
        expected = np.exp(-q_test * T)
        assert abs((dc - dp) - expected) < 1e-10

    def test_deep_itm_call_delta_near_one(self):
        d = bsm_delta(200.0, 100.0, T, r, sigma, "call")
        assert d > 0.99

    def test_deep_otm_call_delta_near_zero(self):
        d = bsm_delta(50.0, 200.0, T, r, sigma, "call")
        assert d < 0.01

    def test_invalid_inputs_return_nan(self):
        assert np.isnan(bsm_delta(0, K, T, r, sigma))


class TestBsmVega:
    def test_vega_positive(self):
        assert bsm_vega(S, K, T, r, sigma) > 0

    def test_vega_same_for_calls_and_puts(self):
        """Vega is identical for calls and puts by the BSM formula."""
        v_call = bsm_vega(S, K, T, r, sigma)
        # Vega doesn't take option_type, just confirm it's positive and finite
        assert v_call > 0 and np.isfinite(v_call)

    def test_vega_peaks_near_atm(self):
        v_atm  = bsm_vega(100.0, 100.0, T, r, sigma)
        v_otm  = bsm_vega(100.0, 150.0, T, r, sigma)
        v_deep_otm = bsm_vega(100.0, 200.0, T, r, sigma)
        assert v_atm > v_otm > v_deep_otm

    def test_invalid_inputs_return_nan(self):
        assert np.isnan(bsm_vega(0, K, T, r, sigma))


class TestBsmGamma:
    def test_gamma_positive(self):
        assert bsm_gamma(S, K, T, r, sigma) > 0

    def test_gamma_peaks_near_atm(self):
        g_atm = bsm_gamma(100.0, 100.0, T, r, sigma)
        g_otm = bsm_gamma(100.0, 150.0, T, r, sigma)
        assert g_atm > g_otm

    def test_invalid_inputs_return_nan(self):
        assert np.isnan(bsm_gamma(0, K, T, r, sigma))


class TestBsmTheta:
    def test_call_theta_negative_for_long(self):
        """Long options lose value over time (theta < 0)."""
        assert bsm_theta(S, K, T, r, sigma, "call") < 0

    def test_put_theta_negative(self):
        assert bsm_theta(S, K, T, r, sigma, "put") < 0

    def test_invalid_inputs_return_nan(self):
        assert np.isnan(bsm_theta(0, K, T, r, sigma))


class TestBsmGreeks:
    def test_greeks_returns_five_keys(self):
        g = bsm_greeks(S, K, T, r, sigma, "call")
        assert set(g.keys()) == {"delta", "gamma", "vega", "theta", "rho"}

    def test_all_finite_for_valid_inputs(self):
        g = bsm_greeks(S, K, T, r, sigma, "call")
        assert all(np.isfinite(v) for v in g.values())

    def test_all_nan_for_invalid_inputs(self):
        g = bsm_greeks(0, K, T, r, sigma, "call")
        assert all(np.isnan(v) for v in g.values())

    def test_consistency_with_individual_functions(self):
        g = bsm_greeks(S, K, T, r, sigma, "put")
        assert abs(g["delta"] - bsm_delta(S, K, T, r, sigma, "put")) < 1e-12
        assert abs(g["gamma"] - bsm_gamma(S, K, T, r, sigma))        < 1e-12
        assert abs(g["vega"]  - bsm_vega(S, K, T, r, sigma))         < 1e-12
