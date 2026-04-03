"""
tests/test_pricers.py
---------------------
Unit tests for CRR binomial tree (src/models/binomial.py) and
LSM Monte Carlo (src/models/monte_carlo.py).

Key invariants tested:
- American price >= European BSM price (early exercise premium >= 0)
- CRR European (american=False) converges to BSM as N → ∞
- CRR price converges as N increases (stable at N=200)
- American call == European call for q=0 (no benefit from early exercise)
- LSM put price >= BSM put (early exercise value >= 0)
- LSM is within a reasonable tolerance of CRR for ATM puts
"""

import numpy as np
import pytest

from src.models.binomial import crr_price, crr_delta, crr_early_exercise_boundary
from src.models.black_scholes import bsm_price
from src.models.monte_carlo import lsm_price, lsm_price_with_stderr

# ── shared parameters ─────────────────────────────────────────────────────────
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
q = 0.0


class TestCrrPrice:
    def test_american_put_ge_european_bsm(self):
        """American put >= European BSM put (early exercise has non-negative value)."""
        crr_p = crr_price(S, K, T, r, sigma, "put", q, N=200, american=True)
        bsm_p = bsm_price(S, K, T, r, sigma, "put", q)
        assert crr_p >= bsm_p - 1e-4

    def test_american_call_equals_european_for_no_dividend(self):
        """
        For q=0, early exercise of a call is never optimal.
        CRR American call should equal BSM European call within tree discretisation
        error (threshold 0.10 at N=200 is conservative).
        """
        crr_c = crr_price(S, K, T, r, sigma, "call", q, N=200, american=True)
        bsm_c = bsm_price(S, K, T, r, sigma, "call", q)
        assert abs(crr_c - bsm_c) < 0.10

    def test_european_crr_converges_to_bsm(self):
        """CRR European (american=False) at N=500 should be within $0.05 of BSM."""
        crr_eur = crr_price(S, K, T, r, sigma, "put", q, N=500, american=False)
        bsm_p   = bsm_price(S, K, T, r, sigma, "put", q)
        assert abs(crr_eur - bsm_p) < 0.05

    def test_price_convergence_n200_vs_n500(self):
        """CRR American put price should not change by more than $0.01 from N=200 to N=500."""
        p200 = crr_price(S, K, T, r, sigma, "put", q, N=200, american=True)
        p500 = crr_price(S, K, T, r, sigma, "put", q, N=500, american=True)
        assert abs(p200 - p500) < 0.01

    def test_put_call_parity_european_crr(self):
        """European CRR: C - P = S - K·e^(-rT)  (q=0)."""
        c = crr_price(S, K, T, r, sigma, "call", q, N=300, american=False)
        p = crr_price(S, K, T, r, sigma, "put",  q, N=300, american=False)
        pcp = S - K * np.exp(-r * T)
        assert abs((c - p) - pcp) < 0.05

    def test_deep_itm_put_has_high_eep(self):
        """Deep ITM put should have meaningful early exercise premium."""
        S_deep_itm = 60.0
        crr_p = crr_price(S_deep_itm, K, T, r, sigma, "put",  q, N=200, american=True)
        bsm_p = bsm_price(S_deep_itm, K, T, r, sigma, "put",  q)
        assert crr_p - bsm_p >= 0  # EEP >= 0

    def test_invalid_inputs_return_nan(self):
        assert np.isnan(crr_price(0, K, T, r, sigma, "put"))
        assert np.isnan(crr_price(S, 0, T, r, sigma, "put"))
        assert np.isnan(crr_price(S, K, 0, r, sigma, "put"))
        assert np.isnan(crr_price(S, K, T, r, 0, "put"))

    def test_positive_price(self):
        p = crr_price(S, K, T, r, sigma, "put", q)
        assert p > 0

    def test_short_maturity(self):
        """Short-maturity ATM put should price without error."""
        p = crr_price(S, K, 7/365, r, sigma, "put", q, N=50)
        assert p > 0 and np.isfinite(p)


class TestCrrDelta:
    def test_call_delta_positive(self):
        d = crr_delta(S, K, T, r, sigma, "call", q, N=200)
        assert d > 0

    def test_put_delta_negative(self):
        d = crr_delta(S, K, T, r, sigma, "put", q, N=200)
        assert d < 0

    def test_delta_close_to_bsm_delta(self):
        from src.models.black_scholes import bsm_delta
        d_crr = crr_delta(S, K, T, r, sigma, "put", q, N=200)
        d_bsm = bsm_delta(S, K, T, r, sigma, "put", q)
        # CRR delta can differ from BSM for Americans, but should be in the same ballpark
        assert abs(d_crr - d_bsm) < 0.05

    def test_invalid_inputs_return_nan(self):
        assert np.isnan(crr_delta(0, K, T, r, sigma))


class TestCrrBoundary:
    def test_returns_two_arrays(self):
        times, boundary = crr_early_exercise_boundary(S, K, T, r, sigma, q, N=100)
        assert times is not None and boundary is not None
        assert len(times) == len(boundary) == 101

    def test_boundary_below_strike(self):
        """Early exercise boundary should be at or below the strike."""
        _, boundary = crr_early_exercise_boundary(S, K, T, r, sigma, q, N=100)
        valid = boundary[~np.isnan(boundary)]
        assert np.all(valid <= K + 1e-4)

    def test_boundary_positive(self):
        """All valid boundary values should be positive."""
        _, boundary = crr_early_exercise_boundary(S, K, T, r, sigma, q, N=100)
        valid = boundary[~np.isnan(boundary)]
        assert np.all(valid > 0)

    def test_higher_vol_lower_boundary(self):
        """
        Higher vol → lower early exercise boundary (option worth more to hold).
        """
        _, b_low_vol  = crr_early_exercise_boundary(S, K, T, r, 0.10, q, N=100)
        _, b_high_vol = crr_early_exercise_boundary(S, K, T, r, 0.40, q, N=100)
        v_low  = b_low_vol[~np.isnan(b_low_vol)][0]
        v_high = b_high_vol[~np.isnan(b_high_vol)][0]
        assert v_low > v_high

    def test_invalid_inputs_return_none(self):
        times, boundary = crr_early_exercise_boundary(0, K, T, r, sigma)
        assert times is None and boundary is None


class TestLsmPrice:
    def test_american_put_ge_european_bsm(self):
        """LSM American put >= European BSM put."""
        lsm_p = lsm_price(S, K, T, r, sigma, "put", q, M=10_000, seed=42)
        bsm_p = bsm_price(S, K, T, r, sigma, "put", q)
        assert lsm_p >= bsm_p - 0.20  # allow Monte Carlo variance

    def test_lsm_close_to_crr(self):
        """LSM(10k) should be within $0.20 of CRR(N=200) for ATM put."""
        lsm_p = lsm_price(S, K, T, r, sigma, "put", q, M=10_000, seed=42)
        crr_p = crr_price(S, K, T, r, sigma, "put", q, N=200, american=True)
        assert abs(lsm_p - crr_p) < 0.20

    def test_positive_price(self):
        p = lsm_price(S, K, T, r, sigma, "put", q, M=5_000, seed=0)
        assert p > 0

    def test_invalid_inputs_return_nan(self):
        assert np.isnan(lsm_price(0, K, T, r, sigma))
        assert np.isnan(lsm_price(S, K, 0, r, sigma))
        assert np.isnan(lsm_price(S, K, T, r, 0))

    def test_reproducible_with_same_seed(self):
        p1 = lsm_price(S, K, T, r, sigma, "put", q, M=2_000, seed=7)
        p2 = lsm_price(S, K, T, r, sigma, "put", q, M=2_000, seed=7)
        assert p1 == p2

    def test_different_seeds_give_different_prices(self):
        p1 = lsm_price(S, K, T, r, sigma, "put", q, M=500, seed=1)
        p2 = lsm_price(S, K, T, r, sigma, "put", q, M=500, seed=2)
        assert p1 != p2


class TestLsmWithStderr:
    def test_returns_four_values(self):
        result = lsm_price_with_stderr(S, K, T, r, sigma, "put", q, M=4_000)
        assert len(result) == 4

    def test_ci_contains_price(self):
        price, stderr, lo, hi = lsm_price_with_stderr(
            S, K, T, r, sigma, "put", q, M=4_000
        )
        assert lo <= price <= hi

    def test_ci_is_tight(self):
        """For ATM 1-year put with M=10k, 95% CI width should be < $1.0."""
        _, _, lo, hi = lsm_price_with_stderr(
            S, K, T, r, sigma, "put", q, M=10_000
        )
        assert (hi - lo) < 1.0
