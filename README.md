# Options Pricing & Hedging

Production-style quantitative finance project built on live GOOG option chains.
Covers the full quant workflow: data ingestion → IV extraction → vol surface
calibration → American option pricing → model validation → interactive dashboard.

**[Live Dashboard →](https://YOUR_USERNAME-options-pricing-hedging-dashboard-app-XXXX.streamlit.app)**
*(replace with your Streamlit Cloud URL after deployment)*

---

## Key Results

| Metric | Value | Notes |
|--------|-------|-------|
| CRR bid-ask hit rate | **81.1%** | April 1 stressed market (tariff selloff) |
| CRR MAE | **$0.30** | vs $3.66 before weighted SVI fix |
| CRR mean error | **−$0.01** | Near-zero systematic bias |
| Hedge variance reduction | **99.8%** | Dynamic vs unhedged, 200 paths |
| Test suite | **167 passing** | 6 test files, 0 failures |

---

## Architecture

```
options-pricing-hedging/
│
├── src/
│   ├── data/
│   │   ├── download.py        ← spot history + option chains (yfinance)
│   │   └── cleaning.py        ← 8-step liquidity filter pipeline
│   │
│   ├── models/
│   │   ├── black_scholes.py   ← BSM price + all 5 Greeks (analytic)
│   │   ├── binomial.py        ← CRR tree: American/European + early exercise boundary
│   │   └── monte_carlo.py     ← LSM Monte Carlo with antithetic variates
│   │
│   ├── surface/
│   │   ├── iv_solver.py       ← American-aware IV extraction (CRR inversion for puts)
│   │   └── svi.py             ← Weighted SVI fitting, surface interpolation
│   │
│   ├── risk/
│   │   ├── delta_hedge.py     ← Hedge simulator (synthetic + historical + benchmark)
│   │   ├── greeks.py          ← Portfolio Greek aggregation
│   │   ├── analytics.py       ← Greeks ladder, spot ladder, smile data, P&L attribution
│   │   └── var.py             ← Historical VaR, parametric VaR, stress scenarios
│   │
│   └── validation/
│       └── metrics.py         ← MAE, MAPE, bid-ask hit rate, vol risk premium
│
├── dashboard/
│   └── app.py                 ← 6-tab Streamlit dashboard
│
├── tests/
│   ├── test_black_scholes.py  ← 30 tests: put-call parity, Greeks, known values
│   ├── test_pricers.py        ← 27 tests: EEP, convergence, boundary properties
│   ├── test_iv_solver.py      ← 20 tests: European + American IV round-trips
│   ├── test_svi.py            ← 20 tests: SVI fit, surface, calendar no-arbitrage
│   ├── test_risk.py           ← 40 tests: hedge simulator, Greeks, VaR
│   └── test_analytics.py      ← 30 tests: Greeks ladder, spot ladder, smile data
│
├── refresh.py                 ← One-command pipeline runner
├── config.toml                ← All parameters (ticker, rates, thresholds)
├── Dockerfile
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
# Full pipeline for GOOG — ~5 min
python refresh.py

# Fast mode — skip LSM Monte Carlo (~30 sec)
python refresh.py --no-lsm

# Different ticker
python refresh.py --ticker AAPL --no-lsm

# Re-run only specific stages
python refresh.py --stages 3 4 5
```

### 3. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501)

### 4. Run the tests

```bash
pytest -v                        # all 167 tests
pytest tests/test_risk.py -v     # single file
pytest -m "not slow"             # skip slow Monte Carlo tests
```

---

## Pipeline Stages

| Stage | What it does |
|-------|-------------|
| 1 — Download | Live spot price + option chains from Yahoo Finance |
| 2 — Clean | 8-step filter: min bid, min mid, ask>bid, positive TTM, open interest, ±15% strike band, spread ratio, staleness |
| 3 — Surface | American-aware IV extraction → weighted SVI fitting → 2D interpolated surface |
| 4 — Price | BSM (European), CRR N=200 (American), LSM M=10,000 (American) |
| 5 — Validate | MAE/MAPE/hit-rate, calibration vs OOS split, vol risk premium |

---

## Pricing Models

### Black-Scholes-Merton (European baseline)
Closed-form with continuous dividend yield `q`. For `q=0` calls, BSM and CRR
prices are identical — no early exercise premium exists.

### CRR Binomial Tree (American)
Cox-Ross-Rubinstein (1979) recombining tree, `N=200` steps. Backward induction
checks early exercise at every node. Converges to BSM for European options
(verified in tests).

### LSM Monte Carlo (American)
Longstaff-Schwartz (2001) least-squares regression on `M=10,000` GBM paths
with antithetic variates. Regression basis `[1, X, X²]` on ITM paths.

---

## Vol Surface

### American-aware IV extraction
Puts are inverted using the CRR binomial tree — the only correct method for
American options. Using BSM inversion for puts artificially inflates their
IV by including the early exercise premium. This fix took CRR hit rate from
14% → 74.7% in normal markets (March 28, 2026).

### Weighted SVI calibration
Per-expiry SVI fitting in total variance space `w = IV² × T` using
**bid-ask spread as inverse weights** — tight ATM options dominate the fit,
illiquid wing options inform it. A minimum weight floor at 20% of ATM weight
prevents wing quotes from becoming invisible in stressed markets. rho is
constrained to `(−0.999, −0.01)` — equity always has left skew.

### Surface interpolation
`LinearNDInterpolator` over `(log_moneyness, ttm) → w` with
`NearestNDInterpolator` fallback for extrapolation. Interpolating in total
variance space preserves calendar no-arbitrage.

---

## Dashboard Tabs

| Tab | Story |
|-----|-------|
| 🌋 Vol Surface | What is the market pricing? 3D SVI surface, ATM term structure, IV smile vs SVI fit |
| 💰 Option Pricer | What is this option worth? Live BSM/CRR/LSM pricing, all 5 Greeks, delta/gamma profiles, theta surface |
| 📊 Model Validation | How honest is our pricing? Hit rates, MAPE, moneyness heatmap, price scatter, vol risk premium |
| 🚧 Early Exercise | When should you exercise an American put? CRR boundary by expiry, vol sensitivity |
| 📉 Hedge Simulator | Does hedging work? Single path P&L, frequency analysis, vol mismatch, 3-strategy benchmark |
| 📐 Risk & Sensitivity | What is the full book's risk? Portfolio Greeks, spot sensitivity, Greeks ladder, historical VaR |

---

## Validation Results

### Normal market — March 28, 2026
| Model | Hit Rate | MAE | Mean Error |
|-------|----------|-----|-----------|
| BSM (European) | 59.3% | $0.25 | −$0.22 |
| CRR (American) | **74.7%** | **$0.23** | **−$0.02** |

### Stressed market — April 1, 2026 (tariff selloff)
| Model | Hit Rate | MAE | Mean Error |
|-------|----------|-----|-----------|
| BSM (European) | 66.9% | $0.43 | −$0.21 |
| CRR (American) | **81.1%** | **$0.30** | **−$0.01** |

Stressed market result is stronger because the weighted SVI with minimum weight
floor correctly captures left skew even when OTM bid-ask spreads are $3-5 wide.

### Volatility Risk Premium (GOOG)
ATM IV consistently 5-12pp above realised vol across all tenors — options are
expensive relative to what GOOG actually does. Positive VRP is the normal equity
market result. Sellers of volatility earn this premium on average.

### Hedge Effectiveness
| Strategy | P&L Std Dev | Variance Reduction |
|----------|-------------|-------------------|
| Unhedged | $27.34 | 0% |
| Static (delta₀, never rebalance) | $15.85 | 66.4% |
| Dynamic (daily rebalance) | $1.12 | **99.8%** |

---

## Configuration

All parameters in `config.toml` — no hardcoded values anywhere.

```toml
ticker            = "GOOG"
history_start     = "2024-01-01"
history_end       = "today"       # always fetches live data
risk_free_rate    = 0.0371        # current US 3-month T-bill
dividend_yield    = 0.0           # GOOG pays no dividend

min_bid                 = 0.05
min_mid                 = 0.05
max_spread_ratio        = 0.50
strike_lower_multiplier = 0.85    # ±15% moneyness band
strike_upper_multiplier = 1.15
```

---

## Docker

```bash
docker build -t options-pricer .
docker run -p 8501:8501 options-pricer
docker run options-pricer python refresh.py --ticker GOOG --no-lsm
```

---

## References

- Black & Scholes (1973). *The Pricing of Options and Corporate Liabilities.*
- Cox, Ross & Rubinstein (1979). *Option Pricing: A Simplified Approach.*
- Longstaff & Schwartz (2001). *Valuing American Options by Simulation.*
- Gatheral (2004). *A Parsimonious Arbitrage-Free Implied Volatility Parametrization.*
