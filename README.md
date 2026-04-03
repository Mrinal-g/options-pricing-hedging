# Options Pricing & Hedging

Production-style quantitative finance project built on live GOOG option chains.
Covers the full quant workflow: data ingestion → IV extraction → vol surface
calibration → American option pricing → delta hedging → portfolio risk analytics →
model validation — with a 6-tab interactive dashboard.

**[Live Dashboard →](https://options-pricing-hedging-uvtsfcmrgeofhuvhg6qg83.streamlit.app)**

---

## Key Results (April 3, 2026)

| Metric | Value | Notes |
|--------|-------|-------|
| CRR bid-ask hit rate | **94.4%** | 177 options, post-tariff-selloff market |
| CRR MAE | **$0.201** | Mean absolute error vs market mid |
| CRR mean error | **−$0.016** | Near-zero systematic bias |
| BSM bid-ask hit rate | 85.9% | European model — underprices American puts |
| LSM bid-ask hit rate | 62.7% | Monte Carlo simulation noise widens errors |
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
│   └── app.py                 ← 6-tab Streamlit dashboard (~1600 lines)
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

### Three-date progression — iterative model improvements

| Date | Market | CRR Hit Rate | CRR MAE | Key improvement |
|------|--------|-------------|---------|----------------|
| March 28, 2026 | Normal | 74.7% | $0.23 | American-aware IV extraction |
| April 1, 2026 | Stressed (tariff selloff) | 81.1% | $0.30 | Weighted SVI + minimum weight floor |
| April 3, 2026 | Post-selloff (live) | **94.4%** | **$0.20** | System running on stable data |

### April 3, 2026 — current live results

| Model | Hit Rate | MAE | Mean Error | MAPE |
|-------|----------|-----|-----------|------|
| CRR (American) | **94.4%** | **$0.201** | −$0.016 | **1.3%** |
| BSM (European) | 85.9% | $0.283 | −$0.207 | 1.6% |
| LSM (Monte Carlo) | 62.7% | $0.397 | −$0.345 | 2.2% |

LSM underperforms CRR due to Monte Carlo sampling variance (M=5,000 paths,
±$0.20–0.40 noise). CRR is the production workhorse for American options.

### Volatility Risk Premium (April 3, 2026)

| Expiry | Days | ATM IV | Realised Vol | VRP (pp) |
|--------|------|--------|-------------|---------|
| 2026-04-10 | 7 | 29.1% | 46.0% | **−16.9** |
| 2026-05-01 | 28 | 36.9% | 31.7% | +5.1 |
| 2026-06-18 | 76 | 35.0% | 25.5% | +9.5 |
| 2026-09-18 | 168 | 35.2% | 27.5% | +7.7 |
| 2027-03-19 | 350 | 35.4% | 29.9% | +5.5 |

7-day negative VRP (−16.9pp): the market underpriced near-term vol during the
tariff selloff — realised vol of 46% exceeded the 29% implied. All other tenors
show the typical equity VRP: implied vol exceeds realised by 5–10pp.

### Hedge Effectiveness (April 3, 2026)

| Strategy | P&L Std Dev | Variance Reduction |
|----------|-------------|-------------------|
| Unhedged | $26.43 | 0% |
| Static (delta₀, never rebalance) | $15.35 | 66.3% |
| Dynamic (daily rebalance) | $1.07 | **99.8%** |

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
