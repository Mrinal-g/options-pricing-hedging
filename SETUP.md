# Setup & Run Guide

Complete instructions to get the project running from scratch.

---

## Prerequisites

- Python 3.11 or 3.12
- pip
- Internet connection (downloads live data from Yahoo Finance)
- ~500 MB disk space

---

## Step 1 — Clone or extract

```bash
git clone https://github.com/YOUR_USERNAME/options-pricing-hedging.git
cd options-pricing-hedging
```

---

## Step 2 — Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

---

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4 — Configure (optional)

```toml
ticker            = "GOOG"
history_start     = "2024-01-01"
history_end       = "today"
risk_free_rate    = 0.0371       # current 3-month T-bill rate
dividend_yield    = 0.0
min_bid                 = 0.05
min_mid                 = 0.05
max_spread_ratio        = 0.50
strike_lower_multiplier = 0.85
strike_upper_multiplier = 1.15
```

---

## Step 5 — Run the pipeline

```bash
python refresh.py              # full pipeline ~5 min
python refresh.py --no-lsm    # fast ~30 sec
python refresh.py --ticker AAPL --no-lsm
python refresh.py --stages 3 4 5
```

### What each stage does

| Stage | Time | What happens |
|-------|------|-------------|
| 1 — Download | ~10s | Live spot + 5 expiry option chains from Yahoo Finance |
| 2 — Clean | ~2s | 8 liquidity filters: bid/mid thresholds, spread ratio, staleness, strike band |
| 3 — Surface | ~15s | American-aware IV extraction (CRR for puts), weighted SVI with minimum weight floor, 2D LinearNDInterpolator |
| 4 — Price | ~3 min / ~10s (no-lsm) | BSM (European), CRR N=200, LSM M=10,000 |
| 5 — Validate | ~5s | MAE/MAPE/hit-rate, calibration vs OOS, vol risk premium |

---

## Step 6 — Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at **http://localhost:8501**

### Dashboard tabs

| Tab | What you see |
|-----|-------------|
| 🌋 Vol Surface | 3D SVI surface + SVI params + ATM term structure + IV smile vs fit |
| 💰 Option Pricer | Live BSM/CRR/LSM + all 5 Greeks + delta/gamma profiles + theta surface |
| 📊 Model Validation | Hit rate, MAPE, moneyness heatmap, price scatter, vol risk premium |
| 🚧 Early Exercise | CRR boundary per expiry + vol sensitivity |
| 📉 Hedge Simulator | Single-path P&L, frequency analysis, vol mismatch, 3-strategy benchmark |
| 📐 Risk & Sensitivity | Portfolio Greeks (interactive sliders), spot sensitivity, Greeks ladder, historical VaR |

---

## Step 7 — Run the tests

```bash
pytest                              # all 167 tests
pytest -v                           # verbose
pytest tests/test_analytics.py -v  # analytics module
pytest tests/test_risk.py -v        # risk module
pytest -m "not slow"                # skip slow Monte Carlo
```

---

## Docker (optional)

```bash
docker build -t options-pricer .
docker run -p 8501:8501 options-pricer
docker run options-pricer python refresh.py --ticker GOOG --no-lsm
```

---

## Common issues

**`All SVI fits failed`** — Too few OTM options after filtering (needs ≥ 8 per expiry). Try during US market hours.

**`ModuleNotFoundError: No module named 'src'`** — Run from project root directory.

**`Dashboard shows "No surface data found"`** — Run `python refresh.py` first.

**`yfinance` rate-limit errors** — Wait 30 seconds and retry.

---

## Project structure

```
options-pricing-hedging/
│
├── config.toml              ← All parameters — edit here only
├── refresh.py               ← One-command pipeline runner
├── requirements.txt
├── Dockerfile
├── pytest.ini
│
├── src/
│   ├── data/
│   │   ├── download.py      ← Live spot + option chain download
│   │   └── cleaning.py      ← 8-step filter pipeline
│   │
│   ├── models/
│   │   ├── black_scholes.py ← BSM price + 5 Greeks
│   │   ├── binomial.py      ← CRR tree + early exercise boundary
│   │   └── monte_carlo.py   ← LSM with antithetic variates
│   │
│   ├── surface/
│   │   ├── iv_solver.py     ← American-aware IV (CRR inversion for puts)
│   │   └── svi.py           ← Weighted SVI, surface interpolation
│   │
│   ├── risk/
│   │   ├── delta_hedge.py   ← Hedge simulator
│   │   ├── greeks.py        ← Portfolio Greek aggregation
│   │   ├── analytics.py     ← Greeks ladder, spot ladder, smile data
│   │   └── var.py           ← Historical VaR, parametric VaR
│   │
│   └── validation/
│       └── metrics.py       ← MAE/MAPE/hit-rate + vol risk premium
│
├── dashboard/
│   └── app.py               ← 6-tab Streamlit dashboard
│
├── tests/
│   ├── test_black_scholes.py  ← 30 tests
│   ├── test_pricers.py        ← 27 tests
│   ├── test_iv_solver.py      ← 20 tests (European + American IV)
│   ├── test_svi.py            ← 20 tests
│   ├── test_risk.py           ← 40 tests
│   └── test_analytics.py     ← 30 tests
│
└── notebooks/
    ├── 01_data_download.ipynb
    ├── 02_market_cleaning.ipynb
    ├── 03_iv_smile.ipynb
    ├── 04_american_option_pricing.ipynb
    └── 05_model_validation.ipynb
```
