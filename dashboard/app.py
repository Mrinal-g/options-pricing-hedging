"""
dashboard/app.py
----------------
Streamlit dashboard for the options pricing and hedging project.

Tabs
----
1. Vol Surface       — interactive 3D SVI surface + term structure
2. Option Pricer     — price any option with BSM / CRR / LSM + all Greeks
3. Model Validation  — MAPE, bid-ask hit rates, residual distributions
4. Early Exercise    — CRR boundary analysis for American puts
5. Hedge Simulator   — interactive delta hedging P&L, frequency & vol analysis
6. Risk & Sensitivity — Example portfolio, spot sensitivity, Greeks ladder, historical VaR

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Make src importable when running from any working directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import griddata

from src.models.binomial import crr_early_exercise_boundary, crr_price
from src.models.black_scholes import bsm_greeks, bsm_price
from src.models.monte_carlo import lsm_price
from src.surface.svi import load_surface, get_engine_iv
from src.risk.delta_hedge import (
    simulate_delta_hedge,
    hedge_summary,
    run_frequency_analysis,
    run_vol_mismatch_analysis,
    compare_hedge_strategies,
)
from src.risk.greeks import OptionPosition, portfolio_greeks
from src.risk.var import stress_scenarios, historical_var
from src.risk.analytics import (
    greeks_ladder,
    greeks_by_expiry_from_df,
    smile_data,
    spot_ladder,
)
from src.validation.metrics import (
    DEFAULT_MODELS,
    add_containment_flags,
    add_price_errors,
    fit_by_moneyness,
    fit_summary,
    moneyness_bucket,
    vol_risk_premium,
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Options Pricing Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = ROOT / "data" / "processed"

# ── cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading options data...")
def load_options() -> pd.DataFrame | None:
    path = DATA_DIR / "options_with_prices.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["expiration"]    = pd.to_datetime(df["expiration"])
    df["valuation_date"] = pd.to_datetime(df["valuation_date"])
    for col in ["strike", "bid", "ask", "mid", "ttm", "spot",
                "log_moneyness", "iv_model", "iv_engine",
                "price_bsm", "price_crr", "price_lsm",
                "delta_bsm", "delta_crr", "eep_crr", "eep_lsm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["moneyness_bucket"] = df["log_moneyness"].apply(moneyness_bucket)
    df = add_price_errors(df)
    df = add_containment_flags(df)
    return df


@st.cache_data(ttl=3600, show_spinner="Loading vol surface...")
def load_surface_points() -> pd.DataFrame | None:
    path = DATA_DIR / "vol_surface_points.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource(show_spinner="Loading surface interpolators...")
def get_interpolators():
    try:
        return load_surface(str(DATA_DIR))
    except FileNotFoundError:
        return None, None


@st.cache_data(ttl=3600, show_spinner="Loading SVI parameters...")
def load_svi_params() -> pd.DataFrame | None:
    path = DATA_DIR / "svi_params.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    try:
        import tomllib
        cfg_path = ROOT / "config.toml"
        if cfg_path.exists():
            with open(cfg_path, "rb") as f:
                cfg = tomllib.load(f)
            st.info(
                f"**Ticker:** {cfg.get('ticker', '?')}  \n"
                f"**Risk-free rate:** {cfg.get('risk_free_rate', '?'):.1%}  \n"
                f"**Dividend yield:** {cfg.get('dividend_yield', '?'):.1%}"
            )
        else:
            cfg = {}
    except Exception:
        cfg = {}

    # ── live market snapshot ──────────────────────────────────────────
    _sb_opts = load_options()
    if _sb_opts is not None:
        _sb_spot  = float(_sb_opts["spot"].iloc[0])
        _sb_vdate = pd.to_datetime(
            _sb_opts["valuation_date"].iloc[0]
        ).strftime("%Y-%m-%d")
        _sb_n    = len(_sb_opts)
        _sb_nexp = _sb_opts["expiration"].nunique()
        _lines = [
            f"**Valuation date:** {_sb_vdate}",
            f"**Spot:** ${_sb_spot:.2f}",
            f"**Options:** {_sb_n} across {_sb_nexp} expiries",
        ]
        st.info("  \n".join(_lines))

    st.markdown("---")
    st.markdown(
        "**Refresh data:**  \n"
        "```bash\npython refresh.py --ticker GOOG\n```"
    )
    st.markdown("---")
    st.caption("Options Pricing & Hedging  \nMrinal Gupta  \nProduction-style quant project")

# ── tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌋 Vol Surface",
    "💰 Option Pricer",
    "📊 Model Validation",
    "🚧 Early Exercise",
    "📉 Hedge Simulator",
    "📐 Risk & Sensitivity",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — VOL SURFACE
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Implied Volatility Surface")

    surface_pts = load_surface_points()
    svi_params  = load_svi_params()

    if surface_pts is None:
        st.warning(
            "No surface data found. "
            "Run `python refresh.py --ticker GOOG` to generate it."
        )
    else:
        ticker_label = cfg.get("ticker", "")

        col1, col2 = st.columns([3, 1])

        with col1:
            # ── 3D interactive surface ────────────────────────────────────────
            # Clip grid to the convex hull of actual market data per expiry.
            # Without clipping, the SVI extrapolates wildly beyond the range
            # of strikes that were actually quoted (especially for short-dated
            # weekly options where the strike range is narrow). This caused
            # the red spike artefacts visible in the raw plot.
            lm_col = "log_moneyness"
            ttm_col = "ttm"

            # Build per-expiry moneyness bounds from the raw surface points
            expiry_bounds = (
                surface_pts.groupby(ttm_col)[lm_col]
                .agg(["min", "max"])
                .reset_index()
            )

            # Grid: use the shared moneyness range across ALL expiries
            # (intersection keeps the surface within data boundaries)
            lm_min_global = expiry_bounds["min"].max()   # tightest left bound
            lm_max_global = expiry_bounds["max"].min()   # tightest right bound

            # Fall back to full range if intersection is too narrow
            if lm_max_global - lm_min_global < 0.05:
                lm_min_global = surface_pts[lm_col].min()
                lm_max_global = surface_pts[lm_col].max()

            x_vals = np.linspace(lm_min_global, lm_max_global, 80)
            y_vals = np.linspace(surface_pts[ttm_col].min(),
                                 surface_pts[ttm_col].max(), 80)
            X, Y   = np.meshgrid(x_vals, y_vals)

            pts = surface_pts[[lm_col, ttm_col]].values
            ws  = surface_pts["w_svi"].values

            W         = griddata(pts, ws, (X, Y), method="linear")
            W_nearest = griddata(pts, ws, (X, Y), method="nearest")
            W         = np.where(np.isnan(W), W_nearest, W)
            W         = np.clip(W, 1e-8, None)
            Z         = np.sqrt(W / Y)

            # Clip extreme IV values that come from thin data at edges
            Z = np.clip(Z, 0.0, 1.5)

            fig = go.Figure()
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                colorscale="Viridis", opacity=0.88,
                showscale=True,
                colorbar=dict(title="IV", tickformat=".0%"),
                name="IV Surface",
                cmin=0.15, cmax=0.65,
            ))

            # Only plot SVI fitted points within the clipped moneyness range
            pts_in_range = surface_pts[
                (surface_pts[lm_col] >= lm_min_global - 0.01) &
                (surface_pts[lm_col] <= lm_max_global + 0.01)
            ]
            iv_z_col = "iv_smooth" if "iv_smooth" in pts_in_range.columns else pts_in_range.columns[-1]
            fig.add_trace(go.Scatter3d(
                x=pts_in_range[lm_col],
                y=pts_in_range[ttm_col],
                z=pts_in_range[iv_z_col],
                mode="markers",
                marker=dict(size=2, color="#FF5252"),
                name="SVI Fitted Points",
            ))
            fig.update_layout(
                title=f"{ticker_label} Implied Volatility Surface (SVI)",
                height=600,
                scene=dict(
                    xaxis_title="Log-Moneyness ln(K/S)",
                    yaxis_title="TTM (years)",
                    zaxis_title="Implied Volatility",
                    yaxis=dict(
                        tickvals=[7/365, 30/365, 90/365, 180/365, 365/365],
                        ticktext=["7d", "30d", "90d", "180d", "1y"],
                    ),
                    camera=dict(
                        eye=dict(x=1.5, y=-1.7, z=1.0),
                        up=dict(x=0, y=0, z=1),
                    ),
                ),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            # ── SVI parameters table ──────────────────────────────────────────
            if svi_params is not None:
                st.subheader("SVI Parameters")
                display = svi_params.copy()
                if "expiration" in display.columns:
                    display["expiry"] = pd.to_datetime(
                        display["expiration"]
                    ).dt.strftime("%Y-%m-%d")
                    display = display.drop(columns=["expiration"])

                show_cols = [c for c in ["expiry", "ttm", "a", "b",
                                          "rho", "m", "sigma", "fit_rmse"]
                             if c in display.columns]
                st.dataframe(
                    display[show_cols].round(4),
                    width="stretch",
                    height=300,
                )
                st.caption(
                    "**ρ** (rho): smile skew — negative = left skew (equity)  \n"
                    "**b**: curvature / slope  \n"
                    "**a**: ATM total variance level"
                )

        # ── term structure ────────────────────────────────────────────────────
        if svi_params is not None and "ttm" in svi_params.columns:
            st.subheader("ATM Term Structure")

            w_linear, w_nearest = get_interpolators()
            if w_linear is not None:
                ttms   = sorted(svi_params["ttm"].values)
                atm_ivs = []
                for t in ttms:
                    iv = get_engine_iv(0.0, t, w_linear, w_nearest)
                    atm_ivs.append(iv)

                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(
                    x=[t * 365 for t in ttms],
                    y=[v * 100 for v in atm_ivs],
                    mode="lines+markers",
                    name="ATM IV",
                    line=dict(width=2.5),
                    marker=dict(size=8),
                ))
                fig_ts.update_layout(
                    title=f"{ticker_label} ATM Implied Vol Term Structure",
                    xaxis_title="Days to Expiry",
                    yaxis_title="Implied Volatility (%)",
                    height=350,
                    hovermode="x unified",
                )
                st.plotly_chart(fig_ts, width="stretch")

        # ── IV Smile: market vs SVI per expiry ───────────────────────────────
        st.subheader("IV Smile — Market vs SVI Fit")
        st.markdown(
            "2D cross-section of the surface. Each expiry slice shows market IV "
            "dots (circles = calls, squares = puts) against the SVI fitted curve. "
            "Dots close to lines = good calibration."
        )

        options_for_smile = load_options()
        if options_for_smile is not None and "iv_model" in options_for_smile.columns:
            sdf = smile_data(options_for_smile)
            if not sdf.empty:
                fig_smile1 = go.Figure()
                smile_colors = ["#2196F3","#4CAF50","#FF9800","#E91E63","#9C27B0"]
                for i, exp in enumerate(sorted(sdf["expiration"].unique())):
                    grp   = sdf[sdf["expiration"] == exp].sort_values("log_moneyness")
                    label = pd.Timestamp(exp).strftime("%Y-%m-%d")
                    days  = int(grp["days_to_expiry"].iloc[0])
                    col   = smile_colors[i % len(smile_colors)]

                    for otype, sym in [("call", "circle"), ("put", "square")]:
                        sub = grp[grp["option_type"] == otype]
                        if sub.empty:
                            continue
                        fig_smile1.add_trace(go.Scatter(
                            x=sub["log_moneyness"] * 100,
                            y=sub["iv_model"] * 100,
                            mode="markers",
                            marker=dict(symbol=sym, size=7,
                                        color=col, opacity=0.75),
                            name=f"{label} ({days}d)",
                            showlegend=(otype == "call"),
                            legendgroup=label,
                        ))

                    # SVI fitted curve
                    fig_smile1.add_trace(go.Scatter(
                        x=grp["log_moneyness"] * 100,
                        y=grp["iv_engine"] * 100,
                        mode="lines",
                        line=dict(color=col, width=2),
                        name=f"{label} SVI",
                        showlegend=False,
                        legendgroup=label,
                    ))

                fig_smile1.add_vline(x=0, line_dash="dot", line_color="gray",
                                     annotation_text="ATM")
                fig_smile1.update_layout(
                    xaxis_title="Log-Moneyness ln(K/S) × 100",
                    yaxis_title="Implied Volatility (%)",
                    title=f"{ticker_label} IV Smile: Market (dots) vs SVI (lines)",
                    height=430,
                    hovermode="x unified",
                    legend=dict(orientation="v", x=1.02, y=1),
                )
                st.plotly_chart(fig_smile1, width="stretch")
                st.caption(
                    "Circles = calls · Squares = puts · "
                    "Left skew (higher IV for lower strikes) is normal for equity."
                )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — OPTION PRICER
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Option Pricer")
    st.caption(
        "Price any option live using three models — European BSM, American CRR binomial tree, "
        "and LSM Monte Carlo. Inputs are pulled from today's vol surface automatically. "
        "Greeks update in real time."
    )

    w_linear, w_nearest = get_interpolators()
    options_df = load_options()

    # Derive sensible defaults from loaded data
    if options_df is not None:
        default_spot = float(options_df["spot"].iloc[0])
        default_r    = cfg.get("risk_free_rate", 0.045)
        default_q    = cfg.get("dividend_yield", 0.0)
    else:
        default_spot = 100.0
        default_r    = 0.045
        default_q    = 0.0

    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.subheader("Inputs")
        spot      = st.number_input("Spot price (S)", value=default_spot,
                                    min_value=0.01, step=1.0, format="%.2f")
        strike    = st.number_input("Strike (K)", value=float(round(default_spot)),
                                    min_value=0.01, step=1.0, format="%.2f")
        ttm_days  = st.slider("Days to expiry", 1, 730, 90)
        r         = st.number_input("Risk-free rate", value=default_r,
                                    step=0.001, format="%.4f")
        q         = st.number_input("Dividend yield", value=default_q,
                                    step=0.001, format="%.4f")
        otype     = st.selectbox("Option type", ["put", "call"])
        model_sel = st.multiselect(
            "Pricing models",
            ["BSM (European)", "CRR (American)", "LSM (American)"],
            default=["BSM (European)", "CRR (American)"],
        )

        # IV: use surface if available, otherwise let user input
        T = ttm_days / 365.0
        log_m = np.log(strike / spot) if spot > 0 else 0.0
        surface_iv = None

        if w_linear is not None:
            try:
                surface_iv = get_engine_iv(log_m, T, w_linear, w_nearest)
            except Exception:
                surface_iv = None

        if surface_iv and not np.isnan(surface_iv):
            sigma = st.number_input(
                "Volatility σ (surface-queried)",
                value=round(surface_iv, 4),
                step=0.005, format="%.4f",
            )
        else:
            sigma = st.number_input(
                "Volatility σ (manual input — run refresh to load surface)",
                value=0.25, step=0.005, format="%.4f",
            )

        st.button("Price option", type="primary")  # triggers Streamlit rerun

    with col_out:
        st.subheader("Results")

        if True:  # always compute — inputs update via Streamlit reruns
            results = {}

            if "BSM (European)" in model_sel:
                p = bsm_price(spot, strike, T, r, sigma, otype, q)
                results["BSM (European)"] = p

            if "CRR (American)" in model_sel:
                p = crr_price(spot, strike, T, r, sigma, otype, q,
                              N=200, american=True)
                results["CRR (American)"] = p

            if "LSM (American)" in model_sel:
                p = lsm_price(spot, strike, T, r, sigma, otype, q,
                              M=5_000, n=min(int(ttm_days), 252), seed=42)
                results["LSM (American)"] = p

            # ── price cards ───────────────────────────────────────────────────
            price_cols = st.columns(max(len(results), 1))
            colors     = {"BSM (European)": "#2196F3",
                          "CRR (American)": "#4CAF50",
                          "LSM (American)": "#FF9800"}

            for i, (name, price) in enumerate(results.items()):
                with price_cols[i]:
                    if not np.isnan(price):
                        st.metric(label=name, value=f"${price:.4f}")
                    else:
                        st.metric(label=name, value="N/A")

            # ── Greeks ────────────────────────────────────────────────────────
            st.subheader("BSM Greeks")
            g = bsm_greeks(spot, strike, T, r, sigma, otype, q)

            gcol1, gcol2, gcol3, gcol4, gcol5 = st.columns(5)
            gcol1.metric("Delta",  f"{g['delta']:.4f}" if not np.isnan(g['delta'])  else "—")
            gcol2.metric("Gamma",  f"{g['gamma']:.6f}" if not np.isnan(g['gamma'])  else "—")
            gcol3.metric("Vega",   f"{g['vega']:.4f}"  if not np.isnan(g['vega'])   else "—")
            gcol4.metric("Theta",  f"{g['theta']:.4f}" if not np.isnan(g['theta'])  else "—")
            gcol5.metric("Rho",    f"{g['rho']:.4f}"   if not np.isnan(g['rho'])    else "—")
            st.caption("Theta = per calendar day  |  Vega = per unit σ")

            # ── Greek profile chart ───────────────────────────────────────────
            st.subheader("Delta & Gamma Profiles")
            strike_grid = np.linspace(spot * 0.80, spot * 1.20, 80)
            deltas      = [bsm_greeks(spot, k, T, r, sigma, otype, q)["delta"]
                           for k in strike_grid]
            gammas      = [bsm_greeks(spot, k, T, r, sigma, otype, q)["gamma"]
                           for k in strike_grid]

            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(
                x=strike_grid, y=deltas,
                name="Delta", line=dict(color="#2196F3", width=2),
                yaxis="y1",
            ))
            fig_g.add_trace(go.Scatter(
                x=strike_grid, y=gammas,
                name="Gamma", line=dict(color="#FF5722", width=2, dash="dash"),
                yaxis="y2",
            ))
            fig_g.add_vline(x=float(strike), line_dash="dot",
                            line_color="gray", annotation_text="K")
            fig_g.add_vline(x=float(spot), line_dash="dot",
                            line_color="green", annotation_text="S")
            fig_g.update_layout(
                xaxis_title="Strike (K)",
                yaxis=dict(title="Delta", side="left"),
                yaxis2=dict(title="Gamma", side="right", overlaying="y"),
                legend=dict(x=0.01, y=0.99),
                height=350,
                hovermode="x unified",
            )
            st.plotly_chart(fig_g, width="stretch")

            # ── Theta Surface ─────────────────────────────────────────────────
            st.subheader("Theta Surface — Daily Decay across Strike & Time")
            st.markdown(
                "How much does each option lose per day? "
                "Calls on the left, puts on the right. "
                "**Red = fastest decay · Green = slowest.** "
                "The spike at ATM near-expiry is where short options bleed the most."
            )

            # ── build theta grid: 50x50 for smooth surface ────────────────────
            _lm_vals  = np.linspace(-0.15, 0.15, 50)
            _day_vals = np.linspace(1, ttm_days, 50)
            _K_vals   = spot * np.exp(_lm_vals)
            _lm_pct   = _lm_vals * 100

            # Plot |theta| so surface peaks UP at ATM near-expiry
            from src.models.black_scholes import bsm_theta as _bsm_theta
            _abs_call = np.zeros((len(_day_vals), len(_lm_vals)))
            _abs_put  = np.zeros((len(_day_vals), len(_lm_vals)))

            for _i, _d in enumerate(_day_vals):
                _t = max(_d / 365.0, 1/365)
                for _j, _K in enumerate(_K_vals):
                    _tc = _bsm_theta(spot, _K, _t, r, sigma, "call", q)
                    _tp = _bsm_theta(spot, _K, _t, r, sigma, "put",  q)
                    _abs_call[_i, _j] = abs(_tc) if not np.isnan(_tc) else 0.0
                    _abs_put[_i, _j]  = abs(_tp) if not np.isnan(_tp) else 0.0

            # Green → Yellow → Orange → Dark Red
            _theta_cs = [
                [0.00, "#1a7a4a"],
                [0.25, "#74c476"],
                [0.50, "#fed976"],
                [0.75, "#fd8d3c"],
                [1.00, "#a50f15"],
            ]

            def _theta_fig(z_data, option_label):
                _f = go.Figure()
                _f.add_trace(go.Surface(
                    x=_lm_pct,
                    y=_day_vals,
                    z=z_data,
                    colorscale=_theta_cs,
                    showscale=True,
                    opacity=0.96,
                    colorbar=dict(
                        title=dict(text="|Theta| $/day", side="right"),
                        len=0.55,
                        thickness=18,
                        tickfont=dict(size=11),
                    ),
                    contours=dict(
                        z=dict(show=True, usecolormap=True,
                               highlightcolor="white", project_z=True),
                    ),
                    hovertemplate=(
                        "Log-moneyness: %{x:.1f}%<br>"
                        "Days to expiry: %{y:.0f}<br>"
                        "|Theta|: $%{z:.3f}/day<extra></extra>"
                    ),
                ))
                _f.update_layout(
                    title=dict(
                        text=f"{option_label} — Theta Surface",
                        x=0.5,
                        font=dict(size=14),
                    ),
                    scene=dict(
                        xaxis=dict(
                            title=dict(text="Log-Moneyness ln(K/S) × 100",
                                       font=dict(size=11)),
                            tickfont=dict(size=10),
                        ),
                        yaxis=dict(
                            title=dict(text="Days to Expiry",
                                       font=dict(size=11)),
                            tickfont=dict(size=10),
                        ),
                        zaxis=dict(
                            title=dict(text="|Theta| $/day",
                                       font=dict(size=11)),
                            tickfont=dict(size=10),
                        ),
                        # Camera: elevated 3/4 view — spike at near-expiry ATM is the focal point
                        camera=dict(
                            eye=dict(x=-1.4, y=-1.5, z=1.4),
                            up=dict(x=0, y=0, z=1),
                        ),
                        bgcolor="rgba(0,0,0,0)",
                        aspectratio=dict(x=1.2, y=1.2, z=0.7),
                    ),
                    height=560,
                    margin=dict(l=0, r=10, t=60, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                return _f

            # Full-width side by side — comparison IS the story
            # Calls and puts nearly identical for no-dividend stock
            _tc1, _tc2 = st.columns(2)
            with _tc1:
                st.plotly_chart(
                    _theta_fig(_abs_call, "Calls"),
                    width="stretch",
                )
            with _tc2:
                st.plotly_chart(
                    _theta_fig(_abs_put, "Puts"),
                    width="stretch",
                )

            st.caption(
                f"S=\\${spot:.2f}, σ={sigma:.0%}, r={r:.1%}, up to {ttm_days} days. "
                "Calls and puts are nearly identical (no dividend) — "
                "the small difference comes from interest rate effects. "
                "Drag to rotate · Scroll to zoom · Hover for exact values."
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Model Validation")
    st.caption(
        "How accurately do the three pricing models reproduce actual market prices? "
        "Measured by bid-ask hit rate (does the model price land inside the quoted spread?), "
        "MAE, and MAPE — broken down by model, moneyness, and expiry."
    )

    options_df = load_options()

    if options_df is None:
        st.warning(
            "No priced data found. "
            "Run `python refresh.py --ticker GOOG` to generate it."
        )
    else:
        # ── headline result metrics ───────────────────────────────────────────
        if "in_spread_crr" in options_df.columns and "in_spread_bsm" in options_df.columns:
            _h1, _h2, _h3, _h4, _h5, _h6 = st.columns(6)
            _crr_hit = options_df["in_spread_crr"].mean() * 100
            _bsm_hit = options_df["in_spread_bsm"].mean() * 100
            _lsm_hit = options_df["in_spread_lsm"].mean() * 100 if "in_spread_lsm" in options_df.columns else None
            _crr_mae = options_df["err_crr"].abs().mean() if "err_crr" in options_df.columns else None
            _bsm_mae = options_df["err_bsm"].abs().mean() if "err_bsm" in options_df.columns else None
            _lsm_mae = options_df["err_lsm"].abs().mean() if "err_lsm" in options_df.columns else None
            _crr_mape = options_df["ape_crr"].mean() if "ape_crr" in options_df.columns else None
            _n = len(options_df)
            _h1.metric(
                "CRR Bid-Ask Hit Rate",
                f"{_crr_hit:.1f}%",
                help="% of CRR prices landing inside the bid-ask spread — the key accuracy test"
            )
            _h2.metric(
                "BSM Bid-Ask Hit Rate",
                f"{_bsm_hit:.1f}%",
                help="BSM underprices puts (European model on American options) — expected"
            )
            _h3.metric(
                "LSM Bid-Ask Hit Rate",
                f"{_lsm_hit:.1f}%" if _lsm_hit is not None else "—",
                help="LSM Monte Carlo — simulation noise widens errors vs CRR"
            )
            _h4.metric(
                "CRR MAE",
                f"${_crr_mae:.3f}" if _crr_mae else "—",
                help="Mean absolute dollar error vs market mid price"
            )
            _h5.metric(
                "LSM MAE",
                f"${_lsm_mae:.3f}" if _lsm_mae else "—",
                help="LSM mean absolute error — higher than CRR due to simulation noise"
            )
            _h6.metric(
                "Options tested",
                str(_n),
                help="Total number of options in the validation dataset"
            )
            _lsm_msg = ""
            if _lsm_hit is not None and _lsm_mae is not None:
                _lsm_msg = f" LSM (Monte Carlo) hits {_lsm_hit:.1f}% (MAE \\${_lsm_mae:.3f})."
            st.success(
                f"**CRR (American) prices {_crr_hit:.1f}% of options inside the bid-ask spread** — "
                f"production quality for a vanilla options pricing engine. "
                f"Mean absolute error: \\${_crr_mae:.3f} per option."
                f"{_lsm_msg}"
            )
            st.markdown("---")

        # ── overall fit summary ───────────────────────────────────────────────
        st.subheader("Overall Fit Summary")
        summary = fit_summary(options_df)
        st.dataframe(summary.style.highlight_min(
            subset=["mae", "mape"], color="#c6efce"
        ).highlight_max(
            subset=["bid_ask_hit_%"], color="#c6efce"
        ), width="stretch")

        col_left, col_right = st.columns(2)

        with col_left:
            # ── MAPE bar chart ────────────────────────────────────────────────
            st.subheader("MAPE by Model")
            fig_mape = go.Figure(go.Bar(
                x=summary.index.tolist(),
                y=summary["mape"].tolist(),
                marker_color=["#2196F3", "#4CAF50", "#FF9800"],
                text=[f"{v:.1f}%" for v in summary["mape"]],
                textposition="outside",
            ))
            fig_mape.update_layout(
                yaxis_title="MAPE (%)", height=300,
                showlegend=False,
            )
            st.plotly_chart(fig_mape, width="stretch")

        with col_right:
            # ── hit rate bar chart ────────────────────────────────────────────
            st.subheader("Bid-Ask Hit Rate by Model")
            fig_hit = go.Figure(go.Bar(
                x=summary.index.tolist(),
                y=summary["bid_ask_hit_%"].tolist(),
                marker_color=["#2196F3", "#4CAF50", "#FF9800"],
                text=[f"{v:.1f}%" for v in summary["bid_ask_hit_%"]],
                textposition="outside",
            ))
            fig_hit.update_layout(
                yaxis_title="Hit Rate (%)", yaxis_range=[0, 105],
                height=300, showlegend=False,
            )
            st.plotly_chart(fig_hit, width="stretch")

        # ── fit by moneyness heatmap ──────────────────────────────────────────
        st.subheader("Bid-Ask Hit Rate (%) by Moneyness Bucket")
        moneyness_df = fit_by_moneyness(options_df)
        hit_pivot    = moneyness_df["hit_%"].unstack(level="model")
        st.dataframe(
            hit_pivot.style.background_gradient(cmap="RdYlGn",
                                                 axis=None,
                                                 vmin=0, vmax=100),
            width="stretch",
        )

        # ── price scatter ─────────────────────────────────────────────────────
        st.subheader("Model Price vs Market Mid")
        model_choice = st.selectbox(
            "Select model", list(DEFAULT_MODELS.values()),
            key="scatter_model",
        )
        price_col = {v: k for k, v in DEFAULT_MODELS.items()}[model_choice]

        if price_col in options_df.columns:
            sub = options_df.dropna(subset=[price_col, "mid"])
            lo  = min(sub["mid"].min(), sub[price_col].min())
            hi  = max(sub["mid"].max(), sub[price_col].max())

            scatter_fig = go.Figure()
            for otype, color in [("call", "#1565C0"), ("put", "#B71C1C")]:
                s = sub[sub["option_type"] == otype]
                scatter_fig.add_trace(go.Scatter(
                    x=s["mid"], y=s[price_col],
                    mode="markers",
                    name=otype.capitalize(),
                    marker=dict(color=color, size=6, opacity=0.65),
                ))
            scatter_fig.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi],
                mode="lines",
                name="45° line",
                line=dict(color="gray", dash="dash"),
            ))
            scatter_fig.update_layout(
                xaxis_title="Market Mid ($)",
                yaxis_title=f"{model_choice} Price ($)",
                height=420,
                hovermode="closest",
            )
            st.plotly_chart(scatter_fig, width="stretch")


        # ── volatility risk premium ───────────────────────────────────────────
        st.markdown("---")
        st.subheader("Volatility Risk Premium (ATM IV vs Realised Vol)")
        st.markdown(
            "Compares near-ATM implied vol against the historical realised vol "
            "over the matching horizon.  **Positive VRP = options are expensive** — "
            "the market charges more than what actually happened. "
            "In equity markets this is almost always positive."
        )

        spot_hist_path = DATA_DIR.parent / "raw" / "spot_history.csv"
        if spot_hist_path.exists() and "iv_model" in options_df.columns:
            spot_h = pd.read_csv(spot_hist_path)
            vrp_df = vol_risk_premium(options_df, spot_h)
            if not vrp_df.empty:
                vrp_display = vrp_df.reset_index()
                vrp_display["expiration"] = pd.to_datetime(
                    vrp_display["expiration"]
                ).dt.strftime("%Y-%m-%d")

                fig_vrp = go.Figure()
                fig_vrp.add_trace(go.Bar(
                    name="ATM Implied Vol",
                    x=vrp_display["expiration"],
                    y=vrp_display["atm_iv"] * 100,
                    marker_color="#2196F3",
                ))
                fig_vrp.add_trace(go.Bar(
                    name="Matched Realised Vol",
                    x=vrp_display["expiration"],
                    y=vrp_display["rv_matched"] * 100,
                    marker_color="#FF9800",
                ))
                fig_vrp.update_layout(
                    barmode="group",
                    xaxis_title="Expiry",
                    yaxis_title="Volatility (%)",
                    title="ATM Implied Vol vs Realised Vol by Expiry",
                    height=380,
                )
                st.plotly_chart(fig_vrp, width="stretch")

                # VRP table
                vrp_show = vrp_display[[
                    "expiration", "days_to_expiry",
                    "atm_iv", "rv_window", "rv_matched",
                    "vrp", "vrp_pct"
                ]].copy()
                vrp_show["atm_iv"]     = (vrp_show["atm_iv"]     * 100).round(2)
                vrp_show["rv_matched"] = (vrp_show["rv_matched"] * 100).round(2)
                vrp_show["vrp"]        = (vrp_show["vrp"]        * 100).round(2)
                vrp_show.columns = [
                    "Expiry", "Days", "ATM IV %",
                    "RV Window", "Realised Vol %", "VRP (pp)", "VRP %"
                ]
                st.dataframe(vrp_show, width="stretch")
                st.caption(
                    "VRP (pp) = ATM IV − Realised Vol in percentage points.  "
                    "Positive means the option market overestimated future volatility.  "
                    "VRP % = VRP as a fraction of ATM IV — how much of implied vol was 'risk premium'."
                )
        else:
            st.info("Run `python refresh.py` first to generate spot history and IV data.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — EARLY EXERCISE
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Early Exercise Boundary")
    st.caption(
        "American puts can be exercised before expiry. The early exercise boundary S*(t) "
        "is the critical spot price below which exercising immediately beats holding. "
        "Computed via CRR binomial tree across all expiries."
    )
    st.markdown(
        "The **early exercise boundary** S\\*(t) is the highest spot price at which "
        "it is optimal to exercise an American put immediately.  "
        "Below S\\*(t), you should exercise; above it, you should hold."
    )

    options_df = load_options()

    if options_df is None:
        st.warning(
            "No priced data found. "
            "Run `python refresh.py --ticker GOOG` first."
        )
    else:
        puts = options_df[options_df["option_type"] == "put"].copy()
        r_ee = cfg.get("risk_free_rate", 0.045)
        q_ee = cfg.get("dividend_yield", 0.0)

        col_ee1, col_ee2 = st.columns([2, 1])

        with col_ee2:
            st.subheader("Controls")
            n_steps = st.slider("CRR steps", 50, 300, 150, step=25)
            sigma_override = st.number_input(
                "Override σ (0 = use iv_engine)",
                value=0.0, min_value=0.0, max_value=1.5, step=0.01,
            )
            show_normalised = st.checkbox("Normalise S*/K", value=True)

        with col_ee1:
            expiries = sorted(puts["expiration"].unique())
            fig_ee = go.Figure()

            for expiry in expiries:
                grp = puts[puts["expiration"] == expiry]
                row = grp.iloc[(grp["log_moneyness"].abs()).argsort()].iloc[0]
                S_  = row["spot"]
                K_  = row["strike"]
                T_  = row["ttm"]
                sig = sigma_override if sigma_override > 0 else row["iv_engine"]

                times, boundary = crr_early_exercise_boundary(
                    S_, K_, T_, r_ee, sig, q=q_ee, N=n_steps
                )
                if times is None:
                    continue

                label = f"{pd.Timestamp(expiry).date()} (K={K_:.0f})"

                if show_normalised:
                    tau   = 1.0 - times / T_
                    y_val = boundary / K_
                    fig_ee.add_trace(go.Scatter(
                        x=tau, y=y_val, mode="lines",
                        name=label, line=dict(width=2),
                    ))
                else:
                    fig_ee.add_trace(go.Scatter(
                        x=times, y=boundary, mode="lines",
                        name=label, line=dict(width=2),
                    ))

            if show_normalised:
                fig_ee.add_hline(y=1.0, line_dash="dash", line_color="gray",
                                 annotation_text="Strike K")
                fig_ee.update_layout(
                    xaxis_title="Time Remaining τ (1=now, 0=expiry)",
                    yaxis_title="S*(t) / K",
                    xaxis_autorange="reversed",
                    height=450,
                    hovermode="x unified",
                )
            else:
                fig_ee.update_layout(
                    xaxis_title="Time t (years from today)",
                    yaxis_title="Early Exercise Boundary S*(t) ($)",
                    height=450,
                    hovermode="x unified",
                )

            fig_ee.update_layout(
                title="CRR Early Exercise Boundary — Near-ATM Puts by Expiry",
                legend=dict(x=0.01, y=0.01),
            )
            st.plotly_chart(fig_ee, width="stretch")

        # ── vol sensitivity ───────────────────────────────────────────────────
        st.subheader("Boundary Sensitivity to Volatility")
        st.markdown(
            "Higher volatility → lower early exercise boundary.  "
            "As vol increases, the option's time value grows, so you prefer "
            "to hold rather than exercise."
        )

        medium_puts = puts[
            (puts["days_to_expiry"] > 60) & (puts["days_to_expiry"] <= 150)
        ]
        if len(medium_puts) > 0:
            rep = medium_puts.iloc[(medium_puts["log_moneyness"].abs()).argsort()].iloc[0]
            sigma_grid  = np.linspace(0.10, 0.60, 25)
            s_star_vals = []
            for sig in sigma_grid:
                times_s, bnd = crr_early_exercise_boundary(
                    rep["spot"], rep["strike"], rep["ttm"],
                    r_ee, sig, q=q_ee, N=100,
                )
                if times_s is not None:
                    valid = ~np.isnan(bnd)
                    s_star_vals.append(
                        bnd[valid][0] / rep["strike"] if valid.any() else np.nan
                    )
                else:
                    s_star_vals.append(np.nan)

            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(
                x=sigma_grid * 100, y=s_star_vals,
                mode="lines+markers", name="S*(t=0)/K",
                line=dict(color="#4CAF50", width=2.5),
            ))
            fig_sens.add_hline(y=1.0, line_dash="dash", line_color="gray",
                               annotation_text="Strike K")
            fig_sens.add_vline(
                x=float(rep["iv_engine"]) * 100 if not np.isnan(rep["iv_engine"]) else 25,
                line_dash="dot", line_color="red",
                annotation_text=f"Market IV",
            )
            fig_sens.update_layout(
                xaxis_title="Volatility σ (%)",
                yaxis_title="S*(t=0) / K",
                title=f"Boundary Sensitivity (K={rep['strike']:.0f}, T={rep['ttm']:.2f}y)",
                height=350,
            )
            st.plotly_chart(fig_sens, width="stretch")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — HEDGE SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Delta Hedge Simulator")
    st.caption(
        "Sell an option, hedge the delta exposure daily by trading the stock, "
        "and watch how P&L evolves. Five sections: single path, hedging frequency, "
        "vol mismatch risk, three-strategy effectiveness benchmark, "
        "and a stress matrix combining vol mismatch with transaction costs."
    )

    options_df = load_options()
    default_spot  = float(options_df["spot"].iloc[0]) if options_df is not None else 100.0
    default_r     = cfg.get("risk_free_rate", 0.045)
    default_q     = cfg.get("dividend_yield", 0.0)
    w_lin_h, w_near_h = get_interpolators()

    # ── inputs ────────────────────────────────────────────────────────────────
    st.subheader("Position Setup")
    hcol1, hcol2, hcol3 = st.columns(3)

    with hcol1:
        h_spot   = st.number_input("Spot (S)", value=default_spot, step=1.0,
                                   format="%.2f", key="h_spot")
        h_strike = st.number_input("Strike (K)", value=float(round(default_spot)),
                                   step=1.0, format="%.2f", key="h_strike")
        h_otype  = st.selectbox("Option type", ["put", "call"], key="h_otype")

    with hcol2:
        h_ttm_days = st.slider("Days to expiry", 10, 365, 90, key="h_ttm")
        h_r        = st.number_input("Risk-free rate", value=default_r,
                                     step=0.001, format="%.4f", key="h_r")

    with hcol3:
        # query surface IV if available
        h_T    = h_ttm_days / 365.0
        h_logm = np.log(h_strike / h_spot) if h_spot > 0 else 0.0
        surf_iv = None
        if w_lin_h is not None:
            try:
                surf_iv = get_engine_iv(h_logm, h_T, w_lin_h, w_near_h)
            except Exception:
                surf_iv = None

        iv_default = round(surf_iv, 4) if surf_iv and not np.isnan(surf_iv) else 0.25
        h_sigma_imp = st.number_input(
            "Implied vol σ (hedge vol)",
            value=iv_default, step=0.005, format="%.4f", key="h_sigma"
        )
        h_sigma_real = st.number_input(
            "Realised vol (path sim)",
            value=iv_default, step=0.005, format="%.4f", key="h_sigma_r",
            help="Set different from implied vol to simulate vol P&L"
        )
        h_tc = st.number_input(
            "Transaction cost (% per trade)",
            value=0.0, min_value=0.0, max_value=1.0,
            step=0.001, format="%.3f", key="h_tc"
        )

    h_steps = st.slider("Hedging steps", 10, 252, 252,
                        help="252 = daily, 52 = weekly, 12 = monthly", key="h_steps")
    h_seed  = st.number_input("Random seed", value=42, step=1, key="h_seed",
                          format="%d")

    # ── single simulation ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Single Path Simulation")

    sim_df = simulate_delta_hedge(
        S0=h_spot, K=h_strike, T=h_T, r=h_r,
        sigma_implied=h_sigma_imp,
        sigma_realised=h_sigma_real,
        option_type=h_otype,
        q=default_q,
        n_steps=h_steps,
        transaction_cost_pct=h_tc / 100,
        seed=int(h_seed),
    )
    summ = hedge_summary(sim_df, h_spot, h_strike, h_T, h_r, h_sigma_imp, h_otype)

    # metric cards
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Option premium sold", f"${summ['initial_option_value']:.4f}")
    mc2.metric("Final P&L",
               f"${summ['final_pnl']:.4f}",
               f"{summ['pnl_as_pct_premium']:.1f}% of premium")
    mc3.metric("Total transaction costs", f"${summ['total_transaction_cost']:.4f}")
    mc4.metric("P&L std (daily)", f"${summ['pnl_std']:.4f}")

    # P&L chart
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=sim_df["t"], y=sim_df["pnl_cumulative"],
        mode="lines", name="Cumulative P&L",
        line=dict(color="#2196F3", width=2),
        fill="tozeroy",
        fillcolor="rgba(33,150,243,0.10)",
    ))
    fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_pnl.update_layout(
        xaxis_title="Time (years)", yaxis_title="Cumulative P&L ($)",
        title=f"Delta Hedge P&L — Short {h_otype.capitalize()} (K={h_strike}, σ_imp={h_sigma_imp:.0%})",
        height=350, hovermode="x unified",
    )
    st.plotly_chart(fig_pnl, width="stretch")

    # Spot path + delta
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=sim_df["t"], y=sim_df["S"],
            mode="lines", name="Spot",
            line=dict(color="#FF9800", width=2),
        ))
        fig_s.add_hline(y=h_strike, line_dash="dot", line_color="gray",
                        annotation_text="Strike")
        fig_s.update_layout(xaxis_title="Time (years)", yaxis_title="Spot ($)",
                            title="Simulated Spot Path", height=300)
        st.plotly_chart(fig_s, width="stretch")

    with pcol2:
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(
            x=sim_df["t"], y=sim_df["delta"],
            mode="lines", name="Delta",
            line=dict(color="#4CAF50", width=2),
        ))
        fig_d.update_layout(xaxis_title="Time (years)", yaxis_title="Delta",
                            title="Hedge Delta Over Time", height=300)
        st.plotly_chart(fig_d, width="stretch")

    # ── hedging frequency analysis ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Hedging Frequency Analysis")
    st.markdown(
        "More frequent rebalancing tightens the hedge but increases transaction costs. "
        "Each bar shows the **distribution of final P&L** across 100 random paths."
    )

    with st.spinner("Running 100 × 4 simulations..."):
        fa_df = run_frequency_analysis(
            S0=h_spot, K=h_strike, T=h_T, r=h_r,
            sigma_implied=h_sigma_imp,
            option_type=h_otype, q=default_q,
            frequencies=[4, 12, 52, 252],
            n_sims=100, seed=int(h_seed),
        )

    fa_display = fa_df.reset_index()
    fig_fa = go.Figure()
    for _, row in fa_display.iterrows():
        fig_fa.add_trace(go.Bar(
            name=row["label"],
            x=[row["label"]],
            y=[row["std_pnl"]],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[row["pnl_95pct"] - row["mean_pnl"]],
                arrayminus=[row["mean_pnl"] - row["pnl_5pct"]],
            ),
            text=f"μ={row['mean_pnl']:.3f}",
            textposition="outside",
        ))
    fig_fa.update_layout(
        title="P&L Std Dev by Hedging Frequency (bars = std, whiskers = 5th/95th pct)",
        yaxis_title="P&L Std Dev ($)", height=380,
        showlegend=False,
    )
    st.plotly_chart(fig_fa, width="stretch")
    st.dataframe(fa_df[["label","mean_pnl","std_pnl","pnl_5pct","pnl_95pct","mean_tc","mean_pnl_net_tc"]].round(4),
                 width="stretch")

    # ── vol mismatch analysis ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Vol Mismatch P&L")
    st.markdown(
        "If realised vol ≠ implied vol, the hedge accumulates a systematic drift.  \n"
        "Short gamma: **realised > implied → you lose money**. "
        "Each point is the mean P&L over 100 paths."
    )

    with st.spinner("Running vol mismatch analysis..."):
        vm_df = run_vol_mismatch_analysis(
            S0=h_spot, K=h_strike, T=h_T, r=h_r,
            sigma_implied=h_sigma_imp,
            option_type=h_otype, q=default_q,
            realised_vols=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
            n_sims=75, seed=int(h_seed),
        )

    fig_vm = go.Figure()
    colors = ["#4CAF50" if v < 0 else "#F44336"
              for v in vm_df["vol_spread"]]
    fig_vm.add_trace(go.Bar(
        x=vm_df.index * 100,
        y=vm_df["mean_pnl"],
        marker_color=colors,
        name="Mean P&L",
        error_y=dict(type="data", array=vm_df["std_pnl"].tolist(), visible=True),
    ))
    fig_vm.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_vm.add_vline(
        x=h_sigma_imp * 100, line_dash="dot", line_color="orange",
        annotation_text="Implied vol",
    )
    fig_vm.update_layout(
        xaxis_title="Realised vol (%)",
        yaxis_title="Mean P&L ($)  [100 paths]",
        title="Hedge P&L vs Realised Vol (green = realised < implied = profit)",
        height=380,
    )
    st.plotly_chart(fig_vm, width="stretch")

    # ── hedge effectiveness benchmark ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Hedge Effectiveness Benchmark")
    st.markdown(
        "Compares **three strategies** on the same simulated paths — the honest "
        "test of whether daily delta hedging is actually worth the cost."
    )
    st.markdown("""
    | Strategy | Description |
    |---|---|
    | **Unhedged** | Sell option, collect premium, pay payoff at expiry. No stock position. |
    | **Static hedge** | Buy delta₀ shares at t=0, never rebalance. |
    | **Dynamic hedge** | Daily rebalance to new BSM delta. |
    """)

    hb_col1, hb_col2, hb_col3 = st.columns(3)
    with hb_col1:
        bm_sims = st.slider("Paths per strategy", 50, 500, 200,
                            step=50, key="bm_sims")
    with hb_col2:
        bm_tc   = st.number_input("Transaction cost %", value=0.0,
                                   min_value=0.0, max_value=1.0,
                                   step=0.001, format="%.3f", key="bm_tc")
    with hb_col3:
        bm_seed = st.number_input("Benchmark seed", value=42,
                                   step=1, key="bm_seed", format="%d")

    with st.spinner(f"Running {bm_sims} × 3 paths..."):
        bm_df = compare_hedge_strategies(
            S0=h_spot, K=h_strike, T=h_T, r=h_r,
            sigma_implied=h_sigma_imp,
            sigma_realised=h_sigma_real,
            option_type=h_otype, q=default_q,
            n_steps=h_steps,
            transaction_cost_pct=bm_tc / 100,
            n_sims=bm_sims,
            seed=int(bm_seed),
        )

    bm_reset  = bm_df.reset_index()
    clr_bm    = ["#EF5350", "#FF9800", "#4CAF50"]

    fig_bm = go.Figure()
    for i, row in bm_reset.iterrows():
        fig_bm.add_trace(go.Bar(
            name=row["strategy"],
            x=[row["strategy"]],
            y=[row["std_pnl"]],
            marker_color=clr_bm[i],
            text=f"σ={row['std_pnl']:.3f}",
            textposition="outside",
            error_y=dict(
                type="data", symmetric=False,
                array=[row["pnl_95pct"] - row["mean_pnl"]],
                arrayminus=[row["mean_pnl"] - row["pnl_5pct"]],
            ),
        ))

    fig_bm.update_layout(
        title="P&L Std Dev by Strategy — lower = better hedge",
        yaxis_title="P&L Std Dev ($)",
        height=380,
        showlegend=False,
    )
    st.plotly_chart(fig_bm, width="stretch")

    bm_display = bm_df.rename(columns={
        "mean_pnl"              : "Mean P&L ($)",
        "std_pnl"               : "Std Dev ($)",
        "pnl_5pct"              : "5th Pct ($)",
        "pnl_95pct"             : "95th Pct ($)",
        "pnl_range"             : "Range ($)",
        "variance_reduction_pct": "Var Reduction %",
    })
    st.dataframe(bm_display.round(4), width="stretch")
    st.caption(
        "**Var Reduction %** = variance removed vs unhedged baseline. "
        "Dynamic should be highest. If static beats dynamic, hedging frequency is too low."
    )

    # ── hedge stress matrix ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Hedge Stress Matrix — Vol Mismatch × Transaction Costs")
    st.markdown(
        "Real desks face **two simultaneous risks**: realised vol differing from implied vol, "
        "and transaction costs eroding hedge P&L. This matrix shows dynamic hedge mean P&L "
        "at each (realised vol, transaction cost) combination — **the honest test of whether "
        "hedging is profitable after frictions.**"
    )

    _stress_cols = st.columns(3)
    with _stress_cols[0]:
        stress_sims = st.slider("Paths per cell", 30, 200, 50,
                                step=10, key="stress_sims")
    with _stress_cols[1]:
        stress_seed = st.number_input("Stress seed", value=42,
                                       step=1, key="stress_seed", format="%d")

    # Define the grid — compact for speed, covers the key regimes
    _rv_grid = [0.15, 0.25, 0.30, 0.35, 0.45]
    _tc_grid = [0.0, 0.05, 0.10, 0.20]  # percent per trade

    with st.spinner(f"Running {len(_rv_grid)}×{len(_tc_grid)} stress scenarios ({stress_sims} paths each)..."):
        _stress_results = []
        for _rv in _rv_grid:
            _row = {}
            for _tc in _tc_grid:
                # Run batch of simulations
                _pnls = []
                for _sim_i in range(stress_sims):
                    _sim = simulate_delta_hedge(
                        S0=h_spot, K=h_strike, T=h_T, r=h_r,
                        sigma_implied=h_sigma_imp,
                        sigma_realised=_rv,
                        option_type=h_otype, q=default_q,
                        n_steps=126,
                        transaction_cost_pct=_tc / 100,
                        seed=int(stress_seed) + _sim_i,
                    )
                    _pnls.append(float(_sim["pnl_cumulative"].iloc[-1]))
                _row[f"TC={_tc:.2f}%"] = np.mean(_pnls)
            _stress_results.append(_row)

    stress_df = pd.DataFrame(
        _stress_results,
        index=[f"{v:.0%}" for v in _rv_grid],
    )
    stress_df.index.name = "Realised Vol"

    # Heatmap
    fig_stress = go.Figure(data=go.Heatmap(
        z=stress_df.values,
        x=stress_df.columns.tolist(),
        y=stress_df.index.tolist(),
        colorscale=[
            [0.0, "#B71C1C"],
            [0.35, "#EF5350"],
            [0.5, "#FFFFFF"],
            [0.65, "#66BB6A"],
            [1.0, "#1B5E20"],
        ],
        zmid=0,
        text=[[f"${v:.2f}" for v in row] for row in stress_df.values],
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorbar=dict(title="Mean P&L ($)"),
        hovertemplate=(
            "Realised Vol: %{y}<br>"
            "Transaction Cost: %{x}<br>"
            "Mean P&L: $%{z:.2f}<extra></extra>"
        ),
    ))

    # Mark the implied vol row in y-axis labels
    _imp_label = f"{h_sigma_imp:.0%}"
    _y_labels = [f"{v:.0%}" for v in _rv_grid]
    _y_labels_annotated = [
        f"{l}  ◄ implied" if l == _imp_label else l for l in _y_labels
    ]

    fig_stress.update_layout(
        title=f"Dynamic Hedge Mean P&L — Short {h_otype.capitalize()} (K={h_strike}, σ_imp={h_sigma_imp:.0%})",
        xaxis_title="Transaction Cost (% per trade)",
        yaxis_title="Realised Volatility",
        yaxis=dict(
            tickvals=_y_labels,
            ticktext=_y_labels_annotated,
            autorange="reversed",
        ),
        height=420,
    )
    st.plotly_chart(fig_stress, width="stretch")

    st.dataframe(
        stress_df.style
        .format("${:.2f}")
        .background_gradient(cmap="RdYlGn", axis=None, vmin=stress_df.min().min(), vmax=stress_df.max().max()),
        width="stretch",
    )
    st.caption(
        "**Green = profitable** (realised vol < implied, low costs). "
        "**Red = losing money** (realised vol > implied, high costs). "
        "The diagonal from top-right to bottom-left is the break-even frontier. "
        "At zero transaction cost, the hedge breaks even when realised vol = implied vol."
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — RISK & SENSITIVITY
# ═════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("Risk & Sensitivity")
    st.caption(
        "From a single option to the full book — build a portfolio, stress-test it "
        "against spot moves, inspect Greeks by strike and expiry, and quantify tail risk."
    )

    options_df = load_options()

    if options_df is None:
        st.warning("Run `python refresh.py` first to generate data.")
    else:
        r6           = cfg.get("risk_free_rate", 0.045)
        q6           = cfg.get("dividend_yield", 0.0)
        S6           = float(options_df["spot"].iloc[0])
        _ticker_name = cfg.get("ticker", "GOOG")

        # ── SECTION 1: Example Portfolio ──────────────────────────────────────
        st.subheader("① Example Portfolio")
        st.markdown(
            "A real desk manages a **book of positions** — not individual options. "
            "Adjust the quantities below and watch the portfolio value and Greeks update instantly. "
            "Positive = long, negative = short."
        )

        _S6 = S6
        _K1 = round(_S6 * 0.95)
        _K2 = round(_S6 * 1.07)
        _K3 = round(_S6)

        _sc1, _sc2, _sc3 = st.columns(3)
        with _sc1:
            st.markdown(f"**Leg 1 — OTM Put** (K=${_K1}, 90d, IV=38%)")
            st.caption("Downside protection — long = insured against a crash")
            _q1 = st.slider("Quantity", -20, 20, 10, step=1, key="pf_q1")
        with _sc2:
            st.markdown(f"**Leg 2 — OTM Call** (K=${_K2}, 90d, IV=33%)")
            st.caption("Covered call — short = cap upside, collect premium")
            _q2 = st.slider("Quantity", -20, 20, -5, step=1, key="pf_q2")
        with _sc3:
            st.markdown(f"**Leg 3 — ATM Put** (K=${_K3}, 30d, IV=37%)")
            st.caption("Short vol — short = earn theta, lose on big moves")
            _q3 = st.slider("Quantity", -20, 20, -10, step=1, key="pf_q3")

        _pos_defs = [
            (_K1, 90/365, 0.38, "put",  _q1, f"OTM Put  K=${_K1}  90d"),
            (_K2, 90/365, 0.33, "call", _q2, f"OTM Call K=${_K2}  90d"),
            (_K3, 30/365, 0.37, "put",  _q3, f"ATM Put  K=${_K3}  30d"),
        ]
        _ex_positions = [
            OptionPosition(K=float(K), T=T, sigma=sig,
                           option_type=ot, quantity=float(qty), label=lbl)
            for K, T, sig, ot, qty, lbl in _pos_defs if qty != 0
        ]

        if not _ex_positions:
            st.warning("All quantities are zero — set at least one position.")
        else:
            _pg = portfolio_greeks(_ex_positions, _S6, r6, q6)

            _pos_rows  = []
            _total_val = 0.0
            for _p in _ex_positions:
                _pval    = bsm_price(_S6, _p.K, _p.T, r6, _p.sigma, _p.option_type, q6)
                _pos_val = _p.quantity * _pval
                _total_val += _pos_val
                _pos_rows.append({
                    "Position"  : _p.label,
                    "Type"      : _p.option_type.capitalize(),
                    "Strike"    : f"${_p.K:.0f}",
                    "TTM"       : f"{round(_p.T*365)}d",
                    "IV"        : f"{_p.sigma:.0%}",
                    "Price ($)" : f"{_pval:.2f}",
                    "Qty"       : f"{int(_p.quantity):+d}",
                    "Value ($)" : f"{_pos_val:.2f}",
                })

            _mv1, _mv2, _mv3 = st.columns(3)
            _mv1.metric("Total Portfolio Value", f"${_total_val:.2f}",
                        help="Sum of qty x price. Negative = net premium received.")
            _mv2.metric("Active Legs", str(len(_ex_positions)))
            _mv3.metric("Net Exposure", "Short" if _total_val < 0 else "Long")

            st.dataframe(pd.DataFrame(_pos_rows), width="stretch")

            st.markdown("**Net Portfolio Greeks**")
            _gc1,_gc2,_gc3,_gc4,_gc5,_gc6 = st.columns(6)
            _gc1.metric("Net Delta",  f"{_pg['net_delta']:.3f}",
                        help="Shares-equivalent. Negative = short the market.")
            _gc2.metric("Net Gamma",  f"{_pg['net_gamma']:.5f}",
                        help="Negative = short gamma, large moves hurt.")
            _gc3.metric("Net Vega",   f"{_pg['net_vega']:.2f}",
                        help="P&L per unit vol move.")
            _gc4.metric("Net Theta",  f"{_pg['net_theta']:.3f}",
                        help="Positive = earning time decay each day.")
            _gc5.metric("$ Delta",    f"${_pg['dollar_delta']:.0f}",
                        help="Dollar value of stock to buy/sell for delta neutrality (delta × spot).")
            _gc6.metric("$ Vega/1pt", f"${_pg['dollar_vega']:.2f}",
                        help="P&L for a 1 vol-point move.")

            _delta_dir  = "short the market" if _pg["net_delta"] < 0 else "long the market"
            _gamma_str  = "short gamma — large moves hurt" if _pg["net_gamma"] < 0 else "long gamma — large moves help"
            _theta_dir  = "earning" if _pg["net_theta"] > 0 else "paying"
            _theta_amt  = abs(_pg["net_theta"])
            _hedge_verb = "sell" if _pg["dollar_delta"] > 0 else "buy"
            _hedge_amt  = abs(_pg["dollar_delta"])
            st.info(
                f"**Reading this book:** "
                f"Net delta {_pg['net_delta']:.2f} — portfolio is {_delta_dir}. "
                f"It is {_gamma_str}. "
                f"Theta {_pg['net_theta']:.2f} — {_theta_dir} "
                f"\\${_theta_amt:.2f}/day in time decay. "
                f"To delta-hedge: {_hedge_verb} \\${_hedge_amt:.0f} of {_ticker_name} stock."
            )

        st.markdown("---")

        # ── SECTION 2: Spot Sensitivity ────────────────────────────────────────
        st.subheader("② Spot Sensitivity (What-If)")
        st.markdown(
            "Re-prices the entire short book at spot levels from -20% to +20%. "
            "This is the first chart a trader checks before market open — "
            "how much do I make or lose if the stock moves?"
        )

        with st.spinner("Computing spot ladder..."):
            sl_df = spot_ladder(options_df, S6, r6, q6)

        if not sl_df.empty:
            fig_ladder = go.Figure()
            colors_l = ["#4CAF50" if v >= 0 else "#EF5350"
                        for v in sl_df["value_change"]]
            fig_ladder.add_trace(go.Bar(
                x=sl_df["spot_shock_pct"],
                y=sl_df["value_change"],
                marker_color=colors_l,
                text=[f"${v:+.2f}" for v in sl_df["value_change"]],
                textposition="outside",
            ))
            fig_ladder.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_ladder.update_layout(
                xaxis_title="Spot Shock (%)",
                yaxis_title="Portfolio Value Change ($)",
                title=f"Portfolio P&L vs Spot Move — Short Book  (S={S6:.2f})",
                height=380, showlegend=False,
            )
            st.plotly_chart(fig_ladder, width="stretch")

            sl_show = sl_df.reset_index()
            sl_show.columns = [
                "Spot ($)","Shock %","Value ($)","Delta Value ($)",
                "Net Delta","Net Gamma","Net Vega","Net Theta","$ Delta"
            ]
            st.dataframe(sl_show.style.format(precision=4), width="stretch")
            st.caption(
                "Green = portfolio gains. Red = portfolio loses. "
                "A short options book typically shows red in both directions — "
                "this is normal for a short gamma position."
            )

        st.markdown("---")

        # ── SECTION 3: Greeks Ladder ───────────────────────────────────────────
        st.subheader("③ Greeks Ladder")
        st.markdown(
            "Delta, gamma, vega, theta for every option in the dataset — "
            "filterable by expiry and type. "
            "The aggregated table below shows where risk concentrates across the term structure."
        )

        ladder_df = greeks_ladder(options_df, S6, r6, q6)

        if not ladder_df.empty:
            _lc1, _lc2 = st.columns(2)
            with _lc1:
                expiry_opts = ["All"] + [
                    str(e.date()) for e in sorted(ladder_df["expiration"].unique())
                ]
                sel_exp = st.selectbox("Filter by expiry", expiry_opts, key="ladder_exp")
            with _lc2:
                otype_f = st.radio("Option type", ["All","call","put"],
                                   horizontal=True, key="ladder_type")

            ldf = ladder_df.copy()
            if sel_exp != "All":
                ldf = ldf[ldf["expiration"].dt.strftime("%Y-%m-%d") == sel_exp]
            if otype_f != "All":
                ldf = ldf[ldf["option_type"] == otype_f]

            show_cols = ["option_type","strike","ttm","iv_engine","mid",
                         "delta","gamma","vega","theta",
                         "dollar_delta","dollar_gamma_1pct","dollar_vega_1pt"]
            ldf_show = ldf[show_cols].copy()
            ldf_show.columns = [
                "Type","Strike","TTM","IV","Mid($)",
                "Delta","Gamma","Vega","Theta",
                "$ Delta","$ Gamma/1%","$ Vega/1pt"
            ]
            st.dataframe(
                ldf_show.style
                .background_gradient(subset=["Delta"],      cmap="RdYlGn", vmin=-1, vmax=1)
                .background_gradient(subset=["$ Vega/1pt"], cmap="Blues")
                .format(precision=4),
                width="stretch", height=350,
            )

            st.markdown("**Aggregated by Expiry — short book (quantity = -1 per option)**")
            st.caption(
                "Red gamma = short gamma at that expiry (large moves hurt). "
                "Positive daily theta = earning time decay. "
                "Blue dollar vega = large vol exposure at that tenor."
            )
            agg_df = greeks_by_expiry_from_df(options_df, S6, r6, q6, quantity=-1.0)
            if not agg_df.empty:
                agg_show = agg_df.copy()
                agg_show.index = pd.to_datetime(agg_show.index).strftime("%Y-%m-%d")
                st.dataframe(
                    agg_show.style
                    .background_gradient(subset=["net_gamma"],      cmap="RdYlGn_r")
                    .background_gradient(subset=["dollar_vega_1pt"], cmap="Blues_r")
                    .format(precision=4),
                    width="stretch",
                )

        st.markdown("---")

        # ── SECTION 4: Historical VaR ──────────────────────────────────────────
        st.subheader("④ Historical VaR")
        st.markdown(
            "Re-prices the short book at every historical daily return since 2024. "
            "The loss distribution is empirical — no normal distribution assumed. "
            "VaR = the loss you would not exceed on 95% of trading days."
        )

        spot_path_file = DATA_DIR.parent / "raw" / "spot_history.csv"
        if spot_path_file.exists():
            spot_hist_var = pd.read_csv(spot_path_file)
            _vc1, _vc2 = st.columns(2)
            with _vc1:
                var_conf = st.slider("Confidence level", 0.90, 0.99, 0.95,
                                     step=0.01, key="var_conf")
            with _vc2:
                var_horizon = st.slider("Horizon (days)", 1, 10, 1, key="var_horizon")

            var_positions = []
            for _, row in options_df.iterrows():
                sig = row.get("iv_engine", np.nan)
                if pd.isna(sig) or sig <= 0 or row["ttm"] <= 0:
                    continue
                var_positions.append(OptionPosition(
                    K=float(row["strike"]), T=float(row["ttm"]),
                    sigma=float(sig), option_type=str(row["option_type"]),
                    quantity=-1.0,
                ))

            if var_positions:
                with st.spinner("Computing historical VaR..."):
                    var_result = historical_var(
                        var_positions, spot_hist_var, S6, r6, q6,
                        confidence=var_conf, horizon_days=var_horizon,
                    )
                _r1, _r2, _r3, _r4 = st.columns(4)
                _r1.metric(f"VaR ({var_conf:.0%}, {var_horizon}d)",
                           f"${var_result['var_horizon']:.2f}",
                           help="Maximum loss not exceeded on this % of days.")
                _r2.metric(f"CVaR ({var_conf:.0%}, {var_horizon}d)",
                           f"${var_result['cvar_horizon']:.2f}",
                           help="Average loss in the worst days beyond VaR.")
                _r3.metric("Scenarios", str(var_result["n_scenarios"]))
                _r4.metric("P&L Std Dev", f"${var_result['pnl_std']:.2f}")
                st.caption(
                    f"Based on {var_result['n_scenarios']} daily returns since 2024-01-01. "
                    f"CVaR (\\${var_result['cvar_horizon']:.2f}) > VaR (\\${var_result['var_horizon']:.2f}) "
                    f"indicates fat tails — the average loss on the worst days far exceeds the VaR threshold."
                )
        else:
            st.info("Run `python refresh.py` to generate spot history for VaR.")