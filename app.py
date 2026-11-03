from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from forecast_engine import (
    HORIZONS, EXPANDED_FEATURES, standardize,
    compute_expanded_features, infer_recipes, feature_candidates
)

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "Final_Dataset_AI_dxy_fixed.csv"
MODEL_FILE = ROOT / "model_artifacts.joblib"
STATUS_FILE = ROOT / "data_refresh_status.json"

st.set_page_config(
    page_title="Crude Oil Scenario Intelligence",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
:root { color-scheme: light; }
html, body, [data-testid="stAppViewContainer"] {
    background: #ffffff !important;
    color: #202a35 !important;
}
[data-testid="stHeader"] { background: rgba(255,255,255,.92) !important; }
.block-container {max-width: 1480px; padding-top: 1.15rem; padding-bottom: 3rem;}
[data-testid="stMarkdownContainer"], [data-testid="stMetricLabel"],
[data-testid="stMetricValue"], [data-testid="stCaptionContainer"] { color: #202a35; }
.hero {
    background: #ffffff; border: 1px solid #e9e4d8; border-left: 6px solid #b89a5a;
    border-radius: 16px; padding: 1.55rem 1.8rem; margin: 0 0 1.15rem 0;
    box-shadow: 0 5px 20px rgba(30,35,40,.045);
}
.hero .eyebrow {color:#927640; font-size:.72rem; font-weight:750;
    letter-spacing:.14em; text-transform:uppercase;}
.hero h1 {color:#1d2935 !important; font-size:2.15rem; margin:.32rem 0 .3rem 0;
    line-height:1.15; letter-spacing:-.035em;}
.hero p {color:#56616d !important; margin:.25rem 0 0 0; font-size:1rem;}
.hero .note {color:#7a838b !important; font-size:.78rem; margin-top:.65rem;}
.price-card {
    background:#fff; border:1px solid #e5e6e7; border-radius:14px;
    padding:1.1rem 1.2rem; min-height:140px;
    box-shadow:0 4px 16px rgba(20,30,40,.035);
}
.price-card .label {color:#69737d; font-size:.81rem; font-weight:650;}
.price-card .value {color:#1e2933; font-size:1.9rem; font-weight:760;
    letter-spacing:-.035em; margin:.25rem 0;}
.price-card .meta {color:#69737d; font-size:.77rem;}
.price-card .tag {display:inline-block; padding:.18rem .5rem; border-radius:30px;
    background:#f5f0e5; color:#80652f; font-size:.69rem; font-weight:700; margin-top:.4rem;}
.section-lead {background:#faf9f6; border-left:4px solid #b89a5a;
    border-radius:8px; padding:.8rem 1rem; color:#4c555e; margin:.45rem 0 1rem 0;}
.small-muted {color:#707982; font-size:.84rem;}
.callout-good {background:#eff7f1; border:1px solid #cfe5d4; color:#285b39;
    padding:.8rem 1rem; border-radius:9px;}
.callout-warn {background:#fff8e9; border:1px solid #ead9aa; color:#70551b;
    padding:.8rem 1rem; border-radius:9px;}
div[data-testid="stTabs"] button {font-weight:700; color:#65707a;}
div[data-testid="stTabs"] button[aria-selected="true"] {color:#222d36; border-bottom-color:#b89a5a;}
div[data-testid="stMetric"] {background:#fff; border:1px solid #e5e6e7;
    border-radius:12px; padding:.75rem 1rem; box-shadow:0 3px 12px rgba(20,30,40,.035);}
div[data-testid="stDataFrame"] {background:#fff; border:1px solid #e8e8e8; border-radius:12px;}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-testid="stNumberInput"] input {
    background:#fff !important; color:#202a35 !important; border-color:#d9dee3 !important;
}
div[data-testid="stNumberInput"] button {background:#f7f7f5 !important; color:#46515b !important;}
h1, h2, h3, h4 {color:#1f2b35 !important;}
.stButton > button {border-radius:9px; border-color:#c7b17f;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<section class="hero">
  <div class="eyebrow">Energy markets · Interactive research prototype</div>
  <h1>Crude Oil Scenario Intelligence</h1>
  <p>Explore how WTI market, supply, currency and geopolitical assumptions change a trained model's conditional estimate.</p>
  <div class="note">64 trained input features · 13 scenario controls · Historical model evaluation · Research use only</div>
</section>
""", unsafe_allow_html=True)


@st.cache_resource
def load_saved_state():
    for path in [DATA_FILE, MODEL_FILE, STATUS_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required project file not found: {path.name}")

    df = standardize(pd.read_csv(DATA_FILE, parse_dates=["Date"]))
    artifacts = joblib.load(MODEL_FILE)
    with open(STATUS_FILE, "r", encoding="utf-8") as handle:
        status = json.load(handle)

    required = ["models", "feature_names", "feature_date", "market_date", "current_close"]
    missing = [key for key in required if key not in artifacts]
    if missing:
        raise ValueError(f"Model artifact is missing keys: {missing}")

    feature_names = list(artifacts["feature_names"])
    if feature_names != EXPANDED_FEATURES:
        raise ValueError("Saved artifact feature order does not match forecast_engine.py.")

    feature_date = pd.Timestamp(artifacts["feature_date"]).normalize()
    market_date = pd.Timestamp(artifacts["market_date"]).normalize()
    current_close = float(artifacts["current_close"])

    matches = df.index[df["Date"].eq(feature_date)].tolist()
    if not matches:
        raise ValueError(f"Feature date {feature_date.date()} is not present in saved CSV.")

    feature_idx = matches[-1]
    csv_close = float(df.loc[feature_idx, "WTI_Close"])
    if not np.isclose(csv_close, current_close, rtol=1e-5, atol=0.01):
        raise ValueError("Saved model and dataset do not match: WTI close differs.")

    models = {}
    for horizon in HORIZONS:
        record = artifacts["models"].get(horizon, artifacts["models"].get(str(horizon)))
        if record is None or "pipeline" not in record or "latest_x" not in record:
            raise ValueError(f"Missing saved pipeline/input for horizon {horizon}.")
        models[horizon] = record

    brent_close = None
    brent_date = None
    dxy_close = None
    dxy_date = None
    if "DXY_Close" in df.columns:
        dxy_rows = df.loc[pd.to_numeric(df["DXY_Close"], errors="coerce").notna(),
                          ["Date", "DXY_Close"]].copy()
        if not dxy_rows.empty:
            dxy_rows["DXY_Close"] = pd.to_numeric(dxy_rows["DXY_Close"], errors="coerce")
            latest_dxy = dxy_rows.iloc[-1]
            dxy_close = float(latest_dxy["DXY_Close"])
            dxy_date = pd.Timestamp(latest_dxy["Date"]).normalize()

    if "Brent_Close" in df.columns:
        brent_rows = df.loc[pd.to_numeric(df["Brent_Close"], errors="coerce").notna(),
                             ["Date", "Brent_Close"]].copy()
        if not brent_rows.empty:
            brent_rows["Brent_Close"] = pd.to_numeric(brent_rows["Brent_Close"], errors="coerce")
            latest_brent = brent_rows.iloc[-1]
            brent_close = float(latest_brent["Brent_Close"])
            brent_date = pd.Timestamp(latest_brent["Date"]).normalize()

    return {
        "df": df, "models": models, "status": status,
        "feature_date": feature_date, "market_date": market_date,
        "current_close": current_close, "feature_names": feature_names,
        "feature_idx": feature_idx, "brent_close": brent_close,
        "brent_date": brent_date, "dxy_close": dxy_close, "dxy_date": dxy_date,
    }


def make_scenario_features(
    state, wti_price_usd, dxy_level, inventory_mmbbl, oil_rig_count,
    brent_price_usd, gpr_index
):
    """Create a temporary 64-feature row from absolute scenario levels."""
    df = state["df"]
models = state["models"]
status = state["status"]
feature_date = state["feature_date"]
market_date = state["market_date"]
current_close = state["current_close"]
brent_close = state["brent_close"]
brent_date = state["brent_date"]
dxy_close = state["dxy_close"]
dxy_date = state["dxy_date"]

market_inputs = {
    "wti": resolve_market_reference("CL=F", current_close, feature_date.date()),
    "brent": resolve_market_reference(
        "BZ=F",
        brent_close if brent_close is not None else current_close,
        brent_date.date() if brent_date is not None else feature_date.date()
    ),
    "dxy": resolve_market_reference(
        "DX-Y.NYB",
        dxy_close if dxy_close is not None else 100.0,
        dxy_date.date() if dxy_date is not None else feature_date.date()
    ),
}

# Latest published reference observations checked on 4 Nov 2026.
# EIA commercial crude stocks exclude the Strategic Petroleum Reserve.
LATEST_PUBLIC_SCENARIO_INPUTS = {
    "inventory_mmbbl": 410.6,   # EIA, week ending 23 Oct 2026; released 28 Oct
    "inventory_as_of": "Week ending 23 Oct 2026",
    "oil_rigs": 462,            # Baker Hughes US oil-directed rigs, 30 Oct 2026
    "rigs_as_of": "30 Oct 2026",
    "gpr_index": 150.52,        # Caldara-Iacoviello GPR monthly reading for Oct 2026
    "gpr_as_of": "Oct 2026 monthly reading",
}

latest_defaults = {
    "scenario_wti_price": float(market_inputs["wti"]["value"]),
    "scenario_brent_price": float(market_inputs["brent"]["value"]),
    "scenario_dxy_level": float(market_inputs["dxy"]["value"]),
    "scenario_inventory_mmbbl": float(LATEST_PUBLIC_SCENARIO_INPUTS["inventory_mmbbl"]),
    "scenario_oil_rigs": int(LATEST_PUBLIC_SCENARIO_INPUTS["oil_rigs"]),
    "scenario_gpr_index": float(LATEST_PUBLIC_SCENARIO_INPUTS["gpr_index"]),
}

st.markdown("### Market snapshot")
snap1, snap2, snap3, snap4 = st.columns(4, gap="medium")
with snap1:
    price_card(
        "WTI · US crude benchmark", f"USD {market_inputs['wti']['value']:,.2f}/bbl",
        f"Latest daily close: {market_inputs['wti']['date']}",
        market_inputs["wti"]["source"], "Main model target"
    )
with snap2:
    price_card(
        "Brent · Global crude benchmark", f"USD {market_inputs['brent']['value']:,.2f}/bbl",
        f"Latest daily close: {market_inputs['brent']['date']}",
        market_inputs["brent"]["source"], "Scenario input"
    )
with snap3:
    price_card(
        "US Dollar Index · DXY", f"{market_inputs['dxy']['value']:,.2f}",
        f"Latest daily close: {market_inputs['dxy']['date']}",
        market_inputs["dxy"]["source"], "Scenario input"
    )
with snap4:
    if market_inputs["wti"]["date"] == market_inputs["brent"]["date"]:
        spread_value = f"USD {market_inputs['brent']['value'] - market_inputs['wti']['value']:+,.2f}/bbl"
        spread_date = f"Synchronized date: {market_inputs['wti']['date']}"
    else:
        spread_value = "Dates differ"
        spread_date = f"WTI {market_inputs['wti']['date']} · Brent {market_inputs['brent']['date']}"
    price_card(
        "Brent − WTI spread", spread_value, spread_date,
        "Difference between benchmark prices", "Market context"
    )

with st.expander("WTI and Brent: what is the difference?"):
    left, right = st.columns(2)
    with left:
        st.markdown("**WTI — West Texas Intermediate**")
        st.write(
            "A light, sweet crude benchmark produced in the United States. "
            "Its futures contract is linked to delivery at Cushing, Oklahoma. "
            "It is widely used as a reference for the North American oil market."
        )
    with right:
        st.markdown("**Brent crude**")
        st.write(
            "A light, sweet crude benchmark based on North Sea crude streams. "
            "It is the main reference for much of the seaborne/global crude market."
        )
    st.caption(
        "The two prices can differ because crude quality, transport costs, delivery location "
        "and regional supply-demand conditions differ. This app displays both benchmarks, "
        "but the saved prediction model currently forecasts WTI only."
    )

if market_date > feature_date:
    st.warning(
        f"The dataset contains WTI data through {market_date.date()}, but the latest complete "
        f"64-feature model row is {feature_date.date()}. Forecast calculations use the latter."
    )
else:
    st.caption(
        f"Forecast reference data date: {feature_date.date()} · "
        f"Last successful refresh (UTC): {status.get('last_successful_run_utc', 'not recorded')}"
    )

labels = {
    1: "Next trading session", 4: "4 trading sessions",
    5: "5 trading sessions", 10: "10 trading sessions",
    20: "20 trading sessions"
}
SCENARIO_HORIZONS = [5, 10, 20]
scenario_tab, market_tab = st.tabs(["01 · Scenario Lab", "02 · Forecast & Backtest"])

with market_tab:
    st.subheader("Next trading session · WTI reference estimate")
    next_record = models[1]
    next_price = float(next_record["price"])
    next_return = float(next_record["return"]) * 100
    a, b, c = st.columns([1.25, 1, 1], gap="medium")
    with a:
        st.metric("Estimated WTI close reference", f"USD {next_price:,.2f}/bbl",
                  f"{next_return:+.3f}% vs reference close")
    with b:
        st.metric("Reference close", f"USD {current_close:,.2f}/bbl")
    with c:
        st.metric("Model reference date", str(feature_date.date()))

    st.markdown(
        """<div class="callout-warn">
        <strong>Important:</strong> This is not a forecast of tomorrow's opening price.
        The dataset target is the future <em>closing</em> price. For the next session,
        the current model uses a zero-return benchmark, so the estimate stays equal to
        the last complete reference close. It is a neutral baseline, not a directional signal.
        </div>""",
        unsafe_allow_html=True
    )

    st.subheader("WTI outlook by forecast horizon")
    st.markdown(
        '<div class="section-lead">Each row answers: “What reference close does the current model produce '
        'after this many observed trading sessions?” It does not mean calendar days.</div>',
        unsafe_allow_html=True
    )
    rows = []
    for horizon in HORIZONS:
        record = models[horizon]
        model_type = record["model_type"]
        method = "Neutral / unchanged-return baseline" if "Zero" in model_type else "Experimental ExtraTrees ML"
        result = "Baseline tie" if not record["beats_zero"] and "Zero" in model_type else (
            "Below baseline" if not record["beats_zero"] else "Beats baseline"
        )
        rows.append({
            "Forecast horizon": labels[horizon],
            "Method": method,
            "Estimated WTI close (USD/bbl)": record["price"],
            "Expected change": record["return"] * 100,
            "Holdout error (pp)": record["test_mae"] * 100,
            "Zero-change error (pp)": record["zero_mae"] * 100,
            "Validation": result
        })
    forecast_df = pd.DataFrame(rows)
    st.dataframe(
        forecast_df.style.format({
            "Estimated WTI close (USD/bbl)": "{:,.2f}",
            "Expected change": "{:+.3f}%",
            "Holdout error (pp)": "{:.3f}",
            "Zero-change error (pp)": "{:.3f}"
        }),
        width="stretch", hide_index=True
    )
    st.caption(
        "pp = percentage points. Holdout error is measured on a later-in-time test period. "
        "Smaller MAE is better. An indicative price range is not a calibrated confidence interval."
    )
    weak_horizons = [str(h) for h in HORIZONS if not models[h]["beats_zero"]]
    if weak_horizons:
        st.warning(
            "Current chronological holdout: the trained approach does not beat the simple "
            "zero-change benchmark at horizons " + ", ".join(weak_horizons) +
            ". Treat the experimental ML estimates as research outputs, not proven improvements."
        )

    with st.expander("What information does the model actually use?"):
        st.write(
            "The model has 64 numeric input features. Many are calculated automatically "
            "from previous market observations; they are not 53 separate sliders."
        )
        feature_families = [
            ("WTI price history & technical indicators", [n for n in state["feature_names"] if n.startswith("WTI_")],
             "Past returns, momentum, volatility, distance from moving averages and drawdowns."),
            ("US Dollar Index (DXY)", [n for n in state["feature_names"] if n.startswith("DXY_")],
             "Daily/past returns, momentum and volatility of the six-currency US dollar index."),
            ("Other commodity markets", [n for n in state["feature_names"] if n.startswith("CMD_")],
             "Lagged Brent, Urals, ESPO, European natural gas (TTF), wheat and urea prices."),
            ("Geopolitical risk (GPR)", [n for n in state["feature_names"] if n.startswith("GPR_")],
             "Lagged total geopolitical risk, threats and acts indices."),
            ("Trade policy uncertainty (TPU)", [n for n in state["feature_names"] if n.startswith("TPU_")],
             "Lagged trade-policy uncertainty index."),
            ("Global oil production", [n for n in state["feature_names"] if n.startswith("EIA_world_prod_")],
             "Global oil production series with a three-month lag."),
            ("US oil rig count", [n for n in state["feature_names"] if n.startswith("RIG_")],
             "Rig-count level and changes across recent periods."),
            ("US crude inventories", [n for n in state["feature_names"] if n.startswith("INVENTORY_") or n.startswith("INV_")],
             "Inventory level, changes and percentage change.")
        ]
        for title, names, detail in feature_families:
            st.markdown(f"**{title} — {len(names)} features**")
            st.caption(detail)
        st.markdown("**Not currently trained model inputs**")
        st.write(
            "GPR is included through three lagged historical indices, and TPU is included as a lagged "
            "historical uncertainty index. They are not live headline/news signals. OPEC+ decisions, "
            "war/peace headlines beyond the historical GPR series, shipping disruptions, weather, "
            "refinery demand and interest rates are not separate trained inputs. The live BZ=F card "
            "is a benchmark quote; the model's CMD_brent_lag1 is a separate stored lagged feature."
        )

    st.subheader("Historical evaluation · actual vs previous-close benchmark")
    history = df[["Date", "WTI_Close"]].dropna().sort_values("Date").copy()
    history["Previous_Close_Benchmark"] = history["WTI_Close"].shift(1)
    history_start = history["Date"].max() - pd.Timedelta(days=90)
    history_view = history.loc[history["Date"] >= history_start].dropna(subset=["Previous_Close_Benchmark"])
    if not history_view.empty:
        fig, ax = plt.subplots(figsize=(13, 4.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.plot(history_view["Date"], history_view["WTI_Close"], color="#23865b",
                linewidth=2.5, label="Actual WTI close")
        ax.plot(history_view["Date"], history_view["Previous_Close_Benchmark"], color="#c6534b",
                linewidth=2, linestyle="--", label="Previous-close benchmark estimate")
        ax.set_ylabel("WTI close (USD/bbl)")
        ax.set_xlabel("Trading date")
        ax.grid(alpha=.18, color="#9aa0a6")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, ncol=2, loc="upper left")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        st.caption(
            f"Showing available observations from {history_view['Date'].min().date()} to {history_view['Date'].max().date()}. "
            "The red dashed line is a retrospective previous-close benchmark replay, not historical predictions made by today's trained ML model."
        )
    else:
        st.info("There are not enough historical observations to display the benchmark replay.")

with scenario_tab:
    st.subheader("Interactive what-if scenario builder")
    st.markdown(
        '<div class="section-lead"><strong>How it works:</strong> Choose a ready-made market scenario or adjust the inputs yourself. '
        'The app updates the 64-feature input row and sends it to the saved 5-, 10- or 20-session ExtraTrees model. '
        'The original dataset and trained model are not overwritten.</div>',
        unsafe_allow_html=True
    )

    for key, value in SCENARIO_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("scenario_preset", "Custom / manual")

    st.selectbox(
        "Choose a demo scenario",
        options=list(SCENARIO_PRESETS.keys()),
        key="scenario_preset",
        on_change=apply_scenario_preset,
        help="Preset values are illustrative assumptions, not claims about a real event."
    )
    st.caption("Select a preset or move any control. The result panel updates automatically.")

    control_col, result_col = st.columns([1.45, 1], gap="large")

    with control_col:
        with st.container(border=True):
            st.markdown("#### 1 · Market prices & currency")
            m1, m2 = st.columns(2, gap="medium")
            with m1:
                wti_shock = st.slider(
                    "WTI price shock (%)", -20.0, 20.0, step=0.5,
                    key="scenario_wti_shock",
                    help="Changes the reference WTI price and recalculates its historical technical features."
                )
                dxy_move = st.slider(
                    "US Dollar Index · DXY (%)", -10.0, 10.0, step=0.25,
                    key="scenario_dxy_move",
                    help="Changes the saved DXY level and its derived return/momentum features."
                )
                brent_move = st.slider(
                    "Brent crude (%)", -30.0, 30.0, step=1.0,
                    key="scenario_brent_move",
                    help="Applies a percentage shock to the model's lagged Brent input."
                )
            with m2:
                urals_move = st.slider(
                    "Urals crude (%)", -30.0, 30.0, step=1.0,
                    key="scenario_urals_move",
                    help="Applies a percentage shock to the model's lagged Urals input."
                )
                espo_move = st.slider(
                    "ESPO crude (%)", -30.0, 30.0, step=1.0,
                    key="scenario_espo_move",
                    help="Applies a percentage shock to the model's lagged ESPO input."
                )

        with st.container(border=True):
            st.markdown("#### 2 · Supply & storage")
            s1, s2 = st.columns(2, gap="medium")
            with s1:
                inventory_change = st.number_input(
                    "US commercial crude stocks (kbbl)",
                    min_value=-50000, max_value=50000, step=1000,
                    key="scenario_inventory_change",
                    help="Change in commercial stocks. +2,000 kbbl means +2 million barrels."
                )
            with s2:
                rig_change = st.number_input(
                    "Active US oil rigs (count)",
                    min_value=-150, max_value=150, step=5,
                    key="scenario_rig_change",
                    help="Hypothetical change in active oil-directed drilling rigs."
                )
            production_move = st.slider(
                "Global oil production (%)", -5.0, 5.0, step=0.5,
                key="scenario_production_move",
                help="Applies a shock to the model's stored global production feature (three-month lagged)."
            )

        with st.container(border=True):
            st.markdown("#### 3 · Geopolitical risk & related indicators")
            r1, r2 = st.columns(2, gap="medium")
            with r1:
                gpr_move = st.slider(
                    "Geopolitical Risk Index · GPR (%)", -50.0, 100.0,
                    step=5.0, key="scenario_gpr_move",
                    help="Changes stored GPR total, threats and acts indices together."
                )
                tpu_move = st.slider(
                    "Trade Policy Uncertainty · TPU (%)", -50.0, 100.0,
                    step=5.0, key="scenario_tpu_move"
                )
                ttf_move = st.slider(
                    "European natural gas · TTF (%)", -30.0, 30.0,
                    step=1.0, key="scenario_ttf_move"
                )
            with r2:
                wheat_move = st.slider(
                    "Wheat price (%)", -30.0, 30.0, step=1.0,
                    key="scenario_wheat_move"
                )
                urea_move = st.slider(
                    "Urea / fertilizer price (%)", -30.0, 30.0,
                    step=1.0, key="scenario_urea_move"
                )
                chosen_horizon = st.selectbox(
                    "Model horizon", SCENARIO_HORIZONS,
                    key="scenario_horizon",
                    format_func=lambda h: labels[h],
                    help="5, 10 or 20 observed trading sessions. 1- and 4-session production methods are zero-return baselines."
                )

        macro_shocks = {
            "brent": brent_move, "urals": urals_move, "espo": espo_move,
            "ttf": ttf_move, "wheat": wheat_move, "urea": urea_move,
            "gpr": gpr_move, "tpu": tpu_move, "world_prod": production_move,
        }

        st.caption(
            "Coverage note: the inventory control is US commercial stock, not Strategic Petroleum Reserve. "
            "SPR releases and direct shipping-route disruption are not separate trained model inputs."
        )

    with result_col:
        with st.container(border=True):
            st.markdown("#### Live scenario output")
            st.caption("Updates automatically when you change the controls.")
            try:
                scenario_x, scenario_close = make_scenario_features(
                    state, wti_shock, dxy_move,
                    int(inventory_change), int(rig_change), macro_shocks
                )
                baseline = models[chosen_horizon]
                baseline_return = float(baseline["return"])
                baseline_price = float(baseline["price"])
                scenario_return = float(baseline["pipeline"].predict(scenario_x)[0])
                scenario_price = scenario_close * (1.0 + scenario_return)

                if (not np.isfinite(scenario_return) or
                    not np.isfinite(scenario_price) or scenario_price <= 0):
                    raise ValueError("Model returned an invalid scenario estimate.")

                delta_return_pp = (scenario_return - baseline_return) * 100
                delta_price = scenario_price - baseline_price

                st.metric(
                    "Scenario WTI reference",
                    f"USD {scenario_close:,.2f}/bbl"
                )
                st.metric(
                    "Model-estimated WTI close",
                    f"USD {scenario_price:,.2f}/bbl",
                    delta=f"USD {delta_price:+,.2f} vs base case"
                )
                st.metric(
                    "Model-estimated return",
                    f"{scenario_return * 100:+.3f}%",
                    delta=f"{delta_return_pp:+.3f} percentage points"
                )
                st.divider()
                st.markdown("**Base case vs selected scenario**")
                st.write(
                    f"Base case: **USD {baseline_price:,.2f}/bbl**  "
                    f"({baseline_return * 100:+.3f}%)"
                )
                st.write(
                    f"Scenario: **USD {scenario_price:,.2f}/bbl**  "
                    f"({scenario_return * 100:+.3f}%)"
                )
                st.caption(
                    f"Forecast horizon: {labels[chosen_horizon]}. "
                    "WTI shock changes the starting reference price directly; other inputs change the model feature row."
                )
                if not baseline["beats_zero"]:
                    st.warning(
                        "Experimental model: this horizon did not beat the zero-return benchmark "
                        "on the current chronological holdout. Treat this as exploratory scenario analysis."
                    )
                else:
                    st.info(
                        "This horizon beat the zero-return benchmark on the current chronological holdout; "
                        "that does not guarantee future accuracy."
                    )
            except Exception as exc:
                st.error(f"Scenario calculation unavailable: {type(exc).__name__}: {exc}")

    st.markdown(
        '<div class="callout-warn"><strong>Model scope:</strong> GPR and TPU are lagged historical indices, '
        'not live news feeds. Supply-chain disruption is represented only indirectly through geopolitical risk, '
        'production and commodity assumptions. Strategic emergency-stock releases and individual shipping routes '
        'are not standalone model inputs.</div>',
        unsafe_allow_html=True
    )

    with st.expander("What the 64 model features contain"):
        st.write(
            "The 13 controls adjust model input families. The remaining features are automatically "
            "calculated from historical observations, including return lags, momentum, volatility, "
            "moving-average distance, drawdown and changes in DXY, inventories and rig counts."
        )
        st.write(
            "Not direct model inputs in this version: country-by-country strategic reserves, IEA emergency-stock release volumes, "
            "named conflict events, tanker-route disruptions, pipeline outages, OPEC+ policy decisions, weather, refinery utilization and interest rates. "
            "These require aligned historical series, a publication-lag policy, retraining and out-of-sample validation before they can affect model output."
        )

with st.expander("Data freshness, sources and model diagnostics"):
    st.markdown("**Last successful automated refresh (UTC)**")
    st.write(status.get("last_successful_run_utc", "Not recorded"))
    st.write("WTI data date:", str(market_date.date()))
    st.write("Forecast feature date:", str(feature_date.date()))
    st.write("Dataset rows:", status.get("dataset_rows", len(df)))
    st.write("WTI source:", status.get("wti_source_status", "Not recorded"))
    st.write("Brent source:", status.get("brent_source_status", "Awaiting first Brent refresh"))
    st.write("DXY source:", status.get("dxy_source_status", "Not recorded"))
    st.write("Inventory source:", status.get("inventory_source_status", "Not recorded"))
    st.write("Rig-count source:", status.get("rig_count_status", "Not recorded"))
    st.write("Lagged macro/commodity inputs:", status.get("macro_feature_status", "Original stored values; no separate live refresh connector."))
    st.write("EIA API configured:", status.get("eia_api_configured", False))

    st.markdown("**Chronological holdout performance**")
    diagnostic_rows = []
    for horizon in HORIZONS:
        record = models[horizon]
        diagnostic_rows.append({
            "Horizon (sessions)": horizon,
            "Method": record["model_type"],
            "Model MAE (pp)": record["test_mae"] * 100,
            "Zero-change MAE (pp)": record["zero_mae"] * 100,
            "Beats zero-change benchmark": record["beats_zero"],
            "Holdout rows": record["test_rows"]
        })
    st.dataframe(
        pd.DataFrame(diagnostic_rows).style.format({
            "Model MAE (pp)": "{:.3f}",
            "Zero-change MAE (pp)": "{:.3f}"
        }),
        width="stretch", hide_index=True
    )
    st.caption(
        "The app reads saved model artifacts. It does not refresh data or retrain while a visitor "
        "uses the dashboard. GitHub Actions handles scheduled data updates and training."
    )

st.caption(
    "Data quality note: displayed benchmark values are saved daily futures closes, not guaranteed live quotes. "
    "WTI forecast calculations use only the saved WTI feature row. Brent is a benchmark comparison until "
    "a separately trained and validated Brent model exists."
)
