from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import os

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
    page_title="Crude Oil Price Predictor",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
:root { color-scheme: light; }
html, body, [data-testid="stAppViewContainer"] {background:#f4f7fb!important;color:#172b46!important;}
[data-testid="stHeader"] {background:transparent!important;}
.block-container {max-width:1250px;padding-top:1.1rem;padding-bottom:2rem;}
.hero {background:linear-gradient(115deg,#102a4c,#245783);border-radius:17px;
       padding:1.35rem 1.7rem;margin-bottom:1rem;box-shadow:0 8px 24px #193d6018;}
.hero .kicker {color:#c6dbef;font-size:.72rem;font-weight:750;letter-spacing:.14em;}
.hero h1 {color:white!important;font-size:2rem;margin:.25rem 0;}
.hero p {color:#e4eff8!important;margin:.2rem 0;font-size:.95rem;}
.price-box {background:white;border:1px solid #dbe5ef;border-radius:14px;
            padding:1rem 1.15rem;min-height:130px;box-shadow:0 3px 12px #1b36520b;}
.price-label {font-size:.82rem;color:#62758c;font-weight:700;}
.price-value {font-size:1.9rem;color:#163354;font-weight:800;letter-spacing:-.035em;margin:.22rem 0;}
.price-sub {font-size:.76rem;color:#6a7d93;}
.stale-note {background:#fff6df;border:1px solid #efdfac;color:#765817;border-radius:9px;padding:.7rem .9rem;}
.info-note {background:#eaf2fb;border-left:4px solid #477cae;border-radius:8px;padding:.7rem .9rem;color:#2c4b6b;}
div[data-testid="stMetric"] {background:white;border:1px solid #dce5ef;border-radius:12px;padding:.8rem 1rem;}
div[data-testid="stTabs"] button {font-weight:700;font-size:.95rem;}
h2,h3,h4 {color:#183858!important;}
[data-testid="stSlider"] label,[data-testid="stNumberInput"] label {font-weight:650;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="kicker">CRUDE OIL · NEXT-SESSION FORECAST</div>
  <h1>Crude Oil Price Predictor</h1>
  <p>Today's market price, tomorrow's model estimate, and a simple what-if simulator.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_saved_state():
    for path in [DATA_FILE, MODEL_FILE, STATUS_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path.name}")
    df = standardize(pd.read_csv(DATA_FILE, parse_dates=["Date"]))
    artifacts = joblib.load(MODEL_FILE)
    with open(STATUS_FILE, "r", encoding="utf-8") as handle:
        status = json.load(handle)

    feature_names = list(artifacts.get("feature_names", []))
    if feature_names != EXPANDED_FEATURES:
        raise ValueError(
            "Saved model feature list does not match the current engine. "
            "Run the one-time model training workflow."
        )
    models = {}
    for horizon in HORIZONS:
        record = artifacts.get("models", {}).get(
            horizon, artifacts.get("models", {}).get(str(horizon))
        )
        if record is None or "pipeline" not in record:
            raise ValueError(f"Saved model missing horizon {horizon}.")
        models[horizon] = record

    matrix = compute_expanded_features(df)
    valid = matrix.notna().all(axis=1) & pd.to_numeric(df["WTI_Close"], errors="coerce").notna()
    valid_indices = np.flatnonzero(valid.to_numpy())
    if not len(valid_indices):
        raise ValueError("No complete feature row is available for the saved model.")
    idx = int(valid_indices[-1])
    latest_x = matrix.iloc[[idx]].loc[:, feature_names]
    current_close = float(df.loc[idx, "WTI_Close"])
    if not np.isfinite(current_close) or current_close <= 0:
        raise ValueError("Latest WTI close is invalid.")

    return {
        "df": df,
        "models": models,
        "status": status,
        "feature_names": feature_names,
        "feature_idx": idx,
        "feature_date": pd.Timestamp(df.loc[idx, "Date"]).normalize(),
        "market_date": pd.Timestamp(df["Date"].max()).normalize(),
        "current_close": current_close,
        "latest_x": latest_x,
    }


@st.cache_data(ttl=60, show_spinner=False)
def fetch_intraday_quote(ticker: str):
    """Fetch the latest available Yahoo Finance 5-minute bar; provider quote may be delayed."""
    if os.getenv("GITHUB_ACTIONS") or os.getenv("CI"):
        return None
    try:
        import yfinance as yf
        history = yf.Ticker(ticker).history(
            period="2d", interval="5m", prepost=True,
            auto_adjust=False, actions=False,
        )
        if history is None or history.empty or "Close" not in history.columns:
            return None
        history = history.dropna(subset=["Close"])
        if history.empty:
            return None
        last = history.iloc[-1]
        ts = pd.Timestamp(history.index[-1])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        timestamp = ts.tz_convert("UTC")
        now = pd.Timestamp.now(tz="UTC")
        age_minutes = max(0.0, (now - timestamp).total_seconds() / 60.0)
        return {
            "price": float(last["Close"]),
            "timestamp_utc": timestamp.isoformat(),
            "age_minutes": age_minutes,
            "open": float(last["Open"]) if pd.notna(last.get("Open")) else None,
            "high": float(last["High"]) if pd.notna(last.get("High")) else None,
            "low": float(last["Low"]) if pd.notna(last.get("Low")) else None,
            "volume": float(last["Volume"]) if pd.notna(last.get("Volume")) else None,
            "source": f"Yahoo Finance {ticker}",
        }
    except Exception:
        return None


def make_scenario_features(state, wti_shock_pct, dxy_move_pct,
                           inventory_change_kbbl, rig_change, macro_shocks=None):
    """Apply live quote adjustments plus user assumptions to a copy of the latest row."""
    df = state["df"]
    idx = state["feature_idx"]
    feature_names = state["feature_names"]
    macro_shocks = macro_shocks or {}

    live_wti_move = float(state.get("live_wti_move_pct", 0.0))
    live_dxy_move = float(state.get("live_dxy_move_pct", 0.0))
    all_zero = (
        abs(live_wti_move) < 1e-12 and abs(live_dxy_move) < 1e-12
        and abs(wti_shock_pct) < 1e-12 and abs(dxy_move_pct) < 1e-12
        and inventory_change_kbbl == 0 and rig_change == 0
        and all(abs(float(v)) < 1e-12 for v in macro_shocks.values())
    )
    if all_zero:
        base_x = state.get("latest_x")
        if base_x is None:
            base_x = state["models"][1]["latest_x"]
        return base_x.loc[:, feature_names].copy(), state["current_close"]

    scenario_df = df.copy()
    wti_factor = (1.0 + live_wti_move / 100.0) * (1.0 + wti_shock_pct / 100.0)
    dxy_factor = (1.0 + live_dxy_move / 100.0) * (1.0 + dxy_move_pct / 100.0)
    if wti_factor <= 0 or dxy_factor <= 0:
        raise ValueError("WTI and DXY assumptions must keep their price indices positive.")

    for column in ["WTI_Open", "WTI_High", "WTI_Low", "WTI_Close"]:
        value = pd.to_numeric(pd.Series([scenario_df.at[idx, column]]), errors="coerce").iloc[0]
        if pd.notna(value):
            scenario_df.at[idx, column] = float(value) * wti_factor

    dxy_value = pd.to_numeric(
        pd.Series([scenario_df.at[idx, "DXY_Close"]]), errors="coerce"
    ).iloc[0]
    if pd.isna(dxy_value):
        raise ValueError("DXY reference value is missing.")
    scenario_df.at[idx, "DXY_Close"] = float(dxy_value) * dxy_factor
    scenario_df["DXY_ret_1d"] = pd.to_numeric(
        scenario_df["DXY_Close"], errors="coerce"
    ).pct_change(fill_method=None)

    inv_column = "INV_US_Crude_Stocks_kbbl"
    rig_column = "RIG_US_Oil_Rigs"
    inventory = pd.to_numeric(pd.Series([scenario_df.at[idx, inv_column]]), errors="coerce").iloc[0]
    rigs = pd.to_numeric(pd.Series([scenario_df.at[idx, rig_column]]), errors="coerce").iloc[0]
    if pd.isna(inventory) or pd.isna(rigs):
        raise ValueError("Stored inventory or rig-count baseline is missing.")
    new_inventory = float(inventory) + inventory_change_kbbl
    new_rigs = float(rigs) + rig_change
    if new_inventory < 0 or new_rigs < 0:
        raise ValueError("Inventory and rig count cannot be negative.")
    scenario_df.at[idx, inv_column] = new_inventory
    scenario_df.at[idx, rig_column] = new_rigs

    feature_groups = {
        "commodity_basket": [
            "CMD_brent_lag1", "CMD_urals_lag1", "CMD_espo_lag1",
            "CMD_ttf_lag1", "CMD_wheat_lag1", "CMD_urea_lag1",
        ],
        "gpr": ["GPR_total_lag1m", "GPR_threats_lag1m", "GPR_acts_lag1m"],
        "tpu": ["TPU_lag1m"],
        "world_prod": ["EIA_world_prod_lag3m"],
    }
    for control, columns in feature_groups.items():
        shock = float(macro_shocks.get(control, 0.0))
        factor = 1.0 + shock / 100.0
        if factor <= 0:
            raise ValueError(f"{control} assumption must keep its input positive.")
        if abs(shock) < 1e-12:
            continue
        for column in columns:
            value = pd.to_numeric(pd.Series([scenario_df.at[idx, column]]), errors="coerce").iloc[0]
            if pd.isna(value) or value <= 0:
                raise ValueError(f"{column} does not have a valid saved value.")
            scenario_df.at[idx, column] = float(value) * factor

    recipes, _ = infer_recipes(df)
    candidates = feature_candidates(scenario_df)
    core_features = [
        "WTI_ret_1d", "WTI_ret_lag1", "WTI_ret_lag2",
        "WTI_ret_lag4", "WTI_ret_5d_mean", "WTI_vol_20d", "WTI_range_pct"
    ]
    missing = []
    for feature in core_features:
        recipe = recipes.get(feature)
        if recipe is None or recipe not in candidates.get(feature, {}):
            missing.append(feature)
        else:
            scenario_df[feature] = candidates[feature][recipe]
    if missing:
        raise ValueError("Could not recalculate inputs: " + ", ".join(missing))

    matrix = compute_expanded_features(scenario_df)
    x = matrix.loc[[idx], feature_names]
    if x.shape != (1, len(feature_names)) or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Scenario produced invalid model inputs.")
    latest_price = state["current_close"] * (1.0 + live_wti_move / 100.0)
    scenario_price = latest_price * (1.0 + wti_shock_pct / 100.0)
    return x, float(scenario_price)


def next_weekday(date_value):
    day = pd.Timestamp(date_value)
    if day.tzinfo is not None:
        day = day.tz_convert("America/New_York").tz_localize(None)
    day = day.normalize() + pd.Timedelta(days=1)
    while day.weekday() >= 5:
        day += pd.Timedelta(days=1)
    return day


try:
    state = load_saved_state()
except Exception as exc:
    st.error(f"Model or market data could not be loaded: {type(exc).__name__}: {exc}")
    st.stop()

models = state["models"]
wti_quote = fetch_intraday_quote("CL=F")
brent_quote = fetch_intraday_quote("BZ=F")
dxy_quote = fetch_intraday_quote("DX-Y.NYB")

def quote_or_close(quote, close):
    if (quote and np.isfinite(quote.get("price", np.nan)) and quote["price"] > 0
            and float(quote.get("age_minutes", 1e12)) <= 24 * 60):
        return float(quote["price"]), quote
    return float(close), None

wti_price, wti_quote_used = quote_or_close(wti_quote, state["current_close"])
brent_fallback = state["df"].loc[
    pd.to_numeric(state["df"].get("Brent_Close", pd.Series(dtype=float)), errors="coerce").notna(),
    "Brent_Close"
] if "Brent_Close" in state["df"].columns else pd.Series(dtype=float)
brent_saved = float(brent_fallback.iloc[-1]) if len(brent_fallback) else None
brent_price, brent_quote_used = quote_or_close(brent_quote, brent_saved) if brent_saved else (None, None)
dxy_saved = pd.to_numeric(state["df"].get("DXY_Close", pd.Series(dtype=float)), errors="coerce").dropna()
dxy_last = float(dxy_saved.iloc[-1]) if len(dxy_saved) else None
dxy_price, dxy_quote_used = quote_or_close(dxy_quote, dxy_last) if dxy_last else (None, None)

state["live_wti_move_pct"] = (wti_price / state["current_close"] - 1.0) * 100.0
state["live_dxy_move_pct"] = (
    (dxy_price / dxy_last - 1.0) * 100.0 if dxy_price is not None and dxy_last else 0.0
)

def quote_status(quote):
    if quote is None:
        return "Saved daily close · live source unavailable"
    stamp = pd.Timestamp(quote["timestamp_utc"]).tz_convert("America/New_York")
    age = quote["age_minutes"]
    freshness = "Recent provider quote" if age <= 30 else "Last available quote"
    return f"{freshness} · {stamp:%d %b %Y, %I:%M %p} ET · may be delayed"

tabs = st.tabs(["01 · Live price & tomorrow forecast", "02 · Hypothetical scenario"])

with tabs[0]:
    @st.fragment(run_every="60s")
    def market_panel():
        # Refresh data inside the fragment while keeping the model and page layout stable.
        live_wti, live_wti_used = quote_or_close(fetch_intraday_quote("CL=F"), state["current_close"])
        live_brent, live_brent_used = quote_or_close(fetch_intraday_quote("BZ=F"), brent_saved) if brent_saved else (None, None)
        live_dxy, live_dxy_used = quote_or_close(fetch_intraday_quote("DX-Y.NYB"), dxy_last) if dxy_last else (None, None)

        current_state = dict(state)
        current_state["live_wti_move_pct"] = (live_wti / state["current_close"] - 1.0) * 100.0
        current_state["live_dxy_move_pct"] = (
            (live_dxy / dxy_last - 1.0) * 100.0 if live_dxy is not None and dxy_last else 0.0
        )
        x, reference_price = make_scenario_features(current_state, 0.0, 0.0, 0, 0, {})
        one_day = models[1]
        predicted_return = float(one_day["pipeline"].predict(x)[0])
        predicted_price = reference_price * (1.0 + predicted_return)
        if not np.isfinite(predicted_price) or predicted_price <= 0:
            st.error("Next-session model returned an invalid price.")
            return

        current_stamp = pd.Timestamp.now(tz="America/New_York")
        if live_wti_used:
            current_stamp = pd.Timestamp(live_wti_used["timestamp_utc"]).tz_convert("America/New_York")
        else:
            current_stamp = state["feature_date"]
        target_date = next_weekday(current_stamp)

        st.subheader("Crude oil market right now")
        q1, q2, q3 = st.columns(3, gap="medium")
        with q1:
            st.markdown(
                f'<div class="price-box"><div class="price-label">WTI · current available quote</div>'
                f'<div class="price-value">USD {live_wti:,.2f}/bbl</div>'
                f'<div class="price-sub">{quote_status(live_wti_used)}</div></div>',
                unsafe_allow_html=True
            )
        with q2:
            brent_text = f"USD {live_brent:,.2f}/bbl" if live_brent is not None else "Unavailable"
            st.markdown(
                f'<div class="price-box"><div class="price-label">Brent · current available quote</div>'
                f'<div class="price-value">{brent_text}</div>'
                f'<div class="price-sub">{quote_status(live_brent_used) if live_brent is not None else "No saved benchmark value"}</div></div>',
                unsafe_allow_html=True
            )
        with q3:
            st.markdown(
                f'<div class="price-box"><div class="price-label">Model estimate · next session close</div>'
                f'<div class="price-value">USD {predicted_price:,.2f}/bbl</div>'
                f'<div class="price-sub">{target_date:%d %b %Y} · {predicted_return * 100:+.2f}% vs current quote</div></div>',
                unsafe_allow_html=True
            )

        if live_wti_used is None:
            st.markdown(
                '<div class="stale-note">Live quote could not be retrieved. The dashboard is showing the most recent saved daily close instead.</div>',
                unsafe_allow_html=True
            )
        st.caption(
            f"Model input data date: {state['feature_date'].date()} · "
            f"Model: {one_day['model_type']} · "
            f"Chronological holdout MAE: {one_day['test_mae'] * 100:.2f} percentage points."
        )
        if not one_day["beats_zero"]:
            st.markdown(
                '<div class="stale-note">This model is experimental and has not beaten the unchanged-price benchmark in the latest chronological test. Treat tomorrow’s value as a research estimate, not a guaranteed forecast.</div>',
                unsafe_allow_html=True
            )

        history = state["df"][["Date", "WTI_Close"]].dropna().tail(12).copy()
        history["Date"] = pd.to_datetime(history["Date"])
        quote_day = pd.Timestamp(current_stamp).tz_localize(None).normalize() if getattr(current_stamp, "tzinfo", None) else pd.Timestamp(current_stamp).normalize()
        if quote_day <= history["Date"].max():
            history.loc[history.index[-1], "WTI_Close"] = live_wti
            current_day = history.iloc[-1]["Date"]
        else:
            history = pd.concat(
                [history, pd.DataFrame([{"Date": quote_day, "WTI_Close": live_wti}])],
                ignore_index=True
            )
            current_day = quote_day

        chart = history.rename(columns={"WTI_Close": "WTI close"})
        chart = pd.concat([
            chart,
            pd.DataFrame([{"Date": target_date, "WTI close": predicted_price}])
        ], ignore_index=True)
        fig, ax = plt.subplots(figsize=(11.5, 4.2))
        ax.plot(chart["Date"].iloc[:-1], chart["WTI close"].iloc[:-1],
                marker="o", linewidth=2, label="Recent / current WTI")
        ax.plot([current_day, target_date], [live_wti, predicted_price],
                linestyle="--", marker="o", linewidth=2.2, label="Next-session model estimate")
        ax.scatter([target_date], [predicted_price], marker="*", s=190,
                   label="Predicted next-session close", zorder=5)
        ax.set_ylabel("USD per barrel")
        ax.set_xlabel("Market date")
        ax.grid(alpha=.22)
        ax.legend(loc="best")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        st.caption("Solid line = recent/current prices · Dashed line = model estimate for the next scheduled weekday session. Exchange holidays can change the next session date.")

    market_panel()

with tabs[1]:
    st.subheader("What-if: change the market assumptions")
    st.write(
        "Move the eight assumptions below. The model recalculates its 64 inputs on a temporary copy "
        "and immediately shows how the next-session WTI estimate changes. The saved model is not retrained."
    )

    control_state = dict(state)
    control_state["live_wti_move_pct"] = state["live_wti_move_pct"]
    control_state["live_dxy_move_pct"] = state["live_dxy_move_pct"]

    with st.container(border=True):
        st.markdown("**Market assumptions**")
        left, right = st.columns(2, gap="large")
        with left:
            wti_shock = st.slider("1 · WTI price change (%)", -15.0, 15.0, 0.0, 0.5)
            dxy_move = st.slider("2 · US Dollar Index (DXY) change (%)", -5.0, 5.0, 0.0, 0.25)
            inventory_million = st.slider("3 · US crude inventory change (million barrels)", -5.0, 5.0, 0.0, 0.25)
            rig_change = st.slider("4 · Active US oil rigs change", -50, 50, 0, 1)
        with right:
            commodity_move = st.slider("5 · Commodity market basket change (%)", -20.0, 20.0, 0.0, 0.5)
            gpr_move = st.slider("6 · Geopolitical Risk Index (GPR) change (%)", -50.0, 100.0, 0.0, 5.0)
            tpu_move = st.slider("7 · Trade Policy Uncertainty (TPU) change (%)", -50.0, 100.0, 0.0, 5.0)
            production_move = st.slider("8 · Global oil production change (%)", -5.0, 5.0, 0.0, 0.25)

    st.caption(
        "Commodity basket changes Brent/Urals/ESPO crude, European gas, wheat and urea together. "
        "GPR changes total geopolitical risk, threats and acts indices together. GPR/TPU are historical lagged indices, not live news."
    )

    macro_shocks = {
        "commodity_basket": commodity_move,
        "gpr": gpr_move,
        "tpu": tpu_move,
        "world_prod": production_move,
    }
    try:
        scenario_x, scenario_reference = make_scenario_features(
            control_state, wti_shock, dxy_move,
            int(round(inventory_million * 1000)), int(rig_change), macro_shocks
        )
        model = models[1]
        scenario_return = float(model["pipeline"].predict(scenario_x)[0])
        scenario_price = scenario_reference * (1.0 + scenario_return)

        baseline_x, baseline_reference = make_scenario_features(
            control_state, 0.0, 0.0, 0, 0, {}
        )
        baseline_return = float(model["pipeline"].predict(baseline_x)[0])
        baseline_price = baseline_reference * (1.0 + baseline_return)

        if not np.isfinite(scenario_price) or scenario_price <= 0:
            raise ValueError("The scenario model returned an invalid price.")

        st.divider()
        out1, out2, out3 = st.columns(3)
        out1.metric("Current WTI quote", f"USD {baseline_reference:,.2f}/bbl")
        out2.metric(
            "Scenario next-session estimate",
            f"USD {scenario_price:,.2f}/bbl",
            delta=f"USD {scenario_price - baseline_price:+,.2f} vs current case"
        )
        out3.metric(
            "Scenario expected move",
            f"{scenario_return * 100:+.2f}%",
            delta=f"{(scenario_return - baseline_return) * 100:+.2f} pp vs current case"
        )

        fig, ax = plt.subplots(figsize=(9.5, 3.4))
        ax.bar(
            ["Current assumptions", "What-if assumptions"],
            [baseline_price, scenario_price],
            width=.55
        )
        ax.set_ylabel("Next-session WTI estimate · USD/bbl")
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    except Exception as exc:
        st.error(f"Scenario prediction unavailable: {type(exc).__name__}: {exc}")

    st.caption(
        "These are hypothetical model inputs, not causal economic laws. The result is a research estimate, "
        "not a guaranteed market price. No retraining happens when you move a control."
    )
