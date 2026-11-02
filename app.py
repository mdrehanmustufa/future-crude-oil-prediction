from pathlib import Path
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
    page_title="Future Crude Oil Prediction",
    page_icon="🛢️",
    layout="wide"
)

st.markdown("""
<style>
.block-container {max-width: 1450px; padding-top: 1.2rem;}
.hero {padding: 1.2rem 1.5rem; border: 1px solid #dbe3ef;
       border-radius: 14px; background: linear-gradient(110deg,#eff6ff,#fff);
       margin-bottom: 1rem;}
.hero h1 {color: #17365d; margin-bottom: .2rem;}
.muted {color: #64748b; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>Future Crude Oil Prediction</h1>
  <div>WTI multi-horizon market forecast and hypothetical scenario simulator</div>
  <p class="muted">Research and educational use only. Not investment or trading advice.</p>
</div>
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

    return {
        "df": df, "models": models, "status": status,
        "feature_date": feature_date, "market_date": market_date,
        "current_close": current_close, "feature_names": feature_names,
        "feature_idx": feature_idx,
    }


def make_scenario_features(state, wti_shock_pct, dxy_move_pct,
                           inventory_change_kbbl, rig_change):
    df = state["df"]
    idx = state["feature_idx"]
    feature_names = state["feature_names"]

    unchanged = (
        abs(wti_shock_pct) < 1e-12
        and abs(dxy_move_pct) < 1e-12
        and inventory_change_kbbl == 0
        and rig_change == 0
    )
    if unchanged:
        baseline_x = state["models"][1]["latest_x"].copy()
        return baseline_x.loc[:, feature_names], state["current_close"]

    scenario_df = df.copy()
    wti_factor = 1.0 + wti_shock_pct / 100.0
    dxy_factor = 1.0 + dxy_move_pct / 100.0
    if wti_factor <= 0 or dxy_factor <= 0:
        raise ValueError("WTI and DXY scenario prices must remain positive.")

    for column in ["WTI_Open", "WTI_High", "WTI_Low", "WTI_Close"]:
        value = pd.to_numeric(
            pd.Series([scenario_df.at[idx, column]]), errors="coerce"
        ).iloc[0]
        if pd.notna(value):
            scenario_df.at[idx, column] = float(value) * wti_factor

    dxy_value = pd.to_numeric(
        pd.Series([scenario_df.at[idx, "DXY_Close"]]), errors="coerce"
    ).iloc[0]
    if pd.isna(dxy_value):
        raise ValueError("DXY value is missing on the forecast feature date.")
    scenario_df.at[idx, "DXY_Close"] = float(dxy_value) * dxy_factor
    scenario_df["DXY_ret_1d"] = pd.to_numeric(
        scenario_df["DXY_Close"], errors="coerce"
    ).pct_change(fill_method=None)

    inventory_column = "INV_US_Crude_Stocks_kbbl"
    rigs_column = "RIG_US_Oil_Rigs"
    inventory = pd.to_numeric(
        pd.Series([scenario_df.at[idx, inventory_column]]), errors="coerce"
    ).iloc[0]
    rigs = pd.to_numeric(
        pd.Series([scenario_df.at[idx, rigs_column]]), errors="coerce"
    ).iloc[0]
    if pd.isna(inventory) or pd.isna(rigs):
        raise ValueError("Inventory or rig-count baseline is missing.")

    new_inventory = float(inventory) + inventory_change_kbbl
    new_rigs = float(rigs) + rig_change
    if new_inventory < 0 or new_rigs < 0:
        raise ValueError("Scenario cannot create negative inventory or rig count.")
    scenario_df.at[idx, inventory_column] = new_inventory
    scenario_df.at[idx, rigs_column] = new_rigs

    recipes, _ = infer_recipes(df)
    candidates = feature_candidates(scenario_df)
    core_features = [
        "WTI_ret_1d", "WTI_ret_lag1", "WTI_ret_lag2",
        "WTI_ret_lag4", "WTI_ret_5d_mean", "WTI_vol_20d", "WTI_range_pct"
    ]
    unmatched = []
    for feature in core_features:
        recipe = recipes.get(feature)
        if recipe is None or recipe not in candidates.get(feature, {}):
            unmatched.append(feature)
        else:
            scenario_df[feature] = candidates[feature][recipe]
    if unmatched:
        raise ValueError("Could not reproduce historical feature formulas: " + ", ".join(unmatched))

    scenario_matrix = compute_expanded_features(scenario_df)
    scenario_x = scenario_matrix.loc[[idx], feature_names]
    if scenario_x.shape != (1, len(feature_names)):
        raise ValueError("Scenario feature matrix has an unexpected shape.")
    if not np.isfinite(scenario_x.to_numpy(dtype=float)).all():
        raise ValueError("Scenario produced incomplete/non-finite model features.")
    return scenario_x, float(scenario_df.at[idx, "WTI_Close"])


try:
    state = load_saved_state()
except Exception as exc:
    st.error(f"Saved forecast files could not be loaded: {type(exc).__name__}: {exc}")
    st.info("The refresh workflow must first create the CSV, model_artifacts.joblib and data_refresh_status.json.")
    st.stop()

df = state["df"]
models = state["models"]
status = state["status"]
feature_date = state["feature_date"]
market_date = state["market_date"]
current_close = state["current_close"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("WTI reference close", f"USD {current_close:,.2f}")
m2.metric("Forecast feature date", str(feature_date.date()))
m3.metric("Latest market date", str(market_date.date()))
m4.metric("Model input features", str(len(state["feature_names"])))

if market_date > feature_date:
    st.warning(
        f"Latest market row is {market_date.date()}, but the latest complete "
        f"feature row is {feature_date.date()}. Forecast uses the complete row."
    )

labels = {
    1: "Next trading session", 4: "4 trading sessions",
    5: "5 trading sessions", 10: "10 trading sessions",
    20: "20 trading sessions"
}
market_tab, scenario_tab = st.tabs(["1. Real Market Forecast", "2. What-if Scenario"])

with market_tab:
    st.subheader("WTI multi-horizon reference forecast")
    rows = []
    for horizon in HORIZONS:
        record = models[horizon]
        rows.append({
            "Horizon": labels[horizon],
            "Model": record["model_type"],
            "Forecast price (USD)": record["price"],
            "Expected return (%)": record["return"] * 100,
            "Indicative lower band": record["low_price"],
            "Indicative upper band": record["high_price"],
            "Holdout MAE (pp)": record["test_mae"] * 100,
            "Zero-baseline MAE (pp)": record["zero_mae"] * 100,
            "Beats zero baseline": record["beats_zero"]
        })
    forecast_df = pd.DataFrame(rows)
    st.dataframe(
        forecast_df.style.format({
            "Forecast price (USD)": "{:,.2f}",
            "Expected return (%)": "{:+.3f}%",
            "Indicative lower band": "{:,.2f}",
            "Indicative upper band": "{:,.2f}",
            "Holdout MAE (pp)": "{:.3f}",
            "Zero-baseline MAE (pp)": "{:.3f}"
        }),
        width="stretch", hide_index=True
    )
    st.caption(
        "Horizons count observed trading sessions, not calendar days. "
        "Price bands are indicative residual bands, not calibrated confidence intervals."
    )
    weak_horizons = [str(h) for h in HORIZONS if not models[h]["beats_zero"]]
    if weak_horizons:
        st.warning(
            "On the current chronological holdout, the model does not beat the "
            "zero-return benchmark at horizon(s): " + ", ".join(weak_horizons) +
            ". ExtraTrees results are experimental, not proven improvements."
        )

    st.subheader("Recent WTI close history")
    history = df[["Date", "WTI_Close"]].dropna().tail(252)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(history["Date"], history["WTI_Close"], label="WTI close")
    ax.axvline(feature_date, linestyle="--", label="Forecast feature date")
    ax.set_ylabel("USD per barrel")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


with scenario_tab:
    st.subheader("Hypothetical market assumptions")
    st.write(
        "Change the latest complete market row. The app recalculates dependent "
        "features on a separate copy; saved baseline data and model files are untouched."
    )
    left, right = st.columns(2)
    with left:
        wti_shock = st.slider(
            "Hypothetical WTI price shock (%)",
            min_value=-20.0, max_value=20.0, value=0.0, step=0.5
        )
        dxy_move = st.slider(
            "Hypothetical DXY movement (%)",
            min_value=-10.0, max_value=10.0, value=0.0, step=0.25
        )
    with right:
        inventory_change = st.number_input(
            "US crude inventory change (kbbl)",
            min_value=-50000, max_value=50000, value=0, step=1000
        )
        rig_change = st.number_input(
            "US oil rig-count change",
            min_value=-150, max_value=150, value=0, step=5
        )

    chosen_horizon = st.selectbox(
        "Forecast horizon", HORIZONS, format_func=lambda h: labels[h]
    )

    try:
        scenario_x, scenario_close = make_scenario_features(
            state, wti_shock, dxy_move, int(inventory_change), int(rig_change)
        )
        baseline = models[chosen_horizon]
        baseline_return = float(baseline["return"])
        baseline_price = float(baseline["price"])
        scenario_return = float(baseline["pipeline"].predict(scenario_x)[0])
        scenario_price = scenario_close * (1.0 + scenario_return)

        if not np.isfinite(scenario_return) or not np.isfinite(scenario_price) or scenario_price <= 0:
            raise ValueError("Model returned an invalid scenario forecast.")

        delta_return_pp = (scenario_return - baseline_return) * 100
        delta_price = scenario_price - baseline_price
        st.divider()
        a, b, c = st.columns(3)
        a.metric("Scenario WTI reference price", f"USD {scenario_close:,.2f}")
        b.metric(
            "Scenario forecast price", f"USD {scenario_price:,.2f}",
            delta=f"USD {delta_price:+,.2f} vs baseline"
        )
        c.metric(
            "Scenario expected return", f"{scenario_return * 100:+.3f}%",
            delta=f"{delta_return_pp:+.3f} pp vs baseline"
        )

        comparison = pd.DataFrame([
            {
                "Output": "Baseline", "Reference WTI close": current_close,
                "Expected return (%)": baseline_return * 100,
                "Forecast price": baseline_price
            },
            {
                "Output": "Hypothetical scenario", "Reference WTI close": scenario_close,
                "Expected return (%)": scenario_return * 100,
                "Forecast price": scenario_price
            }
        ])
        st.dataframe(
            comparison.style.format({
                "Reference WTI close": "{:,.2f}",
                "Expected return (%)": "{:+.3f}%",
                "Forecast price": "{:,.2f}"
            }),
            width="stretch", hide_index=True
        )
        st.caption(
            f"Horizon: {chosen_horizon} observed trading sessions. "
            "Price change includes the direct WTI reference-price shock; compare "
            "expected returns to isolate the model's changed response."
        )
        if chosen_horizon in (5, 10, 20):
            st.warning(
                "This horizon uses an experimental ExtraTrees model. "
                "Its latest holdout MAE did not beat the zero-return baseline."
            )
        else:
            st.info(
                "This horizon uses a zero-return baseline. It does not create a "
                "directional return forecast when assumptions change."
            )
    except Exception as exc:
        st.error(f"Scenario calculation unavailable: {type(exc).__name__}: {exc}")

    st.caption(
        "This is a conditional model scenario, not a causal economic simulation. "
        "Feature relationships are not proof that changing an input causes a price "
        "movement. Geopolitical risk is context-only because it is not a trained input."
    )


with st.expander("Data and model diagnostics"):
    st.subheader("Saved automation run")
    st.write("Last successful refresh (UTC):", status.get("last_successful_run_utc", "Not recorded"))
    st.write("Rows in saved dataset:", status.get("dataset_rows", len(df)))
    st.write("WTI source:", status.get("wti_source_status", "Not recorded"))
    st.write("DXY source:", status.get("dxy_source_status", "Not recorded"))
    st.write("Inventory source:", status.get("inventory_source_status", "Not recorded"))
    st.write("Rig count source:", status.get("rig_count_status", "Not recorded"))
    st.write("EIA API configured:", status.get("eia_api_configured", False))

    st.subheader("Chronological holdout performance")
    diagnostic_rows = []
    for horizon in HORIZONS:
        record = models[horizon]
        diagnostic_rows.append({
            "Horizon": horizon, "Model": record["model_type"],
            "Model MAE (pp)": record["test_mae"] * 100,
            "Zero baseline MAE (pp)": record["zero_mae"] * 100,
            "Beats zero baseline": record["beats_zero"],
            "Holdout rows": record["test_rows"]
        })
    st.dataframe(
        pd.DataFrame(diagnostic_rows).style.format({
            "Model MAE (pp)": "{:.3f}",
            "Zero baseline MAE (pp)": "{:.3f}"
        }),
        width="stretch", hide_index=True
    )
    st.caption(
        "The app loads saved model artifacts. It does not refresh market data or "
        "train models at runtime. Scheduled GitHub Actions handles those tasks."
    )
