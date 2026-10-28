import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from forecast_engine import (
    HORIZONS, PRIOR_WALK_FORWARD, load_base, refresh_market_history,
    train_hybrid_horizons,
)

st.set_page_config(
    page_title="Future Crude Oil Prediction",
    page_icon="🛢️",
    layout="wide",
)

st.markdown("""
<style>
.block-container {max-width: 1450px; padding-top: 1.2rem;}
.hero {padding: 1.25rem 1.5rem; border: 1px solid #dbe3ef; border-radius: 14px;
       background: linear-gradient(110deg, #eff6ff, #ffffff); margin-bottom: 1rem;}
.hero h1 {color: #17365d; margin-bottom: .2rem;}
.small-muted {color: #64748b; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>Future Crude Oil Prediction</h1>
  <div>WTI multi-horizon forecast · Data checks · Experimental machine learning</div>
  <p class="small-muted">Research and educational use only. Forecasts are estimates, not guarantees or trading advice.</p>
</div>
""", unsafe_allow_html=True)

if not os.getenv("EIA_API_KEY"):
    try:
        if st.secrets.get("EIA_API_KEY"):
            os.environ["EIA_API_KEY"] = str(st.secrets["EIA_API_KEY"])
    except Exception:
        pass


@st.cache_resource(ttl=6 * 60 * 60)
def load_and_fit():
    base, base_name = load_base()
    updated, source_status = refresh_market_history(base)
    fitted = train_hybrid_horizons(updated)
    return {
        "base_name": base_name,
        "source": source_status,
        "loaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        **fitted,
    }


with st.sidebar:
    st.header("Controls")
    if st.button("↻ Refresh data and refit models", width="stretch"):
        load_and_fit.clear()
        st.rerun()
    st.caption("Refresh/refit happens on app access after the 6-hour cache expires, or when you click Refresh. This is not a background cron job.")
    st.divider()
    st.markdown("**Data sources**")
    st.markdown("- WTI: Yahoo Finance `CL=F` (source overlap must pass)")
    st.markdown("- DXY: Yahoo Finance `DX-Y.NYB` (source overlap must pass)")
    st.markdown("- Crude inventory: optional EIA API if `EIA_API_KEY` is configured")
    st.markdown("- Oil rigs: last known oil-specific value carried forward")

try:
    with st.spinner("Loading historical data, checking source overlap and fitting horizon models..."):
        state = load_and_fit()
except Exception as exc:
    st.error(f"Forecast initialization failed: {type(exc).__name__}: {exc}")
    st.info("Confirm that `Final_Dataset_AI_dxy_fixed.csv` or `Final_Dataset_AI.csv` is in the project folder, or that the GitHub raw fallback is reachable. Check requirements.txt and the error above.")
    st.stop()

df = state["df"]
source = state["source"]
feature_date = pd.Timestamp(state["feature_date"])
market_date = pd.Timestamp(state["market_date"])
current_close = float(state["current_close"])
models = state["models"]

if source.get("wti", "").startswith("Yahoo CL=F:") and "failed" not in source.get("wti", "").lower():
    st.success(f"WTI source check: {source.get('wti')}")
else:
    st.warning(f"WTI source status: {source.get('wti', 'unknown')}")
if source.get("dxy", "").startswith("Yahoo DX-Y.NYB:") and "failed" not in source.get("dxy", "").lower():
    st.success(f"DXY source check: {source.get('dxy')}")
else:
    st.warning(f"DXY source status: {source.get('dxy', 'unknown')}")

lag_days = (market_date - feature_date).days
if lag_days > 0:
    st.warning(f"Latest market row is {market_date.date()}, but the latest complete 53-feature row is {feature_date.date()}. Forecasts use the complete feature row, not the newer incomplete row.")
if lag_days > 3:
    st.error("Feature data is more than three calendar days behind the latest market row. Treat forecasts as stale until inputs are refreshed/repaired.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("WTI reference close", f"${current_close:,.2f}")
m2.metric("Forecast feature date", str(feature_date.date()))
m3.metric("Latest market date", str(market_date.date()))
m4.metric("Model input features", str(state["feature_count"]))

tab_forecast, tab_scenario, tab_checks = st.tabs(["Market Forecast", "What-if Scenario", "Data & Model Checks"])
labels = {1: "Next session", 4: "4 trading sessions", 5: "5 sessions (~1 week)", 10: "10 sessions (~2 weeks)", 20: "20 sessions (~1 month)"}

with tab_forecast:
    st.subheader("Multi-horizon reference forecasts")
    rows = []
    for h in HORIZONS:
        model = models[h]
        rows.append({
            "Horizon": labels[h],
            "Model status": model["model_type"],
            "Reference forecast price": model["price"],
            "Predicted cumulative return (%)": model["return"] * 100,
            "Indicative lower band": model["low_price"],
            "Indicative upper band": model["high_price"],
            "Holdout MAE (pp)": model["test_mae"] * 100,
            "Zero-return MAE (pp)": model["zero_mae"] * 100,
            "Beats zero baseline": model["beats_zero"],
        })
    forecast_table = pd.DataFrame(rows)
    st.dataframe(forecast_table.style.format({
        "Reference forecast price": "${:,.2f}",
        "Predicted cumulative return (%)": "{:+.3f}%",
        "Indicative lower band": "${:,.2f}",
        "Indicative upper band": "${:,.2f}",
        "Holdout MAE (pp)": "{:.3f}",
        "Zero-return MAE (pp)": "{:.3f}",
    }), width="stretch", hide_index=True)
    st.caption("Horizon is observed trading sessions, not calendar days. Bands use a chronological holdout residual distribution and are indicative—not calibrated confidence intervals.")
    st.warning("The ExtraTrees outputs for 5, 10 and 20 sessions are experimental. In earlier experiments, their MAE improvements over the zero-return benchmark were small and did not consistently carry over to the earlier held-out test.")

    st.subheader("Recent WTI close history")
    history = df[["Date", "WTI_Close"]].dropna().tail(252)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(history["Date"], history["WTI_Close"], label="WTI close")
    ax.axvline(feature_date, linestyle="--", label="Forecast feature cutoff")
    ax.set_ylabel("USD per barrel")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

with tab_scenario:
    st.subheader("Scenario controls")
    st.info("Prediction-changing sliders are intentionally disabled for now. The deployed tree models are experimental, and changing feature values independently can create impossible combinations (for example, changing price momentum without recalculating its dependent lags, volatility and drawdown).")
    chosen_h = st.selectbox("Forecast horizon", HORIZONS, format_func=lambda h: labels[h])
    chosen_model = models[chosen_h]
    st.metric("Current model type", chosen_model["model_type"])
    st.metric("Reference forecast price", f"${chosen_model['price']:,.2f}")
    st.slider("WTI volatility (display only; not connected to forecast)", 0.0, 0.15, 0.03, disabled=True, help="Disabled intentionally; this input does not alter model predictions.")
    st.slider("DXY momentum (display only; not connected to forecast)", -0.10, 0.10, 0.0, disabled=True, help="Disabled intentionally; this input does not alter model predictions.")
    st.caption("Feature importance and correlations are not causal effects. We will only enable scenario calculations after a consistent scenario feature-recalculation method is built and tested.")

with tab_checks:
    st.subheader("Current run: chronological holdout diagnostics")
    diag = state["diagnostics"].copy()
    st.dataframe(diag.style.format({"Model MAE (pp)": "{:.3f}", "Zero-return MAE (pp)": "{:.3f}"}), width="stretch", hide_index=True)
    if not diag["Beats zero baseline"].all():
        st.info("At least one horizon does not beat the zero-return baseline on this holdout. The app reports this rather than masking it.")

    st.subheader("Previous four-fold walk-forward experiment")
    prior_rows = []
    for h in HORIZONS:
        r = PRIOR_WALK_FORWARD[h]
        prior_rows.append({
            "Horizon sessions": h,
            "Historical candidate": r["candidate"],
            "Candidate mean MAE (pp)": r["candidate_mae_pp"],
            "Zero-return mean MAE (pp)": r["zero_mae_pp"],
            "Folds beating zero": f"{r['folds_beating_zero']}/{r['folds']}",
            "Evidence status": "Historical experiment; not this run's test result",
        })
    st.dataframe(pd.DataFrame(prior_rows).style.format({"Candidate mean MAE (pp)": "{:.4f}", "Zero-return mean MAE (pp)": "{:.4f}"}), width="stretch", hide_index=True)

    st.subheader("Source and freshness status")
    source_rows = [
        {"Check": "Historical file", "Result": state["base_name"]},
        {"Check": "WTI source", "Result": source.get("wti", "")},
        {"Check": "DXY source", "Result": source.get("dxy", "")},
        {"Check": "Inventory", "Result": source.get("inventory", "")},
        {"Check": "Rig count", "Result": source.get("rig_count", "")},
        {"Check": "Base market date", "Result": source.get("base_date", "")},
        {"Check": "Refresh cutoff date", "Result": source.get("cutoff_date", "")},
        {"Check": "Latest market date", "Result": source.get("latest_market_date", str(market_date.date()))},
        {"Check": "Latest complete feature date", "Result": source.get("latest_feature_date", str(feature_date.date()))},
        {"Check": "Rows after refresh", "Result": source.get("rows_after_refresh", len(df))},
        {"Check": "Refreshed/refit at (UTC)", "Result": state["loaded_at"]},
        {"Check": "Features returned in frozen order", "Result": f"{state['feature_count']} ordered columns"},
    ]
    source_status_df = pd.DataFrame(source_rows)
    source_status_df["Result"] = source_status_df["Result"].astype(str)
    st.dataframe(source_status_df, width="stretch", hide_index=True)
    if source.get("formula_notes"):
        st.subheader("Historical feature-formula checks for refreshed rows")
        st.dataframe(pd.DataFrame([{"Feature": f, "Formula check": note} for f, note in source["formula_notes"].items()]), width="stretch", hide_index=True)
    st.caption("Refresh is cache-driven (six hours or button click), not an unattended scheduled job. Inventory may stay at its last-known value without an EIA key; the oil-specific rig count is carried forward.")
