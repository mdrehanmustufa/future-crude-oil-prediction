from pathlib import Path
import json

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "Final_Dataset_AI_dxy_fixed.csv"
MODEL_FILE = ROOT / "model_artifacts.joblib"
STATUS_FILE = ROOT / "data_refresh_status.json"

st.set_page_config(page_title="WTI Prediction vs Actual", page_icon="📊", layout="wide")
st.title("WTI · Price History & Prediction Comparison")
st.caption("25 October 2026 onwards · USD per barrel")

missing = [p.name for p in [DATA_FILE, MODEL_FILE, STATUS_FILE] if not p.exists()]
if missing:
    st.error("Required project files missing: " + ", ".join(missing))
    st.stop()

df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["WTI_Close"] = pd.to_numeric(df["WTI_Close"], errors="coerce")
df = (
    df.dropna(subset=["Date", "WTI_Close"])
      .sort_values("Date")
      .drop_duplicates("Date", keep="last")
      .reset_index(drop=True)
)
if len(df) < 3:
    st.error("Not enough valid WTI price observations to draw the comparison.")
    st.stop()

with open(STATUS_FILE, "r", encoding="utf-8") as f:
    status = json.load(f)
artifacts = joblib.load(MODEL_FILE)
record = artifacts.get("models", {}).get(1, artifacts.get("models", {}).get("1"))
if record is None:
    st.error("One-session forecast artifact is unavailable.")
    st.stop()

# Transparent one-session benchmark replay: today's estimate is yesterday's
# completed close. Shift is performed BEFORE filtering so the first displayed
# date can use the preceding available session as its estimate.
df["Predicted_Close"] = df["WTI_Close"].shift(1)
df["Absolute_Difference"] = (df["WTI_Close"] - df["Predicted_Close"]).abs()
df["Absolute_Percent_Error"] = (
    df["Absolute_Difference"] / df["WTI_Close"].abs() * 100
)

latest_date = pd.Timestamp(df["Date"].max()).normalize()
start_date = pd.Timestamp(year=latest_date.year, month=latest_date.month, day=25)
window = df[(df["Date"] >= start_date) & (df["Date"] <= latest_date)].copy()
comparison = window.dropna(subset=["Predicted_Close"]).copy()

if window.empty:
    st.warning("No observations found for the requested date window in the latest dataset.")
    st.stop()

latest_close = float(df.iloc[-1]["WTI_Close"])
latest_model_price = float(record.get("price", latest_close))
latest_model_type = str(record.get("model_type", "Not recorded"))
m1, m2, m3 = st.columns(3)
m1.metric("Latest saved WTI close", f"${latest_close:,.2f}/bbl", str(latest_date.date()))
m2.metric("Next-session reference estimate", f"${latest_model_price:,.2f}/bbl")
m3.metric("Completed sessions compared", str(len(comparison)))

st.subheader("1. Actual WTI price history")
fig1, ax1 = plt.subplots(figsize=(11, 4.2))
ax1.plot(window["Date"], window["WTI_Close"], color="green", marker="o",
         linewidth=2.4, markersize=5, label="Actual WTI close")
ax1.set_ylabel("Price (USD/barrel)")
ax1.set_xlabel("Trading date")
ax1.grid(True, alpha=0.25)
ax1.legend(loc="best")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
fig1.autofmt_xdate()
fig1.tight_layout()
st.pyplot(fig1, use_container_width=True)
plt.close(fig1)

st.subheader("2. Predicted vs Actual")
st.caption("Green = actual closing price · Red = one-session benchmark estimate")
fig2, ax2 = plt.subplots(figsize=(11, 4.5))
ax2.plot(comparison["Date"], comparison["WTI_Close"], color="green",
         marker="o", linewidth=2.4, markersize=5, label="Actual price")
ax2.plot(comparison["Date"], comparison["Predicted_Close"], color="red",
         marker="o", linewidth=2.2, linestyle="--", markersize=5,
         label="Predicted / previous-close benchmark")
ax2.set_ylabel("Price (USD/barrel)")
ax2.set_xlabel("Trading date")
ax2.grid(True, alpha=0.25)
ax2.legend(loc="best")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
fig2.autofmt_xdate()
fig2.tight_layout()
st.pyplot(fig2, use_container_width=True)
plt.close(fig2)

st.subheader("Prediction error by trading session")
table = comparison[[
    "Date", "Predicted_Close", "WTI_Close",
    "Absolute_Difference", "Absolute_Percent_Error"
]].rename(columns={
    "Date": "Date",
    "Predicted_Close": "Predicted (USD)",
    "WTI_Close": "Actual (USD)",
    "Absolute_Difference": "Absolute difference (USD)",
    "Absolute_Percent_Error": "Absolute error (%)",
})
table["Date"] = table["Date"].dt.strftime("%d %b %Y")
st.dataframe(
    table.style.format({
        "Predicted (USD)": "${:,.2f}",
        "Actual (USD)": "${:,.2f}",
        "Absolute difference (USD)": "${:,.2f}",
        "Absolute error (%)": "{:.2f}%"
    }),
    use_container_width=True,
    hide_index=True
)

if not comparison.empty:
    mean_error = float(comparison["Absolute_Percent_Error"].mean())
    mean_abs_error = float(comparison["Absolute_Difference"].mean())
    e1, e2 = st.columns(2)
    e1.metric("Mean absolute percentage error (MAPE)", f"{mean_error:.2f}%")
    e2.metric("Mean absolute price difference", f"${mean_abs_error:.2f}/bbl")

st.info(
    "How to read this chart: for each trading session, the estimate is the previous "
    "completed WTI close. This is a historical zero-return benchmark replay. It is "
    "not a directional prediction generated by a successfully validated AI model, "
    "so the displayed MAPE must not be presented as AI model accuracy."
)

st.subheader("Model limitations")
st.markdown(
    "- **Sudden geopolitical shocks:** wars, military strikes, sanctions, trade restrictions "
    "or unexpected statements by political leaders can move crude prices sharply.\n"
    "- **News timing and interpretation:** geopolitical indicators and news-derived signals "
    "may be delayed, incomplete or unable to capture the full impact of a new event.\n"
    "- **Unusual market conditions:** a model trained on historical patterns may perform "
    "poorly when supply routes, production, inventories or market sentiment change suddenly.\n"
    "- **Further validation required:** performance must be tested with a larger, strictly "
    "out-of-sample walk-forward backtest, including separate normal-market and shock periods."
)

st.caption(
    f"Data window: {start_date.date()} to {latest_date.date()} · "
    f"Model artifact type: {latest_model_type} · "
    f"Latest dataset refresh (UTC): {status.get('last_successful_run_utc', 'not recorded')}"
)
st.caption(
    "Source: project WTI dataset. Dates without a recorded trading-session observation "
    "are not filled or fabricated. Research prototype only; not investment advice."
)
