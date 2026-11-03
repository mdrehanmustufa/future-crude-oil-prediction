from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "Final_Dataset_AI_dxy_fixed.csv"
MODEL_FILE = ROOT / "model_artifacts.joblib"
STATUS_FILE = ROOT / "data_refresh_status.json"

st.set_page_config(page_title="Prediction vs Actual | WTI", page_icon="📊", layout="wide")
st.title("WTI · Prediction vs Actual")
st.caption("One-session historical benchmark check · USD per barrel")

missing = [p.name for p in [DATA_FILE, MODEL_FILE, STATUS_FILE] if not p.exists()]
if missing:
    st.error("Required project files missing: " + ", ".join(missing))
    st.stop()

df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["WTI_Close"] = pd.to_numeric(df["WTI_Close"], errors="coerce")
df = (df.dropna(subset=["Date", "WTI_Close"])
        .sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True))
if len(df) < 5:
    st.error("At least five valid daily WTI rows are required.")
    st.stop()

with open(STATUS_FILE, "r", encoding="utf-8") as f:
    status = json.load(f)
artifacts = joblib.load(MODEL_FILE)
record = artifacts.get("models", {}).get(1, artifacts.get("models", {}).get("1"))
if record is None:
    st.error("One-session forecast artifact is unavailable.")
    st.stop()

# The deployed one-session selector currently retains a zero-return benchmark.
# Historical rows below reproduce that same rule using only the previous close.
recent = df.tail(5).copy()
recent["Predicted_Close"] = recent["WTI_Close"].shift(1)
history = recent.dropna(subset=["Predicted_Close"]).tail(4).copy()
history["Absolute_Difference"] = (history["WTI_Close"] - history["Predicted_Close"]).abs()
history["Absolute_Percent_Error"] = (
    history["Absolute_Difference"] / history["WTI_Close"].abs() * 100
)
history = history.rename(columns={
    "Date": "Actual trading date",
    "Predicted_Close": "Predicted close (USD)",
    "WTI_Close": "Actual close (USD)",
    "Absolute_Difference": "Difference (USD)",
    "Absolute_Percent_Error": "Absolute error (%)",
})
display = history[[
    "Actual trading date", "Predicted close (USD)", "Actual close (USD)",
    "Difference (USD)", "Absolute error (%)"
]].copy()
display["Actual trading date"] = display["Actual trading date"].dt.strftime("%d %b %Y")

latest_close = float(df.iloc[-1]["WTI_Close"])
latest_date = pd.Timestamp(df.iloc[-1]["Date"]).date()
model_price = float(record.get("price", latest_close))
model_date = pd.Timestamp(artifacts.get("feature_date", df.iloc[-1]["Date"])).date()

m1, m2, m3 = st.columns(3)
m1.metric("Latest saved WTI close", f"${latest_close:,.2f}/bbl", str(latest_date))
m2.metric("Next-session model estimate", f"${model_price:,.2f}/bbl", f"Reference date: {model_date}")
m3.metric("Model method", str(record.get("model_type", "Not recorded")))

st.subheader("Last 4 completed trading sessions")
st.dataframe(
    display.style.format({
        "Predicted close (USD)": "${:,.2f}",
        "Actual close (USD)": "${:,.2f}",
        "Difference (USD)": "${:,.2f}",
        "Absolute error (%)": "{:.2f}%"
    }),
    use_container_width=True,
    hide_index=True
)

mean_error = float(display["Absolute error (%)"].mean())
st.metric("4-session mean absolute percentage error", f"{mean_error:.2f}%")

st.info(
    "How this comparison is calculated: before each trading session, the model's selected "
    "one-session rule uses the previous completed WTI close as its estimate. The actual close "
    "is then compared with that estimate. This is a real historical benchmark replay, not a "
    "claim that a directional AI model predicted the exact movement."
)

st.caption(
    "Source: project dataset and saved model artifacts. Latest refresh (UTC): "
    + str(status.get("last_successful_run_utc", "not recorded"))
)
st.caption(
    "Research prototype only. WTI futures are volatile; this display is not financial advice "
    "and does not establish predictive accuracy."
)
