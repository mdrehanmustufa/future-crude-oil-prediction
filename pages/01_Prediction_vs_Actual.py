from datetime import timedelta
from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "Final_Dataset_AI_dxy_fixed.csv"
MODEL_FILE = ROOT / "model_artifacts.joblib"
STATUS_FILE = ROOT / "data_refresh_status.json"

st.set_page_config(page_title="Crude Oil Market Dashboard", page_icon="🛢️", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1450px; padding-top: 1.2rem; padding-bottom: 2.5rem;}
.market-hero {padding: 1.25rem 1.5rem; border-radius: 16px;
background: linear-gradient(110deg,#102b45,#1e5260); color: white; margin-bottom: 1rem;}
.market-hero h1 {color:white !important; margin:0 0 .25rem 0; font-size:2rem;}
.market-hero p {color:#d9e9ed; margin:0;}
div[data-testid="stMetric"] {background:rgba(128,128,128,.06); border:1px solid rgba(128,128,128,.18);
border-radius:12px; padding:.75rem 1rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="market-hero">
<h1>Crude Oil Market Dashboard</h1>
<p>WTI forecasting · Global benchmarks · Actual vs model estimate</p>
</div>
""", unsafe_allow_html=True)

required = [DATA_FILE, MODEL_FILE, STATUS_FILE]
missing = [p.name for p in required if not p.exists()]
if missing:
    st.error("Required project files missing: " + ", ".join(missing))
    st.stop()

@st.cache_data(ttl=900, show_spinner=False)
def market_snapshot(symbol):
    ticker = yf.Ticker(symbol)
    quote = None
    quote_type = "Latest available quote"
    change_pct = None
    try:
        info = ticker.fast_info
        quote = info.get("last_price")
        previous = info.get("previous_close")
        if quote is not None and pd.notna(quote):
            quote = float(quote)
            quote_type = "Latest market quote · provider feed"
            if previous is not None and pd.notna(previous) and float(previous) != 0:
                change_pct = (quote / float(previous) - 1) * 100
    except Exception:
        quote = None
    if quote is None or change_pct is None:
        hist = ticker.history(period="5d", interval="1d", auto_adjust=False)
        if hist is not None and not hist.empty:
            closes = hist["Close"].dropna()
            if quote is None:
                quote = float(closes.iloc[-1])
                quote_type = "Latest available daily close · delayed"
            if len(closes) >= 2 and float(closes.iloc[-2]) != 0:
                change_pct = (float(closes.iloc[-1]) / float(closes.iloc[-2]) - 1) * 100
    return quote, quote_type, change_pct

@st.cache_data(ttl=900, show_spinner=False)
def get_market_cards():
    result = {}
    for name, symbol in [
        ("WTI Crude", "CL=F"),
        ("Brent Crude", "BZ=F"),
        ("US Dollar Index", "DX-Y.NYB"),
    ]:
        try:
            result[name] = market_snapshot(symbol)
        except Exception:
            result[name] = (None, "Quote unavailable", None)
    return result

df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["WTI_Close"] = pd.to_numeric(df["WTI_Close"], errors="coerce")
df = (df.dropna(subset=["Date", "WTI_Close"]).sort_values("Date")
        .drop_duplicates("Date", keep="last").reset_index(drop=True))
if df.empty:
    st.error("WTI history is empty.")
    st.stop()

with open(STATUS_FILE, "r", encoding="utf-8") as handle:
    refresh_status = json.load(handle)
artifacts = joblib.load(MODEL_FILE)
record = artifacts.get("models", {}).get(1, artifacts.get("models", {}).get("1"))
if record is None:
    st.error("One-session model artifact is unavailable.")
    st.stop()

feature_date = pd.Timestamp(artifacts.get("feature_date", df["Date"].max())).normalize()
model_price = float(record.get("price", float(df.iloc[-1]["WTI_Close"])))
reference_close = float(record.get("current_close", df.iloc[-1]["WTI_Close"]))
model_type = str(record.get("model_type", "Not recorded"))
latest_dataset_date = pd.Timestamp(df["Date"].max()).normalize()

target_date = feature_date + timedelta(days=1)
while target_date.weekday() >= 5:
    target_date += timedelta(days=1)

st.subheader("Market Snapshot")
quotes = get_market_cards()
cols = st.columns(3)
for col, name in zip(cols, ["WTI Crude", "Brent Crude", "US Dollar Index"]):
    value, label, change_pct = quotes[name]
    with col:
        if value is None:
            st.metric(name, "Unavailable")
        else:
            unit = "USD/bbl" if name != "US Dollar Index" else "Index"
            delta = None if change_pct is None else f"{'▲' if change_pct >= 0 else '▼'} {change_pct:+.2f}%"
            st.metric(name, f"{value:,.2f} {unit}", delta=delta, delta_color="normal")
        st.caption(label)
st.caption("Market quotes are sourced from Yahoo Finance through yfinance. Quote availability and delay depend on the provider.")

st.divider()
st.subheader("Next Trading Session · WTI Prediction")
p1, p2, p3 = st.columns([1, 1, 1.2])
with p1:
    st.metric("Predicted WTI Close", f"${model_price:,.2f}/bbl")
with p2:
    st.metric("Reference WTI Close", f"${reference_close:,.2f}/bbl")
with p3:
    st.metric("Target Session Date", target_date.strftime("%d %b %Y"))
st.caption(f"Forecast feature date: {feature_date.strftime('%d %b %Y')} · Method: {model_type}")
st.info(
    "Model transparency: the current one-session selection retained a zero-return / previous-close "
    "baseline because the tested ML candidates did not beat it in validation. Therefore this estimate "
    "is currently equal to the reference close; it is not a validated directional AI prediction. "
    "The target date is a weekday estimate and may shift for exchange holidays."
)

st.divider()
# Build the previous-close benchmark replay over the full dataset before applying the selected display window.
df["Predicted_Close"] = df["WTI_Close"].shift(1)
df["Absolute_Error_USD"] = (df["WTI_Close"] - df["Predicted_Close"]).abs()
df["Absolute_Error_Pct"] = df["Absolute_Error_USD"] / df["WTI_Close"].abs() * 100

st.subheader("Model Estimate vs Actual WTI Price")
range_days = st.select_slider(
    "Chart time range",
    options=[30, 60, 90, 180, 365, 0],
    value=90,
    format_func=lambda x: "All available data" if x == 0 else f"Last {x} days"
)
period_start = df["Date"].min() if range_days == 0 else latest_dataset_date - pd.Timedelta(days=range_days)
window = df[(df["Date"] >= period_start) & (df["Date"] <= latest_dataset_date)].copy()
comparison = window.dropna(subset=["Predicted_Close"]).copy()

if comparison.empty:
    st.warning("Not enough historical rows for the selected comparison window.")
else:
    st.caption(
        f"{comparison['Date'].min().strftime('%d %b %Y')} – "
        f"{comparison['Date'].max().strftime('%d %b %Y')} · Daily WTI close · USD per barrel"
    )
    compare_fig = go.Figure()
    compare_fig.add_trace(go.Scatter(
        x=comparison["Date"], y=comparison["WTI_Close"], mode="lines+markers",
        name="Actual WTI Close", line=dict(color="#159447", width=3.2),
        marker=dict(size=5), connectgaps=False,
        customdata=comparison[["Predicted_Close", "Absolute_Error_USD", "Absolute_Error_Pct"]].to_numpy(),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual close: $%{y:,.2f}<br>Estimate: $%{customdata[0]:,.2f}<br>Absolute error: $%{customdata[1]:,.2f}<br>Error: %{customdata[2]:.2f}%<extra></extra>"
    ))
    compare_fig.add_trace(go.Scatter(
        x=comparison["Date"], y=comparison["Predicted_Close"], mode="lines+markers",
        name="Model Estimate · Previous-close baseline",
        line=dict(color="#d93636", width=2.7, dash="dash"),
        marker=dict(size=5),
        customdata=comparison[["WTI_Close", "Absolute_Error_USD", "Absolute_Error_Pct"]].to_numpy(),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Estimate: $%{y:,.2f}<br>Actual close: $%{customdata[0]:,.2f}<br>Absolute error: $%{customdata[1]:,.2f}<br>Error: %{customdata[2]:.2f}%<extra></extra>"
    ))
    compare_fig.update_layout(
        height=540, margin=dict(l=12, r=16, t=35, b=12),
        hovermode="x unified", xaxis_title="Trading date", yaxis_title="WTI close (USD/bbl)",
        legend=dict(orientation="h", y=1.12, x=0),
        template="plotly_white", plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=13),
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.09), type="date"),
        yaxis=dict(gridcolor="rgba(120,130,140,.16)", zeroline=False)
    )
    st.plotly_chart(compare_fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

st.subheader("Prediction Details · Selected Period")
details = comparison[["Date", "Predicted_Close", "WTI_Close", "Absolute_Error_USD", "Absolute_Error_Pct"]].copy()
details = details.rename(columns={
    "Date": "Trading Date", "Predicted_Close": "Predicted Price (USD)",
    "WTI_Close": "Actual Price (USD)", "Absolute_Error_USD": "Absolute Error (USD)",
    "Absolute_Error_Pct": "Absolute Error (%)"
})
details["Trading Date"] = details["Trading Date"].dt.strftime("%d %b %Y")
st.dataframe(details.style.format({
    "Predicted Price (USD)": "${:,.2f}",
    "Actual Price (USD)": "${:,.2f}",
    "Absolute Error (USD)": "${:,.2f}",
    "Absolute Error (%)": "{:.2f}%"
}), use_container_width=True, hide_index=True)

if not comparison.empty:
    e1, e2 = st.columns(2)
    e1.metric("30-day benchmark MAPE", f"{comparison['Absolute_Error_Pct'].mean():.2f}%")
    e2.metric("30-day mean absolute error", f"${comparison['Absolute_Error_USD'].mean():.2f}/bbl")
    st.caption("These metrics describe the previous-close benchmark replay for the selected period, not validated AI model accuracy. Changing the display window does not change the underlying errors.")

st.caption(
    f"Dataset last date: {latest_dataset_date.date()} · "
    f"Last successful data refresh (UTC): {refresh_status.get('last_successful_run_utc', 'not recorded')}"
)
st.caption("Research prototype only · Not investment or trading advice.")
