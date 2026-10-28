import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso

# Configure Streamlit page
st.set_page_config(
    page_title="WTI Next-Trading-Day Forecaster",
    page_icon="o",
    layout="centered"
)

# App Title
st.title("WTI Next-Trading-Day Forecasting")
st.markdown("""This interactive dashboard provides next-trading-day forecasts for West Texas Intermediate (WTI) crude oil returns and prices using the exact supervised machine learning methodology of the **WTI forecasting notebook**.""")

# Constants
DATA_FILENAME = "Final_Dataset_AI.csv"
CORE_FEATURES = [
    'WTI_ret_1d', 'WTI_ret_lag1', 'WTI_ret_lag2', 'WTI_ret_lag4',
    'WTI_ret_5d_mean', 'WTI_vol_20d', 'WTI_range_pct', 'WTI_volume_zero_flag',
    'DXY_ret_1d', 'DXY_alt_source_flag', 'DXY_missing_flag',
    'RIG_US_Oil_Rigs', 'INV_US_Crude_Stocks_kbbl'
]
TARGET_RET = "TARGET_ret_next"
SEED = 42

@st.cache_data
def load_and_forecast():
    # Resolve dataset path safely
    data_paths = [Path(DATA_FILENAME), Path("/content") / DATA_FILENAME, Path("../") / DATA_FILENAME]
    data_file = None
    for p in data_paths:
        if p.exists():
            data_file = p
            break

    if data_file is None:
        return {"error": f"Dataset '{DATA_FILENAME}' not found in runtime paths."}

    # Read data
    df = pd.read_csv(data_file, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Isolate forecast target date and inputs
    latest_candidates = df[df[CORE_FEATURES].notna().all(axis=1)].copy()
    if latest_candidates.empty:
        return {"error": "No rows found where all CORE_FEATURES are available."}

    latest_row = latest_candidates.iloc[[-1]].copy()
    forecast_date = latest_row["Date"].iloc[0]

    # Filter training data
    train_all = df[
        (df["Date"] < forecast_date)
        & (df["ELIGIBLE_core"] == 1)
        & df[TARGET_RET].notna()
        & df[CORE_FEATURES].notna().all(axis=1)
    ].copy().sort_values("Date")

    if len(train_all) < 100:
        return {"error": f"Insufficient training rows found: {len(train_all)}"}

    # Build the exact Lasso Pipeline selected during validation
    final_model = Pipeline([
        ("scale", StandardScaler()),
        ("model", Lasso(alpha=0.001, max_iter=20000, random_state=SEED))
    ])

    # Train and predict
    final_model.fit(train_all[CORE_FEATURES], train_all[TARGET_RET])
    pred_return = float(final_model.predict(latest_row[CORE_FEATURES])[0])
    current_close = float(latest_row["WTI_Close"].iloc[0])
    pred_price = current_close * (1 + pred_return)

    return {
        "error": None,
        "forecast_date": forecast_date.date(),
        "training_rows": len(train_all),
        "training_date_min": train_all["Date"].min().date(),
        "training_date_max": train_all["Date"].max().date(),
        "model_name": "Lasso (alpha=0.001)",
        "current_close": current_close,
        "pred_return": pred_return,
        "pred_price": pred_price
    }

# Run forecasting pipeline safely (cached)
result = load_and_forecast()

if result["error"]:
    st.error(result["error"])
    st.info("Please ensure 'Final_Dataset_AI.csv' is placed alongside app.py in your working directory.")
else:
    # Overview Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Observed WTI Close", f"${result['current_close']:.2f}")
    col2.metric("Predicted Return", f"{result['pred_return']*100:+.4f}%")
    col3.metric("Predicted Next-Day WTI", f"${result['pred_price']:.2f}")

    st.success(f"Forecast generated successfully for next trading day after **{result['forecast_date']}**!")

    # Metadata & Parameters Card
    with st.expander("Model Parameters & Training Metadata"):
        st.markdown("""- **Model Family**: {result['model_name']}
- **Dataset Size**: {result['training_rows']} rows
- **Training Interval**: {result['training_date_min']} to {result['training_date_max']}
- **Selected Feature Set**: Core Market & DXY Macro (`CORE_FEATURES`)""")

    # Disclaimer Warning
    st.warning("""WARNING - Experimental Disclaimer: This prediction is computed for research purposes only, not commercial trading. The strategy ignores execution slippage, bid-ask spread, financing/borrow costs, and potential point-to-time timeline leakage.""")