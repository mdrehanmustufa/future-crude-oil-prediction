import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediction vs Actual | WTI", page_icon="📊", layout="wide")
st.title("WTI · Prediction vs Actual")
st.caption("Historical one-session benchmark replay · USD per barrel")

# Yahoo Finance CL=F historical daily closes. Nov 1, 2026 was Sunday,
# so there was no regular daily trading-session close for this benchmark.
rows = [
    {"Date": "01 Nov 2026", "Predicted": None, "Actual": None,
     "Status": "Sunday · no regular trading session"},
    {"Date": "02 Nov 2026", "Predicted": 104.12, "Actual": 96.46,
     "Status": "Previous completed session close used as estimate"},
    {"Date": "03 Nov 2026", "Predicted": 96.46, "Actual": 97.34,
     "Status": "Previous completed session close used as estimate"},
]
data = pd.DataFrame(rows)
valid = data.dropna(subset=["Predicted", "Actual"]).copy()
valid["Difference (USD)"] = (valid["Actual"] - valid["Predicted"]).abs()
valid["Absolute error (%)"] = valid["Difference (USD)"] / valid["Actual"] * 100

latest_actual = 97.34
next_reference_estimate = latest_actual
c1, c2, c3 = st.columns(3)
c1.metric("WTI close · 03 Nov", "$97.34/bbl")
c2.metric("Benchmark estimate for next session", f"${next_reference_estimate:,.2f}/bbl",
          "Previous-close baseline · not directional")
c3.metric("Dates with a completed session", "2 of 3")

st.subheader("1–3 November · predicted vs actual")
show = data.merge(
    valid[["Date", "Difference (USD)", "Absolute error (%)"]],
    on="Date", how="left"
)
st.dataframe(
    show.style.format({
        "Predicted": lambda x: "—" if pd.isna(x) else f"${x:,.2f}",
        "Actual": lambda x: "—" if pd.isna(x) else f"${x:,.2f}",
        "Difference (USD)": lambda x: "—" if pd.isna(x) else f"${x:,.2f}",
        "Absolute error (%)": lambda x: "—" if pd.isna(x) else f"{x:.2f}%"
    }),
    use_container_width=True, hide_index=True
)

if not valid.empty:
    mean_error = valid["Absolute error (%)"].mean()
    st.metric("Mean absolute percentage error · completed sessions", f"{mean_error:.2f}%")

st.info(
    "Important model note: the current one-session model selection retained a zero-return "
    "baseline. Therefore each historical estimate above equals the previous completed "
    "session's close. This is a valid baseline benchmark replay, not evidence that a "
    "directional AI model predicted the market movement."
)
st.caption(
    "Price source: Yahoo Finance continuous WTI futures (CL=F), historical daily close. "
    "Nov 1 is shown for the requested calendar range but has no regular session close. "
    "Prices can differ slightly across data providers and continuous-futures conventions."
)
st.caption("Research prototype only · not investment or trading advice.")
