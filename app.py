from pathlib import Path
from datetime import datetime, timezone, timedelta, time as dt_time
from zoneinfo import ZoneInfo
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from forecast_engine import (
    HORIZONS, EXPANDED_FEATURES, standardize,
    compute_expanded_features, infer_recipes, feature_candidates
)

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "Final_Dataset_AI_dxy_fixed.csv"
MODEL_FILE = ROOT / "model_artifacts.joblib"
STATUS_FILE = ROOT / "data_refresh_status.json"

st.set_page_config(
    page_title="Crude Oil Intelligence",
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
    display:flex; align-items:center; gap:1.2rem;
    background:linear-gradient(110deg,#ffffff 0%,#fcfbf8 100%);
    border:1px solid #e9e4d8; border-left:6px solid #b89a5a;
    border-radius:18px; padding:1.25rem 1.55rem; margin:0 0 1rem 0;
    box-shadow:0 7px 24px rgba(30,35,40,.055);
}
.hero-mark {width:76px; min-width:76px; height:76px; display:flex; align-items:center; justify-content:center;}
.hero-copy {min-width:0;}
.hero .eyebrow {color:#927640; font-size:.7rem; font-weight:800;
    letter-spacing:.16em; text-transform:uppercase;}
.hero h1 {color:#1d2935 !important; font-size:2.1rem; margin:.16rem 0 .18rem 0;
    line-height:1.1; letter-spacing:-.035em;}
.hero p {color:#56616d !important; margin:.16rem 0 0 0; font-size:.98rem;}
.hero .note {color:#7a838b !important; font-size:.76rem; margin-top:.48rem;}
.market-heading {display:flex; align-items:center; gap:.5rem; margin:.2rem 0 .7rem 0;}
.market-heading h3 {margin:0; color:#1f2b35;}
.driver-note {background:#f7f8fa; border:1px solid #e5e9ee; border-radius:10px;
    padding:.65rem .8rem; color:#59636d; font-size:.82rem; margin-top:.55rem;}
.chart-card {background:#fff; border:1px solid #e5e8ec; border-radius:13px; padding:.7rem .8rem;}
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
  <div class="hero-mark">
    <svg viewBox="0 0 100 100" width="74" height="74" role="img" aria-label="Crude oil barrel logo">
      <defs>
        <linearGradient id="barrelBody" x1="0" x2="1">
          <stop offset="0" stop-color="#12263c"/><stop offset=".5" stop-color="#243b52"/><stop offset="1" stop-color="#0b1b2d"/>
        </linearGradient>
        <linearGradient id="barrelGold" x1="0" x2="1">
          <stop offset="0" stop-color="#8f6b2d"/><stop offset=".45" stop-color="#e5c47b"/><stop offset="1" stop-color="#9c783c"/>
        </linearGradient>
      </defs>
      <path d="M18 18 Q50 7 82 18 L82 81 Q50 94 18 81 Z" fill="url(#barrelBody)" stroke="#152b41" stroke-width="2"/>
      <ellipse cx="50" cy="18" rx="32" ry="9" fill="#e7edf2" stroke="#c3ccd5" stroke-width="2"/>
      <ellipse cx="50" cy="18" rx="24" ry="5.5" fill="#20374f"/>
      <ellipse cx="71" cy="16.5" rx="4.2" ry="2.2" fill="#d9b86f"/>
      <path d="M18 35 Q50 42 82 35 L82 41 Q50 49 18 41 Z" fill="url(#barrelGold)"/>
      <path d="M18 61 Q50 68 82 61 L82 67 Q50 75 18 67 Z" fill="url(#barrelGold)"/>
      <path d="M18 81 Q50 92 82 81" fill="none" stroke="#d8b46a" stroke-width="4" stroke-linecap="round"/>
      <path d="M50 45 C44 54 38 60 38 67 C38 75 43.5 80 50 80 C56.5 80 62 75 62 67 C62 60 56 54 50 45 Z"
            fill="#d99028" stroke="#f2cb76" stroke-width="2"/>
      <path d="M47 60 C44 65 44 69 47 72" fill="none" stroke="#fff4d9" stroke-width="2.5" stroke-linecap="round"/>
    </svg>
  </div>
  <div class="hero-copy">
    <div class="eyebrow">Energy markets · Research dashboard</div>
    <h1>Crude Oil Intelligence</h1>
    <p>WTI market overview · global supply scenarios · model evaluation</p>
    <div class="note">64 trained input features · 5 interactive scenario drivers · conditional research estimates</div>
  </div>
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
    state, wti_price_usd, dxy_level, brent_price_usd,
    global_production_value, gpr_index
):
    """Build a temporary 64-feature row using five trained, user-facing scenario drivers."""
    df = state["df"]
    idx = state["feature_idx"]
    feature_names = state["feature_names"]
    scenario_df = df.copy()

    values = {
        "WTI price": wti_price_usd,
        "DXY level": dxy_level,
        "Brent price": brent_price_usd,
        "Global oil production": global_production_value,
        "Geopolitical Risk Index": gpr_index,
    }
    if any(not np.isfinite(float(value)) for value in values.values()):
        raise ValueError("Scenario inputs must be finite numbers.")
    if any(float(values[key]) <= 0 for key in ["WTI price", "DXY level", "Brent price", "Global oil production"]):
        raise ValueError("WTI, DXY, Brent and global production must remain positive.")
    if float(gpr_index) < 0:
        raise ValueError("Geopolitical Risk Index cannot be negative.")

    old_wti = float(scenario_df.at[idx, "WTI_Close"])
    if old_wti <= 0:
        raise ValueError("Saved WTI reference price is invalid.")
    wti_factor = float(wti_price_usd) / old_wti
    for column in ["WTI_Open", "WTI_High", "WTI_Low", "WTI_Close"]:
        value = pd.to_numeric(pd.Series([scenario_df.at[idx, column]]), errors="coerce").iloc[0]
        if pd.notna(value):
            scenario_df.at[idx, column] = float(value) * wti_factor

    old_dxy = pd.to_numeric(pd.Series([scenario_df.at[idx, "DXY_Close"]]), errors="coerce").iloc[0]
    if pd.isna(old_dxy) or old_dxy <= 0:
        raise ValueError("DXY value is missing or invalid on the model reference date.")
    scenario_df.at[idx, "DXY_Close"] = float(dxy_level)
    scenario_df["DXY_ret_1d"] = pd.to_numeric(
        scenario_df["DXY_Close"], errors="coerce"
    ).pct_change(fill_method=None)

    # Brent, worldwide production and GPR are existing trained exogenous features.
    brent_column = "CMD_brent_lag1"
    old_brent = pd.to_numeric(pd.Series([scenario_df.at[idx, brent_column]]), errors="coerce").iloc[0]
    if pd.isna(old_brent) or old_brent <= 0:
        raise ValueError("Stored lagged Brent input is missing or invalid.")
    scenario_df.at[idx, brent_column] = float(brent_price_usd)

    production_column = "EIA_world_prod_lag3m"
    old_production = pd.to_numeric(pd.Series([scenario_df.at[idx, production_column]]), errors="coerce").iloc[0]
    if pd.isna(old_production) or old_production <= 0:
        raise ValueError("Stored global production input is missing or invalid.")
    scenario_df.at[idx, production_column] = float(global_production_value)

    gpr_columns = ["GPR_total_lag1m", "GPR_threats_lag1m", "GPR_acts_lag1m"]
    old_gpr = pd.to_numeric(pd.Series([scenario_df.at[idx, gpr_columns[0]]]), errors="coerce").iloc[0]
    if pd.isna(old_gpr) or old_gpr <= 0:
        raise ValueError("Stored GPR baseline is missing or invalid.")
    gpr_factor = float(gpr_index) / float(old_gpr)
    for column in gpr_columns:
        base_value = pd.to_numeric(pd.Series([scenario_df.at[idx, column]]), errors="coerce").iloc[0]
        if pd.isna(base_value) or base_value < 0:
            raise ValueError(f"{column} is missing or invalid for this scenario.")
        scenario_df.at[idx, column] = float(base_value) * gpr_factor

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


@st.cache_data(ttl=900, show_spinner=False)
def latest_yahoo_daily_close(symbol):
    """Return the latest and previous completed daily closes for a market reference."""
    try:
        history = yf.Ticker(symbol).history(period="10d", interval="1d", auto_adjust=False)
        if history is None or history.empty or "Close" not in history:
            return None, None, None, "Yahoo Finance quote unavailable"
        closes = history["Close"].dropna()
        if closes.empty:
            return None, None, None, "Yahoo Finance quote unavailable"
        timestamp = pd.Timestamp(closes.index[-1])
        previous = float(closes.iloc[-2]) if len(closes) >= 2 else None
        return float(closes.iloc[-1]), previous, timestamp.strftime("%d %b %Y"), "Yahoo Finance daily close"
    except Exception as exc:
        return None, None, None, f"Saved dataset fallback · {type(exc).__name__}"


def resolve_market_reference(symbol, fallback_value, fallback_date, fallback_previous=None):
    value, previous, quote_date, source = latest_yahoo_daily_close(symbol)
    if value is not None and np.isfinite(value) and value > 0:
        return {
            "value": float(value), "previous_close": previous,
            "date": quote_date, "source": source
        }
    return {
        "value": float(fallback_value),
        "previous_close": float(fallback_previous) if fallback_previous is not None else None,
        "date": str(fallback_date),
        "source": "Latest saved dataset value · quote fallback"
    }


def next_scheduled_refresh_ist(now=None):
    """Next GitHub Actions schedule: weekdays at 23:00 UTC (04:30 IST next day)."""
    ist = ZoneInfo("Asia/Kolkata")
    now = now or datetime.now(ist)
    now = now.astimezone(ist)
    utc_today = now.astimezone(timezone.utc).date()
    for offset in range(8):
        run_day = utc_today + timedelta(days=offset)
        if run_day.weekday() >= 5:
            continue
        candidate_utc = datetime.combine(run_day, dt_time(23, 0), tzinfo=timezone.utc)
        if candidate_utc > now.astimezone(timezone.utc):
            return candidate_utc.astimezone(ist)
    raise RuntimeError("Could not calculate the next scheduled refresh.")


GPR_SCENARIO_SPECS = [
    (0, "Calm / no major escalation", 0.05),
    (1, "Routine diplomatic tension", 0.25),
    (2, "Direct military clash between countries", 0.50),
    (3, "Sanctions / trade restrictions", 0.70),
    (4, "Sustained regional conflict", 0.85),
    (5, "Major escalation / serious energy-supply threat", 0.97),
]


def build_gpr_scenario_ladder(df, current_value):
    """Map six illustrative event-severity tiers to the training history's GPR distribution."""
    values = pd.to_numeric(df.get("GPR_total_lag1m"), errors="coerce").dropna()
    values = values[np.isfinite(values) & (values >= 0)]
    if len(values) < 30:
        raise ValueError("Insufficient historical GPR observations to calibrate the 0–5 scenario scale.")
    ladder = {}
    for level, label, percentile in GPR_SCENARIO_SPECS:
        ladder[level] = {
            "label": label,
            "percentile": percentile,
            "index_value": float(values.quantile(percentile)),
        }
    current_percentile = float((values <= float(current_value)).mean())
    current_level = min(
        ladder,
        key=lambda level: abs(ladder[level]["percentile"] - current_percentile)
    )
    return ladder, current_level, current_percentile


def reset_scenario_to_latest(defaults):
    """Restore each scenario widget to the latest published reference value."""
    for key, value in defaults.items():
        st.session_state[key] = value


def reset_scenario_to_latest(defaults):
    """Restore each scenario widget to the latest published reference value."""
    for key, value in defaults.items():
        st.session_state[key] = value


SCENARIO_HORIZONS = [5, 10, 20]


def price_card(label, value, date_label, description, tag):
    safe_value = value if value is not None else "Not available"
    st.markdown(
        f"""<div class="price-card">
        <div class="label">{label}</div>
        <div class="value">{safe_value}</div>
        <div class="meta">{date_label}</div>
        <div class="meta">{description}</div>
        <div class="tag">{tag}</div>
        </div>""",
        unsafe_allow_html=True
    )


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
brent_close = state["brent_close"]
brent_date = state["brent_date"]
dxy_close = state["dxy_close"]
dxy_date = state["dxy_date"]

# Saved-data fallbacks and previous closes for the market status cards.
wti_saved = df[["Date", "WTI_Close"]].dropna().sort_values("Date")
wti_saved_previous = float(wti_saved["WTI_Close"].iloc[-2]) if len(wti_saved) >= 2 else None

brent_saved = (
    df[["Date", "Brent_Close"]].dropna().sort_values("Date")
    if "Brent_Close" in df.columns else pd.DataFrame()
)
brent_saved_previous = float(brent_saved["Brent_Close"].iloc[-2]) if len(brent_saved) >= 2 else None

dxy_saved = (
    df[["Date", "DXY_Close"]].dropna().sort_values("Date")
    if "DXY_Close" in df.columns else pd.DataFrame()
)
dxy_saved_previous = float(dxy_saved["DXY_Close"].iloc[-2]) if len(dxy_saved) >= 2 else None

market_inputs = {
    "wti": resolve_market_reference("CL=F", current_close, feature_date.date(), wti_saved_previous),
    "brent": resolve_market_reference(
        "BZ=F",
        brent_close if brent_close is not None else current_close,
        brent_date.date() if brent_date is not None else feature_date.date(),
        brent_saved_previous
    ),
    "dxy": resolve_market_reference(
        "DX-Y.NYB",
        dxy_close if dxy_close is not None else 100.0,
        dxy_date.date() if dxy_date is not None else feature_date.date(),
        dxy_saved_previous
    ),
}

scenario_idx = state["feature_idx"]
global_production_reference = float(df.at[scenario_idx, "EIA_world_prod_lag3m"])
gpr_reference = float(df.at[scenario_idx, "GPR_total_lag1m"])
gpr_ladder, gpr_current_level, gpr_current_percentile = build_gpr_scenario_ladder(df, gpr_reference)

latest_defaults = {
    "scenario_wti_price": float(market_inputs["wti"]["value"]),
    "scenario_brent_price": float(market_inputs["brent"]["value"]),
    "scenario_dxy_level": float(market_inputs["dxy"]["value"]),
    "scenario_global_production": global_production_reference,
    "scenario_gpr_level": int(gpr_current_level),
}

st.markdown(
    '<div class="market-heading"><span style="color:#b38a3b;font-size:1.25rem">▥</span>'
    '<h3>Global Market Overview · WTI reference</h3></div>',
    unsafe_allow_html=True
)
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
        market_inputs["dxy"]["source"], "Currency indicator"
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

with st.expander("Market terms & global data coverage"):
    st.markdown(
        "**WTI:** US crude-oil benchmark. **Brent:** global seaborne crude-oil benchmark. "
        "**DXY:** US Dollar Index, measuring the dollar against a basket of major currencies."
    )
    st.markdown(
        "**Global oil production:** included in the trained model as a lagged EIA world-production feature. "
        "**GPR:** a lagged geopolitical-risk index."
    )
    st.caption(
        "The model also retains regional US commercial inventory and US oil-rig features from its training data, "
        "but these are held fixed in the Expo scenario panel. Global observed inventory and worldwide rig-count "
        "series are not currently trained inputs, so they are not presented as sliders that change the model output."
    )
    st.markdown(
        "Global inventory reference: [IEA observed oil inventories](https://www.iea.org/data-and-statistics/charts/global-observed-oil-inventories-2021-2026). "
        "Worldwide rig-count reference: [Baker Hughes Worldwide Rig Count](https://rigcount.bakerhughes.com/intl-rig-count/)."
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
    st.subheader("What-if Scenario Analysis")
    st.markdown(
        '<div class="section-lead"><strong>Start from the latest reference values.</strong> '
        'Adjust a market or global factor and compare its conditional WTI estimate with the reset case. '
        'The small graph updates with the selected inputs.</div>',
        unsafe_allow_html=True
    )

    global_prod_step = max(abs(global_production_reference) * 0.002, 0.01)
    global_prod_min = max(0.001, global_production_reference * 0.75)
    global_prod_max = max(global_prod_min + global_prod_step, global_production_reference * 1.25)
    gpr_step = max(abs(gpr_reference) * 0.01, 0.1)
    gpr_max = max(100.0, gpr_reference * 3.0, gpr_reference + 50.0)

    for key, value in latest_defaults.items():
        st.session_state.setdefault(key, value)

    st.button(
        "↺ Reset to current references",
        key="reset_scenario_latest",
        type="secondary",
        on_click=reset_scenario_to_latest,
        kwargs={"defaults": latest_defaults},
        help="Restore the latest completed WTI, Brent and DXY closes and the latest stored global production/GPR feature values."
    )

    control_col, result_col = st.columns([1.15, 1], gap="large")

    with control_col:
        with st.container(border=True):
            st.markdown("#### 1 · Market prices & currency")
            market_left, market_right = st.columns(2, gap="medium")
            with market_left:
                wti_price = st.slider(
                    "WTI crude price · USD/bbl",
                    min_value=float(max(1.0, market_inputs["wti"]["value"] * 0.75)),
                    max_value=float(market_inputs["wti"]["value"] * 1.25),
                    step=0.10, key="scenario_wti_price",
                    help=f"Latest daily close: USD {market_inputs['wti']['value']:,.2f} on {market_inputs['wti']['date']}. Main target price."
                )
                brent_price = st.slider(
                    "Brent crude price · USD/bbl",
                    min_value=float(max(1.0, market_inputs["brent"]["value"] * 0.75)),
                    max_value=float(market_inputs["brent"]["value"] * 1.25),
                    step=0.10, key="scenario_brent_price",
                    help=f"Latest daily close: USD {market_inputs['brent']['value']:,.2f} on {market_inputs['brent']['date']}. Global benchmark."
                )
            with market_right:
                dxy_level = st.slider(
                    "US Dollar Index · DXY",
                    min_value=float(max(1.0, market_inputs["dxy"]["value"] * 0.90)),
                    max_value=float(market_inputs["dxy"]["value"] * 1.10),
                    step=0.05, key="scenario_dxy_level",
                    help=f"Latest daily close: {market_inputs['dxy']['value']:,.2f} on {market_inputs['dxy']['date']}."
                )
            st.caption(
                f"Reference dates — WTI: {market_inputs['wti']['date']} · "
                f"Brent: {market_inputs['brent']['date']} · DXY: {market_inputs['dxy']['date']}."
            )

        with st.container(border=True):
            st.markdown("#### 2 · Global supply")
            global_production = st.slider(
                "World oil production · EIA historical input",
                min_value=float(global_prod_min),
                max_value=float(global_prod_max),
                step=float(global_prod_step),
                key="scenario_global_production",
                help="Uses the existing EIA_world_prod_lag3m feature and its original dataset unit. The feature is lagged by three months, not a live daily production feed."
            )
            st.caption(
                f"Current model reference: {global_production_reference:,.3f} · "
                "EIA world-production feature · 3-month lag · source units retained."
            )

        with st.container(border=True):
            st.markdown("#### 3 · Geopolitical event severity")
            gpr_level = st.select_slider(
                "Choose an event scenario · 0 to 5",
                options=list(range(6)),
                value=int(latest_defaults["scenario_gpr_level"]),
                format_func=lambda level: f"{level} · {gpr_ladder[level]['label']}",
                key="scenario_gpr_level",
                help="These are illustrative scenario tiers. Each tier maps to a percentile of the model's historical GPR index distribution."
            )
            gpr_index = (
                gpr_reference
                if gpr_level == gpr_current_level
                else gpr_ladder[gpr_level]["index_value"]
            )
            st.metric(
                "Mapped GPR index input",
                f"{gpr_index:,.2f}",
                f"Historical distribution · P{gpr_ladder[gpr_level]['percentile'] * 100:.0f}"
            )
            st.caption(
                "0 = calm conditions · 1 = routine diplomatic tension · 2 = direct military clash · "
                "3 = sanctions/trade restrictions · 4 = sustained regional conflict · "
                "5 = major escalation or serious energy-supply threat. "
                f"Latest stored GPR rank: P{gpr_current_percentile * 100:.0f}; "
                f"current conditions are mapped to tier {gpr_current_level}. The underlying GPR series is one month lagged."
            )

    with result_col:
        with st.container(border=True):
            st.markdown("#### Live scenario outcome")
            st.caption("The estimate and mini graph update whenever a scenario input changes.")
            chosen_horizon = st.selectbox(
                "Forecast horizon", SCENARIO_HORIZONS, index=0,
                format_func=lambda h: labels[h],
                help="Choose 5, 10 or 20 observed trading sessions."
            )
            try:
                base_x, base_close = make_scenario_features(
                    state,
                    latest_defaults["scenario_wti_price"],
                    latest_defaults["scenario_dxy_level"],
                    latest_defaults["scenario_brent_price"],
                    latest_defaults["scenario_global_production"],
                    gpr_reference,
                )
                scenario_x, scenario_close = make_scenario_features(
                    state, wti_price, dxy_level, brent_price,
                    global_production, gpr_index
                )

                horizon_results = []
                for horizon in SCENARIO_HORIZONS:
                    horizon_model = models[horizon]["pipeline"]
                    ref_return = float(horizon_model.predict(base_x)[0])
                    whatif_return = float(horizon_model.predict(scenario_x)[0])
                    ref_price = base_close * (1.0 + ref_return)
                    whatif_price = scenario_close * (1.0 + whatif_return)
                    horizon_results.append({
                        "horizon": horizon,
                        "reference_price": ref_price,
                        "scenario_price": whatif_price,
                        "reference_return": ref_return,
                        "scenario_return": whatif_return,
                    })

                selected = next(item for item in horizon_results if item["horizon"] == chosen_horizon)
                scenario_price = selected["scenario_price"]
                reference_price = selected["reference_price"]
                scenario_return = selected["scenario_return"]
                reference_return = selected["reference_return"]
                delta_price = scenario_price - reference_price
                delta_return_pp = (scenario_return - reference_return) * 100.0

                if (not np.isfinite(scenario_price) or scenario_price <= 0 or
                    not np.isfinite(scenario_return)):
                    raise ValueError("Model returned an invalid scenario estimate.")

                st.metric(
                    f"Scenario WTI estimate · {chosen_horizon} sessions",
                    f"USD {scenario_price:,.2f}/bbl",
                    delta=f"USD {delta_price:+,.2f} vs reset case"
                )
                st.metric(
                    "Estimated return",
                    f"{scenario_return * 100:+.3f}%",
                    delta=f"{delta_return_pp:+.3f} pp vs reset case"
                )
                st.caption(
                    f"Reset/reference case: USD {reference_price:,.2f}/bbl · "
                    f"Changed scenario: USD {scenario_price:,.2f}/bbl."
                )

                # Test all six GPR levels against the same selected market assumptions.
                # This reveals whether the fitted ExtraTrees model actually responds to GPR.
                gpr_sensitivity_rows = []
                for level in range(6):
                    level_gpr = (
                        gpr_reference
                        if level == gpr_current_level
                        else gpr_ladder[level]["index_value"]
                    )
                    level_x, level_close = make_scenario_features(
                        state, wti_price, dxy_level, brent_price,
                        global_production, level_gpr
                    )
                    level_return = float(models[chosen_horizon]["pipeline"].predict(level_x)[0])
                    level_price = level_close * (1.0 + level_return)
                    gpr_sensitivity_rows.append({
                        "Level": level,
                        "Scenario": gpr_ladder[level]["label"],
                        "Mapped GPR index": level_gpr,
                        "Estimated WTI close": level_price,
                    })
                gpr_sensitivity_df = pd.DataFrame(gpr_sensitivity_rows)
                st.markdown("**GPR impact check · other inputs held constant**")
                st.dataframe(
                    gpr_sensitivity_df.style.format({
                        "Mapped GPR index": "{:,.2f}",
                        "Estimated WTI close": "USD {:,.2f}",
                    }),
                    width="stretch", hide_index=True
                )
                distinct_gpr_outputs = gpr_sensitivity_df["Estimated WTI close"].round(4).nunique()
                if distinct_gpr_outputs == 1:
                    st.warning(
                        "The GPR input changes across all six levels, but this trained model gives the same WTI estimate. "
                        "That means the current fitted trees are not using GPR splits for this selected horizon and market setup. "
                        "The dashboard will not invent a price movement; this is a model limitation to investigate."
                    )
                else:
                    st.success(
                        f"The model produces {distinct_gpr_outputs} distinct WTI estimates across the six GPR tiers. "
                        "Tree-based models can still remain flat between some adjacent tiers."
                    )

                graph_horizons = [item["horizon"] for item in horizon_results]
                reference_prices = [item["reference_price"] for item in horizon_results]
                scenario_prices = [item["scenario_price"] for item in horizon_results]
                fig, ax = plt.subplots(figsize=(6.0, 2.55))
                fig.patch.set_facecolor("#ffffff")
                ax.set_facecolor("#ffffff")
                ax.plot(
                    graph_horizons, reference_prices, marker="o", markersize=5,
                    linewidth=2.1, color="#738294", label="Reset case"
                )
                ax.plot(
                    graph_horizons, scenario_prices, marker="o", markersize=5,
                    linewidth=2.6, color="#b58a3b", label="Changed scenario"
                )
                ax.set_xticks(graph_horizons, [f"{h} sessions" for h in graph_horizons])
                ax.set_ylabel("WTI estimate · USD/bbl", fontsize=8)
                ax.tick_params(axis="both", labelsize=8, colors="#66717c")
                ax.grid(axis="y", alpha=.18, color="#9aa4ad")
                ax.spines[["top", "right", "left"]].set_visible(False)
                ax.spines["bottom"].set_color("#d8dee4")
                ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
                fig.tight_layout(pad=1.0)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

                if not models[chosen_horizon]["beats_zero"]:
                    st.caption(
                        "Research note: this experimental ExtraTrees horizon did not beat the zero-return "
                        "benchmark in the latest chronological holdout. Treat this as an exploratory what-if result."
                    )
            except Exception as exc:
                st.error(f"Scenario calculation unavailable: {type(exc).__name__}: {exc}")

    st.markdown(
        '<div class="driver-note"><strong>Global-data coverage:</strong> Global oil production and GPR are trained inputs. '
        'The existing model also contains US-specific inventory and rig-count history, but these are held at their stored values. '
        'Global inventories and worldwide rig counts are not yet trained as model drivers, so they are not mislabeled as global sliders.</div>',
        unsafe_allow_html=True
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
    st.write("Global production input:", f"{global_production_reference:,.3f} · EIA world-production feature · 3-month lag")
    st.write("GPR input:", f"{gpr_reference:,.3f} · stored feature · 1-month lag")
    st.write("Regional inputs held at saved values:", "US commercial inventories and US oil-rig history")
    st.write("Other stored model inputs:", status.get("macro_feature_status", "Latest stored dataset values."))

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
    "Data quality note: market cards use the latest completed daily close when available, or clearly labeled saved-data fallback. "
    "Scenario inputs are WTI, Brent, DXY, lagged global oil production and lagged GPR. "
    "Regional US inventory/rig features remain in the 64-feature model row at their stored values."
)
