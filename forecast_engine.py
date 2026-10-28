"""Forecast engine for Future Crude Oil Prediction.

Important design choices:
- The ordered 53-feature list is frozen and shared by training and inference.
- Stored core features are retained from the source dataset where present, matching
  the user's earlier experiment; newly appended market rows have those columns
  recomputed using historical formula inference.
- Every training label must have target_date <= latest complete feature date.
- 1- and 4-session forecasts use a zero-return baseline.
- 5-, 10-, and 20-session ExtraTrees forecasts remain experimental.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

HORIZONS = [1, 4, 5, 10, 20]
RAW_COLUMNS = ["WTI_Open", "WTI_High", "WTI_Low", "WTI_Close", "WTI_Volume"]
TARGETS = ["TARGET_close_next", "TARGET_ret_next", "TARGET_up_next"]
CORE_FEATURES = [
    "WTI_ret_1d", "WTI_ret_lag1", "WTI_ret_lag2", "WTI_ret_lag4",
    "WTI_ret_5d_mean", "WTI_vol_20d", "WTI_range_pct", "DXY_ret_1d",
    "RIG_US_Oil_Rigs", "INV_US_Crude_Stocks_kbbl",
]
EXPANDED_FEATURES = [
    # Core features from the existing dataset
    "WTI_ret_1d", "WTI_ret_lag1", "WTI_ret_lag2", "WTI_ret_lag4",
    "WTI_ret_5d_mean", "WTI_vol_20d", "WTI_range_pct", "DXY_ret_1d",
    "RIG_US_Oil_Rigs", "INV_US_Crude_Stocks_kbbl",
    # WTI return lags
    "WTI_past_return_1", "WTI_past_return_2", "WTI_past_return_3",
    "WTI_past_return_5", "WTI_past_return_10", "WTI_past_return_15", "WTI_past_return_20",
    # WTI momentum
    "WTI_momentum_3", "WTI_momentum_5", "WTI_momentum_10", "WTI_momentum_20", "WTI_momentum_60",
    # WTI volatility
    "WTI_volatility_3", "WTI_volatility_5", "WTI_volatility_10", "WTI_volatility_20", "WTI_volatility_60",
    # Distance from moving average
    "WTI_MA_distance_3", "WTI_MA_distance_5", "WTI_MA_distance_10", "WTI_MA_distance_20", "WTI_MA_distance_60",
    # Drawdown from rolling high
    "WTI_drawdown_3", "WTI_drawdown_5", "WTI_drawdown_10", "WTI_drawdown_20", "WTI_drawdown_60",
    # DXY features
    "DXY_return_current", "DXY_past_return_1", "DXY_past_return_2", "DXY_past_return_5",
    "DXY_momentum_5", "DXY_momentum_20", "DXY_volatility_5", "DXY_volatility_20",
    # Rig count changes
    "RIG_change_1", "RIG_change_5", "RIG_change_20", "RIG_pct_change_5",
    # Inventory changes
    "INVENTORY_change_1", "INVENTORY_change_5", "INVENTORY_change_20", "INVENTORY_pct_change_5",
]
assert len(EXPANDED_FEATURES) == 53 and len(set(EXPANDED_FEATURES)) == 53

GITHUB_CSV = (
    "https://raw.githubusercontent.com/mdrehanmustufa/"
    "future-crude-oil-prediction/main/Final_Dataset_AI.csv"
)

# Results copied from the user's earlier 4-fold walk-forward experiment.
# These are historical experiment results, NOT recalculated metrics for a new deployment fit.
PRIOR_WALK_FORWARD = {
    1: {"candidate": "Zero-return baseline", "candidate_mae_pp": 1.7919, "zero_mae_pp": 1.7919, "folds_beating_zero": 0, "folds": 4},
    4: {"candidate": "Historical-mean baseline", "candidate_mae_pp": 3.5452, "zero_mae_pp": 3.5473, "folds_beating_zero": 2, "folds": 4},
    5: {"candidate": "ExtraTrees expanded", "candidate_mae_pp": 3.8998, "zero_mae_pp": 3.9116, "folds_beating_zero": 2, "folds": 4},
    10: {"candidate": "ExtraTrees expanded", "candidate_mae_pp": 5.1182, "zero_mae_pp": 5.1305, "folds_beating_zero": 2, "folds": 4},
    20: {"candidate": "ExtraTrees expanded", "candidate_mae_pp": 7.1106, "zero_mae_pp": 7.1323, "folds_beating_zero": 3, "folds": 4},
}


class ZeroReturnBaseline(BaseEstimator, RegressorMixin):
    """Predict a cumulative return of exactly zero for any row."""
    def fit(self, X, y=None):
        self.n_features_in_ = getattr(X, "shape", [0, 0])[1]
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" not in df.columns:
        raise ValueError("Dataset mein Date column nahi mila.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    df = (df.dropna(subset=["Date"]).sort_values("Date")
          .drop_duplicates("Date", keep="last").reset_index(drop=True))
    for col in RAW_COLUMNS + CORE_FEATURES + ["DXY_Close", "DXY_alt_source_flag", "DXY_missing_flag", "WTI_volume_zero_flag"] + TARGETS + ["ELIGIBLE_core", "ELIGIBLE_extended"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["WTI_Close"].notna().sum() < 100:
        raise ValueError("Dataset mein WTI_Close ke 100 se kam valid observations hain.")
    return df


def load_base():
    """Load local repaired CSV, local original CSV, then GitHub raw as fallback."""
    errors = []
    for name in ["Final_Dataset_AI_dxy_fixed.csv", "Final_Dataset_AI.csv"]:
        if os.path.exists(name):
            try:
                return standardize(pd.read_csv(name, parse_dates=["Date"])), name
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
    try:
        response = requests.get(GITHUB_CSV, timeout=25)
        response.raise_for_status()
        from io import BytesIO
        df = pd.read_csv(BytesIO(response.content), parse_dates=["Date"])
        return standardize(df), GITHUB_CSV
    except Exception as exc:
        errors.append(f"GitHub fallback: {type(exc).__name__}: {exc}")
    raise RuntimeError("Historical dataset load nahi ho paya. " + " | ".join(errors))


def completed_cutoff():
    """Conservative cutoff: do not use a same-day WTI daily bar before 17:30 New York time."""
    now = datetime.now(ZoneInfo("America/New_York"))
    cutoff_day = now.date() if now.time() >= time(17, 30) else now.date() - timedelta(days=1)
    return pd.Timestamp(cutoff_day)


def download_daily(ticker, start, end):
    import yfinance as yf
    history = yf.Ticker(ticker).history(
        start=pd.Timestamp(start).strftime("%Y-%m-%d"),
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d", auto_adjust=False, actions=False,
    )
    if history is None or history.empty or "Close" not in history.columns:
        raise RuntimeError(f"{ticker}: daily source returned no data.")
    idx = pd.DatetimeIndex(pd.to_datetime(history.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out = history.copy()
    out["Date"] = idx.normalize()
    out = out.reset_index(drop=True)
    return out[out["Date"] <= pd.Timestamp(end)].sort_values("Date")


def source_matches(base, remote, base_col, remote_col="Close"):
    recent_cutoff = base["Date"].max() - pd.Timedelta(days=180)
    old = base.loc[base["Date"] >= recent_cutoff, ["Date", base_col]].dropna()
    new = remote[["Date", remote_col]].dropna().drop_duplicates("Date")
    overlap = old.merge(new, on="Date", how="inner")
    overlap = overlap[(overlap[base_col] != 0) & (overlap[remote_col] != 0)]
    if len(overlap) < 5:
        return False, f"Only {len(overlap)} overlap rows; source not verified."
    err = (overlap[remote_col] / overlap[base_col] - 1).abs()
    med, p90 = float(err.median()), float(err.quantile(.90))
    ok = med <= .03 and p90 <= .06
    return ok, f"{len(overlap)} overlap rows; median difference {med:.3%}; 90th-percentile difference {p90:.3%}"


def feature_candidates(df):
    close = pd.to_numeric(df["WTI_Close"], errors="coerce")
    high = pd.to_numeric(df["WTI_High"], errors="coerce")
    low = pd.to_numeric(df["WTI_Low"], errors="coerce")
    open_ = pd.to_numeric(df["WTI_Open"], errors="coerce")
    volume = pd.to_numeric(df.get("WTI_Volume", pd.Series(np.nan, index=df.index)), errors="coerce")
    ret = close.pct_change(fill_method=None)
    out = {
        "WTI_ret_1d": {"pct_change": ret},
        "WTI_ret_lag1": {"shift_1": ret.shift(1)},
        "WTI_ret_lag2": {"shift_2": ret.shift(2)},
        "WTI_ret_lag4": {"shift_4": ret.shift(4)},
        "WTI_ret_5d_mean": {
            "rolling5_current": ret.rolling(5, min_periods=5).mean(),
            "rolling5_previous": ret.shift(1).rolling(5, min_periods=5).mean(),
            "rolling5_min1": ret.rolling(5, min_periods=1).mean(),
            "rolling5_previous_min1": ret.shift(1).rolling(5, min_periods=1).mean(),
        },
        "WTI_vol_20d": {},
        "WTI_range_pct": {
            "range_over_close": (high-low)/close.replace(0, np.nan),
            "range_over_low": (high-low)/low.replace(0, np.nan),
            "range_over_open": (high-low)/open_.replace(0, np.nan),
            "range_over_close_x100": 100*(high-low)/close.replace(0, np.nan),
        },
        "WTI_volume_zero_flag": {
            "zero_volume": volume.eq(0).astype(float),
            "missing_volume": volume.isna().astype(float),
            "nonpositive_volume": volume.le(0).astype(float),
        },
    }
    for minp in [5, 10, 15, 20]:
        vol = ret.rolling(20, min_periods=minp).std()
        prev = ret.shift(1).rolling(20, min_periods=minp).std()
        out["WTI_vol_20d"][f"std20_{minp}"] = vol
        out["WTI_vol_20d"][f"prev_std20_{minp}"] = prev
        out["WTI_vol_20d"][f"annual_std20_{minp}"] = vol*np.sqrt(252)
        out["WTI_vol_20d"][f"prev_annual_std20_{minp}"] = prev*np.sqrt(252)
    return out


def infer_recipes(base):
    candidates = feature_candidates(base)
    chosen, notes = {}, {}
    limits = {"WTI_ret_1d": .00025, "WTI_ret_lag1": .00025, "WTI_ret_lag2": .00025,
              "WTI_ret_lag4": .00025, "WTI_ret_5d_mean": .001, "WTI_vol_20d": .003,
              "WTI_range_pct": .003, "WTI_volume_zero_flag": .05}
    for feature, options in candidates.items():
        actual = pd.to_numeric(base.get(feature, pd.Series(np.nan, index=base.index)), errors="coerce")
        scores = []
        for name, candidate in options.items():
            pair = pd.concat([actual.rename("actual"), candidate.rename("candidate")], axis=1).dropna()
            if len(pair) < 30:
                continue
            error = (pair.actual - pair.candidate).abs()
            med_error = float(error.median())
            scale = max(float(pair.actual.abs().median()), 1e-6)
            scores.append((med_error/scale, med_error, name))
        if not scores:
            chosen[feature], notes[feature] = None, "No reliable historical formula match."
            continue
        scores.sort()
        rel_error, med_error, name = scores[0]
        if med_error <= limits[feature] and rel_error <= .05:
            chosen[feature] = name
            notes[feature] = f"Matched {name}; median absolute error={med_error:.6g}"
        else:
            chosen[feature] = None
            notes[feature] = "Formula did not match historical values confidently."
    return chosen, notes


def fetch_eia_inventory():
    """Optional weekly EIA WCESTUS1 commercial crude inventory series."""
    key = os.getenv("EIA_API_KEY", "").strip()
    if not key:
        return None, "No EIA_API_KEY configured; using latest stored inventory."
    try:
        response = requests.get(
            "https://api.eia.gov/v2/seriesid/PET.WCESTUS1.W",
            params={"api_key": key, "length": 5000}, timeout=15,
        )
        response.raise_for_status()
        rows = response.json().get("response", {}).get("data", [])
        eia = pd.DataFrame(rows)
        if not {"period", "value"}.issubset(eia.columns):
            return None, "EIA response format unexpected; using stored inventory."
        eia["Date"] = pd.to_datetime(eia["period"], errors="coerce")
        eia["value"] = pd.to_numeric(eia["value"], errors="coerce")
        eia = eia.dropna(subset=["Date", "value"]).sort_values("Date")
        if eia.empty:
            return None, "EIA returned no numeric inventory rows."
        eia["available_date"] = eia["Date"] + pd.Timedelta(days=5)
        return eia[["Date", "available_date", "value"]], (
            f"EIA WCESTUS1 through week-ending {eia['Date'].max().date()}; "
            "five-day publication lag applied conservatively."
        )
    except Exception as exc:
        return None, f"EIA refresh failed ({type(exc).__name__}); using stored inventory."


def refresh_market_history(base):
    """Refresh WTI/DXY only after source-overlap verification. Falls back to stored data on errors."""
    df = standardize(base)
    base_max = df["Date"].max()
    cutoff = completed_cutoff()
    start = max(df["Date"].min(), base_max - pd.Timedelta(days=180))
    status = {"base_date": str(base_max.date()), "cutoff_date": str(cutoff.date(),),
              "wti": "Not refreshed", "dxy": "Not refreshed", "inventory": "",
              "new_wti_rows": 0, "dxy_repaired": 0, "formula_notes": {}}
    wti, dxy = pd.DataFrame(), pd.DataFrame()
    wti_ok = dxy_ok = False
    try:
        remote = download_daily("CL=F", start, cutoff)
        wti = remote.rename(columns={"Open": "WTI_Open", "High": "WTI_High", "Low": "WTI_Low", "Volume": "WTI_Volume"})
        wti_ok, msg = source_matches(df, wti, "WTI_Close", "Close")
        status["wti"] = f"Yahoo CL=F: {msg}"
    except Exception as exc:
        status["wti"] = f"WTI fetch failed ({type(exc).__name__}); stored history retained."
    try:
        dxy = download_daily("DX-Y.NYB", start, cutoff)
        dxy_ok, msg = source_matches(df, dxy, "DXY_Close", "Close")
        status["dxy"] = f"Yahoo DX-Y.NYB: {msg}"
    except Exception as exc:
        status["dxy"] = f"DXY fetch failed ({type(exc).__name__}); stored values retained."

    dxy_map = {}
    if dxy_ok:
        dxy_map = dxy.drop_duplicates("Date").set_index("Date")["Close"].to_dict()
        mapped = df["Date"].map(dxy_map)
        fill = df["DXY_Close"].isna() & mapped.notna()
        status["dxy_repaired"] = int(fill.sum())
        df.loc[fill, "DXY_Close"] = mapped[fill]
        df.loc[fill, "DXY_alt_source_flag"] = 0
        calc_ret = df["DXY_Close"].pct_change(fill_method=None)
        ret_fill = df["DXY_ret_1d"].isna() & calc_ret.notna()
        df.loc[ret_fill, "DXY_ret_1d"] = calc_ret[ret_fill]
        df["DXY_missing_flag"] = df["DXY_Close"].isna().astype(float)

    if wti_ok and not wti.empty:
        recipes, notes = infer_recipes(base)
        status["formula_notes"] = notes
        fresh = wti[(wti["Date"] > base_max) & (wti["Date"] <= cutoff)].drop_duplicates("Date").sort_values("Date")
        if len(fresh):
            last_values = {}
            for col in df.columns:
                if col == "Date":
                    continue
                vals = df[col].dropna()
                last_values[col] = vals.iloc[-1] if len(vals) else np.nan
            new_rows = []
            for _, live in fresh.iterrows():
                row = dict(last_values)
                row["Date"] = pd.Timestamp(live["Date"]).normalize()
                for col in ["WTI_Open", "WTI_High", "WTI_Low", "WTI_Volume"]:
                    row[col] = pd.to_numeric(live.get(col, np.nan), errors="coerce")
                row["WTI_Close"] = pd.to_numeric(live["Close"], errors="coerce")
                row["DXY_Close"] = dxy_map.get(row["Date"], np.nan) if dxy_ok else np.nan
                row["DXY_missing_flag"] = float(pd.isna(row["DXY_Close"]))
                if pd.notna(row["DXY_Close"]):
                    row["DXY_alt_source_flag"] = 0.0
                for target in TARGETS:
                    row[target] = np.nan
                row["ELIGIBLE_core"] = 0
                row["ELIGIBLE_extended"] = 0
                new_rows.append(row)
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
            df = df.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
            status["new_wti_rows"] = len(new_rows)
            new_mask = df["Date"] > base_max
            options = feature_candidates(df)
            for feature, recipe in recipes.items():
                if recipe is None or recipe not in options[feature]:
                    df.loc[new_mask, feature] = np.nan
                else:
                    df.loc[new_mask, feature] = options[feature][recipe].loc[new_mask]
            if dxy_ok:
                dxy_ret = df["DXY_Close"].pct_change(fill_method=None)
                df.loc[new_mask, "DXY_ret_1d"] = dxy_ret.loc[new_mask]
                df.loc[new_mask, "DXY_missing_flag"] = df.loc[new_mask, "DXY_Close"].isna().astype(float)

    eia, eia_note = fetch_eia_inventory()
    status["inventory"] = eia_note
    if eia is not None:
        new_mask = df["Date"] > base_max
        for idx in df.index[new_mask]:
            known = eia[eia["available_date"] <= df.at[idx, "Date"]]
            if not known.empty:
                df.at[idx, "INV_US_Crude_Stocks_kbbl"] = float(known.iloc[-1]["value"])

    close = pd.to_numeric(df["WTI_Close"], errors="coerce")
    next_close = close.shift(-1)
    next_return = next_close / close - 1
    df["TARGET_close_next"] = next_close
    df["TARGET_ret_next"] = next_return
    df["TARGET_up_next"] = (next_return > 0).astype(float)
    df.loc[next_return.isna(), "TARGET_up_next"] = np.nan
    df["ELIGIBLE_core"] = (df[CORE_FEATURES].notna().all(axis=1) & next_return.notna()).astype(int)
    status["latest_market_date"] = str(df["Date"].max().date())
    # Feature date is determined by the complete 53-feature matrix, not core-only features.
    X = compute_expanded_features(df)
    valid = X.notna().all(axis=1) & close.notna()
    status["latest_feature_date"] = str(df.loc[valid, "Date"].max().date()) if valid.any() else "none"
    status["rows_after_refresh"] = len(df)
    status["rig_count"] = "US oil-specific rig count is carried forward from the dataset; not a live daily series."
    return standardize(df), status


def compute_expanded_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the same ordered 53-column matrix used by this pipeline.

    For the ten core columns, retain stored dataset calculations when those columns
    exist, matching the user's prior model experiment. For extra columns, reproduce
    that experiment's non-centered rolling calculations on raw price/indicator series.
    All rolling windows are backward-looking; there is no centered window or negative shift.
    """
    source = df.copy()
    close = pd.to_numeric(source["WTI_Close"], errors="coerce")
    high = pd.to_numeric(source["WTI_High"], errors="coerce")
    low = pd.to_numeric(source["WTI_Low"], errors="coerce")
    wti_ret = close.pct_change(fill_method=None)
    feat = pd.DataFrame(index=source.index)

    calculated_core = {
        "WTI_ret_1d": wti_ret,
        "WTI_ret_lag1": wti_ret.shift(1),
        "WTI_ret_lag2": wti_ret.shift(2),
        "WTI_ret_lag4": wti_ret.shift(4),
        "WTI_ret_5d_mean": wti_ret.rolling(5, min_periods=5).mean(),
        "WTI_vol_20d": wti_ret.rolling(20, min_periods=20).std(),
        "WTI_range_pct": (high-low)/close.replace(0, np.nan),
    }
    for name, calculated in calculated_core.items():
        feat[name] = (pd.to_numeric(source[name], errors="coerce")
                      if name in source.columns and pd.to_numeric(source[name], errors="coerce").notna().any()
                      else calculated)
    dxy_close = pd.to_numeric(source.get("DXY_Close", pd.Series(np.nan, index=source.index)), errors="coerce")
    dxy_ret = dxy_close.pct_change(fill_method=None)
    feat["DXY_ret_1d"] = (pd.to_numeric(source["DXY_ret_1d"], errors="coerce")
                          if "DXY_ret_1d" in source.columns and pd.to_numeric(source["DXY_ret_1d"], errors="coerce").notna().any()
                          else dxy_ret)
    for col in ["RIG_US_Oil_Rigs", "INV_US_Crude_Stocks_kbbl"]:
        feat[col] = pd.to_numeric(source[col], errors="coerce") if col in source.columns else np.nan

    for lag in [1, 2, 3, 5, 10, 15, 20]:
        feat[f"WTI_past_return_{lag}"] = wti_ret.shift(lag)
    for window in [3, 5, 10, 20, 60]:
        feat[f"WTI_momentum_{window}"] = close.pct_change(window, fill_method=None)
        feat[f"WTI_volatility_{window}"] = wti_ret.rolling(window).std()
        feat[f"WTI_MA_distance_{window}"] = close / close.rolling(window).mean() - 1
        feat[f"WTI_drawdown_{window}"] = close / close.rolling(window).max() - 1

    feat["DXY_return_current"] = dxy_ret
    for lag in [1, 2, 5]:
        feat[f"DXY_past_return_{lag}"] = dxy_ret.shift(lag)
    for window in [5, 20]:
        feat[f"DXY_momentum_{window}"] = dxy_close.pct_change(window, fill_method=None)
        feat[f"DXY_volatility_{window}"] = dxy_ret.rolling(window).std()

    rigs = pd.to_numeric(source.get("RIG_US_Oil_Rigs", pd.Series(np.nan, index=source.index)), errors="coerce")
    inv = pd.to_numeric(source.get("INV_US_Crude_Stocks_kbbl", pd.Series(np.nan, index=source.index)), errors="coerce")
    feat["RIG_change_1"] = rigs.diff(1)
    feat["RIG_change_5"] = rigs.diff(5)
    feat["RIG_change_20"] = rigs.diff(20)
    feat["RIG_pct_change_5"] = rigs.pct_change(5, fill_method=None)
    feat["INVENTORY_change_1"] = inv.diff(1)
    feat["INVENTORY_change_5"] = inv.diff(5)
    feat["INVENTORY_change_20"] = inv.diff(20)
    feat["INVENTORY_pct_change_5"] = inv.pct_change(5, fill_method=None)

    # Deterministic order, and keep only the frozen model-input list.
    return feat.loc[:, EXPANDED_FEATURES].replace([np.inf, -np.inf], np.nan)


def _trees_model(horizon: int):
    # Settings correspond to the best model candidates from prior validation:
    # h=5 ExtraTrees(depth=6, leaf=10); h=10/20 ExtraTrees(depth=3, leaf=30).
    depth, leaf = (6, 10) if horizon == 5 else (3, 30)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=100, max_depth=depth, min_samples_leaf=leaf,
            random_state=42, n_jobs=-1
        )),
    ])


def train_hybrid_horizons(df: pd.DataFrame):
    """Train each deployed model and calculate a chronological holdout diagnostic.

    All labels used for fitting must have target_date <= latest complete feature date.
    A final model is refit on all labels known by that cutoff after the diagnostic split.
    """
    df = standardize(df)
    df = df.sort_values("Date").reset_index(drop=True)
    close = pd.to_numeric(df["WTI_Close"], errors="coerce")
    feature_matrix = compute_expanded_features(df)
    feature_complete = feature_matrix.notna().all(axis=1) & close.notna()
    valid_indices = np.flatnonzero(feature_complete.to_numpy())
    if len(valid_indices) == 0:
        raise RuntimeError("Koi complete 53-feature row nahi mili. DXY/indicator freshness check karein.")
    latest_idx = int(valid_indices[-1])
    feature_date = pd.Timestamp(df.loc[latest_idx, "Date"])
    current_close = float(close.iloc[latest_idx])
    latest_x = feature_matrix.iloc[[latest_idx]].loc[:, EXPANDED_FEATURES]
    if not np.isfinite(current_close) or current_close <= 0:
        raise RuntimeError("Latest complete feature row ka WTI close invalid hai.")

    models: dict[int, dict[str, Any]] = {}
    diagnostics = []
    for h in HORIZONS:
        y = close.shift(-h) / close - 1.0
        target_date = df["Date"].shift(-h)
        # Explicit anti-leakage bound: every training/evaluation label must already be observable at feature_date.
        labelled = feature_complete & y.notna() & target_date.notna() & (target_date <= feature_date) & np.isfinite(y)
        idx = np.flatnonzero(labelled.to_numpy())
        if len(idx) < 300:
            raise RuntimeError(f"{h}-session horizon ke liye labelled rows kam hain: {len(idx)}")
        X_all = feature_matrix.iloc[idx].loc[:, EXPANDED_FEATURES].reset_index(drop=True)
        y_all = y.iloc[idx].astype(float).to_numpy()
        dates = df.loc[idx, "Date"].reset_index(drop=True)
        target_dates = target_date.iloc[idx].reset_index(drop=True)

        # Chronological holdout: validation settings are already fixed, so don't tune on this test slice.
        split = int(len(idx) * .85)
        split = min(max(split, 50), len(idx) - 30)
        test_start = pd.Timestamp(dates.iloc[split])
        train_local = np.flatnonzero((dates < test_start).to_numpy() & (target_dates < test_start).to_numpy())
        test_local = np.flatnonzero((dates >= test_start).to_numpy())
        if len(train_local) < 100 or len(test_local) < 30:
            raise RuntimeError(f"{h}-session horizon ke temporal holdout rows insufficient hain.")
        if not (target_dates.iloc[train_local] < test_start).all():
            raise AssertionError(f"Temporal leakage guard failed for horizon {h}.")
        if not (target_dates.iloc[test_local] <= feature_date).all():
            raise AssertionError(f"Test target cutoff exceeded for horizon {h}.")

        y_test = y_all[test_local]
        zero_mae = float(mean_absolute_error(y_test, np.zeros_like(y_test)))
        residuals = y_test.copy()
        if h in (1, 4):
            holdout_model = ZeroReturnBaseline().fit(X_all.iloc[train_local], y_all[train_local])
            model_type = "Zero-return baseline"
            params = "Cumulative return = 0"
            test_pred = holdout_model.predict(X_all.iloc[test_local])
        else:
            holdout_model = _trees_model(h)
            holdout_model.fit(X_all.iloc[train_local], y_all[train_local])
            model_type = "ExtraTrees experimental"
            params = f"depth={6 if h == 5 else 3}, min_leaf={10 if h == 5 else 30}, trees=100"
            test_pred = holdout_model.predict(X_all.iloc[test_local])
        test_mae = float(mean_absolute_error(y_test, test_pred))
        residuals = y_test - test_pred

        # Refit final inference model only after measuring the chronological holdout.
        if h in (1, 4):
            final_model = ZeroReturnBaseline().fit(X_all, y_all)
            predicted_return = 0.0
        else:
            final_model = _trees_model(h)
            final_model.fit(X_all, y_all)
            predicted_return = float(final_model.predict(latest_x)[0])
        if not np.isfinite(predicted_return):
            raise RuntimeError(f"{h}-session model returned a non-finite prediction.")
        q10, q90 = np.quantile(residuals, [.10, .90])
        predicted_price = current_close * (1 + predicted_return)
        low_price = max(.01, current_close * (1 + predicted_return + q10))
        high_price = max(low_price, current_close * (1 + predicted_return + q90))
        models[h] = {
            "pipeline": final_model,
            "latest_x": latest_x.copy(),
            "feature_date": feature_date,
            "current_close": current_close,
            "model_type": model_type,
            "params": params,
            "labeled_rows": int(len(idx)),
            "train_rows": int(len(train_local)),
            "test_rows": int(len(test_local)),
            "test_start_date": pd.Timestamp(dates.iloc[test_local[0]]),
            "test_mae": test_mae,
            "zero_mae": zero_mae,
            "beats_zero": bool(test_mae < zero_mae),
            "return": predicted_return,
            "price": predicted_price,
            "low_price": low_price,
            "high_price": high_price,
            "active_features": [],
            "target_cutoff": feature_date,
        }
        diagnostics.append({
            "Horizon sessions": h,
            "Model": model_type,
            "Parameters": params,
            "Training rows": int(len(train_local)),
            "Holdout rows": int(len(test_local)),
            "Holdout start": pd.Timestamp(dates.iloc[test_local[0]]).date(),
            "Model MAE (pp)": test_mae * 100,
            "Zero-return MAE (pp)": zero_mae * 100,
            "Beats zero baseline": test_mae < zero_mae,
            "Latest permissible target date": feature_date.date(),
        })

    state = {
        "df": df,
        "feature_matrix": feature_matrix,
        "models": models,
        "latest": df.loc[latest_idx].copy(),
        "latest_x": latest_x,
        "feature_date": feature_date,
        "market_date": pd.Timestamp(df["Date"].max()),
        "current_close": current_close,
        "diagnostics": pd.DataFrame(diagnostics),
        "feature_count": len(EXPANDED_FEATURES),
        "feature_names": list(EXPANDED_FEATURES),
    }
    # Hard guards for the final inference row and feature ordering.
    assert state["latest_x"].columns.tolist() == EXPANDED_FEATURES
    assert state["latest_x"].shape == (1, 53)
    for h, record in models.items():
        assert pd.Timestamp(record["target_cutoff"]) == feature_date
        assert record["price"] > 0
    return state


def train_direct_horizons(df):
    """Backwards-compatible alias used by older app/notebook cells."""
    state = train_hybrid_horizons(df)
    return state["latest"], state["models"]
