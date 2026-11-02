#!/usr/bin/env python3
"""Refresh market observations and run inference with the saved model; never retrains."""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import forecast_engine as fe

DATA_FILE = ROOT / "Final_Dataset_AI_dxy_fixed.csv"
MODEL_FILE = ROOT / "model_artifacts.joblib"
STATUS_FILE = ROOT / "data_refresh_status.json"
FORECAST_FILE = ROOT / "daily_forecast.json"


def verify_source(label: str, message: str) -> None:
    if not message.startswith(label + ":") or "fetch failed" in message.lower():
        raise RuntimeError(f"{label} source check failed: {message}")
    overlap = re.search(r"(\d+)\s+overlap rows", message)
    median = re.search(r"median difference\s+([0-9.]+)%", message)
    p90 = re.search(r"90th-percentile difference\s+([0-9.]+)%", message)
    if not overlap or not median or not p90:
        raise RuntimeError(f"{label} overlap check could not be parsed: {message}")
    rows, med, p90_value = int(overlap.group(1)), float(median.group(1)), float(p90.group(1))
    if rows < 5 or med > 3.0 or p90_value > 6.0:
        raise RuntimeError(
            f"{label} prices failed overlap validation: rows={rows}, median={med}%, p90={p90_value}%"
        )


def verify_brent(message: str) -> None:
    match = re.search(
        r"(\d+) valid daily bars; latest (\d{4}-\d{2}-\d{2}) close ([0-9.]+)",
        message
    )
    if not message.startswith("Yahoo BZ=F:") or not match:
        raise RuntimeError(f"Brent benchmark refresh failed: {message}")
    if int(match.group(1)) < 30 or not (5 < float(match.group(3)) < 1000):
        raise RuntimeError(f"Brent benchmark values failed validation: {message}")


def atomic_json(path: Path, payload: dict) -> None:
    fd, temp_path = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=ROOT)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    fd, temp_path = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=ROOT)
    os.close(fd)
    try:
        df.to_csv(temp_path, index=False)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def next_weekday(date_value: pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(date_value).normalize() + pd.Timedelta(days=1)
    while result.weekday() >= 5:
        result += pd.Timedelta(days=1)
    return result


def main() -> None:
    if not DATA_FILE.exists() or not MODEL_FILE.exists() or not STATUS_FILE.exists():
        raise FileNotFoundError("Historical data, saved model and status file must all exist.")

    print("Loading saved historical data and frozen model...")
    base = fe.standardize(pd.read_csv(DATA_FILE, parse_dates=["Date"]))
    artifacts = joblib.load(MODEL_FILE)
    feature_names = list(artifacts.get("feature_names", []))
    if feature_names != fe.EXPANDED_FEATURES:
        raise RuntimeError("Saved model feature list does not match current 64-feature engine.")
    model_record = artifacts.get("models", {}).get(1, artifacts.get("models", {}).get("1"))
    if model_record is None or "pipeline" not in model_record:
        raise RuntimeError("Saved next-session prediction pipeline is missing.")

    print("Fetching current market observations...")
    updated, source = fe.refresh_market_history(base)
    verify_source("Yahoo CL=F", str(source.get("wti", "")))
    verify_source("Yahoo DX-Y.NYB", str(source.get("dxy", "")))
    verify_brent(str(source.get("brent", "")))

    print("Running inference with the existing saved model (no training)...")
    matrix = fe.compute_expanded_features(updated)
    close = pd.to_numeric(updated["WTI_Close"], errors="coerce")
    valid = matrix.notna().all(axis=1) & close.notna()
    indices = np.flatnonzero(valid.to_numpy())
    if not len(indices):
        raise RuntimeError("No complete 64-feature row exists after market refresh.")
    idx = int(indices[-1])
    feature_date = pd.Timestamp(updated.loc[idx, "Date"]).normalize()
    current_close = float(close.iloc[idx])
    x = matrix.iloc[[idx]].loc[:, feature_names]
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise RuntimeError("Latest feature row contains missing or infinite inputs.")

    predicted_return = float(model_record["pipeline"].predict(x)[0])
    predicted_close = current_close * (1.0 + predicted_return)
    if not np.isfinite(predicted_close) or predicted_close <= 0:
        raise RuntimeError("Saved model returned an invalid next-session estimate.")

    forecast_date = next_weekday(feature_date)
    cutoff_value = source.get("cutoff_date")
    if cutoff_value:
        cutoff_date = pd.Timestamp(cutoff_value).normalize()
        if feature_date < cutoff_date - pd.Timedelta(days=4):
            raise RuntimeError(
                f"Latest verified WTI feature row ({feature_date.date()}) is stale "
                f"relative to completed-session cutoff ({cutoff_date.date()}); "
                "refusing to publish a stale next-session forecast."
            )
    brent_close = None
    if "Brent_Close" in updated.columns:
        same_day = pd.to_numeric(
            pd.Series([updated.loc[idx, "Brent_Close"]]), errors="coerce"
        ).iloc[0]
        if pd.notna(same_day):
            brent_close = float(same_day)

    now = datetime.now(timezone.utc)
    forecast = {
        "generated_at_utc": now.isoformat(),
        "source_data_date": str(feature_date.date()),
        "next_session_date_estimate": str(forecast_date.date()),
        "forecast_target": "WTI next-session closing price",
        "wti_reference_close_usd_bbl": current_close,
        "wti_next_session_estimate_usd_bbl": predicted_close,
        "predicted_return_pct": predicted_return * 100.0,
        "brent_reference_close_usd_bbl": brent_close,
        "model_type": str(model_record.get("model_type", "Saved model")),
        "holdout_mae_percentage_points": float(model_record.get("test_mae", np.nan) * 100),
        "zero_change_mae_percentage_points": float(model_record.get("zero_mae", np.nan) * 100),
        "beats_zero_change_baseline": bool(model_record.get("beats_zero", False)),
        "feature_count": len(feature_names),
        "model_retrained_this_run": False,
        "source": {
            "wti": str(source.get("wti", "")),
            "dxy": str(source.get("dxy", "")),
            "brent": str(source.get("brent", "")),
            "inventory": str(source.get("inventory", "")),
            "rig_count": str(source.get("rig_count", "")),
        },
        "next_session_date_note": "Weekday estimate; exchange holidays may alter the next session.",
    }

    old_status = {}
    with open(STATUS_FILE, "r", encoding="utf-8") as handle:
        old_status = json.load(handle)
    old_status.update({
        "last_successful_data_update_utc": now.isoformat(),
        "last_successful_run_utc": now.isoformat(),
        "dataset_rows": int(len(updated)),
        "latest_market_date": str(pd.Timestamp(updated["Date"].max()).date()),
        "latest_feature_date": str(feature_date.date()),
        "current_wti_close": current_close,
        "wti_source_status": str(source.get("wti", "")),
        "dxy_source_status": str(source.get("dxy", "")),
        "brent_source_status": str(source.get("brent", "")),
        "brent_last_date": str(feature_date.date()) if brent_close is not None else None,
        "brent_last_close": brent_close,
        "inventory_source_status": str(source.get("inventory", "")),
        "rig_count_status": str(source.get("rig_count", "")),
        "macro_feature_status": str(source.get("macro_feature_status", "")),
        "feature_count": len(feature_names),
        "model_retrained_this_run": False,
        "last_model_training_utc": old_status.get("last_model_training_utc", old_status.get("last_successful_run_utc")),
    })

    atomic_csv(DATA_FILE, updated)
    atomic_json(FORECAST_FILE, forecast)
    atomic_json(STATUS_FILE, old_status)

    print("DAILY UPDATE AND INFERENCE SUCCESS")
    print(f"WTI source date: {feature_date.date()}")
    print(f"WTI current reference close: {current_close:.4f}")
    print(f"Next-session WTI estimate ({forecast_date.date()}): {predicted_close:.4f}")
    print(f"Model retrained this run: {forecast['model_retrained_this_run']}")
    print("Saved market CSV, daily_forecast.json and data_refresh_status.json")


if __name__ == "__main__":
    main()
