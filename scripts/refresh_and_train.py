#!/usr/bin/env python3
"""Refresh verified WTI/DXY data, retrain horizon models and persist artifacts."""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import forecast_engine as fe


def verify_source(label: str, message: str) -> None:
    if not message.startswith(label + ":"):
        raise RuntimeError(f"{label} was not fetched successfully: {message}")
    if "source not verified" in message.lower() or "fetch failed" in message.lower():
        raise RuntimeError(f"{label} source verification failed: {message}")
    overlap = re.search(r"(\d+)\s+overlap rows", message)
    median = re.search(r"median difference\s+([0-9.]+)%", message)
    p90 = re.search(r"90th-percentile difference\s+([0-9.]+)%", message)
    if not overlap or not median or not p90:
        raise RuntimeError(f"{label} overlap metrics could not be parsed: {message}")
    count, median_pct, p90_pct = int(overlap.group(1)), float(median.group(1)), float(p90.group(1))
    if count < 5 or median_pct > 3.0 or p90_pct > 6.0:
        raise RuntimeError(
            f"{label} overlap outside acceptance thresholds: "
            f"rows={count}, median={median_pct}%, p90={p90_pct}%"
        )


def verify_brent(message: str) -> None:
    if not message.startswith("Yahoo BZ=F:"):
        raise RuntimeError(f"Brent benchmark refresh failed: {message}")
    match = re.search(
        r"(\d+) valid daily bars; latest (\d{4}-\d{2}-\d{2}) close ([0-9.]+)",
        message
    )
    if not match:
        raise RuntimeError(f"Brent refresh metadata could not be parsed: {message}")
    count, latest_date, latest_close = int(match.group(1)), match.group(2), float(match.group(3))
    if count < 30 or not (5 < latest_close < 1000):
        raise RuntimeError(
            f"Brent data failed basic validation: rows={count}, close={latest_close}"
        )


def atomic_json(path: Path, payload: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def next_weekday(date_value: datetime) -> datetime:
    result = datetime.fromisoformat(str(date_value)[:10]) + timedelta(days=1)
    while result.weekday() >= 5:
        result += timedelta(days=1)
    return result


def main() -> None:
    dataset_path = ROOT / "Final_Dataset_AI_dxy_fixed.csv"
    model_path = ROOT / "model_artifacts.joblib"
    status_path = ROOT / "data_refresh_status.json"
    backup_dir = ROOT / "data_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Required dataset missing: {dataset_path.name}")

    now = datetime.now(timezone.utc)
    backup_path = backup_dir / f"Final_Dataset_AI_dxy_fixed_{now:%Y%m%d_%H%M%S}.csv"
    shutil.copy2(dataset_path, backup_path)

    print("Loading current historical dataset...")
    base, base_name = fe.load_base()
    print(f"Loaded {base_name}: {len(base)} rows")
    print("Refreshing WTI and DXY from Yahoo Finance...")
    updated, source = fe.refresh_market_history(base)
    verify_source("Yahoo CL=F", str(source.get("wti", "")))
    verify_source("Yahoo DX-Y.NYB", str(source.get("dxy", "")))
    verify_brent(str(source.get("brent", "")))

    print("Training and evaluating all forecast horizons...")
    state = fe.train_hybrid_horizons(updated)
    metrics = {}
    for horizon, record in state["models"].items():
        metrics[str(horizon)] = {
            "model_type": str(record["model_type"]),
            "holdout_mae_pp": float(record["test_mae"] * 100),
            "zero_return_mae_pp": float(record["zero_mae"] * 100),
            "beats_zero_baseline": bool(record["beats_zero"]),
            "training_rows": int(record["train_rows"]),
            "holdout_rows": int(record["test_rows"]),
        }

    artifacts = {
        "models": state["models"],
        "feature_names": list(state["feature_names"]),
        "feature_date": state["feature_date"],
        "market_date": state["market_date"],
        "current_close": float(state["current_close"]),
    }

    fd, temporary_csv = tempfile.mkstemp(prefix=dataset_path.name, suffix=".tmp", dir=ROOT)
    os.close(fd)
    fd, temporary_model = tempfile.mkstemp(prefix=model_path.name, suffix=".tmp", dir=ROOT)
    os.close(fd)
    try:
        state["df"].to_csv(temporary_csv, index=False)
        joblib.dump(artifacts, temporary_model, compress=3)
        os.replace(temporary_csv, dataset_path)
        os.replace(temporary_model, model_path)
    finally:
        for temporary in [temporary_csv, temporary_model]:
            if os.path.exists(temporary):
                os.unlink(temporary)

    run_status = {
        "last_successful_run_utc": now.isoformat(),
        "last_model_training_utc": now.isoformat(),
        "historical_source": base_name,
        "dataset_rows": int(len(state["df"])),
        "latest_market_date": str(state["market_date"].date()),
        "latest_feature_date": str(state["feature_date"].date()),
        "current_wti_close": float(state["current_close"]),
        "wti_source_status": str(source.get("wti", "")),
        "dxy_source_status": str(source.get("dxy", "")),
        "brent_source_status": str(source.get("brent", "")),
        "brent_last_date": source.get("brent_last_date"),
        "brent_last_close": source.get("brent_last_close"),
        "inventory_source_status": str(source.get("inventory", "")),
        "rig_count_status": str(source.get("rig_count", "")),
        "macro_feature_status": str(source.get("macro_feature_status", "")),
        "eia_api_configured": bool(os.getenv("EIA_API_KEY", "").strip()),
        "feature_count": int(state["feature_count"]),
        "model_metrics": metrics,
    }
    atomic_json(status_path, run_status)

    next_record = state["models"][1]
    forecast = {
        "generated_at_utc": now.isoformat(),
        "source_data_date": str(state["feature_date"].date()),
        "next_session_date_estimate": str(next_weekday(state["feature_date"]).date()),
        "forecast_target": "WTI next-session closing price",
        "wti_reference_close_usd_bbl": float(state["current_close"]),
        "wti_next_session_estimate_usd_bbl": float(next_record["price"]),
        "predicted_return_pct": float(next_record["return"] * 100),
        "model_type": str(next_record["model_type"]),
        "holdout_mae_percentage_points": float(next_record["test_mae"] * 100),
        "zero_change_mae_percentage_points": float(next_record["zero_mae"] * 100),
        "beats_zero_change_baseline": bool(next_record["beats_zero"]),
        "feature_count": int(state["feature_count"]),
        "model_retrained_this_run": True,
        "next_session_date_note": "Weekday estimate; exchange holidays may alter the next session.",
    }
    atomic_json(ROOT / "daily_forecast.json", forecast)
    print("REFRESH AND RETRAIN SUCCESS")
    print(f"Dataset rows: {len(state['df'])}")
    print(f"Latest market date: {state['market_date'].date()}")
    print(f"Latest feature date: {state['feature_date'].date()}")
    print(f"WTI close: {state['current_close']:.4f}")
    print(f"Saved: {dataset_path.name}, {model_path.name}, {status_path.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
