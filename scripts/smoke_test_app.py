#!/usr/bin/env python3
"""CI smoke tests for saved artifacts, dashboard rendering and scenario calculations."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import forecast_engine as fe

# Validate next-session selection with both signal and no-signal synthetic series.
synthetic_dates = pd.Series(pd.bdate_range("2020-01-01", periods=500))
synthetic_target_dates = synthetic_dates.shift(-1)
synthetic_x = pd.DataFrame({"signal": np.linspace(-1.0, 1.0, 500)})
synthetic_y = 0.015 * synthetic_x["signal"].to_numpy()
selected, candidate_scores, fold_scores, zero_score, _ = fe._evaluate_next_session_candidates(
    synthetic_x, synthetic_y, synthetic_dates, synthetic_target_dates
)
assert len(fold_scores) == 4
assert np.isfinite(zero_score) and zero_score > 0
assert all(np.isfinite(item["pooled_mae"]) for item in candidate_scores.values())
flat_selected, _, flat_folds, flat_zero, _ = fe._evaluate_next_session_candidates(
    synthetic_x, np.zeros(500), synthetic_dates, synthetic_target_dates
)
assert flat_selected == "Zero-return baseline"
assert len(flat_folds) == 4 and flat_zero == 0.0

APP = ROOT / "app.py"
DATA = ROOT / "Final_Dataset_AI_dxy_fixed.csv"
MODEL = ROOT / "model_artifacts.joblib"
STATUS = ROOT / "data_refresh_status.json"

for path in [APP, DATA, MODEL, STATUS]:
    assert path.exists(), f"Required file missing: {path.name}"

# Validate the actual Streamlit app and user-facing controls.
app_test = AppTest.from_file(str(APP), default_timeout=90)
app_test.run(timeout=120)
assert not app_test.exception, [exc.message for exc in app_test.exception]
assert len(app_test.tabs) == 2, f"Expected two main tabs, found {len(app_test.tabs)}"
assert len(app_test.slider) == 4, f"Expected four numeric market/global sliders plus the GPR tier selector, found {len(app_test.slider)}"
assert len(app_test.number_input) == 0
assert len(app_test.selectbox) == 2
assert len(app_test.button) == 1, "Expected a single Reset to latest data button."
assert len(app_test.select_slider) == 2, "Expected a backtest range selector and 0–5 GPR severity selector."

initial_slider_values = [float(item.value) for item in app_test.slider]
initial_gpr_level = int(app_test.select_slider[-1].value)

app_test.slider[0].set_value(initial_slider_values[0] * 1.05)
app_test.slider[1].set_value(initial_slider_values[1] * 1.05)
app_test.slider[2].set_value(initial_slider_values[2] * 1.02)
app_test.slider[3].set_value(initial_slider_values[3] * 1.05)
app_test.select_slider[-1].set_value(5)
app_test.run(timeout=120)
assert not app_test.exception, [exc.message for exc in app_test.exception]
assert not app_test.error, [error.value for error in app_test.error]

# Exercise the highest GPR event tier and verify the app recalculates without error.
app_test.button[0].click()
app_test.run(timeout=120)
assert not app_test.exception, [exc.message for exc in app_test.exception]
reset_slider_values = [float(item.value) for item in app_test.slider]
assert np.allclose(reset_slider_values, initial_slider_values), (
    f"Reset did not restore latest-reference values: {reset_slider_values} vs {initial_slider_values}"
)
assert int(app_test.select_slider[-1].value) == initial_gpr_level, (
    "Reset did not restore the current GPR severity tier."
)

# Extract the exact scenario function from app.py, rather than duplicating its logic.
tree = ast.parse(APP.read_text(encoding="utf-8"))
functions = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "make_scenario_features"
]
assert len(functions) == 1

namespace = {
    "pd": pd,
    "np": np,
    "compute_expanded_features": fe.compute_expanded_features,
    "infer_recipes": fe.infer_recipes,
    "feature_candidates": fe.feature_candidates,
}
module = ast.Module(body=functions, type_ignores=[])
exec(compile(module, str(APP), "exec"), namespace)
scenario_function = namespace["make_scenario_features"]

df = fe.standardize(pd.read_csv(DATA, parse_dates=["Date"]))
production_values = pd.to_numeric(df["EIA_world_prod_lag3m"], errors="coerce").dropna()
production_values = production_values[production_values > 0]
production_p01 = float(production_values.quantile(0.01))
production_p99 = float(production_values.quantile(0.99))
assert production_p01 <= initial_slider_values[3] <= production_p99
assert 64725 < production_values.min(), "Test scenario 64,725 should be recognized as out of training support."
source_text = APP.read_text(encoding="utf-8")
assert "Estimated cumulative price change" in source_text
assert "Change vs level 0 (%)" in source_text
assert "Oil Threats and Oil Acts" in source_text
artifacts = joblib.load(MODEL)
with open(STATUS, encoding="utf-8") as handle:
    status = json.load(handle)

assert list(artifacts["feature_names"]) == fe.EXPANDED_FEATURES
feature_date = pd.Timestamp(artifacts["feature_date"]).normalize()
matches = df.index[df["Date"].eq(feature_date)].tolist()
assert matches, "Artifact feature date is absent from the saved CSV."

state = {
    "df": df,
    "models": artifacts["models"],
    "feature_date": feature_date,
    "current_close": float(artifacts["current_close"]),
    "feature_names": list(artifacts["feature_names"]),
    "feature_idx": matches[-1],
}
assert np.isclose(
    float(status["current_wti_close"]), state["current_close"],
    rtol=1e-5, atol=0.01
)

# Use the latest saved feature-row levels as the test baseline.
idx = state["feature_idx"]
base_wti = float(df.loc[idx, "WTI_Close"])
base_dxy = float(df.loc[idx, "DXY_Close"])
base_brent = float(df.loc[idx, "CMD_brent_lag1"])
base_global_production = float(df.loc[idx, "EIA_world_prod_lag3m"])
base_gpr = float(df.loc[idx, "GPR_total_lag1m"])

baseline_x, baseline_close = scenario_function(
    state, base_wti, base_dxy, base_brent, base_global_production, base_gpr
)
saved_latest_x = state["models"][1]["latest_x"].loc[:, state["feature_names"]]
assert np.allclose(
    baseline_x.to_numpy(dtype=float),
    saved_latest_x.to_numpy(dtype=float),
    rtol=1e-6, atol=1e-8, equal_nan=True
)
assert np.isclose(baseline_close, base_wti)

scenario_specs = [
    ("WTI", scenario_function(state, base_wti * 1.10, base_dxy, base_brent, base_global_production, base_gpr), ["WTI_"]),
    ("Brent", scenario_function(state, base_wti, base_dxy, base_brent * 1.15, base_global_production, base_gpr), ["CMD_brent_lag1"]),
    ("DXY", scenario_function(state, base_wti, base_dxy * 1.05, base_brent, base_global_production, base_gpr), ["DXY_"]),
    ("Global oil production", scenario_function(state, base_wti, base_dxy, base_brent, base_global_production * 1.15, base_gpr), ["EIA_world_prod_lag3m"]),
]
gpr_history = pd.to_numeric(df["GPR_total_lag1m"], errors="coerce").dropna()
gpr_levels = [
    float(gpr_history.quantile(q))
    for q in (0.05, 0.25, 0.50, 0.70, 0.85, 0.97)
]
gpr_feature_rows = []
for level, gpr_value in enumerate(gpr_levels):
    features, close_value = scenario_function(
        state, base_wti, base_dxy, base_brent, base_global_production, gpr_value
    )
    assert features.shape == (1, len(fe.EXPANDED_FEATURES))
    assert np.isfinite(features.to_numpy(dtype=float)).all()
    gpr_cols = ["GPR_total_lag1m", "GPR_threats_lag1m", "GPR_acts_lag1m"]
    assert not np.allclose(
        features[gpr_cols].to_numpy(dtype=float),
        baseline_x[gpr_cols].to_numpy(dtype=float),
        rtol=1e-9, atol=1e-12
    ), f"GPR level {level} did not alter the three trained GPR features."
    gpr_feature_rows.append((level, gpr_value, features))
# All six levels must update the GPR feature family when they differ from the current observation.
assert any(
    not np.allclose(
        features[["GPR_total_lag1m", "GPR_threats_lag1m", "GPR_acts_lag1m"]].to_numpy(dtype=float),
        baseline_x[["GPR_total_lag1m", "GPR_threats_lag1m", "GPR_acts_lag1m"]].to_numpy(dtype=float),
        rtol=1e-9, atol=1e-12
    )
    for _, _, features in gpr_feature_rows
), "None of the six GPR tiers changed the trained GPR features."

sensitivity_predictions = {h: [] for h in (5, 10, 20)}
for label, (features, scenario_close), expected_prefixes in scenario_specs:
    assert features.shape == (1, len(fe.EXPANDED_FEATURES)), f"{label} feature shape mismatch."
    assert np.isfinite(features.to_numpy(dtype=float)).all(), f"{label} features invalid."
    mapped = [
        col for col in features.columns
        if any(col == prefix or col.startswith(prefix) for prefix in expected_prefixes)
    ]
    assert mapped, f"{label}: no mapped trained feature columns were found."
    changed = [
        col for col in mapped
        if not np.isclose(
            float(features.iloc[0][col]),
            float(baseline_x.iloc[0][col]),
            rtol=1e-9, atol=1e-12
        )
    ]
    assert changed, f"{label}: scenario level changed no expected model input features ({mapped})."
    for h in (5, 10, 20):
        record = state["models"][h]
        prediction = float(record["pipeline"].predict(features)[0])
        assert np.isfinite(prediction), f"{label} {h}-session prediction invalid."
        sensitivity_predictions[h].append(prediction)

# Require each scenario-capable horizon to respond to at least one non-WTI input shock.
# WTI is excluded here because its direct reference-price change can move the displayed price
# without changing the model-estimated return.
for h in (5, 10, 20):
    base_prediction = float(state["models"][h]["pipeline"].predict(baseline_x)[0])
    moved = any(
        not np.isclose(value, base_prediction, rtol=1e-10, atol=1e-12)
        for value in sensitivity_predictions[h][1:]
    )
    assert moved, f"{h}-session model output did not respond to any non-WTI scenario shock."

print("PASS: Next-session walk-forward selector handles signal and no-signal test data.")
print("PASS: Streamlit app renders two tabs, five market/global scenario sliders, and the reset button.")
print("PASS: Reset button restores latest-reference control values.")
print("PASS: Reset scenario levels reproduce the saved 64-feature baseline row.")
print("PASS: WTI, Brent, DXY, global oil production and GPR levels each update their mapped trained features.")
# Print the actual GPR sensitivity diagnostics. A flat prediction is a model finding,
# not a reason to fabricate a price response.
for h in (5, 10, 20):
    gpr_predictions = [
        float(state["models"][h]["pipeline"].predict(features)[0])
        for _, _, features in gpr_feature_rows
    ]
    distinct = len(np.unique(np.round(gpr_predictions, 8)))
    print(f"INFO: GPR 0–5 sensitivity at {h} sessions: {distinct} distinct model returns.")

for h in (1, 4, 5, 10, 20):
    rows = state["models"][h].get("holdout_backtest", [])
    assert len(rows) >= 30, f"{h}-session holdout backtest rows missing."
    for row in rows[:5]:
        assert pd.Timestamp(row["target_date"]) > pd.Timestamp(row["feature_date"])
        assert row["actual_close"] > 0 and row["estimated_close"] > 0
        assert np.isfinite(row["absolute_error"]) and row["absolute_error"] >= 0

print("PASS: Interactive backtest series persisted for every forecast horizon.")
print("PASS: GPR 0–5 tiers map into all three trained GPR features; actual model response is reported, never forced.")
print("PASS: Holdout points contain feature date, target date, actual close, estimated close and error.")
