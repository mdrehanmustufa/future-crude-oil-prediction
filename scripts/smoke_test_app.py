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
assert len(app_test.slider) == 6, f"Expected six simple scenario sliders, found {len(app_test.slider)}"
assert len(app_test.number_input) == 0
assert len(app_test.selectbox) == 1
assert len(app_test.button) == 1, "Expected a single Reset to latest data button."

initial_slider_values = [float(item.value) for item in app_test.slider]
app_test.slider[0].set_value(initial_slider_values[0] * 1.05)
app_test.slider[1].set_value(initial_slider_values[1] * 1.05)
app_test.slider[2].set_value(initial_slider_values[2] * 1.02)
app_test.slider[3].set_value(initial_slider_values[3] + 5.0)
app_test.slider[4].set_value(initial_slider_values[4] + 5)
app_test.slider[5].set_value(initial_slider_values[5] + 5.0)
app_test.run(timeout=120)
assert not app_test.exception, [exc.message for exc in app_test.exception]
assert not app_test.error, [error.value for error in app_test.error]

app_test.button[0].click()
app_test.run(timeout=120)
assert not app_test.exception, [exc.message for exc in app_test.exception]
reset_slider_values = [float(item.value) for item in app_test.slider]
assert np.allclose(reset_slider_values, initial_slider_values), (
    f"Reset did not restore latest-reference values: {reset_slider_values} vs {initial_slider_values}"
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

# Use the latest saved feature-row levels as a test baseline; web reference values
# are separately shown as current defaults in the user interface.
idx = state["feature_idx"]
base_wti = float(df.loc[idx, "WTI_Close"])
base_dxy = float(df.loc[idx, "DXY_Close"])
base_inventory_mmbbl = float(df.loc[idx, "INV_US_Crude_Stocks_kbbl"]) / 1000.0
base_rigs = float(df.loc[idx, "RIG_US_Oil_Rigs"])
base_brent = float(df.loc[idx, "CMD_brent_lag1"])
base_gpr = float(df.loc[idx, "GPR_total_lag1m"])

baseline_x, baseline_close = scenario_function(
    state, base_wti, base_dxy, base_inventory_mmbbl, base_rigs, base_brent, base_gpr
)
saved_latest_x = state["models"][1]["latest_x"].loc[:, state["feature_names"]]
assert np.allclose(
    baseline_x.to_numpy(dtype=float),
    saved_latest_x.to_numpy(dtype=float),
    rtol=1e-6, atol=1e-8, equal_nan=True
)
assert np.isclose(baseline_close, base_wti)

scenario_specs = [
    ("WTI", scenario_function(state, base_wti * 1.10, base_dxy, base_inventory_mmbbl, base_rigs, base_brent, base_gpr), ["WTI_"]),
    ("Brent", scenario_function(state, base_wti, base_dxy, base_inventory_mmbbl, base_rigs, base_brent * 1.15, base_gpr), ["CMD_brent_lag1"]),
    ("DXY", scenario_function(state, base_wti, base_dxy * 1.05, base_inventory_mmbbl, base_rigs, base_brent, base_gpr), ["DXY_"]),
    ("Commercial crude inventory", scenario_function(state, base_wti, base_dxy, base_inventory_mmbbl + 20.0, base_rigs, base_brent, base_gpr), ["INV_US_Crude_Stocks_kbbl"]),
    ("US oil rig count", scenario_function(state, base_wti, base_dxy, base_inventory_mmbbl, base_rigs + 50, base_brent, base_gpr), ["RIG_US_Oil_Rigs"]),
    ("GPR", scenario_function(state, base_wti, base_dxy, base_inventory_mmbbl, base_rigs, base_brent, base_gpr * 1.5), ["GPR_"]),
]
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
print("PASS: Streamlit app renders two tabs, six scenario sliders, and the reset button.")
print("PASS: Reset button restores latest-reference control values.")
print("PASS: Reset scenario levels reproduce the saved 64-feature baseline row.")
print("PASS: WTI, Brent, DXY, commercial inventory, oil rigs and GPR levels each update their mapped trained features.")
print("PASS: All 5/10/20-session scenario predictions remain finite and at least one non-WTI driver changes each horizon.")
