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
assert len(app_test.slider) == 11, f"Expected 11 WTI/DXY/macro sliders, found {len(app_test.slider)}"
assert len(app_test.number_input) == 2
assert len(app_test.selectbox) == 1

app_test.slider[0].set_value(1.0)
app_test.slider[1].set_value(1.0)
app_test.number_input[0].set_value(1000)
app_test.number_input[1].set_value(5)
app_test.run(timeout=120)
assert not app_test.exception, [exc.message for exc in app_test.exception]
assert not app_test.error, [error.value for error in app_test.error]

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

zero_x, zero_close = scenario_function(state, 0.0, 0.0, 0, 0)
baseline_x = state["models"][1]["latest_x"].loc[:, state["feature_names"]]
assert np.allclose(
    zero_x.to_numpy(dtype=float),
    baseline_x.to_numpy(dtype=float),
    equal_nan=True
)
assert np.isclose(zero_close, state["current_close"])

scenarios = [
    ("WTI", scenario_function(state, 1.0, 0.0, 0, 0)),
    ("DXY", scenario_function(state, 0.0, 1.0, 0, 0)),
    ("inventory", scenario_function(state, 0.0, 0.0, 1000, 0)),
    ("rig count", scenario_function(state, 0.0, 0.0, 0, 5)),
    ("Brent commodity", scenario_function(state, 0.0, 0.0, 0, 0, {"brent": 1.0})),
    ("GPR", scenario_function(state, 0.0, 0.0, 0, 0, {"gpr": 5.0})),
    ("TPU", scenario_function(state, 0.0, 0.0, 0, 0, {"tpu": 5.0})),
    ("global production", scenario_function(state, 0.0, 0.0, 0, 0, {"world_prod": 1.0})),
]
for label, (features, _) in scenarios:
    assert features.shape == (1, len(fe.EXPANDED_FEATURES)), f"{label} feature shape mismatch."
    assert np.isfinite(features.to_numpy(dtype=float)).all(), f"{label} features invalid."
    for record in state["models"].values():
        prediction = float(record["pipeline"].predict(features)[0])
        assert np.isfinite(prediction), f"{label} prediction invalid."

print("PASS: Next-session walk-forward selector handles signal and no-signal test data.")
print("PASS: Streamlit app renders two tabs and scenario controls.")
print("PASS: Zero-change scenario matches saved baseline.")
print("PASS: WTI, DXY, inventory, rig, Brent, GPR, TPU and global-production scenarios produce finite features.")
print("PASS: All horizon predictions remain finite.")
