# Future Crude Oil Prediction

A Streamlit research and education dashboard for WTI reference forecasts and conditional market scenarios.

## Main sections

1. Real Market Forecast — saved WTI close, data date, 1/4/5/10/20 observed-trading-session forecasts, recent price history and holdout metrics.
2. What-if Scenario — a separate-copy scenario engine for hypothetical WTI, DXY, US crude inventory and oil-rig assumptions. Dependent features are recalculated before inference. The scenario is conditional, not causal.

Geopolitical risk is context-only because it is not an input to the trained model.

## Models and limitations

- 1 and 4 sessions use a zero-return baseline.
- 5, 10 and 20 sessions use experimental ExtraTrees pipelines.
- In the latest chronological holdout, ExtraTrees did not beat the zero-return benchmark. The app reports this plainly.
- Indicative residual bands are not calibrated confidence intervals.
- Forecasts are estimates for research/education, not trading advice.

## Data sources

- WTI: Yahoo Finance CL=F, accepted only after recent overlap checks.
- DXY: Yahoo Finance DX-Y.NYB, accepted only after recent overlap checks.
- US crude inventory: EIA weekly commercial stocks when EIA_API_KEY is configured; otherwise the latest stored observation is retained.
- US oil rig count: carried forward from the stored dataset, not a live daily series.

## Automated refresh and retraining

The GitHub Actions workflow at .github/workflows/market-refresh.yml is intended to run on pushes to integration-test for initial validation; on weekdays at 23:00 UTC after it is merged into the default main branch; and manually through workflow_dispatch on the default branch.

The workflow needs GitHub Actions permission to write repository contents. EIA_API_KEY is optional.

Each successful run refreshes the dataset, retrains all five horizons, saves model_artifacts.joblib and data_refresh_status.json, runs dashboard/scenario smoke tests, then commits generated outputs to the branch where the workflow ran.

The Streamlit app loads the saved dataset, model artifacts and metadata. It does not independently refresh or retrain models at runtime.

## Local setup

Install dependencies: pip install -r requirements.txt

Refresh and train: python scripts/refresh_and_train.py

Run the app: streamlit run app.py

The integration-test branch is used to validate the workflow without changing main or triggering a Render deployment.
