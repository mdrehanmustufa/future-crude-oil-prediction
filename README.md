# Crude Oil Market Intelligence

A Streamlit research dashboard that explains WTI and Brent benchmarks, provides a multi-horizon WTI reference outlook, and runs a conditional what-if scenario.

## Dashboard guide

1. **Market snapshot** — last complete saved WTI and Brent daily futures closes, their dates, and the Brent-minus-WTI spread when the dates match.
2. **Market forecast** — WTI next-session closing-price reference estimate and 1/4/5/10/20 observed-trading-session outlook.
3. **What-if analysis** — four exposed controls for WTI price shock, US Dollar Index (DXY), US crude inventories and US oil rig count. All changes are applied to a temporary data copy.

The dashboard labels saved daily futures closes and model estimates explicitly; it does not call them live quotes or tomorrow's opening price.

## Understand the inputs

The saved WTI model uses 53 engineered features grouped into four families:
- WTI price history and technical indicators (returns, momentum, volatility, moving-average distance and drawdown).
- DXY returns, momentum and volatility.
- US oil-rig level and changes.
- US crude-inventory level and changes.

Only WTI, DXY, inventory and rig assumptions are exposed as scenario controls. Geopolitical Risk Index (GPR), war headlines, OPEC+ decisions, weather, shipping disruption, refinery demand and interest rates are not trained inputs in this model. Adding a UI control does not add predictive capability; each new factor requires historical data, date alignment, retraining and out-of-sample validation.

Brent is shown as a market benchmark comparison. The model currently forecasts WTI only; Brent is not passed into the saved WTI model.

## Model limitations

- 1 and 4 sessions use a zero-return baseline.
- 5, 10 and 20 sessions use experimental ExtraTrees pipelines.
- In the current chronological holdout, the ExtraTrees models did not beat the zero-change benchmark.
- The next-session estimate is a closing-price reference based on the dataset target; it is not an opening-price prediction.
- Indicative residual bands are not calibrated confidence intervals.
- Forecasts are for research and education, not trading advice.

## Data sources and freshness

- WTI: Yahoo Finance continuous futures ticker CL=F, accepted only after recent overlap checks against the existing WTI series.
- DXY: Yahoo Finance DX-Y.NYB, accepted only after recent overlap checks.
- Brent: Yahoo Finance continuous futures ticker BZ=F, validated for sufficient recent positive daily bars. Stored for benchmark comparison only.
- US crude inventories: EIA weekly commercial stocks when EIA_API_KEY is configured; otherwise the latest stored observation is retained.
- US oil rig count: carried forward from the stored dataset, not a live daily series.

Displayed prices are saved daily futures closes, not guaranteed live quotes. Always compare the same contract/month, source and timestamp when checking another website; continuous futures and named monthly contracts can differ around contract rolls.

## Automated refresh and retraining

The GitHub Actions workflow runs on pushes to the default branch and the temporary test branches, on weekdays at 23:00 UTC when enabled on the default branch, and manually through workflow_dispatch.

Each successful run refreshes WTI, DXY and Brent data, validates source freshness/overlap, retrains the WTI horizons, saves model_artifacts.joblib and data_refresh_status.json, runs dashboard/scenario smoke tests, then commits updated generated outputs. The workflow uses the optional EIA_API_KEY secret for inventory updates.

The Streamlit app reads the saved dataset, model artifacts and metadata. It does not refresh data or retrain models during a visitor session. Render auto-deploys from main when the branch receives an update.

## Local setup

```bash
pip install -r requirements.txt
python scripts/refresh_and_train.py
streamlit run app.py
```
