# Crude Oil Price Predictor

A two-screen Streamlit app with one clear purpose:

1. **Live price & tomorrow forecast** — latest available WTI quote, Brent reference quote, and a graph connecting recent WTI closes to the model's next-session closing-price estimate.
2. **Hypothetical scenario** — eight controls for WTI, DXY, US crude inventory, oil rigs, a commodity basket, GPR, TPU and global oil production. Moving a control recalculates the next-session estimate using the saved model.

The next-session target is the WTI closing price, not the opening price. The forecast date is the next scheduled weekday; exchange holidays can change the actual next session.

## Data and live quote behaviour

- WTI intraday quote: Yahoo Finance continuous futures ticker CL=F, refreshed on the app approximately every 60 seconds when a recent provider quote is available.
- Brent intraday reference: Yahoo Finance continuous futures ticker BZ=F.
- DXY intraday input: Yahoo Finance ticker DX-Y.NYB.
- When an intraday quote is unavailable or older than 24 hours, the app falls back to the most recent saved daily close and labels the fallback.
- Yahoo Finance intraday quotes may be delayed and are not a guaranteed exchange real-time feed. CME's real-time futures API is a separately provisioned market-data service.
- Daily market history is refreshed by GitHub Actions on weekdays at 23:00 UTC, after the US market close. The next-session model estimate is regenerated from the latest full daily feature row.
- The app updates its displayed intraday current quote independently; the saved-model estimate is recalculated from that quote when the app refreshes.

## Eight what-if parameters

1. WTI price move (%)
2. US Dollar Index (DXY) move (%)
3. US commercial crude inventory change (million barrels)
4. Active US oil rig count change
5. Commodity basket move (%) — Brent, Urals, ESPO, European gas (TTF), wheat and urea together
6. Geopolitical Risk Index (GPR) move (%)
7. Trade Policy Uncertainty (TPU) move (%)
8. Global oil production move (%)

GPR and TPU are lagged historical indices, not live news sentiment. Inventory is weekly; rig counts and several macro/commodity series are carried forward between their source releases. Scenario controls are hypothetical feature shocks, not causal economic laws.

## Model lifecycle — train once, predict daily

- The 64-feature model is trained on code changes or an intentional manual workflow run.
- Scheduled weekday runs do **not** retrain or overwrite model_artifacts.joblib. They fetch new WTI/DXY/Brent observations, rebuild the latest feature row, run inference through the saved model and save daily_forecast.json.
- The scenario tab also uses the saved pipeline; moving sliders never retrains the model.
- A normal daily run updates the dataset, forecast JSON and freshness metadata. Render auto-deploys from main when those generated files are committed.

## Model quality

- The next-session model is an experimental ExtraTrees model selected so the app can show a genuine model output rather than a flat zero-return baseline.
- The dashboard displays chronological holdout error and warns when the model does not beat the unchanged-price benchmark. An estimate is not a guaranteed market prediction.
- WTI is the forecast target. Brent is shown as a current market reference; the historical lagged Brent input is a separate feature.

## Local setup

```bash
pip install -r requirements.txt
python scripts/refresh_and_train.py  # train once / intentional retrain
python scripts/daily_update_and_predict.py  # daily refresh + inference only
streamlit run app.py
```
