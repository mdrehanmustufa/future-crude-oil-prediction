# Crude Oil Price Predictor

A two-screen Streamlit app with one clear purpose:

1. **Live price & next-session forecast** — latest available WTI quote, Brent reference quote, and a graph connecting recent WTI closes to the model's next-session closing-price estimate.
2. **Hypothetical scenario** — eight controls for WTI, DXY, US crude inventory, oil rigs, a commodity basket, GPR, TPU and global oil production. Moving a control recalculates the next-session estimate using the saved model.

The target is the WTI closing price, not the opening price. The forecast date is the next scheduled weekday; exchange holidays can change the actual next session.

## Data and quote behaviour

- WTI intraday quote: Yahoo Finance continuous futures ticker CL=F, refreshed in the app approximately every 60 seconds when a recent provider quote is available.
- Brent intraday reference: Yahoo Finance continuous futures ticker BZ=F.
- DXY intraday input: Yahoo Finance ticker DX-Y.NYB.
- If an intraday quote is unavailable or older than 24 hours, the app falls back to the most recent saved daily close and labels the fallback.
- Yahoo Finance intraday quotes may be delayed; they are not a guaranteed exchange real-time feed. CME's real-time futures API is a separately provisioned market-data service.
- GitHub Actions refreshes daily market history on weekdays at 23:00 UTC, after the US market close. Each completed daily row feeds the next-session inference.
- The app independently refreshes the intraday quote; it adjusts the saved model's latest feature row and recalculates the next-session estimate without training.

## Eight what-if parameters

1. WTI price move (%)
2. US Dollar Index (DXY) move (%)
3. US commercial crude inventory change (million barrels)
4. Active US oil rig count change
5. Commodity basket move (%) — Brent, Urals, ESPO, European gas (TTF), wheat and urea together
6. Geopolitical Risk Index (GPR) move (%)
7. Trade Policy Uncertainty (TPU) move (%)
8. Global oil production move (%)

GPR and TPU are lagged historical indices, not live news sentiment. Inventory is weekly; rig counts and several macro/commodity series are carried forward between their source releases. These controls are hypothetical feature shocks, not causal economic laws.

## Model lifecycle — train once, predict daily

- A code branch that changes the model engine trains and validates a new model once. Its tested model artifact is included in the PR.
- Merging that PR into `main` does **not** train the model again. The production push runs the daily update/inference path using the model artifact from the PR.
- Scheduled weekday runs do **not** retrain or overwrite `model_artifacts.joblib`. They fetch new WTI/DXY/Brent observations, rebuild the latest feature row, run inference through the frozen model and save `daily_forecast.json`.
- Manual `workflow_dispatch` is the intentional retraining path.
- Moving a What-if slider never retrains the model.
- Daily generated market data and forecast metadata are committed to the repository; Render auto-deploys from `main` when those files change.

## Forecast validation

The 1-session ExtraTrees model only marginally beat the zero-change baseline on the latest chronological holdout (about 0.2% relative MAE improvement). That is weak evidence, not proof of forecasting skill. The dashboard displays the holdout error and does not promise guaranteed prices. The 4-session model remains the zero-return baseline; longer ExtraTrees horizons remain experimental.

WTI is the forecast target. Brent is a current market reference; the historical lagged Brent column is a separate model input. Several macro/commodity inputs are historical lagged values and are not refreshed intraday.

## Local setup

```bash
pip install -r requirements.txt
python scripts/refresh_and_train.py  # train once / intentional retrain
python scripts/daily_update_and_predict.py  # daily refresh + inference only
streamlit run app.py
```
