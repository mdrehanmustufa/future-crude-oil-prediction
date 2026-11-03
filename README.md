# Crude Oil Market Intelligence

A Streamlit research dashboard explaining WTI and Brent benchmarks, providing multi-horizon WTI reference estimates, and supporting a market what-if simulator.

## Dashboard guide

1. **Market snapshot** — saved daily WTI and Brent futures closes, source dates, and Brent-minus-WTI spread when both dates match.
2. **Scenario Lab** — primary Expo view, with presets and 13 adjustable inputs covering WTI, DXY, US commercial crude inventories, US oil rigs, six commodity markets, GPR, TPU and global oil production. The scenario estimate is displayed beside the controls and updates when any control changes. The current market snapshot shows WTI, Brent, DXY and the synchronized spread.
3. **Forecast & Backtest** — multi-horizon estimates, chronological holdout diagnostics and a 90-day previous-close benchmark replay.

The dashboard labels saved daily futures closes and model estimates explicitly; it does not call them live quotes or tomorrow's opening price. The target is WTI future close, not open.

## Trained model inputs

The WTI model now uses 64 numeric features across eight families:

- WTI price history and technical indicators (returns, momentum, volatility, moving-average distance and drawdown).
- US Dollar Index (DXY) returns, momentum and volatility.
- Lagged commodity prices: Brent, Urals, ESPO, European natural gas (TTF), wheat and urea.
- Lagged Geopolitical Risk (GPR) total, threats and acts indices.
- Lagged Trade Policy Uncertainty (TPU) index.
- Lagged global oil production.
- US oil rig-count level and changes.
- US crude inventory level and changes.

The dataset contains the historical lagged columns for these added macro/commodity factors. Their values are carried forward on appended WTI dates unless their own source is refreshed. GPR and TPU are lagged historical indices, not live news sentiment. The Brent card uses Yahoo continuous futures BZ=F for a visible benchmark quote; the model's CMD_brent_lag1 is a separate stored historical lagged feature.

Strategic Petroleum Reserve (SPR) and IEA emergency-stock release volumes, OPEC+ policy, current war/peace headlines outside the lagged GPR series, tanker/shipping disruptions, weather, refinery utilization and interest rates are not separate trained inputs in this version. US commercial crude inventory is not the same as strategic emergency stocks. A UI slider alone is not a valid model feature; each new factor requires historical data, date alignment, lag policy, retraining and out-of-sample validation.

## Scenario controls and units

- WTI and DXY shocks are percentages applied to the latest complete feature row.
- Inventory is in thousand barrels (kbbl); +2,000 kbbl is +2 million barrels.
- Rig count is a number of active oil-directed rigs.
- The six commodity controls are percentage shocks to the corresponding stored lagged values.
- The GPR control applies the same proportional shock to total, threats and acts indices.
- TPU and global production controls apply proportional shocks to their stored lagged values.

Scenarios are conditional feature experiments, not causal economic simulations. A changed input does not prove it causes an oil-price move.

## Model limitations

- 1 and 4 sessions use a zero-return baseline.
- 5, 10 and 20 sessions use experimental ExtraTrees pipelines.
- The current chronological holdout must be checked after retraining; extra inputs do not guarantee better accuracy.
- The next-session estimate is a closing-price reference based on the dataset target; it is not an opening-price prediction.
- Indicative residual bands are not calibrated confidence intervals.
- Forecasts are for research and education, not trading advice.

## Data sources and freshness

- WTI: Yahoo Finance continuous futures ticker CL=F, accepted only after recent overlap checks against the existing WTI series.
- DXY: Yahoo Finance DX-Y.NYB, accepted only after recent overlap checks.
- Brent: Yahoo Finance continuous futures ticker BZ=F, validated for sufficient recent positive daily bars. Stored for benchmark comparison.
- US crude inventories: EIA weekly commercial stocks when EIA_API_KEY is configured; otherwise the latest stored observation is retained.
- US oil rig count: carried forward from the stored dataset, not a live daily series.
- Macro/commodity columns: the original dataset's lagged series; they are not refreshed by dedicated current source connectors in this workflow.

Displayed prices are saved daily futures closes, not guaranteed live quotes. Always compare the same contract/month, source and timestamp when checking another website; continuous futures and named monthly contracts can differ around contract rolls.

## Automated refresh and retraining

The GitHub Actions workflow runs on pushes to the default branch and test branch, on weekdays at 23:00 UTC when enabled on the default branch, and manually through workflow_dispatch.

Each successful run refreshes WTI, DXY and Brent data, validates source freshness/overlap, retrains all WTI horizons using the 64-feature matrix, saves model_artifacts.joblib and data_refresh_status.json, runs dashboard/scenario smoke tests, then commits updated generated outputs. The workflow uses the optional EIA_API_KEY secret for inventory updates.

The Streamlit app reads the saved dataset, model artifacts and metadata. It does not refresh data or retrain models during a visitor session. Render auto-deploys from main when the branch receives an update.

## Local setup

```bash
pip install -r requirements.txt
python scripts/refresh_and_train.py
streamlit run app.py
```
