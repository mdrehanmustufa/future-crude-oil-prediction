# Crude Oil Intelligence

A Streamlit research dashboard explaining WTI and Brent benchmarks, providing multi-horizon WTI reference estimates, and supporting a market what-if simulator.

## Dashboard guide

1. **Global market overview** — latest completed daily WTI, Brent and DXY closes from Yahoo Finance when available, with saved-dataset fallback and source dates.
2. **What-if Scenario Analysis** — five simple absolute-level controls: WTI, Brent, DXY, the existing EIA world-oil-production feature, and GPR. A Reset button restores the current market references and saved lagged global inputs.
3. **Forecast & Backtest** — multi-horizon estimates, chronological holdout diagnostics and the 90-day previous-close benchmark replay.

The dashboard title is **Crude Oil Intelligence**. The main page keeps the market overview first, then places the scenario controls beside the model result and a compact chart comparing reset/reference and changed-scenario WTI estimates at 5, 10 and 20 trading sessions.

All 64 trained model features still go into the saved models. The visible scenario sliders expose only features that can be mapped honestly to existing trained inputs. US commercial inventory and US oil-rig history remain in the model's feature row at their saved values; they are not relabelled as global data.

### Scenario reference levels

- **Market prices:** latest completed Yahoo Finance daily close for CL=F (WTI), BZ=F (Brent) and DX-Y.NYB (DXY), refreshed in the app cache every 15 minutes. When Yahoo Finance is unavailable, the app falls back to the saved dataset and shows that fallback in the source label.
- **Global oil production:** the existing EIA world-production feature, `EIA_world_prod_lag3m`, uses the dataset's original units and three-month lag. The dashboard initializes its slider from the latest complete model feature row; it is not a live daily feed.
- **Geopolitical Risk Index:** the existing GPR total/threats/acts features are adjusted together from the latest complete model row. The training feature is lagged by one month, not a real-time event feed.

Global observed inventories and worldwide rig-count reports are available from external sources, but neither is currently a separately trained input in this model. The IEA publishes a global observed oil-inventories series, and Baker Hughes publishes a worldwide rig-count report. Those series need historical alignment, source-lag handling, inclusion in the training feature matrix, retraining and chronological validation before they can control scenario outputs. See [IEA global observed inventories](https://www.iea.org/data-and-statistics/charts/global-observed-oil-inventories-2021-2026) and [Baker Hughes Worldwide Rig Count](https://rigcount.bakerhughes.com/intl-rig-count/).

The visible controls use **absolute levels**, not percentage shocks. GPR scales the stored total, threats and acts feature trio proportionally. Small changes may not move an ExtraTrees estimate when a new value remains inside the same tree-leaf regions.

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

- WTI and Brent controls are absolute market levels in USD per barrel.
- DXY is an absolute index level.
- Global oil production is adjusted as an absolute value in the original EIA dataset units; the stored feature is lagged by three months.
- GPR is an absolute index level; the stored total, threats and acts features move proportionally together.
- US commercial inventory and US oil-rig features remain fixed at the latest saved historical row and are not presented as global scenario sliders.

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
