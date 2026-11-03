# Crude Oil Market Intelligence

A Streamlit research dashboard explaining WTI and Brent benchmarks, providing multi-horizon WTI reference estimates, and supporting a market what-if simulator.

## Dashboard guide

1. **Market snapshot** — latest completed daily WTI, Brent and DXY closes from Yahoo Finance when available, with saved-dataset fallback and source dates.
2. **Scenario Lab** — six simple, level-based controls: WTI price, Brent price, DXY, US commercial crude stocks, US oil-directed rig count and the Geopolitical Risk Index (GPR). A Reset button restores the latest reference levels.
3. **Forecast & Backtest** — multi-horizon estimates, chronological holdout diagnostics and the 90-day previous-close benchmark replay.

The six visible controls do not replace the model inputs. All 64 trained features still go into the saved model. The other stored features (for example Urals, ESPO, TTF, wheat, urea, TPU and global production) are held at their latest saved row values while the six user-facing scenario assumptions are varied.

### Scenario reference levels

- **Market prices:** latest completed Yahoo Finance daily close for CL=F (WTI), BZ=F (Brent) and DX-Y.NYB (DXY), refreshed in the app cache every 15 minutes. When Yahoo Finance is unavailable, the app falls back to the saved dataset and shows that fallback in the source label.
- **US commercial crude stocks:** 410.6 million barrels, EIA week ending 23 October 2026 (published 28 October 2026). This excludes the Strategic Petroleum Reserve.
- **US oil-directed rigs:** 462, Baker Hughes count for 30 October 2026. This is the oil-specific count, not the total U.S. oil-and-gas rig count.
- **Geopolitical Risk Index:** 150.52, October 2026 monthly reading (Caldara–Iacoviello index, as reported by MacroDataHub, last updated 2 November 2026).

These reference series have different release frequencies. The market quotes are daily; commercial stocks and rig counts are weekly; GPR is monthly. The Reset button restores the latest published reference level for each series, not a fabricated same-day observation.

The simple UI controls use **absolute levels**, not percentage shocks. Inventory is entered in million barrels and converted to the model's thousand-barrel feature. The GPR value scales the stored total, threats and acts feature trio together.

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

- WTI, Brent and DXY controls are absolute market/index levels in USD per barrel or index points.
- Inventory is an absolute level in million barrels; the feature transformation converts it to the model's thousand-barrel (kbbl) unit.
- Rig count is an absolute number of active U.S. oil-directed rigs.
- GPR is an absolute monthly index level; the stored total, threats and acts features move proportionally together.

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
