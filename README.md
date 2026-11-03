# Crude Oil Intelligence

A Streamlit research dashboard explaining WTI and Brent benchmarks, providing multi-horizon WTI reference estimates, and supporting a market what-if simulator.

## Dashboard guide

1. **Global market overview** — latest completed daily WTI, Brent and DXY closes from Yahoo Finance when available, with saved-dataset fallback and source dates.
2. **What-if Scenario Analysis** — five simple absolute-level controls: WTI, Brent, DXY, the existing EIA world-oil-production feature, and GPR. A Reset button restores the current market references and saved lagged global inputs.
3. **Forecast & Backtest** — four market/model status cards, multi-horizon estimates and a point-by-point interactive chronological holdout chart (actual close vs the estimate saved from the model that was fit before the holdout).

The dashboard title is **Crude Oil Intelligence**. The main page keeps the market overview first, places scenario controls beside the model result, and includes a compact 5/10/20-session scenario comparison chart. In Forecast & Backtest, the top status cards show WTI price/change, data source, loaded model status and the next scheduled GitHub Actions refresh.

All 64 trained model features still go into the saved models. The visible scenario sliders expose only features that can be mapped honestly to existing trained inputs. US commercial inventory and US oil-rig history remain in the model's feature row at their saved values; they are not relabelled as global data.

### Scenario reference levels

- **Market prices:** latest completed Yahoo Finance daily close for CL=F (WTI), BZ=F (Brent) and DX-Y.NYB (DXY), refreshed in the app cache every 15 minutes. When Yahoo Finance is unavailable, the app falls back to the saved dataset and shows that fallback in the source label.
- **Global oil production:** the existing EIA world-production feature, `EIA_world_prod_lag3m`, uses the dataset's original units and three-month lag. The dashboard initializes its slider from the latest complete model feature row; it is not a live daily feed.
- **Geopolitical Risk Index:** the existing GPR total/threats/acts features are adjusted together from the latest complete model row. The training feature is lagged by one month, not a real-time event feed.

Global observed inventories and worldwide rig-count reports are available from external sources, but neither is currently a separately trained input in this model. The IEA publishes a global observed oil-inventories series, and Baker Hughes publishes a worldwide rig-count report. Those series need historical alignment, source-lag handling, inclusion in the training feature matrix, retraining and chronological validation before they can control scenario outputs. See [IEA global observed inventories](https://www.iea.org/data-and-statistics/charts/global-observed-oil-inventories-2021-2026) and [Baker Hughes Worldwide Rig Count](https://rigcount.bakerhughes.com/intl-rig-count/).

The visible controls use **absolute levels**, not percentage shocks.

- **Global oil production** has a broad 60,000–100,000 prototype slider range so users can test hypothetical supply shocks, including 64,725. The existing ML feature is still `EIA_world_prod_lag3m`, lagged by three months; values outside historical support are not reliable ML extrapolations.
- **Prototype stress overlay:** each 1% production decline relative to the saved reference adds 0.8% to the displayed scenario price (with a 50% lower multiplier floor); each 1% production rise subtracts 0.8%. Geopolitical level 0–5 adds 0%, 2.5%, 5%, 7.5%, 10%, or 12.5%, respectively. Supply and geopolitical multipliers are combined.
- **Reset-anchored WTI scenario price** starts at the latest completed WTI market close. When inputs change, the app applies the relative change between the ML scenario estimate and its reset-input estimate, multiplied by the relative supply/GPR stress factor. This keeps the unchanged case aligned to the observed WTI reference instead of displaying a model-only offset as the starting market price.
- The supply and geopolitical multipliers remain explicit scenario assumptions; they are not coefficients learned or validated by the ML model.

The geopolitical control is an event-scenario ladder from 0 to 5:

- **0:** Calm / no major escalation
- **1:** Routine diplomatic tension
- **2:** Direct military clash between countries
- **3:** Sanctions / trade restrictions
- **4:** Sustained regional conflict
- **5:** Major escalation / serious energy-supply threat

Each level maps to a percentile of the model's historical GPR distribution (P5, P25, P50, P70, P85, P97); the selected index is applied proportionally to the stored total, threats and acts GPR features. These labels are illustrative what-if scenarios, not an official conflict classification. The Caldara–Iacoviello GPR index is a news-coverage measure based on the share of newspaper articles about geopolitical risk, with separate threats and acts components; it is not a literal 0–5 war scale. See the [official GPR index and methodology](https://www.matteoiacoviello.com/gpr.htm). The latest observed GPR is preserved as the reset reference. ExtraTrees can return the same estimate across several tiers when inputs fall within the same tree leaves; the app displays the dollar and percentage change from level 0 and warns when the full-ladder response is below 0.10%, rather than forcing a price movement.

The current 64-feature model uses a generic monthly GPR index lagged by one month. It does **not** yet include an oil-specific event/supply disruption feature. A useful research extension is the [AI-GPR daily dataset](https://www.matteoiacoviello.com/ai_gpr.html), which provides separate **Oil Threats GPR** and **Oil Acts GPR** series. Those features must be historically aligned and added to training, then pass chronological validation before being treated as model inputs.

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
- The current chronological holdout has not shown a reliable improvement over the zero-return benchmark.
- The Forecast & Backtest chart uses saved chronological holdout predictions. For 5/10/20-session ExtraTrees horizons, the model is refit every 21 forecast origins using only labels whose target dates are strictly before each refit block. This expanding-window walk-forward design lets the model adapt as new observations become available without using future labels. The 1/4-session baselines remain unchanged.
- GPR is a monthly lagged news-based index, not a live event detector. The 0–5 event ladder maps illustrative scenario labels to empirical percentiles of the training-set GPR distribution. It does not claim that a specific war or sanction always corresponds to one exact GPR number.
- ExtraTrees is piecewise constant; a changed GPR input may produce no change in the model estimate. The app reports this instead of fabricating a movement.
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
