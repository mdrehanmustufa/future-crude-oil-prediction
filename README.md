# Future Crude Oil Prediction

A Streamlit research/educational dashboard for multi-horizon WTI crude oil reference forecasts.

## Forecast horizons

- 1 and 4 sessions: zero-return baseline.
- 5, 10 and 20 sessions: experimental Extra Trees models.

The Extra Trees models are experimental. In the current chronological holdout evaluation they did not beat the zero-return baseline. Forecasts are estimates, not guarantees or trading advice.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data and refresh

The app prefers `Final_Dataset_AI_dxy_fixed.csv`, then falls back to `Final_Dataset_AI.csv` or the public GitHub raw CSV. It attempts to refresh WTI (`CL=F`) and DXY (`DX-Y.NYB`) from Yahoo Finance after checking recent overlap with the stored series. Inventory refresh requires an `EIA_API_KEY`; otherwise the last stored inventory observation is retained. Oil-specific rig counts are carried forward from the dataset.

The app refreshes/refits on access after its six-hour cache expires or when the refresh button is clicked. This is not an unattended scheduled job.
