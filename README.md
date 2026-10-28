# WTI Next-Trading-Day Forecasting App

This is a lightweight Streamlit web application that deploys the exact trained machine learning pipeline of the WTI next-trading-day forecasting project.

## Local Setup Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```

## Deploying to Render
1. Upload `app.py`, `requirements.txt`, `Final_Dataset_AI.csv`, and this `README.md` to your GitHub Repository: https://github.com/mdrehanmustufa/future-crude-oil-prediction
2. Log in to **Render** and create a new **Web Service**.
3. Connect your repository.
4. Set the following configuration details:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`
5. Click deploy.