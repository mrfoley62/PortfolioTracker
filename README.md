# Portfolio Tracker Dashboard

A Streamlit dashboard for tracking stock portfolio performance using Yahoo Finance data.

## Features

- Tracks portfolio performance vs S&P 500
- Calculates annualized returns, alpha, beta, and Sharpe ratio
- Graphs rolling 5-day volatility vs S&P 500
- Automatically updates data (cached for 5 minutes)

## Setup

1. Activate the virtual environment and install dependencies:
   ```
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Update `portfolio.csv` with your stocks:
   - ticker: Stock symbol
   - shares: Number of shares
   - purchase_price: Price per share at purchase

3. Run the app:
   ```
   streamlit run streamlit_app.py
   ```

## Notes

- Data is cached for 5 minutes to avoid rate limits
- Click "Refresh Data" to force update
- Uses Yahoo Finance for free data (no API key required)
