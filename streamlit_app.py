import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import date


st.set_page_config(layout="wide", page_title="Portfolio vs S&P500")

st.title("📈 Portfolio vs S&P500 Tracker")

with st.sidebar:
    st.header("Inputs")
    tickers_input = st.text_input(
        "Tickers (comma separated)", value="AAPL, MSFT, AMZN"
    )
    weights_input = st.text_input(
        "Weights (comma separated, optional — sum to 1)", value="0.4,0.4,0.2"
    )
    start_date = st.date_input("Start date", value=date(2020, 1, 1))
    end_date = st.date_input("End date", value=date.today())
    initial_capital = st.number_input("Initial capital ($)", value=10000.0)


@st.cache_data
def fetch_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how="all")
    return data


def compute_portfolio(prices: pd.DataFrame, weights: np.ndarray, capital: float):
    norm = prices / prices.iloc[0]
    weighted = norm.mul(weights, axis=1)
    port_index = weighted.sum(axis=1)
    port_values = port_index * capital
    returns = port_values.pct_change().fillna(0)
    return port_values, returns


def performance_metrics(series: pd.Series):
    total_ret = series.iloc[-1] / series.iloc[0] - 1
    days = (series.index[-1] - series.index[0]).days
    annual_ret = (1 + total_ret) ** (365.0 / days) - 1 if days > 0 else 0.0
    daily_rets = series.pct_change().dropna()
    vol = daily_rets.std() * np.sqrt(252)
    roll_max = series.cummax()
    drawdown = (series / roll_max) - 1
    max_dd = drawdown.min()
    return {
        "Total Return": total_ret,
        "Annualized Return": annual_ret,
        "Volatility (ann.)": vol,
        "Max Drawdown": max_dd,
    }


def parse_weights(n_tickers, weights_text):
    try:
        parts = [float(x) for x in weights_text.split(",") if x.strip() != ""]
        arr = np.array(parts, dtype=float)
        if arr.size == 0:
            return np.repeat(1.0 / n_tickers, n_tickers)
        if arr.size != n_tickers:
            return np.repeat(1.0 / n_tickers, n_tickers)
        s = arr.sum()
        if s == 0:
            return np.repeat(1.0 / n_tickers, n_tickers)
        return arr / s
    except Exception:
        return np.repeat(1.0 / n_tickers, n_tickers)


if st.button("Run"):
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]
    if len(tickers) == 0:
        st.error("Please enter at least one ticker.")
    else:
        prices = fetch_prices(tickers + ["^GSPC"], start_date, end_date)
        if prices.empty:
            st.error("No price data found for the given inputs.")
        else:
            # ensure all tickers present
            common = [c for c in tickers if c in prices.columns]
            if len(common) == 0:
                st.error("None of the provided tickers returned data.")
            else:
                port_prices = prices[common]
                sp_prices = prices["^GSPC"]

                weights = parse_weights(len(common), weights_input)

                port_values, port_returns = compute_portfolio(port_prices, weights, initial_capital)

                sp_norm = sp_prices / sp_prices.iloc[0]
                sp_values = sp_norm * initial_capital

                df_plot = pd.DataFrame({"Portfolio": port_values, "S&P500": sp_values})

                fig = px.line(df_plot, labels={"value": "Value ($)", "index": "Date"})
                st.plotly_chart(fig, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Portfolio Metrics")
                    pm = performance_metrics(port_values)
                    for k, v in pm.items():
                        st.write(f"**{k}:** {v:.2%}")
                with col2:
                    st.subheader("S&P500 Metrics")
                    sm = performance_metrics(sp_values)
                    for k, v in sm.items():
                        st.write(f"**{k}:** {v:.2%}")

                st.subheader("Holdings & Weights")
                holdings = pd.DataFrame({"Ticker": common, "Weight": weights})
                st.table(holdings)

                st.success("Done — chart shows portfolio vs S&P500.")
