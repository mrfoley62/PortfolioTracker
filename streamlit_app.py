import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from scipy import stats
from scipy.stats import gaussian_kde

# Configure page
st.set_page_config(
    page_title="Portfolio Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get sector info
def get_sector(ticker):
    try:
        return yf.Ticker(ticker).info.get('sector', 'Unknown')
    except:
        return 'Unknown'

# Load portfolio
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_portfolio():
    df = pd.read_csv('portfolio.csv')
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    return df

# Get current price
def get_current_price(ticker):
    try:
        return yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
    except:
        return None

# Get Yahoo Finance beta
def get_yf_beta(ticker):
    try:
        return yf.Ticker(ticker).info.get('beta', None)
    except:
        return None

# Get EV/EBITDA metric
def get_ev_ebitda(ticker):
    try:
        return yf.Ticker(ticker).info.get('enterpriseToEbitda', None)
    except:
        return None

# Get historical data
@st.cache_data(ttl=300)
def get_historical_data(ticker):
    try:
        # Always use auto_adjust=True to get dividend-adjusted prices
        data = yf.Ticker(ticker).history(period='3y', auto_adjust=True)
        return data['Close']
    except:
        return None

def calculate_metrics(portfolio, historical_prices, spy_prices, rf_percent=4.2):
    portfolio['current_price'] = pd.to_numeric(portfolio['current_price'], errors='coerce')
    portfolio['current_value'] = portfolio['shares'] * portfolio['current_price']
    portfolio['purchase_value'] = portfolio['shares'] * portfolio['purchase_price']
    total_current = portfolio['current_value'].sum()
    total_purchase = portfolio['purchase_value'].sum()
    total_return = (total_current - total_purchase) / total_purchase if total_purchase != 0 else 0

    if historical_prices.empty or spy_prices.empty:
        return None

    # Historical portfolio returns (vectorized)
    dates = historical_prices.index
    if len(dates) == 0:
        return None

    # Use only tickers that exist in historical_prices
    tickers_in_prices = [t for t in portfolio['ticker'] if t in historical_prices.columns]
    if tickers_in_prices:
        # Shares indexed by ticker, aligned with historical_prices columns
        shares_by_ticker = portfolio.set_index('ticker').loc[tickers_in_prices]['shares']
        price_subset = historical_prices[tickers_in_prices]
        # Multiply prices by shares for each ticker and sum across tickers per date
        port_prices = price_subset.mul(shares_by_ticker, axis=1).sum(axis=1)
    else:
        # No overlapping tickers; portfolio value is zero across all dates
        port_prices = pd.Series(0.0, index=dates)

    if port_prices.empty or port_prices.isna().all():
        return None
    
    # Remove timezone info to match purchase dates
    if port_prices.index.tz is not None:
        port_prices.index = port_prices.index.tz_localize(None)
    if spy_prices.index.tz is not None:
        spy_prices.index = spy_prices.index.tz_localize(None)
    
    # Resample to monthly data (before filtering)
    port_monthly = port_prices.resample('ME').last()
    spy_monthly = spy_prices.resample('ME').last()
    
    port_monthly_returns = port_monthly.pct_change().dropna()
    spy_monthly_returns = spy_monthly.pct_change().dropna()

    # Align dates for full dataset (for volatility calculation)
    common_dates_full = port_monthly_returns.index.intersection(spy_monthly_returns.index)
    if len(common_dates_full) == 0:
        return None
    port_ret_full = port_monthly_returns.loc[common_dates_full]
    spy_ret_full = spy_monthly_returns.loc[common_dates_full]
    
    # Calculate rolling volatility on full dataset
    port_vol_full = port_ret_full.rolling(6).std() * np.sqrt(12)
    spy_vol_full = spy_ret_full.rolling(6).std() * np.sqrt(12)
    
    # Now filter prices to start from earliest purchase date
    earliest_purchase = portfolio['purchase_date'].min()
    port_prices = port_prices[port_prices.index >= earliest_purchase]
    spy_prices = spy_prices[spy_prices.index >= earliest_purchase]
    
    if port_prices.empty or spy_prices.empty:
        return None
    
    port_prices_norm = port_prices / port_prices.iloc[0] * 100
    spy_prices_norm = spy_prices / spy_prices.iloc[0] * 100
    
    # Filter volatility to match the filtered price range
    port_vol = port_vol_full[port_vol_full.index >= earliest_purchase]
    spy_vol = spy_vol_full[spy_vol_full.index >= earliest_purchase]
    
    # Align dates for regression
    common_dates = port_ret_full.index.intersection(spy_ret_full.index)
    if len(common_dates) == 0:
        return None
    port_ret = port_ret_full.loc[common_dates]
    spy_ret = spy_ret_full.loc[common_dates]

    # Beta and Alpha
    if len(common_dates) >= 2 and spy_ret.nunique() > 1:
        slope, intercept, _, _, _ = stats.linregress(spy_ret, port_ret)
        beta = slope
        alpha = intercept * 12  # Annualized (12 months)
    else:
        beta = np.nan
        alpha = np.nan

    # Sharpe Ratio (using configurable rf)
    rf_annual = rf_percent / 100
    rf_monthly = rf_annual / 12
    port_std = port_ret.std()
    if port_std == 0:
        sharpe = 0.0
    else:
        sharpe = ((port_ret.mean() - rf_monthly) * 12) / (port_std * np.sqrt(12))

    # Annualized return
    current_date = pd.Timestamp.now()
    holding_start = portfolio['purchase_date'].min()
    holding_period_days = (current_date - holding_start).days
    ann_return = (1 + total_return) ** (365 / holding_period_days) - 1 if holding_period_days > 0 else 0

    # Calculate Value at Risk (VaR) - 95% confidence level
    # VaR is the maximum expected loss at a given confidence level
    var_95 = np.percentile(port_ret, 5)  # 5th percentile for 95% confidence
    var_99 = np.percentile(port_ret, 1)  # 1st percentile for 99% confidence

    # Calculate Maximum Drawdown
    # Find the peak value and maximum decline from peak
    cumulative_returns = (1 + port_ret).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # Best and worst monthly returns
    best_month = port_ret.max()
    worst_month = port_ret.min()

    return {
        'total_current': total_current,
        'total_purchase': total_purchase,
        'total_return': total_return,
        'ann_return': ann_return,
        'beta': beta,
        'alpha': alpha,
        'sharpe': sharpe,
        'port_vol': port_vol,
        'spy_vol': spy_vol,
        'port_prices': port_prices_norm,
        'spy_prices': spy_prices_norm,
        'var_95': var_95,
        'var_99': var_99,
        'max_drawdown': max_drawdown,
        'best_month': best_month,
        'worst_month': worst_month,
        'portfolio': portfolio,
        'port_returns': port_ret  # Monthly returns for VaR visualization
    }

# Main app
st.title("Portfolio Analysis Dashboard")

# Custom CSS inspired by professional analytics platforms like Kpler
st.markdown("""
<style>
    /* Overall background and font - Clean professional palette */
    * {
        margin: 0;
        padding: 0;
    }
    
    body {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 50%, #0f1419 100%);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
        color: #e1e4e8;
        letter-spacing: 0.3px;
    }
    
    /* Main container styling - Refined with Kpler-inspired design */
    .main .block-container {
        background: rgba(15, 20, 25, 0.95);
        border-radius: 12px;
        padding: 3.5rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.03);
        margin: 1.5rem 1rem;
        width: 100%;
        max-width: none;
        border: 1px solid rgba(48, 128, 255, 0.12);
        backdrop-filter: blur(8px);
    }
    
    /* Reduce overall page margins */
    .main {
        padding: 0 !important;
    }
    
    /* Title styling - Sophisticated and clean */
    h1 {
        color: #f0f6fc;
        font-weight: 800;
        text-align: left;
        margin-bottom: 0.5rem;
        font-size: 2.2rem;
        letter-spacing: -0.8px;
        text-transform: none;
    }
    
    /* Subtitle line under title */
    h1::after {
        content: '';
        display: block;
        height: 3px;
        background: linear-gradient(90deg, #3080ff 0%, #60a5fa 100%);
        width: 60px;
        margin-top: 1.5rem;
        margin-bottom: 3rem;
        border-radius: 2px;
    }
    
    /* Subheader styling with refined spacing and underline */
    h3 {
        color: #f0f6fc;
        font-weight: 600;
        border-bottom: 1px solid rgba(48, 128, 255, 0.2);
        padding-bottom: 1rem;
        margin-top: 3.5rem;
        margin-bottom: 2.5rem;
        font-size: 1.3rem;
        letter-spacing: -0.3px;
        text-transform: none;
    }
    
    /* First subheader has less top margin */
    h3:first-of-type {
        margin-top: 1rem;
    }
    
    /* Dataframe styling - Clean professional look */
    .dataframe {
        border: none;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        background-color: rgba(19, 24, 34, 0.8);
        border: 1px solid rgba(48, 128, 255, 0.08);
    }
    
    .dataframe th {
        background: rgba(48, 128, 255, 0.08);
        color: #f0f6fc;
        font-weight: 700;
        padding: 14px 14px;
        text-align: left;
        font-size: 0.85rem;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        border-bottom: 2px solid rgba(48, 128, 255, 0.15);
    }
    
    .dataframe td {
        padding: 12px 14px;
        border-bottom: 1px solid rgba(48, 128, 255, 0.06);
        color: #d0d7de;
        font-size: 0.9rem;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: rgba(48, 128, 255, 0.02);
    }
    
    .dataframe tr:hover {
        background-color: rgba(48, 128, 255, 0.08);
        transition: background-color 0.2s ease;
    }
    
    /* Metric styling - Kpler-inspired card design */
    .stMetric {
        background: linear-gradient(135deg, rgba(48, 128, 255, 0.08) 0%, rgba(96, 165, 250, 0.05) 100%);
        color: #e1e4e8;
        padding: 1.8rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.8rem 0;
        border: 1px solid rgba(48, 128, 255, 0.15);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(8px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stMetric:hover {
        border-color: rgba(48, 128, 255, 0.3);
        background: linear-gradient(135deg, rgba(48, 128, 255, 0.12) 0%, rgba(96, 165, 250, 0.08) 100%);
        box-shadow: 0 6px 20px rgba(48, 128, 255, 0.12);
        transform: translateY(-2px);
    }
    
    .stMetric label {
        color: #8b949e;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    
    .stMetric .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.8rem;
        color: #58a6ff;
        word-break: break-word;
        overflow-wrap: break-word;
        white-space: normal;
        line-height: 1.3;
    }
    
    .stMetric .metric-delta {
        font-size: 0.85rem;
        margin-top: 0.6rem;
        color: #79c0ff;
    }
    
    /* Button styling - Professional Kpler-style */
    .stButton > button {
        background: linear-gradient(135deg, #3080ff 0%, #60a5fa 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 1.2rem 3rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 24px rgba(48, 128, 255, 0.25);
        letter-spacing: 0.3px;
        font-size: 0.95rem;
        width: 100%;
        margin-top: 3rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 36px rgba(48, 128, 255, 0.35);
        background: linear-gradient(135deg, #4890ff 0%, #75b5ff 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Input fields */
    .stNumberInput, .stSelectbox {
        background-color: rgba(22, 27, 34, 0.4);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(48, 128, 255, 0.1) 0%, rgba(96, 165, 250, 0.06) 100%);
        border-radius: 10px;
        color: #f0f6fc;
        border: 1px solid rgba(48, 128, 255, 0.15);
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(48, 128, 255, 0.15) 0%, rgba(96, 165, 250, 0.1) 100%);
        border-color: rgba(48, 128, 255, 0.25);
    }
    
    /* Plotly charts */
    .plotly-graph-div {
        background-color: transparent !important;
    }
    
    /* Section separator for visual hierarchy - Subtle Kpler style */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, rgba(48, 128, 255, 0.2) 50%, transparent 100%);
        margin: 2.5rem 0;
        border-radius: 1px;
    }
    
    /* Subtitle styling for chart titles within sections */
    [data-testid="stWrite"] > p {
        color: #f0f6fc;
        font-weight: 600;
        font-size: 1rem;
        margin: 2rem 0 1rem 0 !important;
        letter-spacing: -0.3px;
    }
</style>
""", unsafe_allow_html=True)

portfolio = load_portfolio()

# Settings Section with refined styling
with st.expander("⚙️ Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        rf_input = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=4.2, step=0.1)
    with col2:
        beta_source = st.selectbox("Beta Calculation", options=["Custom (3-Year Monthly)", "Yahoo Finance"], index=1)

# Get historical data for all tickers + SPY
tickers = portfolio['ticker'].tolist() + ['SPY']
historical_data = {}
for ticker in tickers:
    historical_data[ticker] = get_historical_data(ticker)

# Add additional columns to portfolio
portfolio['current_price'] = portfolio['ticker'].apply(get_current_price)
portfolio['cumulative_return'] = ((portfolio['current_price'] - portfolio['purchase_price']) / portfolio['purchase_price'] * 100).round(2)

def get_ltm_return(row):
    ticker = row['ticker']
    current = row['current_price']
    hist = historical_data.get(ticker, pd.Series())
    if hist.empty or pd.isna(current):
        return None
    # Get price from approximately 1 year ago (252 trading days)
    # The data is daily, so count back 252 days from the end
    if len(hist) > 252:
        past = hist.iloc[-252]  # Approximately 1 year ago (252 trading days per year)
    else:
        past = hist.iloc[0]  # Fall back to oldest if less than 1 year of data
    if pd.isna(past) or past == 0:
        return None
    return ((current - past) / past * 100).round(2)

portfolio['ltm_return'] = portfolio.apply(get_ltm_return, axis=1)

# Add total row
total_shares = portfolio['shares'].sum()
total_purchase_value = (portfolio['shares'] * portfolio['purchase_price']).sum()
total_current_value = (portfolio['shares'] * portfolio['current_price']).sum()
total_cumulative_return = ((total_current_value - total_purchase_value) / total_purchase_value * 100).round(2) if total_purchase_value > 0 else 0

# Calculate weighted average LTM return
portfolio_no_total = portfolio[portfolio['ticker'] != 'Total']
if total_current_value > 0:
    weights = (portfolio_no_total['shares'] * portfolio_no_total['current_price']) / total_current_value
    valid_ltm = portfolio_no_total['ltm_return'].notna()
    if valid_ltm.any():
        weighted_ltm = (weights[valid_ltm] * portfolio_no_total.loc[valid_ltm, 'ltm_return']).sum()
    else:
        weighted_ltm = None
else:
    weighted_ltm = None

total_row = {
    'ticker': 'Total',
    'shares': total_shares,
    'purchase_date': '',
    'purchase_price': None,
    'current_price': None,
    'cumulative_return': total_cumulative_return,
    'ltm_return': weighted_ltm.round(2) if weighted_ltm is not None else None,
    'weight': 100.00,
    'current_value': None,
    'historical_value': None
}

# Calculate weights for stocks
portfolio.loc[portfolio['ticker'] != 'Total', 'weight'] = (portfolio.loc[portfolio['ticker'] != 'Total', 'shares'] * portfolio.loc[portfolio['ticker'] != 'Total', 'current_price'] / total_current_value * 100).round(2)

# Add position value columns BEFORE creating total row
portfolio['historical_value'] = portfolio.apply(
    lambda row: row['shares'] * row['purchase_price'],
    axis=1
)
portfolio['current_value'] = portfolio.apply(
    lambda row: row['shares'] * row['current_price'],
    axis=1
)

# Calculate totals for the Total row (sum of individual values)
total_historical_value = portfolio.loc[portfolio['ticker'] != 'Total', 'historical_value'].sum()
total_current_value_calc = portfolio.loc[portfolio['ticker'] != 'Total', 'current_value'].sum()

# Update total_row with the calculated values
total_row['historical_value'] = total_historical_value
total_row['current_value'] = total_current_value_calc

# Create total row DataFrame with proper column alignment
total_df = pd.DataFrame([total_row])
# Ensure all columns match the portfolio DataFrame
for col in portfolio.columns:
    if col not in total_df.columns:
        total_df[col] = None

portfolio = pd.concat([portfolio, total_df[portfolio.columns]], ignore_index=True)

# Format columns for display
display_portfolio = portfolio.copy()
display_portfolio['shares'] = display_portfolio['shares'].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else '')
display_portfolio['purchase_price'] = display_portfolio['purchase_price'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else '')
display_portfolio['current_price'] = display_portfolio['current_price'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else '')
display_portfolio['weight'] = display_portfolio['weight'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else '')
display_portfolio['current_value'] = display_portfolio['current_value'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else '')
display_portfolio['historical_value'] = display_portfolio['historical_value'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else '')
display_portfolio['cumulative_return'] = display_portfolio['cumulative_return'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else '')
display_portfolio['ltm_return'] = display_portfolio['ltm_return'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else '')

# Reorder columns
display_portfolio = display_portfolio[['ticker', 'shares', 'purchase_date', 'purchase_price', 'historical_value', 'current_price', 'current_value', 'weight', 'cumulative_return', 'ltm_return']]

st.subheader("Portfolio")
st.dataframe(display_portfolio)

# Section divider
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Combine into df - moved earlier so we can use it in the allocation section
historical_data_dict = {}
for ticker in historical_data:
    historical_data_dict[ticker] = historical_data[ticker]

hist_df = pd.DataFrame(historical_data_dict)

if hist_df.empty or hist_df.shape[1] == 0:
    st.error("Unable to fetch historical data. Please check your Finnhub API key and internet connection.")
    st.stop()

# Allocation pie charts
st.subheader("Portfolio Allocation")
col1, col2 = st.columns([1, 1.3])

# Get sector data for pie charts
portfolio_data = portfolio[portfolio['ticker'] != 'Total'].copy()
portfolio_data['sector'] = portfolio_data['ticker'].apply(get_sector)
portfolio_data['value'] = portfolio_data['shares'] * portfolio_data['current_price']

# Sector allocation pie chart
sector_allocation = portfolio_data.groupby('sector')['value'].sum()
fig_sector = go.Figure(data=[go.Pie(labels=sector_allocation.index, values=sector_allocation.values, hole=0,
                                     marker=dict(colors=['#0c2340', '#1e40af', '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe', '#e0f2fe']))])
fig_sector.update_layout(height=320, width=320, margin=dict(l=20, r=20, t=20, b=20))

# Stock allocation pie chart - generate colors
stock_allocation = portfolio_data[['ticker', 'value']].groupby('ticker')['value'].sum()
stock_colors = ['#1e40af', '#1d4ed8', '#2563eb', '#3b82f6', '#4a90e2', '#60a5fa', '#7ab8f8', '#93c5fd', '#2563eb', '#3b82f6']
# Extend colors if more stocks than predefined colors
while len(stock_colors) < len(stock_allocation):
    stock_colors.extend(stock_colors)
stock_colors = stock_colors[:len(stock_allocation)]

fig_stock = go.Figure(data=[go.Pie(labels=stock_allocation.index, values=stock_allocation.values, hole=0,
                                    marker=dict(colors=stock_colors))])
fig_stock.update_layout(height=320, width=320, margin=dict(l=20, r=20, t=20, b=20))

with col1:
    st.write("**Allocation by Sector**")
    st.plotly_chart(fig_sector, width='stretch')
    st.write("**Allocation by Stock**")
    st.plotly_chart(fig_stock, width='stretch')
with col2:
    # Correlation Matrix with heatmap
    st.write("**Correlation Matrix**")
    # Get daily returns for each holding
    holdings = portfolio.loc[portfolio['ticker'] != 'Total', 'ticker'].tolist()
    correlation_data = {}
    for ticker in holdings:
        if ticker in hist_df.columns:
            returns = hist_df[ticker].pct_change().dropna()
            correlation_data[ticker] = returns

    if correlation_data:
        corr_df = pd.DataFrame(correlation_data).corr()
        
        # Create heatmap with color coding
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_df.values, 2),
            texttemplate='%{text}',
            textfont={"size": 9},
            colorbar=dict(title="Correlation", thickness=15, len=0.7)
        ))
        fig_corr.update_layout(
            xaxis_title="",
            yaxis_title="",
            width=630,
            height=630,
            margin=dict(l=80, r=20, t=40, b=60)
        )
        st.plotly_chart(fig_corr, use_container_width=False)
    else:
        st.info("Not enough holdings")

# Section divider
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

metrics = calculate_metrics(portfolio[portfolio['ticker'] != 'Total'], hist_df.drop('SPY', axis=1), hist_df['SPY'], rf_percent=rf_input)

if metrics is None:
    st.error("Unable to calculate metrics due to missing historical data. Please check your API key and try again.")
    st.stop()

# Calculate Yahoo Finance beta if selected
if beta_source == "Yahoo Finance":
    port_data = metrics['portfolio']
    yf_betas = []
    weights = []
    total_value = 0
    
    for _, row in port_data.iterrows():
        beta = get_yf_beta(row['ticker'])
        if beta is not None and not pd.isna(beta):
            value = row['shares'] * row['current_price']
            yf_betas.append(beta)
            weights.append(value)
            total_value += value
    
    if yf_betas and total_value > 0:
        # Calculate weighted average beta
        weighted_beta = sum(b * w for b, w in zip(yf_betas, weights)) / total_value
        metrics['beta'] = weighted_beta
        
        # Calculate alpha using the formula: Alpha = (Portfolio Return - RF) - Beta * (Market Return - RF)
        rf_annual = rf_input / 100
        
        # Calculate market (S&P 500) annualized return
        spy_start = hist_df['SPY'].iloc[0]
        spy_end = hist_df['SPY'].iloc[-1]
        spy_days = (hist_df['SPY'].index[-1] - hist_df['SPY'].index[0]).days
        market_return = (spy_end / spy_start) ** (365 / spy_days) - 1 if spy_days > 0 else 0
        
        # Calculate alpha
        portfolio_return = metrics['ann_return']
        alpha = (portfolio_return - rf_annual) - weighted_beta * (market_return - rf_annual)
        metrics['alpha'] = alpha
    else:
        st.warning("Unable to fetch Yahoo Finance beta for all holdings. Using calculated beta instead.")

st.subheader("Portfolio Performance")

# Performance chart full width
fig = go.Figure()
fig.add_trace(go.Scatter(x=metrics['port_prices'].index, y=metrics['port_prices'], mode='lines', name='Portfolio'))
fig.add_trace(go.Scatter(x=metrics['spy_prices'].index, y=metrics['spy_prices'], mode='lines', name='S&P 500'))
fig.update_layout(
    title="Portfolio vs S&P 500 Performance", 
    xaxis_title="Date", 
    yaxis_title="Normalized Value (Base 100)",
    xaxis=dict(tickformat='%m/%y', nticks=20),
    height=550
)
st.plotly_chart(fig, use_container_width=True)

# Portfolio Performance metrics - Returns cards
def format_currency(value):
    """Format currency with abbreviations for large numbers"""
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:.2f}K"
    else:
        return f"${value:,.2f}"

col1, col2, col3 = st.columns(3)
col1.metric("Total Value", format_currency(metrics['total_current']), help=f"${metrics['total_current']:,.2f}")
col2.metric("Total Return", f"{metrics['total_return']:.2%}")
col3.metric("Annualized Return", f"{metrics['ann_return']:.2%}")

# Risk Metrics cards
col1, col2, col3 = st.columns(3)
col1.metric("Beta", f"{metrics['beta']:.2f}")
col2.metric("Alpha", f"{metrics['alpha']:.2%}")
col3.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")

# Section divider
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.subheader("Rolling Volatility & Value at Risk Distribution")

# Rolling Volatility with VaR Distribution on the right
col1, col2 = st.columns([1.3, 1.2])

with col1:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=metrics['port_vol'].index, y=metrics['port_vol'], mode='lines', name='Portfolio Vol'))
    fig2.add_trace(go.Scatter(x=metrics['spy_vol'].index, y=metrics['spy_vol'], mode='lines', name='S&P 500 Vol'))
    fig2.update_layout(
        title="Rolling 6-Month Volatility (Annualized)", 
        xaxis_title="Date", 
        yaxis_title="Volatility",
        xaxis=dict(tickformat='%m/%y', nticks=20),
        height=500
    )
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    # Value at Risk Visualization
    if 'port_returns' in metrics and metrics['port_returns'] is not None and len(metrics['port_returns']) > 0:
        port_ret_for_var = metrics['port_returns']
    else:
        port_ret_for_var = None

    if port_ret_for_var is not None and len(port_ret_for_var) > 0:
        # Create histogram with smooth curve overlay
        fig_var = go.Figure()
        
        fig_var.add_trace(go.Histogram(
            x=port_ret_for_var,
            nbinsx=25,
            name='Monthly Returns',
            marker=dict(color='rgba(102, 126, 234, 0.6)'),
            opacity=0.7,
            showlegend=True
        ))
        
        # Add smooth KDE curve
        kde = gaussian_kde(port_ret_for_var.dropna())
        x_range = np.linspace(port_ret_for_var.min(), port_ret_for_var.max(), 200)
        kde_vals = kde(x_range)
        
        # Scale KDE to match histogram
        hist_max = np.histogram(port_ret_for_var, bins=25)[0].max()
        kde_vals_scaled = kde_vals * hist_max / kde_vals.max()
        
        fig_var.add_trace(go.Scatter(
            x=x_range,
            y=kde_vals_scaled,
            name='Distribution Curve',
            mode='lines',
            line=dict(color='rgba(118, 75, 162, 1)', width=3),
            fill='tozeroy',
            fillcolor='rgba(118, 75, 162, 0.3)',
            showlegend=True
        ))
        
        # Add VaR lines
        var_95 = metrics['var_95']
        var_99 = metrics['var_99']
        
        fig_var.add_vline(x=var_95, line_dash="dash", line_color="#ffa500")
        fig_var.add_vline(x=var_99, line_dash="dash", line_color="#ff4444")
        
        # Add annotations for VaR lines with custom positioning
        # Get y-axis max for positioning
        y_max = np.histogram(port_ret_for_var, bins=25)[0].max()
        
        fig_var.add_annotation(x=var_99, y=y_max * 1.08, text=f"VaR 99%<br>{var_99:.2%}", 
                              showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#ff4444",
                              ax=-40, ay=-20,
                              bgcolor="rgba(255, 68, 68, 0.8)", font=dict(color="white", size=10))
        fig_var.add_annotation(x=var_95, y=y_max * 1.08, text=f"VaR 95%<br>{var_95:.2%}", 
                              showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#ffa500",
                              ax=40, ay=-20,
                              bgcolor="rgba(255, 165, 0, 0.8)", font=dict(color="white", size=10))
        
        fig_var.update_layout(
            title="Monthly Returns Distribution",
            xaxis_title="Monthly Return",
            yaxis_title="Frequency",
            height=500,
            hovermode='x unified',
            template='plotly',
            showlegend=True
        )
        fig_var.update_xaxes(tickformat='.0%')
        
        st.plotly_chart(fig_var, use_container_width=True)
    else:
        st.info("Insufficient data for VaR visualization")

col1, col2, col3 = st.columns(3)
col1.metric("VaR (95% Confidence)", f"{metrics['var_95']:.2%}", help="Max expected monthly loss at 95% confidence")
col2.metric("VaR (99% Confidence)", f"{metrics['var_99']:.2%}", help="Max expected monthly loss at 99% confidence")
col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}", help="Largest peak-to-trough decline")

# Section divider
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# EV/EBITDA Comparison
st.subheader("EV/EBITDA Comparison")
holdings = portfolio.loc[portfolio['ticker'] != 'Total', 'ticker'].tolist()
ev_ebitda_data = []

for ticker in holdings:
    ev_ebitda = get_ev_ebitda(ticker)
    if ev_ebitda is not None and not pd.isna(ev_ebitda) and ev_ebitda > 0:
        # Get portfolio position value
        position = portfolio[portfolio['ticker'] == ticker]
        if not position.empty:
            current_price = position['current_price'].iloc[0]
            shares = position['shares'].iloc[0]
            position_value = current_price * shares
            ev_ebitda_data.append({
                'ticker': ticker,
                'ev_ebitda': ev_ebitda,
                'weight': position_value
            })

if ev_ebitda_data:
    # Calculate weighted average EV/EBITDA
    total_weight = sum(item['weight'] for item in ev_ebitda_data)
    portfolio_ev_ebitda = sum(item['ev_ebitda'] * item['weight'] for item in ev_ebitda_data) / total_weight if total_weight > 0 else None
    
    if portfolio_ev_ebitda is not None:
        # Create column chart showing individual holdings
        holdings_sorted = sorted(ev_ebitda_data, key=lambda x: x['ev_ebitda'], reverse=True)
        tickers = [item['ticker'] for item in holdings_sorted]
        ev_values = [item['ev_ebitda'] for item in holdings_sorted]
        
        fig_ev_col = go.Figure(data=[
            go.Bar(
                x=tickers,
                y=ev_values,
                marker=dict(color='#667eea'),
                text=[f"{val:.2f}" for val in ev_values],
                textposition='auto'
            )
        ])
        fig_ev_col.update_layout(
            title="Holdings EV/LTM EBITDA",
            yaxis_title="EV/LTM EBITDA",
            xaxis_title="Ticker",
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig_ev_col)
        
        # Show portfolio average
        col1, col2 = st.columns(2)
        col1.metric("Portfolio Avg EV/LTM EBITDA (Weighted)", f"{portfolio_ev_ebitda:.2f}")
        col2.metric("Holdings Count", len(holdings_sorted))
    else:
        st.info("Unable to calculate weighted portfolio EV/EBITDA")
else:
    st.info("EV/EBITDA data not available for any holdings")

# Section divider before refresh button
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Centered refresh button with professional styling
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()