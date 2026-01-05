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
    page_icon="",
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
        info = yf.Ticker(ticker).info
        # First try the direct metric
        ev_ebitda = info.get('enterpriseToEbitda', None)
        if ev_ebitda is not None:
            return ev_ebitda
        # Fall back to manual calculation (useful for MLPs like USAC)
        market_cap = info.get('marketCap', None)
        total_debt = info.get('totalDebt', 0)
        total_cash = info.get('totalCash', 0)
        ebitda = info.get('ebitda', None)
        if market_cap is not None and ebitda is not None and ebitda > 0:
            enterprise_value = market_cap + total_debt - total_cash
            return enterprise_value / ebitda
        return None
    except:
        return None

# Get P/E ratio metric
def get_pe_ratio(ticker):
    try:
        info = yf.Ticker(ticker).info
        # First try trailing P/E, fall back to forward P/E
        pe_ratio = info.get('trailingPE', None)
        if pe_ratio is not None:
            return pe_ratio
        pe_ratio = info.get('forwardPE', None)
        if pe_ratio is not None:
            return pe_ratio
        return None
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

    # Sharpe Ratio (using configurable rf)
    rf_annual = rf_percent / 100
    rf_monthly = rf_annual / 12

    # Beta and Alpha (Jensen's Alpha)
    if len(common_dates) >= 2 and spy_ret.nunique() > 1:
        slope, intercept, _, _, _ = stats.linregress(spy_ret, port_ret)
        raw_beta = slope
        # Apply Blume adjustment: betas tend to regress toward 1 over time
        beta = (2/3) * raw_beta + (1/3) * 1.0
        # Jensen's Alpha: α = Rp – [Rf + (Rm – Rf) × β]
        rp = port_ret.mean()  # Mean portfolio return (monthly)
        rm = spy_ret.mean()   # Mean market return (monthly)
        alpha = (rp - (rf_monthly + (rm - rf_monthly) * beta)) * 12  # Annualized
    else:
        beta = np.nan
        alpha = np.nan

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

# Get live market data for ticker bar
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_ticker_bar_data(tickers_list):
    """Get current price and day change for ticker bar"""
    ticker_data = {}
    for ticker in tickers_list:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='2d')
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = ((current - prev) / prev) * 100
                ticker_data[ticker] = {'price': current, 'change': change}
            elif len(hist) == 1:
                current = hist['Close'].iloc[-1]
                ticker_data[ticker] = {'price': current, 'change': 0}
        except:
            pass
    return ticker_data

# Load portfolio early for ticker bar
portfolio_for_ticker = pd.read_csv('portfolio.csv')

# Get top holdings by current value for ticker bar
def get_top_holdings_for_ticker_bar(portfolio_df, n=8):
    """Get top n holdings by current value"""
    holdings = []
    for _, row in portfolio_df.iterrows():
        try:
            price = yf.Ticker(row['ticker']).history(period='1d')['Close'].iloc[-1]
            value = row['shares'] * price
            holdings.append({'ticker': row['ticker'], 'value': value})
        except:
            pass
    holdings.sort(key=lambda x: x['value'], reverse=True)
    return [h['ticker'] for h in holdings[:n]]

# Get tickers for ticker bar (S&P 500 + top holdings)
top_holdings = get_top_holdings_for_ticker_bar(portfolio_for_ticker)
ticker_bar_symbols = ['^GSPC'] + top_holdings
ticker_bar_data = get_ticker_bar_data(ticker_bar_symbols)

# Build ticker bar HTML
def build_ticker_bar_html(data):
    items = []
    display_names = {'^GSPC': 'S&P 500'}
    for ticker in ticker_bar_symbols:
        if ticker in data:
            info = data[ticker]
            display_name = display_names.get(ticker, ticker)
            price = f"${info['price']:,.2f}"
            change = info['change']
            change_class = 'positive' if change >= 0 else 'negative'
            change_sign = '+' if change >= 0 else ''
            items.append(f'<div class="ticker-item"><span class="ticker-symbol">{display_name}</span><span class="ticker-price">{price}</span><span class="ticker-change {change_class}">{change_sign}{change:.2f}%</span></div>')
    return ''.join(items)

ticker_bar_html = build_ticker_bar_html(ticker_bar_data)

# CSS for hiding default Streamlit elements
st.markdown("""
<style>
    /* Hide default Streamlit header and sidebar */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    [data-testid="stSidebar"] {
        display: none;
    }
    
    [data-testid="stSidebarCollapsedControl"] {
        display: none;
    }
    
    /* Add padding to main content to account for fixed header + ticker bar */
    .main .block-container {
        padding-top: 110px !important;
    }
</style>
""", unsafe_allow_html=True)

# Main app - Professional header with interactive ticker bar using components.html
import streamlit.components.v1 as components

header_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }}
    
    /* Navigation bar matching theme */
    .nav-header {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999999;
        background: linear-gradient(180deg, #0f1419 0%, #131820 100%);
        border-bottom: 1px solid rgba(48, 128, 255, 0.15);
        padding: 0;
        margin: 0;
    }}
    
    .nav-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        max-width: 100%;
        padding: 0 24px;
        height: 54px;
    }}
    
    .nav-logo {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    
    .nav-logo-text {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        font-size: 20px;
        font-weight: 700;
        color: #f0f6fc;
        letter-spacing: -0.5px;
        text-decoration: none;
    }}
    
    .nav-logo-accent {{
        color: #3080ff;
    }}
    
    .nav-links {{
        display: flex;
        align-items: center;
        gap: 0;
    }}
    
    .nav-link {{
        color: #8b949e;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        font-size: 13px;
        font-weight: 500;
        text-decoration: none;
        padding: 18px 20px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
        border-bottom: 2px solid transparent;
    }}
    
    .nav-link:hover {{
        color: #f0f6fc;
        background-color: rgba(48, 128, 255, 0.08);
        border-bottom: 2px solid rgba(48, 128, 255, 0.5);
    }}
    
    .nav-link.active {{
        color: #f0f6fc;
        border-bottom: 2px solid #3080ff;
    }}
    
    /* Ticker bar below nav */
    .ticker-bar {{
        position: fixed;
        top: 54px;
        left: 0;
        right: 0;
        z-index: 999998;
        background: rgba(15, 20, 25, 0.95);
        border-bottom: 1px solid rgba(48, 128, 255, 0.1);
        padding: 10px 0;
        display: flex;
        align-items: center;
        backdrop-filter: blur(8px);
    }}
    
    .ticker-arrow {{
        background: none;
        border: none;
        color: #8b949e;
        font-size: 18px;
        padding: 8px 16px;
        cursor: pointer;
        transition: all 0.2s ease;
        flex-shrink: 0;
    }}
    
    .ticker-arrow:hover {{
        color: #3080ff;
    }}
    
    .ticker-content {{
        display: flex;
        align-items: center;
        gap: 32px;
        overflow-x: auto;
        scroll-behavior: smooth;
        flex: 1;
        padding: 0 8px;
        -ms-overflow-style: none;
        scrollbar-width: none;
    }}
    
    .ticker-content::-webkit-scrollbar {{
        display: none;
    }}
    
    .ticker-item {{
        display: flex;
        align-items: center;
        gap: 10px;
        flex-shrink: 0;
    }}
    
    .ticker-symbol {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        font-size: 12px;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }}
    
    .ticker-price {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        font-size: 13px;
        font-weight: 600;
        color: #f0f6fc;
    }}
    
    .ticker-change {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        font-size: 12px;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
    }}
    
    .ticker-change.positive {{
        color: #3fb950;
        background: rgba(63, 185, 80, 0.15);
    }}
    
    .ticker-change.negative {{
        color: #f85149;
        background: rgba(248, 81, 73, 0.15);
    }}
</style>
</head>
<body>
<div class="nav-header">
    <div class="nav-container">
        <div class="nav-logo">
            <span class="nav-logo-text">Portfolio<span class="nav-logo-accent">Tracker</span></span>
        </div>
    </div>
</div>

<div class="ticker-bar">
    <button class="ticker-arrow" id="leftArrow">&#9664;</button>
    <div class="ticker-content" id="tickerContent">
        {ticker_bar_html}
    </div>
    <button class="ticker-arrow" id="rightArrow">&#9654;</button>
</div>

<script>
    document.getElementById('leftArrow').addEventListener('click', function() {{
        document.getElementById('tickerContent').scrollLeft -= 200;
    }});
    document.getElementById('rightArrow').addEventListener('click', function() {{
        document.getElementById('tickerContent').scrollLeft += 200;
    }});
</script>
</body>
</html>
"""
components.html(header_html, height=100)

# Navigation buttons (3 across: Dashboard, Peer Comparison, Earnings & News)
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    n1, n2, n3 = st.columns(3)
    with n1:
        # current page
        st.button("Dashboard", use_container_width=True, disabled=True)
    with n2:
        if st.button("Peer Comparison", use_container_width=True):
            st.switch_page("pages/Relative_Valuation.py")
    with n3:
        if st.button("Earnings & News", use_container_width=True):
            st.switch_page("pages/Earnings_&_News.py")

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
        padding-top: 120px !important;
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
    
    /* Button styling - unified with Earnings & News */
    .stButton > button {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        color: #e5e7eb;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        font-weight: 700;
        padding: 1rem 2.4rem;
        transition: all 0.25s ease;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35);
        letter-spacing: 0.2px;
        font-size: 0.95rem;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.45);
        background: linear-gradient(135deg, #233044 0%, #192030 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
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

# Initialize session state for settings (allows settings at bottom to persist)
if 'rf_input' not in st.session_state:
    st.session_state.rf_input = 4.2
if 'beta_source' not in st.session_state:
    st.session_state.beta_source = "Yahoo Finance"

rf_input = st.session_state.rf_input
beta_source = st.session_state.beta_source

# Get historical data for all tickers + benchmark indices
index_tickers = {'S&P 500': 'SPY', 'NASDAQ': 'QQQ', 'Russell 2000': 'IWM', 'Dow Jones': 'DIA'}
tickers = portfolio['ticker'].tolist() + list(index_tickers.values())
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

# Calculate daily return for each stock using already-fetched historical data
def get_daily_return(row):
    ticker = row['ticker']
    hist = historical_data.get(ticker, pd.Series())
    if hist.empty or len(hist) < 2:
        return None
    try:
        current = hist.iloc[-1]
        prev = hist.iloc[-2]
        if pd.isna(current) or pd.isna(prev) or prev == 0:
            return None
        return ((current - prev) / prev * 100).round(2)
    except:
        return None

portfolio['daily_return'] = portfolio.apply(get_daily_return, axis=1)

# Add total row
total_shares = portfolio['shares'].sum()
total_purchase_value = (portfolio['shares'] * portfolio['purchase_price']).sum()
total_current_value = (portfolio['shares'] * portfolio['current_price']).sum()
total_cumulative_return = ((total_current_value - total_purchase_value) / total_purchase_value * 100).round(2) if total_purchase_value > 0 else 0

# Calculate weighted average LTM return and daily return
portfolio_no_total = portfolio[portfolio['ticker'] != 'Total']
if total_current_value > 0:
    weights = (portfolio_no_total['shares'] * portfolio_no_total['current_price']) / total_current_value
    valid_ltm = portfolio_no_total['ltm_return'].notna()
    if valid_ltm.any():
        weighted_ltm = (weights[valid_ltm] * portfolio_no_total.loc[valid_ltm, 'ltm_return']).sum()
    else:
        weighted_ltm = None
    # Calculate weighted average daily return
    valid_daily = portfolio_no_total['daily_return'].notna()
    if valid_daily.any():
        weighted_daily = (weights[valid_daily] * portfolio_no_total.loc[valid_daily, 'daily_return']).sum()
    else:
        weighted_daily = None
else:
    weighted_ltm = None
    weighted_daily = None

total_row = {
    'ticker': 'Total',
    'shares': total_shares,
    'purchase_date': '',
    'purchase_price': None,
    'current_price': None,
    'cumulative_return': total_cumulative_return,
    'ltm_return': weighted_ltm.round(2) if weighted_ltm is not None else None,
    'daily_return': weighted_daily.round(2) if weighted_daily is not None else None,
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
display_portfolio['daily_return'] = display_portfolio['daily_return'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else '')

# Reorder columns
display_portfolio = display_portfolio[['ticker', 'shares', 'purchase_date', 'purchase_price', 'historical_value', 'current_price', 'current_value', 'weight', 'cumulative_return', 'ltm_return', 'daily_return']]

# Separate the Total row from sortable data
display_portfolio_data = display_portfolio[display_portfolio['ticker'] != 'Total']
display_portfolio_total = display_portfolio[display_portfolio['ticker'] == 'Total']

st.subheader("Portfolio")

# Build custom HTML table with sortable headers and fixed Total row
column_headers = {
    'ticker': 'Ticker',
    'shares': 'Shares', 
    'purchase_date': 'Purchase Date',
    'purchase_price': 'Purchase Price',
    'historical_value': 'Historical Value',
    'current_price': 'Current Price',
    'current_value': 'Current Value',
    'weight': 'Weight',
    'cumulative_return': 'Cumulative Return',
    'ltm_return': 'LTM Return',
    'daily_return': 'Daily Return'
}

# Generate table HTML
def generate_sortable_table(data_df, total_df, columns, headers):
    # Build header row
    header_cells = ''.join([f'<th onclick="sortTable({i})" style="cursor:pointer;">{headers[col]} <span class="sort-icon">⇅</span></th>' for i, col in enumerate(columns)])
    
    # Build data rows
    data_rows = ''
    for _, row in data_df.iterrows():
        cells = ''.join([f'<td>{row[col]}</td>' for col in columns])
        data_rows += f'<tr>{cells}</tr>'
    
    # Build total row
    total_cells = ''
    for col in columns:
        val = total_df[col].values[0] if len(total_df) > 0 else ''
        total_cells += f'<td>{val}</td>'
    total_row = f'<tr class="total-row">{total_cells}</tr>'
    
    return f'''
    <style>
        .portfolio-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            font-size: 14px;
        }}
        .portfolio-table th {{
            background: rgba(48, 128, 255, 0.08);
            color: #f0f6fc;
            font-weight: 600;
            padding: 12px 10px;
            text-align: left;
            border-bottom: 2px solid rgba(48, 128, 255, 0.15);
            white-space: nowrap;
        }}
        .portfolio-table th:hover {{
            background: rgba(48, 128, 255, 0.15);
        }}
        .sort-icon {{
            opacity: 0.5;
            font-size: 12px;
            margin-left: 4px;
        }}
        .portfolio-table td {{
            padding: 10px;
            border-bottom: 1px solid rgba(48, 128, 255, 0.06);
            color: #d0d7de;
        }}
        .portfolio-table tbody tr:hover {{
            background-color: rgba(48, 128, 255, 0.08);
        }}
        .portfolio-table .total-row {{
            background: rgba(48, 128, 255, 0.12);
            font-weight: 700;
        }}
        .portfolio-table .total-row td {{
            border-top: 2px solid rgba(48, 128, 255, 0.3);
            color: #f0f6fc;
        }}
        .portfolio-table .total-row:hover {{
            background: rgba(48, 128, 255, 0.12);
        }}
    </style>
    <table class="portfolio-table">
        <thead>
            <tr>{header_cells}</tr>
        </thead>
        <tbody id="tableBody">
            {data_rows}
        </tbody>
        <tfoot>
            {total_row}
        </tfoot>
    </table>
    <script>
        let sortDirections = {{}};
        function sortTable(columnIndex) {{
            const table = document.querySelector('.portfolio-table');
            const tbody = document.getElementById('tableBody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            // Toggle sort direction
            sortDirections[columnIndex] = !sortDirections[columnIndex];
            const ascending = sortDirections[columnIndex];
            
            rows.sort((a, b) => {{
                let aVal = a.cells[columnIndex].textContent.trim();
                let bVal = b.cells[columnIndex].textContent.trim();
                
                // Remove $ and % for numeric comparison
                const aNum = parseFloat(aVal.replace(/[$,%]/g, ''));
                const bNum = parseFloat(bVal.replace(/[$,%]/g, ''));
                
                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    return ascending ? aNum - bNum : bNum - aNum;
                }}
                
                // String comparison
                return ascending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            }});
            
            // Reorder rows
            rows.forEach(row => tbody.appendChild(row));
        }}
        
        // Default sort by Current Value (column 6) descending on page load
        document.addEventListener('DOMContentLoaded', function() {{
            sortDirections[6] = true;  // Set to true so toggle makes it false (descending)
            sortTable(6);
        }});
    </script>
    '''

table_html = generate_sortable_table(display_portfolio_data, display_portfolio_total, list(column_headers.keys()), column_headers)
components.html(table_html, height=450, scrolling=True)

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
col1, col2 = st.columns([1, 1.5], gap="large")

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
    st.plotly_chart(fig_sector, use_container_width=True)
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.write("**Allocation by Stock**")
    st.plotly_chart(fig_stock, use_container_width=True)
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
            height=680,
            margin=dict(l=60, r=40, t=30, b=50)
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough holdings")

# Section divider
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Drop all index tickers from portfolio data for metrics calculation
index_ticker_list = list(index_tickers.values())
portfolio_cols = [col for col in hist_df.columns if col not in index_ticker_list]
metrics = calculate_metrics(portfolio[portfolio['ticker'] != 'Total'], hist_df[portfolio_cols], hist_df['SPY'], rf_percent=rf_input)

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

# Index selector for benchmark comparison
selected_index = st.selectbox("Compare against:", options=list(index_tickers.keys()), index=0)
selected_index_ticker = index_tickers[selected_index]

# Get benchmark prices and normalize them
benchmark_prices = hist_df[selected_index_ticker]
if benchmark_prices.index.tz is not None:
    benchmark_prices.index = benchmark_prices.index.tz_localize(None)
earliest_purchase = portfolio[portfolio['ticker'] != 'Total']['purchase_date'].min()
benchmark_prices = benchmark_prices[benchmark_prices.index >= earliest_purchase]
if not benchmark_prices.empty:
    benchmark_prices_norm = benchmark_prices / benchmark_prices.iloc[0] * 100
else:
    benchmark_prices_norm = metrics['spy_prices']  # Fallback to SPY

# Performance chart full width
fig = go.Figure()
fig.add_trace(go.Scatter(x=metrics['port_prices'].index, y=metrics['port_prices'], mode='lines', name='Portfolio'))
fig.add_trace(go.Scatter(x=benchmark_prices_norm.index, y=benchmark_prices_norm, mode='lines', name=selected_index))
fig.update_layout(
    title=f"Portfolio vs {selected_index}", 
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
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig_ev_col)
        
        # Show portfolio average
        st.metric("Portfolio Avg EV/LTM EBITDA (Weighted)", f"{portfolio_ev_ebitda:.2f}")
    else:
        st.info("Unable to calculate weighted portfolio EV/EBITDA")
else:
    st.info("EV/EBITDA data not available for any holdings")

# P/E Ratio Comparison
st.subheader("P/E Ratio Comparison")
pe_data = []

for ticker in holdings:
    info = yf.Ticker(ticker).info
    pe_ratio = None
    pe_type = None
    
    # Check for trailing P/E first
    pe_ratio = info.get('trailingPE', None)
    if pe_ratio is not None and pe_ratio > 0:
        pe_type = 'Trailing'
    else:
        # Fall back to forward P/E
        pe_ratio = info.get('forwardPE', None)
        if pe_ratio is not None and pe_ratio > 0:
            pe_type = 'Forward'
    
    if pe_ratio is not None:
        # Get portfolio position value
        position = portfolio[portfolio['ticker'] == ticker]
        if not position.empty:
            current_price = position['current_price'].iloc[0]
            shares = position['shares'].iloc[0]
            position_value = current_price * shares
            pe_data.append({
                'ticker': ticker,
                'pe_ratio': pe_ratio,
                'weight': position_value,
                'pe_type': pe_type
            })

if pe_data:
    # Calculate weighted average P/E
    total_weight = sum(item['weight'] for item in pe_data)
    portfolio_pe = sum(item['pe_ratio'] * item['weight'] for item in pe_data) / total_weight if total_weight > 0 else None
    
    if portfolio_pe is not None:
        # Create column chart showing individual holdings
        holdings_sorted_pe = sorted(pe_data, key=lambda x: x['pe_ratio'], reverse=True)
        tickers_pe = [item['ticker'] for item in holdings_sorted_pe]
        pe_values = [item['pe_ratio'] for item in holdings_sorted_pe]
        colors = ['#764ba2' if item['pe_type'] == 'Trailing' else '#a389d4' for item in holdings_sorted_pe]
        
        fig_pe_col = go.Figure(data=[
            go.Bar(
                x=tickers_pe,
                y=pe_values,
                marker=dict(color=colors),
                text=[f"{val:.2f}" for val in pe_values],
                textposition='auto'
            )
        ])
        fig_pe_col.update_layout(
            title="Holdings P/E Ratio (Trailing or Forward)",
            yaxis_title="P/E Ratio",
            height=500,
            showlegend=False,
            annotations=[
                dict(
                    text="<b>Color Key:</b> Dark purple = Trailing P/E | Light purple = Forward P/E",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.15,
                    showarrow=False,
                    font=dict(size=11),
                    xanchor='center'
                )
            ]
        )
        st.plotly_chart(fig_pe_col)
        
        # Show portfolio average
        st.metric("Portfolio Avg P/E (Weighted)", f"{portfolio_pe:.2f}")
    else:
        st.info("Unable to calculate weighted portfolio P/E")
else:
    st.info("P/E data not available for any holdings")

# Section divider before settings
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Settings Section
with st.expander("Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        new_rf = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=st.session_state.rf_input, step=0.1, key="rf_bottom")
        if new_rf != st.session_state.rf_input:
            st.session_state.rf_input = new_rf
            st.rerun()
    with col2:
        new_beta = st.selectbox("Beta Calculation", options=["Custom (3-Year Monthly)", "Yahoo Finance"], index=1 if st.session_state.beta_source == "Yahoo Finance" else 0, key="beta_bottom")
        if new_beta != st.session_state.beta_source:
            st.session_state.beta_source = new_beta
            st.rerun()

# Centered refresh button with professional styling
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()