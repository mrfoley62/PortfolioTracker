import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import os
import streamlit.components.v1 as components

# Page config
st.set_page_config(
    page_title="Relative Valuation",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=300)
def load_portfolio():
    import os
    path = 'portfolio.csv' if os.path.exists('portfolio.csv') else '../portfolio.csv'
    df = pd.read_csv(path)
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    return df

# Ticker bar data helpers (match styling on other pages)
@st.cache_data(ttl=60)
def get_ticker_bar_data(tickers_list):
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
        except Exception:
            pass
    return ticker_data

def get_top_holdings_for_ticker_bar(portfolio_df, n=8):
    holdings = []
    for _, row in portfolio_df.iterrows():
        try:
            price = yf.Ticker(row['ticker']).history(period='1d')['Close'].iloc[-1]
            value = row['shares'] * price
            holdings.append({'ticker': row['ticker'], 'value': value})
        except Exception:
            pass
    holdings.sort(key=lambda x: x['value'], reverse=True)
    return [h['ticker'] for h in holdings[:n]]

def build_ticker_bar_html(data, ticker_bar_symbols):
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
            items.append(
                f"<div class='ticker-item'><span class='ticker-symbol'>{display_name}</span><span class='ticker-price'>{price}</span><span class='ticker-change {change_class}'>{change_sign}{change:.2f}%</span></div>"
            )
    return ''.join(items)

@st.cache_data(ttl=1800)
def get_finnhub_peers(symbol: str, api_key: str | None):
    if not api_key or not symbol:
        return []
    try:
        url = "https://finnhub.io/api/v1/stock/peers"
        resp = requests.get(url, params={"symbol": symbol, "token": api_key}, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        if isinstance(data, list):
            # filter out empty or same symbol
            return [p for p in data if isinstance(p, str) and p and p != symbol]
        return []
    except Exception:
        return []

@st.cache_data(ttl=300)
def fetch_multiples(ticker: str):
    """Return enterprise value, EV/Revenue, EV/EBITDA for a ticker."""
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return None

    try:
        ev = info.get('enterpriseValue')
        if ev is None:
            mcap = info.get('marketCap')
            debt = info.get('totalDebt', 0) or 0
            cash = info.get('totalCash', 0) or 0
            if mcap is not None:
                ev = mcap + debt - cash

        revenue = info.get('totalRevenue') or info.get('revenue')
        ebitda = info.get('ebitda')

        ev_rev = ev / revenue if ev is not None and revenue and revenue > 0 else None
        ev_ebitda = ev / ebitda if ev is not None and ebitda and ebitda > 0 else None

        # P/E ratio: prefer trailing, fall back to forward
        pe_ratio = info.get('trailingPE')
        if pe_ratio is None:
            pe_ratio = info.get('forwardPE')

        industry = info.get('industry') or 'Unknown'

        if ev is None and ev_rev is None and ev_ebitda is None:
            return None

        return {
            'ticker': ticker,
            'enterprise_value': ev,
            'ev_revenue': ev_rev,
            'ev_ebitda': ev_ebitda,
            'pe_ratio': pe_ratio,
            'industry': industry,
        }
    except Exception:
        return None

# Hide default Streamlit chrome to match other pages
st.markdown("""
<style>
    header[data-testid="stHeader"] { display: none; }
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stSidebarCollapsedControl"] { display: none; }
    .main .block-container { padding-top: 110px !important; }
</style>
""", unsafe_allow_html=True)

# Build ticker bar content
portfolio_for_ticker = load_portfolio()
top_holdings = get_top_holdings_for_ticker_bar(portfolio_for_ticker)
ticker_bar_symbols = ['^GSPC'] + top_holdings
ticker_bar_data = get_ticker_bar_data(ticker_bar_symbols)
ticker_bar_html = build_ticker_bar_html(ticker_bar_data, ticker_bar_symbols)
finnhub_api_key = None
if hasattr(st, "secrets"):
    try:
        # Try top-level first, then [general] section
        finnhub_api_key = st.secrets.get("FINNHUB_API_KEY")
        if not finnhub_api_key and isinstance(st.secrets, dict):
            finnhub_api_key = st.secrets.get("general", {}).get("FINNHUB_API_KEY")
    except Exception:
        finnhub_api_key = None
if not finnhub_api_key:
    finnhub_api_key = os.getenv("FINNHUB_API_KEY")

# Header with interactive ticker bar
header_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif; }}
    .nav-header {{ position: fixed; top: 0; left: 0; right: 0; z-index: 999999; background: linear-gradient(180deg, #0f1419 0%, #131820 100%); border-bottom: 1px solid rgba(48, 128, 255, 0.15); padding: 0; margin: 0; }}
    .nav-container {{ display: flex; align-items: center; justify-content: space-between; max-width: 100%; padding: 0 24px; height: 54px; }}
    .nav-logo {{ display: flex; align-items: center; gap: 12px; }}
    .nav-logo-text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif; font-size: 20px; font-weight: 700; color: #f0f6fc; letter-spacing: -0.5px; text-decoration: none; }}
    .nav-logo-accent {{ color: #3080ff; }}
    .ticker-bar {{ position: fixed; top: 54px; left: 0; right: 0; height: 52px; background: #0b1017; border-bottom: 1px solid rgba(48, 128, 255, 0.1); display: flex; align-items: center; gap: 8px; padding: 0 12px; z-index: 9999; }}
    .ticker-content {{ flex: 1; display: flex; overflow-x: auto; gap: 12px; scrollbar-width: none; }}
    .ticker-content::-webkit-scrollbar {{ display: none; }}
    .ticker-item {{ display: flex; align-items: center; gap: 10px; background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); padding: 8px 12px; border-radius: 8px; color: #c9d1d9; white-space: nowrap; }}
    .ticker-symbol {{ font-weight: 700; color: #f0f6fc; }}
    .ticker-price {{ color: #9fb3c8; }}
    .ticker-change {{ font-weight: 700; padding: 2px 6px; border-radius: 4px; }}
    .ticker-change.positive {{ color: #3fb950; background: rgba(63, 185, 80, 0.15); }}
    .ticker-change.negative {{ color: #f85149; background: rgba(248, 81, 73, 0.15); }}
    .ticker-arrow {{ background: rgba(255, 255, 255, 0.04); color: #9fb3c8; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; cursor: pointer; }}
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
    document.getElementById('leftArrow').addEventListener('click', function() {{ document.getElementById('tickerContent').scrollLeft -= 200; }});
    document.getElementById('rightArrow').addEventListener('click', function() {{ document.getElementById('tickerContent').scrollLeft += 200; }});
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
        if st.button("Dashboard", use_container_width=True):
            st.switch_page("streamlit_app.py")
    with n2:
        st.button("Peer Comparison", use_container_width=True, disabled=True)
    with n3:
        if st.button("Earnings & News", use_container_width=True):
            st.switch_page("pages/Earnings_&_News.py")

# Custom CSS matching main app style
st.markdown("""
<style>
    body { background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 50%, #0f1419 100%); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif; color: #e1e4e8; }
    .main .block-container { background: rgba(15, 20, 25, 0.95); border-radius: 12px; padding: 3.5rem; padding-top: 120px !important; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35); }
    h1, h2, h3, h4, h5, h6 { color: #f5f7fb; font-weight: 700; letter-spacing: -0.25px; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Button styling - unified with Dashboard/Earnings */
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
    .stButton > button:active { transform: translateY(0px); }
</style>
""", unsafe_allow_html=True)

# --- Layout ---
st.title("Relative Valuation")
st.caption("Data source: Yahoo Finance (auto-adjusted, cached 5 minutes).")

portfolio = load_portfolio()
holdings = portfolio['ticker'].tolist()

records = []
for t in holdings:
    data = fetch_multiples(t)
    if data:
        records.append(data)

if records:
    df = pd.DataFrame(records)
    df['EV ($mm)'] = df['enterprise_value'] / 1_000_000
    df['EV / Revenue'] = df['ev_revenue']
    df['EV / LTM EBITDA'] = df['ev_ebitda']
    df['P/E'] = df['pe_ratio']
    df['Industry'] = df['industry']

    display_cols = ['Industry', 'ticker', 'EV ($mm)', 'EV / Revenue', 'EV / LTM EBITDA', 'P/E']
    detail_table = df[display_cols].copy()

    # Aggregate rows (separate summary table)
    summary_rows = []
    for label, func in [('Mean', pd.Series.mean), ('Median', pd.Series.median)]:
        summary_rows.append({
            'ticker': label,
            'EV ($mm)': func(detail_table['EV ($mm)'].dropna()) if not detail_table['EV ($mm)'].dropna().empty else None,
            'EV / Revenue': func(detail_table['EV / Revenue'].dropna()) if not detail_table['EV / Revenue'].dropna().empty else None,
            'EV / LTM EBITDA': func(detail_table['EV / LTM EBITDA'].dropna()) if not detail_table['EV / LTM EBITDA'].dropna().empty else None,
            'P/E': func(detail_table['P/E'].dropna()) if not detail_table['P/E'].dropna().empty else None,
        })
    summary_table = pd.DataFrame(summary_rows)

    # Ensure numeric dtypes so column formatting applies
    for col in ['EV ($mm)', 'EV / Revenue', 'EV / LTM EBITDA', 'P/E']:
        detail_table[col] = pd.to_numeric(detail_table[col], errors='coerce')
        summary_table[col] = pd.to_numeric(summary_table[col], errors='coerce')

    # Render by industry
    st.subheader("Holdings by Industry")
    col_cfg = {
        'Industry': st.column_config.TextColumn(width=120),
        'ticker': st.column_config.TextColumn(label='Ticker', width=120),
        'EV ($mm)': st.column_config.NumberColumn(format="%.2f", width=120),
        'EV / Revenue': st.column_config.NumberColumn(format="%.1f", width=120),
        'EV / LTM EBITDA': st.column_config.NumberColumn(format="%.1f", width=120),
        'P/E': st.column_config.NumberColumn(format="%.1f", width=120),
    }
    missing_peer_key_noted = False
    for industry, group in detail_table.sort_values(['Industry', 'ticker']).groupby('Industry', dropna=False):
        st.markdown(f"**{industry}**")
        st.dataframe(
            group.rename(columns={'ticker': 'Ticker'}),
            hide_index=True,
            use_container_width=True,
            column_config=col_cfg,
        )

        # Finnhub peers for tickers in this industry
        peer_lines = []
        for t in group['ticker']:
            peers = get_finnhub_peers(t, finnhub_api_key)
            if peers:
                peer_lines.append(f"- **{t}** peers: {', '.join(peers[:10])}")
        if peer_lines:
            st.markdown("Peers (Finnhub):\n" + "\n".join(peer_lines))
        elif not finnhub_api_key and not missing_peer_key_noted:
            st.info("Add FINNHUB_API_KEY to Streamlit secrets or env to show peer lists.")
            missing_peer_key_noted = True

    st.subheader("Summary (Mean/Median)")
    st.dataframe(
        summary_table.rename(columns={'ticker': 'Statistic'}),
        hide_index=True,
        use_container_width=True,
        column_config={
            'Statistic': st.column_config.TextColumn(width=120),
            'EV ($mm)': st.column_config.NumberColumn(format="%.2f", width=120),
            'EV / Revenue': st.column_config.NumberColumn(format="%.1f", width=120),
            'EV / LTM EBITDA': st.column_config.NumberColumn(format="%.1f", width=120),
            'P/E': st.column_config.NumberColumn(format="%.1f", width=120),
        },
    )
else:
    st.info("No valuation data available for the current holdings.")

# Refresh button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
