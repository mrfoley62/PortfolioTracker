import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="Earnings & News",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Header with interactive ticker bar using components.html
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
        if st.button("Dashboard", use_container_width=True):
            st.switch_page("streamlit_app.py")
    with n2:
        if st.button("Peer Comparison", use_container_width=True):
            st.switch_page("pages/Relative_Valuation.py")
    with n3:
        # current page
        st.button("Earnings & News", use_container_width=True, disabled=True)

# Custom CSS matching main app style
st.markdown("""
<style>
    body {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 50%, #0f1419 100%);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
        color: #e1e4e8;
    }
    
    .main .block-container {
        background: rgba(15, 20, 25, 0.95);
        border-radius: 12px;
        padding: 3.5rem;
        padding-top: 120px !important;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
        margin: 1.5rem 1rem;
        border: 1px solid rgba(48, 128, 255, 0.12);
    }
    
    h1 {
        color: #f0f6fc;
        font-weight: 800;
        font-size: 2.2rem;
    }
    
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
    
    h3 {
        color: #f0f6fc;
        font-weight: 600;
        border-bottom: 1px solid rgba(48, 128, 255, 0.2);
        padding-bottom: 1rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
    }

    /* Button styling - unified with Dashboard */
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
    
    .news-card {
        background: linear-gradient(135deg, rgba(48, 128, 255, 0.08) 0%, rgba(96, 165, 250, 0.05) 100%);
        border: 1px solid rgba(48, 128, 255, 0.15);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .news-card:hover {
        border-color: rgba(48, 128, 255, 0.3);
        transform: translateY(-2px);
    }
    
    .news-title {
        color: #58a6ff;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .news-meta {
        color: #8b949e;
        font-size: 0.8rem;
    }
    
    .news-publisher {
        color: #79c0ff;
    }
    
    .earnings-upcoming {
        background: rgba(34, 197, 94, 0.1);
        border-left: 3px solid #22c55e;
    }
    
    .earnings-past {
        background: rgba(107, 114, 128, 0.1);
        border-left: 3px solid #6b7280;
    }
    
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, rgba(48, 128, 255, 0.2) 50%, transparent 100%);
        margin: 2.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load portfolio
@st.cache_data(ttl=300)
def load_portfolio():
    import os
    # Handle both running from root and from pages folder
    if os.path.exists('portfolio.csv'):
        df = pd.read_csv('portfolio.csv')
    else:
        df = pd.read_csv('../portfolio.csv')
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    return df

# Get earnings events with EPS context
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_earnings_events(ticker):
    """Return list of earnings entries with date/eps info for a ticker."""
    entries = []
    try:
        stock = yf.Ticker(ticker)

        # Preferred: get_earnings_dates DataFrame (may be unavailable for some tickers)
        ed = stock.get_earnings_dates(limit=8)
        if isinstance(ed, pd.DataFrame) and not ed.empty:
            for idx, row in ed.iterrows():
                event_date = pd.to_datetime(idx)
                eps_est = row.get('EPS Estimate') if 'EPS Estimate' in row else row.get('epsestimate')
                eps_act = row.get('Reported EPS') if 'Reported EPS' in row else row.get('epsactual')
                # Surprise can be percent or difference depending on provider
                surprise = row.get('Surprise(%)') if 'Surprise(%)' in row else row.get('surprisepercent')
                entries.append({
                    'ticker': ticker,
                    'earnings_date': event_date,
                    'eps_estimate': eps_est,
                    'eps_actual': eps_act,
                    'surprise_percent': surprise
                })

        # Fallback: calendar (often only next event, sometimes only estimates)
        calendar = stock.calendar
        cal_date = None
        eps_avg = None
        if isinstance(calendar, pd.DataFrame) and not calendar.empty:
            if 'Earnings Date' in calendar.columns:
                cal_date = calendar['Earnings Date'].iloc[0]
            elif 'Earnings Date' in calendar.index:
                cal_date = calendar.loc['Earnings Date'].iloc[0]
            eps_avg = calendar['Earnings Average'].iloc[0] if 'Earnings Average' in calendar else None
        elif isinstance(calendar, dict):
            ed_val = calendar.get('Earnings Date')
            if isinstance(ed_val, list) and ed_val:
                cal_date = ed_val[0]
            elif ed_val:
                cal_date = ed_val
            eps_avg = calendar.get('Earnings Average')

        if cal_date is not None:
            try:
                cal_date = pd.to_datetime(cal_date)
            except Exception:
                cal_date = None
        if cal_date is not None:
            # Only add if not already present for this date
            if not any(abs((e['earnings_date'] - cal_date).days) < 1 for e in entries):
                entries.append({
                    'ticker': ticker,
                    'earnings_date': cal_date,
                    'eps_estimate': eps_avg,
                    'eps_actual': None,
                    'surprise_percent': None
                })

    except Exception:
        pass

    return entries

# Get news for ticker (Yahoo + Finnhub merged, deduped, latest 15)
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_news(ticker):
    items_combined = []

    # Source 1: Yahoo Finance
    try:
        stock = yf.Ticker(ticker)
        news_raw = stock.news or []
        for item in news_raw[:20]:
            content = item.get('content', {}) if isinstance(item, dict) else {}
            title = content.get('title') or item.get('title')
            link = (content.get('canonicalUrl', {}) or {}).get('url')
            if not link:
                link = (content.get('clickThroughUrl', {}) or {}).get('url')
            if not link:
                link = item.get('link')
            publisher = (content.get('provider', {}) or {}).get('displayName') or item.get('publisher')
            pub_date = content.get('pubDate') or item.get('providerPublishTime')
            if not title or not link:
                continue
            items_combined.append({
                'title': title,
                'link': link,
                'publisher': publisher or 'Unknown',
                'providerPublishTime': pub_date,
                'ticker': ticker,
                'source': 'yahoo'
            })
    except Exception:
        pass

    # Source 2: Finnhub company-news
    api_key = st.secrets.get("FINNHUB_API_KEY") if hasattr(st, "secrets") else None
    if api_key:
        try:
            today = datetime.utcnow().date()
            start = today - timedelta(days=21)
            url = "https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": ticker,
                "from": start.isoformat(),
                "to": today.isoformat(),
                "token": api_key,
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                items = resp.json() or []
                for it in items[:30]:
                    title = it.get("headline") or it.get("title")
                    link = it.get("url")
                    if not title or not link:
                        continue
                    pub_ts = it.get("datetime")  # seconds epoch
                    items_combined.append({
                        'title': title,
                        'link': link,
                        'publisher': it.get("source") or 'Unknown',
                        'providerPublishTime': pub_ts,
                        'ticker': ticker,
                        'source': 'finnhub'
                    })
        except Exception:
            pass

    # Deduplicate by link (primary) then title
    seen_links = set()
    seen_titles = set()
    deduped = []
    for it in items_combined:
        link_key = (it.get('link') or '').strip().lower()
        title_key = (it.get('title') or '').strip().lower()
        if link_key and link_key in seen_links:
            continue
        if not link_key and title_key in seen_titles:
            continue
        if link_key:
            seen_links.add(link_key)
        if title_key:
            seen_titles.add(title_key)
        deduped.append(it)

    # Sort by publish time desc
    def _ts(item):
        ts = item.get('providerPublishTime')
        if isinstance(ts, (int, float)):
            return ts
        if isinstance(ts, str):
            try:
                # Attempt ISO parse
                return datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
            except Exception:
                return 0
        return 0

    deduped.sort(key=_ts, reverse=True)

    return deduped[:15]

# Main app
st.title("Earnings & News")

portfolio = load_portfolio()
holdings = portfolio['ticker'].tolist()

# Earnings Calendar Section
st.subheader("Upcoming Earnings")

earnings_events = []
missing_earnings = []
for ticker in holdings:
    events = get_earnings_events(ticker)
    if events:
        earnings_events.extend(events)
    else:
        missing_earnings.append(ticker)

if earnings_events:
    # Build dataframe of events
    earnings_df = pd.DataFrame(earnings_events)
    earnings_df['earnings_date'] = pd.to_datetime(earnings_df['earnings_date'])
    earnings_df = earnings_df.sort_values('earnings_date')
    today = datetime.now()

    # Split into upcoming and past
    upcoming = earnings_df[earnings_df['earnings_date'] >= today]
    past = earnings_df[earnings_df['earnings_date'] < today]

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Upcoming Earnings**")
        if not upcoming.empty:
            for _, row in upcoming.iterrows():
                days_until = (row['earnings_date'] - today).days
                date_str = row['earnings_date'].strftime('%b %d, %Y')

                if days_until <= 7:
                    urgency = "🔴"
                elif days_until <= 14:
                    urgency = "🟡"
                else:
                    urgency = "🟢"

                est = row.get('eps_estimate')
                eps_line = f"Est. EPS: {est:.2f}" if pd.notna(est) else "Est. EPS: N/A"

                st.markdown(f"""
                <div class="news-card earnings-upcoming">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #f0f6fc; font-size: 1.1rem;">{row['ticker']}</span>
                        <span style="color: #22c55e;">{date_str}</span>
                    </div>
                    <div style="color: #8b949e; margin-top: 0.3rem;">
                        {urgency} {days_until} days away
                    </div>
                    <div style="color: #8b949e; margin-top: 0.3rem;">{eps_line}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No upcoming earnings dates found")

    with col2:
        st.write("**Recent Earnings**")
        if not past.empty:
            # Take latest 5 events
            for _, row in past.tail(5).iterrows():
                days_ago = (today - row['earnings_date']).days
                date_str = row['earnings_date'].strftime('%b %d, %Y')

                eps_est = row.get('eps_estimate')
                eps_act = row.get('eps_actual')
                if pd.notna(eps_act) and pd.notna(eps_est):
                    diff = eps_act - eps_est
                    status = "Meet"
                    if diff > 0.01:
                        status = "Beat"
                    elif diff < -0.01:
                        status = "Miss"
                    diff_str = f"{diff:+.2f}"
                    eps_line = f"EPS: {eps_act:.2f} vs {eps_est:.2f} ({status}, {diff_str})"
                elif pd.notna(eps_act):
                    eps_line = f"EPS: {eps_act:.2f} (estimate unavailable)"
                    status = ""
                else:
                    eps_line = "EPS data unavailable"
                    status = ""

                st.markdown(f"""
                <div class="news-card earnings-past">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #f0f6fc; font-size: 1.1rem;">{row['ticker']}</span>
                        <span style="color: #6b7280;">{date_str}</span>
                    </div>
                    <div style="color: #8b949e; margin-top: 0.3rem;">
                        {days_ago} days ago
                    </div>
                    <div style="color: #8b949e; margin-top: 0.3rem;">{eps_line}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent earnings data")
else:
    st.info("Unable to fetch earnings data for holdings")
    if missing_earnings:
        st.caption("No earnings data available from Yahoo Finance for: " + ", ".join(missing_earnings))

# Section divider
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# News Feed Section
st.subheader("News Feed")

# Ticker selector for news
selected_ticker = st.selectbox(
    "Select ticker for news:",
    options=["All Holdings"] + holdings,
    index=0
)

if selected_ticker == "All Holdings":
    tickers_to_fetch = holdings
else:
    tickers_to_fetch = [selected_ticker]

all_news = []
for ticker in tickers_to_fetch:
    news_items = get_news(ticker)
    for item in news_items:
        item['ticker'] = ticker
        all_news.append(item)

# Sort by publish time if available
if all_news:
    # Sort by providerPublishTime (Unix timestamp)
    all_news_sorted = sorted(
        all_news,
        key=lambda x: x.get('providerPublishTime', 0),
        reverse=True
    )
    
    # Display news cards
    for item in all_news_sorted[:20]:  # Show top 20 news items
        title = item.get('title', 'No title')
        publisher = item.get('publisher', 'Unknown')
        link = item.get('link', '#')
        ticker = item.get('ticker', '')
        
        # Convert timestamp/ISO to readable relative time
        pub_time = item.get('providerPublishTime', None)
        pub_date = None
        if isinstance(pub_time, (int, float)):
            pub_date = datetime.fromtimestamp(pub_time)
        elif isinstance(pub_time, str):
            try:
                pub_date = datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
            except Exception:
                pub_date = None

        if pub_date:
            time_ago = datetime.now(pub_date.tzinfo) - pub_date
            if time_ago.days > 0:
                time_str = f"{time_ago.days}d ago"
            elif time_ago.seconds // 3600 > 0:
                time_str = f"{time_ago.seconds // 3600}h ago"
            else:
                time_str = f"{max(time_ago.seconds // 60,1)}m ago"
        else:
            time_str = ""
        
        st.markdown(f"""
        <div class="news-card">
            <div class="news-title">
                <a href="{link}" target="_blank" style="color: #58a6ff; text-decoration: none;">
                    {title}
                </a>
            </div>
            <div class="news-meta">
                <span style="background: rgba(48, 128, 255, 0.2); padding: 2px 8px; border-radius: 4px; margin-right: 8px;">{ticker}</span>
                <span class="news-publisher">{publisher}</span>
                <span style="margin-left: 8px;">{time_str}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No news available for selected holdings")

# Section divider
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Refresh button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
