import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="Earnings & News",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Get earnings dates
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_earnings_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        calendar = stock.calendar
        
        if calendar is None or calendar.empty:
            return None
            
        # Handle different calendar formats
        if isinstance(calendar, pd.DataFrame):
            if 'Earnings Date' in calendar.columns:
                earnings_date = calendar['Earnings Date'].iloc[0]
            elif 'Earnings Date' in calendar.index:
                earnings_date = calendar.loc['Earnings Date'].iloc[0]
            else:
                return None
        else:
            return None
            
        return {
            'ticker': ticker,
            'earnings_date': earnings_date
        }
    except Exception as e:
        return None

# Get news for ticker
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if news:
            return news[:5]  # Return top 5 news items
        return []
    except:
        return []

# Main app
st.title("Earnings & News")

portfolio = load_portfolio()
holdings = portfolio['ticker'].tolist()

# Earnings Calendar Section
st.subheader("Upcoming Earnings")

earnings_data = []
for ticker in holdings:
    data = get_earnings_data(ticker)
    if data and data['earnings_date']:
        earnings_data.append(data)

if earnings_data:
    # Convert to DataFrame and sort by date
    earnings_df = pd.DataFrame(earnings_data)
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
                
                st.markdown(f"""
                <div class="news-card earnings-upcoming">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #f0f6fc; font-size: 1.1rem;">{row['ticker']}</span>
                        <span style="color: #22c55e;">{date_str}</span>
                    </div>
                    <div style="color: #8b949e; margin-top: 0.3rem;">
                        {urgency} {days_until} days away
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No upcoming earnings dates found")
    
    with col2:
        st.write("**Recent Earnings**")
        if not past.empty:
            for _, row in past.tail(5).iterrows():
                days_ago = (today - row['earnings_date']).days
                date_str = row['earnings_date'].strftime('%b %d, %Y')
                
                st.markdown(f"""
                <div class="news-card earnings-past">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #f0f6fc; font-size: 1.1rem;">{row['ticker']}</span>
                        <span style="color: #6b7280;">{date_str}</span>
                    </div>
                    <div style="color: #8b949e; margin-top: 0.3rem;">
                        {days_ago} days ago
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent earnings data")
else:
    st.info("Unable to fetch earnings data for holdings")

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
        
        # Convert timestamp to readable date
        pub_time = item.get('providerPublishTime', None)
        if pub_time:
            pub_date = datetime.fromtimestamp(pub_time)
            time_ago = datetime.now() - pub_date
            
            if time_ago.days > 0:
                time_str = f"{time_ago.days}d ago"
            elif time_ago.seconds // 3600 > 0:
                time_str = f"{time_ago.seconds // 3600}h ago"
            else:
                time_str = f"{time_ago.seconds // 60}m ago"
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
