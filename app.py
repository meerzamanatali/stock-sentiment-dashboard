import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import plotly.graph_objects as go
from transformers import pipeline
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

name_to_ticker = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
    "alphabet": "GOOGL", "amazon": "AMZN", "tesla": "TSLA",
    "nvidia": "NVDA", "meta": "META", "facebook": "META",
    "netflix": "NFLX", "amd": "AMD", "intel": "INTC",
    "jpmorgan": "JPM", "jp morgan": "JPM", "goldman sachs": "GS",
    "bank of america": "BAC", "visa": "V", "mastercard": "MA",
    "coinbase": "COIN", "alibaba": "BABA", "samsung": "005930.KS",
    "tsmc": "TSM", "taiwan semiconductor": "TSM"
}

def resolve_ticker(user_input: str) -> str:
    cleaned = user_input.strip().lower()
    if cleaned in name_to_ticker:
        return name_to_ticker[cleaned]
    return user_input.strip().upper()

st.set_page_config(
    page_title="Stock Sentiment Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Stock Sentiment & Signal Dashboard")
st.markdown("Enter any stock ticker to get live price data, news sentiment analysis, and an AI-generated market signal.")

@st.cache_resource(show_spinner="Loading AI sentiment model...")
def load_sentiment_model():
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        return_all_scores=False
    )

def fetch_stock_data(ticker: str, days: int = 30) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=days)
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)
    return df

def fetch_news(ticker: str, company_name: str) -> list:
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{ticker} OR {company_name} stock",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": NEWSAPI_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        return [
            {
                "title": a["title"],
                "source": a["source"]["name"],
                "published": a["publishedAt"][:10],
                "url": a["url"]
            }
            for a in articles if a["title"] and "[Removed]" not in a["title"]
        ]
    return []

def analyze_sentiment(headlines: list, model) -> list:
    results = []
    for item in headlines:
        prediction = model(item["title"][:512])[0]
        label = prediction["label"].lower()
        score = round(prediction["score"] * 100, 1)
        results.append({**item, "sentiment": label, "confidence": score})
    return results

def compute_signal(sentiment_results: list, price_df: pd.DataFrame) -> dict:
    if not sentiment_results or price_df.empty:
        return {"signal": "INSUFFICIENT DATA", "color": "gray", "reason": "Not enough data to generate signal."}

    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for r in sentiment_results:
        counts[r["sentiment"]] = counts.get(r["sentiment"], 0) + 1

    total = len(sentiment_results)
    positive_pct = counts["positive"] / total
    negative_pct = counts["negative"] / total

    prices = price_df["Close"]
    price_trend = "up" if prices.iloc[-1] > prices.iloc[0] else "down"

    if positive_pct >= 0.5 and price_trend == "up":
        return {"signal": "BUY", "color": "#1D9E75",
                "reason": f"{int(positive_pct*100)}% positive news + upward price trend over 30 days."}
    elif negative_pct >= 0.5 and price_trend == "down":
        return {"signal": "CAUTION", "color": "#D85A30",
                "reason": f"{int(negative_pct*100)}% negative news + downward price trend over 30 days."}
    else:
        return {"signal": "WATCH", "color": "#BA7517",
                "reason": f"Mixed signals — {counts['positive']} positive, {counts['negative']} negative, {counts['neutral']} neutral headlines."}

def plot_price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines",
        name="Close Price",
        line=dict(color="#534AB7", width=2)
    ))
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume",
        yaxis="y2",
        marker_color="rgba(83,74,183,0.15)"
    ))
    fig.update_layout(
        title=f"{ticker} — Last 30 Days",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#2C2C2A"),
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

ticker_input_raw = st.text_input(
    "Search by company name or ticker",
    value="Apple",
    placeholder="e.g. Apple, Tesla, NVDA, Microsoft",
    help="Enter company name or ticker symbol"
)
ticker_input = resolve_ticker(ticker_input_raw)

if ticker_input_raw.strip().lower() in name_to_ticker:
    st.caption(f"Resolved to ticker: {ticker_input}")

company_map = {
    "AAPL": "Apple", "TSLA": "Tesla", "NVDA": "Nvidia",
    "MSFT": "Microsoft", "GOOGL": "Google", "AMZN": "Amazon",
    "META": "Meta", "NFLX": "Netflix"
}

company_name = company_map.get(ticker_input, ticker_input)

analyze_btn = st.button("Analyze", type="primary", use_container_width=False)

if analyze_btn and ticker_input:
    with st.spinner("Fetching stock data..."):
        price_df = fetch_stock_data(ticker_input)

    if price_df.empty:
        st.error(f"Could not find stock data for '{ticker_input}'. Please check the ticker symbol.")
        st.stop()

    with st.spinner("Fetching latest news..."):
        news = fetch_news(ticker_input, company_name)

    if not news:
        st.warning("No news articles found. Check your NewsAPI key or try a different ticker.")
        st.stop()

    sentiment_model = load_sentiment_model()

    with st.spinner("Running AI sentiment analysis..."):
        analyzed = analyze_sentiment(news, sentiment_model)

    signal = compute_signal(analyzed, price_df)

    col1, col2, col3, col4 = st.columns(4)
    current_price = price_df["Close"].iloc[-1]
    price_change = price_df["Close"].iloc[-1] - price_df["Close"].iloc[0]
    price_change_pct = (price_change / price_df["Close"].iloc[0]) * 100
    pos_count = sum(1 for r in analyzed if r["sentiment"] == "positive")
    neg_count = sum(1 for r in analyzed if r["sentiment"] == "negative")

    col1.metric("Current Price", f"${current_price:.2f}", f"{price_change_pct:+.1f}% (30d)")
    col2.metric("Positive Headlines", pos_count, f"out of {len(analyzed)}")
    col3.metric("Negative Headlines", neg_count, f"out of {len(analyzed)}")
    col4.metric("AI Signal", signal["signal"])

    st.markdown("---")

    st.plotly_chart(plot_price_chart(price_df, ticker_input), use_container_width=True)

    st.markdown("---")
    st.subheader("AI Market Signal")
    st.markdown(
        f"""<div style='padding:16px 20px;background:#F1EFE8;border-left:4px solid {signal["color"]};border-radius:6px;'>
        <span style='font-size:22px;font-weight:600;color:{signal["color"]};'>{signal["signal"]}</span>
        <p style='margin:6px 0 0;color:#444441;font-size:14px;'>{signal["reason"]}</p>
        <p style='margin:6px 0 0;color:#888780;font-size:12px;'>⚠️ This is not financial advice. Always do your own research.</p>
        </div>""",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.subheader(f"Latest News & Sentiment — {ticker_input}")

    sentiment_colors = {"positive": "#E1F5EE", "negative": "#FAECE7", "neutral": "#F1EFE8"}
    sentiment_text_colors = {"positive": "#085041", "negative": "#712B13", "neutral": "#444441"}

    for item in analyzed:
        bg = sentiment_colors.get(item["sentiment"], "#F1EFE8")
        tc = sentiment_text_colors.get(item["sentiment"], "#444441")
        st.markdown(
            f"""<div style='padding:12px 16px;background:{bg};border-radius:8px;margin-bottom:8px;'>
            <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
              <a href='{item["url"]}' target='_blank' style='font-size:13px;font-weight:500;color:#2C2C2A;text-decoration:none;flex:1;margin-right:12px;'>{item["title"]}</a>
              <span style='font-size:11px;font-weight:500;color:{tc};background:white;padding:2px 8px;border-radius:20px;white-space:nowrap;'>{item["sentiment"].upper()} {item["confidence"]}%</span>
            </div>
            <div style='font-size:11px;color:#888780;margin-top:6px;'>{item["source"]} · {item["published"]}</div>
            </div>""",
            unsafe_allow_html=True
        )

elif not analyze_btn:
    st.info("Enter a stock ticker above and click Analyze to get started.")