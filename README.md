# AI Stock Sentiment & Signal Dashboard

A Streamlit web app that combines live stock price data with AI-powered news sentiment analysis to generate market signals.

## Features
- Live 30-day price chart with volume
- Latest news headlines fetched from NewsAPI
- AI sentiment analysis using FinBERT (trained on financial text)
- BUY / WATCH / CAUTION signal based on sentiment + price trend
- Confidence score per headline

## Setup
1. Clone this repo
2. Install dependencies: `pip install streamlit yfinance transformers torch requests pandas plotly python-dotenv`
3. Create a `.env` file with your NewsAPI key: `NEWSAPI_KEY=your_key_here`
4. Run: `streamlit run app.py`

## Tech Stack
- Streamlit · yfinance · HuggingFace Transformers (FinBERT) · NewsAPI · Plotly · Pandas

## Disclaimer
This tool is for educational purposes only. Not financial advice.

## Live Demo
[View Live App] (https://stock-sentiment-dashboard-nxr6hxna78abo7fgej8lhp.streamlit.app/)
