import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

from pipeline_runner import run_pipeline_for_ticker, clean_ticker_name, COMPANY_NAMES

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

CACHE_BASE_DIR = os.path.join(PROJECT_DIR, "rawData", "cache")


TOP_PRECOMPUTED = [
    {"ticker": "TCS.NS", "name": "Tata Consultancy Services", "region": "NSE India"},
    {"ticker": "RELIANCE.NS", "name": "Reliance Industries", "region": "NSE India"},
    {"ticker": "INFY.NS", "name": "Infosys", "region": "NSE India"},
    {"ticker": "HDFCBANK.NS", "name": "HDFC Bank", "region": "NSE India"},
    {"ticker": "AAPL", "name": "Apple Inc", "region": "NASDAQ"},
    {"ticker": "NVDA", "name": "NVIDIA Corporation", "region": "NASDAQ"},
    {"ticker": "TSLA", "name": "Tesla Inc", "region": "NASDAQ"},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "region": "NASDAQ"},
    {"ticker": "GOOGL", "name": "Alphabet Google", "region": "NASDAQ"}
]

def resolve_ticker(requested_ticker):
    if not requested_ticker:
        return "TCS.NS"
    ticker = clean_ticker_name(requested_ticker)
    
    # Direct match in cache
    cache_path = os.path.join(CACHE_BASE_DIR, ticker)
    if os.path.exists(cache_path):
        return ticker
        
    # Append .NS check
    if not ticker.endswith(".NS") and len(ticker) <= 5:
        alt_ticker = ticker + ".NS"
        if os.path.exists(os.path.join(CACHE_BASE_DIR, alt_ticker)):
            return alt_ticker
            
    return ticker

def get_ticker_cache_dir(ticker):
    ticker = resolve_ticker(ticker)
    cache_path = os.path.join(CACHE_BASE_DIR, ticker)
    
    if not os.path.exists(cache_path):
        print(f"[CACHE MISS] Executing live on-demand pipeline for: {ticker}")
        try:
            cache_path = run_pipeline_for_ticker(ticker)
        except Exception as e:
            print(f"Error running pipeline for {ticker}: {e}")
            # Fallback to TCS.NS if error
            ticker = "TCS.NS"
            cache_path = os.path.join(CACHE_BASE_DIR, ticker)
            if not os.path.exists(cache_path):
                run_pipeline_for_ticker(ticker)
                
    return cache_path, ticker

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stocks")
def get_top_stocks():
    return jsonify({
        "stocks": TOP_PRECOMPUTED
    })

@app.route("/api/summary")
def get_summary():
    req_ticker = request.args.get("ticker", "TCS.NS")
    cache_dir, ticker = get_ticker_cache_dir(req_ticker)
    
    matrix_path = os.path.join(cache_dir, "feature_matrix.csv")
    metrics_path = os.path.join(cache_dir, "metrics.json")
    model_path = os.path.join(cache_dir, "model.joblib")
    scaler_path = os.path.join(cache_dir, "scaler.joblib")
    
    if not os.path.exists(matrix_path):
        return jsonify({"error": f"Dataset not ready for {ticker}"}), 400
        
    df = pd.read_csv(matrix_path)
    latest_row = df.iloc[-1]
    
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    feature_cols = metrics.get("feature_columns", [
        "Daily_Return", "SMA_5", "SMA_20", "SMA_Ratio", "Volatility_5d",
        "RSI_14", "MACD", "MACD_Signal", "Volume_Change",
        "sent_compound_mean", "sent_pos_mean", "sent_neg_mean", "sent_polarity", "news_count"
    ])
    
    X_latest = df[feature_cols].iloc[-1:].copy()
    X_latest.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_latest.fillna(0.0, inplace=True)
    
    prediction = 1
    prob_up = 0.5
    prob_down = 0.5
    
    if model and scaler:
        X_scaled = scaler.transform(X_latest)
        prediction = int(model.predict(X_scaled)[0])
        probs = model.predict_proba(X_scaled)[0]
        prob_down = float(probs[0])
        prob_up = float(probs[1])
        
    latest_info = {
        "ticker": ticker,
        "company_name": COMPANY_NAMES.get(ticker, ticker),
        "date": str(latest_row.get("Date", "")),
        "close": float(latest_row.get("Close", 0.0)),
        "daily_return": float(latest_row.get("Daily_Return", 0.0)),
        "rsi": float(latest_row.get("RSI_14", 0.0)),
        "sent_compound": float(latest_row.get("sent_compound_mean", 0.0)),
        "prediction": prediction,
        "confidence": float(prob_up if prediction == 1 else prob_down),
        "prob_up": prob_up,
        "prob_down": prob_down
    }
    
    return jsonify({
        "ticker": ticker,
        "metrics": metrics,
        "latest": latest_info,
        "total_trading_days": len(df)
    })

@app.route("/api/stock_data")
def get_stock_data():
    req_ticker = request.args.get("ticker", "TCS.NS")
    cache_dir, ticker = get_ticker_cache_dir(req_ticker)
    matrix_path = os.path.join(cache_dir, "feature_matrix.csv")
    
    if not os.path.exists(matrix_path):
        return jsonify({"error": "Data not available"}), 400
        
    df = pd.read_csv(matrix_path)
    df.fillna(0.0, inplace=True)
    
    records = df[[
        "Date", "Open", "High", "Low", "Close", "Volume",
        "SMA_5", "SMA_20", "RSI_14", "sent_compound_mean", "sent_polarity", "news_count"
    ]].to_dict(orient="records")
    
    return jsonify({
        "ticker": ticker,
        "data": records
    })

@app.route("/api/news")
def get_news():
    req_ticker = request.args.get("ticker", "TCS.NS")
    cache_dir, ticker = get_ticker_cache_dir(req_ticker)
    news_path = os.path.join(cache_dir, "news_raw.csv")
    
    if not os.path.exists(news_path):
        return jsonify([])
        
    df = pd.read_csv(news_path)
    df.fillna("", inplace=True)
    news_items = df.head(30).to_dict(orient="records")
    return jsonify({
        "ticker": ticker,
        "news": news_items
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
