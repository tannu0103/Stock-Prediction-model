import os
import pandas as pd
import numpy as np

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def build_feature_matrix():
    print("Starting Module 3: Technical Feature Engineering & Data Merging...")
    
    base_dir = os.path.dirname(__file__)
    stock_path = os.path.join(base_dir, "..", "rawData", "numericalData", "TCS_1month.csv")
    sentiment_path = os.path.join(base_dir, "..", "rawData", "sentimentData", "TCS_daily_sentiment.csv")
    output_path = os.path.join(base_dir, "..", "rawData", "processed_feature_matrix.csv")
    
    if not os.path.exists(stock_path):
        raise FileNotFoundError(f"Stock data file not found: {stock_path}")
        
    df_stock = pd.read_csv(stock_path)
    print(f"Loaded stock dataset with {len(df_stock)} rows.")
    
    # Ensure Date format
    df_stock["Date"] = pd.to_datetime(df_stock["Date"]).dt.strftime("%Y-%m-%d")
    df_stock.sort_values("Date", inplace=True)
    df_stock.reset_index(drop=True, inplace=True)
    
    # Calculate Technical Features
    df_stock["Daily_Return"] = df_stock["Close"].pct_change() * 100
    df_stock["SMA_5"] = df_stock["Close"].rolling(window=5).mean()
    df_stock["SMA_20"] = df_stock["Close"].rolling(window=20).mean()
    df_stock["SMA_Ratio"] = df_stock["SMA_5"] / (df_stock["SMA_20"] + 1e-9)
    df_stock["Volatility_5d"] = df_stock["Daily_Return"].rolling(window=5).std()
    df_stock["RSI_14"] = calculate_rsi(df_stock["Close"], period=14)
    
    # MACD & MACD Signal
    ema12 = df_stock["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df_stock["Close"].ewm(span=26, adjust=False).mean()
    df_stock["MACD"] = ema12 - ema26
    df_stock["MACD_Signal"] = df_stock["MACD"].ewm(span=9, adjust=False).mean()
    df_stock["Volume_Change"] = df_stock["Volume"].pct_change() * 100
    
    # Target Variable: 1 if Next Day Close > Today Close else 0
    df_stock["Next_Close"] = df_stock["Close"].shift(-1)
    df_stock["Target_Up"] = (df_stock["Next_Close"] > df_stock["Close"]).astype(int)
    
    # Load daily sentiment features
    if os.path.exists(sentiment_path):
        df_sent = pd.read_csv(sentiment_path)
        df_sent["date"] = pd.to_datetime(df_sent["date"]).dt.strftime("%Y-%m-%d")
        
        # Merge on date
        merged_df = pd.merge(df_stock, df_sent, left_on="Date", right_on="date", how="left")
        merged_df.drop(columns=["date"], errors="ignore", inplace=True)
    else:
        merged_df = df_stock.copy()
        
    # Fill NA sentiment values for non-news days
    sent_cols = ["sent_compound_mean", "sent_pos_mean", "sent_neg_mean", "sent_neu_mean", "sent_polarity"]
    for col in sent_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(0.0)
    if "news_count" in merged_df.columns:
        merged_df["news_count"] = merged_df["news_count"].fillna(0)
        
    # Backward fill technical indicators where windows warm up
    merged_df.bfill(inplace=True)
    merged_df.fillna(0.0, inplace=True)

    
    # Save processed feature matrix
    merged_df.to_csv(output_path, index=False)
    print(f"Feature engineering completed! Final matrix rows: {len(merged_df)}")
    print(f"Saved processed dataset: {output_path}")
    print("\nFeature Matrix Columns:")
    print(merged_df.columns.tolist())
    print("\nFirst 3 rows preview:")
    print(merged_df[["Date", "Close", "Daily_Return", "SMA_5", "RSI_14", "sent_compound_mean", "Target_Up"]].head(3))

if __name__ == "__main__":
    build_feature_matrix()
