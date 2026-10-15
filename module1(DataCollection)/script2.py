import os
import yfinance as yf
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TICKER = "TCS.NS"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "rawData", "numericalData")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "TCS_1month.csv")


# --------------------------------------------------
# Fetch stock data
# --------------------------------------------------

def fetch_stock_data(ticker):

    print(f"Fetching data for {ticker}...")

    try:
        df = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

    except Exception as e:
        print(f"Error while fetching data: {e}")
        return pd.DataFrame()

    if df.empty:
        print("No data found.")
        return pd.DataFrame()

    return df


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    df = fetch_stock_data(TICKER)

    if df.empty:
        return

    # Remove unnecessary ticker level if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Make Date a normal column
    df.reset_index(inplace=True)

    # Save CSV
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 50)
    print("DATA FETCHED SUCCESSFULLY")
    print("=" * 50)

    print(f"Ticker: {TICKER}")
    print(f"Rows: {len(df)}")
    print(f"File: {OUTPUT_FILE}")

    print("\nData Preview:")
    print(df.head())


if __name__ == "__main__":
    main()