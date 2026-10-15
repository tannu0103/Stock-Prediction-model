import time
from pipeline_runner import run_pipeline_for_ticker

TOP_STOCKS = [
    "TCS.NS",
    "RELIANCE.NS",
    "INFY.NS",
    "TATAMOTORS.NS",
    "HDFCBANK.NS",
    "AAPL",
    "NVDA",
    "TSLA",
    "MSFT",
    "GOOGL"
]

def main():
    print("==================================================")
    print("PRE-COMPUTING PIPELINE & CACHE FOR TOP STOCKS")
    print(f"Total Stocks: {len(TOP_STOCKS)}")
    print("==================================================\n")
    
    start_total = time.time()
    successful = []
    failed = []
    
    for idx, ticker in enumerate(TOP_STOCKS, 1):
        print(f"[{idx}/{len(TOP_STOCKS)}] Processing {ticker}...")
        try:
            run_pipeline_for_ticker(ticker)
            successful.append(ticker)
        except Exception as e:
            print(f"ERROR processing {ticker}: {e}")
            failed.append((ticker, str(e)))
            
    elapsed = time.time() - start_total
    print("\n==================================================")
    print(f"PRE-COMPUTATION COMPLETED IN {elapsed:.2f} SECONDS")
    print(f"Successful: {len(successful)} -> {successful}")
    if failed:
        print(f"Failed:     {len(failed)} -> {failed}")
    print("==================================================")

if __name__ == "__main__":
    main()
