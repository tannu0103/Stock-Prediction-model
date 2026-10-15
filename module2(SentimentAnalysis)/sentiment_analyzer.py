import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment():
    print("Starting Module 2: NLP Sentiment Analysis...")
    
    # Define file paths
    base_dir = os.path.dirname(__file__)
    input_file = os.path.join(base_dir, "..", "rawData", "sentimentData", "TCS_news_1month.csv")
    output_file = os.path.join(base_dir, "..", "rawData", "sentimentData", "TCS_daily_sentiment.csv")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} news articles from {input_file}")
    
    if df.empty:
        print("Empty news dataset. Creating empty daily sentiment file.")
        pd.DataFrame().to_csv(output_file, index=False)
        return
        
    analyzer = SentimentIntensityAnalyzer()
    
    # Process text: combine title and description for richer context
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["text_full"] = df["title"] + ". " + df["description"]
    
    # Calculate sentiment scores per article
    compounds = []
    positives = []
    negatives = []
    neutrals = []
    
    for text in df["text_full"]:
        scores = analyzer.polarity_scores(text)
        compounds.append(scores["compound"])
        positives.append(scores["pos"])
        negatives.append(scores["neg"])
        neutrals.append(scores["neu"])
        
    df["sent_compound"] = compounds
    df["sent_pos"] = positives
    df["sent_neg"] = negatives
    df["sent_neu"] = neutrals
    
    # Parse published_at date
    df["date"] = pd.to_datetime(df["published_at"], format="mixed", errors="coerce", utc=True).dt.strftime("%Y-%m-%d")

    
    # Aggregate daily metrics
    daily_sentiment = df.groupby("date").agg(
        sent_compound_mean=("sent_compound", "mean"),
        sent_pos_mean=("sent_pos", "mean"),
        sent_neg_mean=("sent_neg", "mean"),
        sent_neu_mean=("sent_neu", "mean"),
        news_count=("sent_compound", "count")
    ).reset_index()
    
    daily_sentiment["sent_polarity"] = daily_sentiment["sent_pos_mean"] - daily_sentiment["sent_neg_mean"]
    
    daily_sentiment.to_csv(output_file, index=False)
    print(f"Daily sentiment aggregated for {len(daily_sentiment)} unique dates.")
    print(f"Saved: {output_file}")
    print("\nSample Daily Sentiment Data:")
    print(daily_sentiment.head())

if __name__ == "__main__":
    analyze_sentiment()
