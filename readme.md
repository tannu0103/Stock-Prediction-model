# MARP — Stock Sentiment & Numerical Data Machine Learning Prediction System

**MARP** (Market Analysis and Regime Prediction) is an end-to-end Machine Learning and Natural Language Processing (NLP) framework designed to classify and predict stock market directional movement (predicting whether a stock price will go **UP** or **DOWN** on the next trading day). 

By combining **Quantitative Technical Indicators** (numerical market data) with **NLP Financial News Sentiment Scores**, the application leverages multi-modal feature fusion to train robust ML models and present real-time visual analytics via a modern web dashboard.

### 🌟 Key Platform Features:
- 🔍 **Interactive Stock Search**: Search for any stock ticker (e.g. `INFY.NS`, `AAPL`, `RELIANCE.NS`, `WIPRO.NS`, `AMZN`) to trigger live data fetching, NLP sentiment analysis, technical feature extraction, and AI prediction.
- ⚡ **Pre-Processed Top Stocks Cache Layer**: Pre-computes and caches top Indian and US stocks (`TCS.NS`, `RELIANCE.NS`, `INFY.NS`, `HDFCBANK.NS`, `AAPL`, `NVDA`, `TSLA`, `MSFT`, `GOOGL`) to eliminate load times.
- 🤖 **On-Demand Live Execution**: If a searched ticker is not cached, the server automatically executes the full end-to-end pipeline in real time and caches the result for future requests.


---

## 📐 Project Architecture & Pipeline

```
 ┌─────────────────────────────────────────────────────────────┐
 │                     MODULE 1: DATA COLLECTION               │
 │  NewsAPI / News Fetcher (NLP)     │  yfinance Stock OHLCV   │
 └──────────────────────────────┬──────────────────────────────┘
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
┌───────────────────────────┐               ┌───────────────────────────┐
│ MODULE 2: SENTIMENT NLP   │               │ MODULE 3: TECH INDICATORS │
│ VADER Polarity Scoring    │               │ RSI, SMA5, SMA20, MACD    │
│ Daily Sentiment Metrics   │               │ Volatility, Daily Returns │
└─────────┬─────────────────┘               └─────────┬─────────────────┘
          │                                           │
          └─────────────────────┬─────────────────────┘
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │            MODULE 3: FEATURE MATRIX FUSION                  │
 │     Merged Technical Indicators + Daily NLP Sentiment       │
 └──────────────────────────────┬──────────────────────────────┘
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │               MODULE 4: MACHINE LEARNING PIPELINE           │
 │ Random Forest / Gradient Boosting Classifiers & Weight Tuning│
 └──────────────────────────────┬──────────────────────────────┘
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │            MODULE 5: INTERACTIVE WEB DASHBOARD              │
 │  Flask API + Plotly Interactive Charts + AI Prediction UI   │
 └─────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Directory Structure

```text
MARP/
│
├── .env                                  # API keys (NEWS_API_KEY, ALPHA_VANTAGE_API_KEY)
├── .gitignore                            # Git ignore rules
├── readme.md                             # Comprehensive project documentation
│
├── module1(DataCollection)/              # Data Fetching Module
│   ├── script1.py                        # NLP News Data Fetcher (NewsAPI + Fallback)
│   └── script2.py                        # Numerical Stock Data Fetcher (yfinance)
│
├── module2(SentimentAnalysis)/           # NLP Sentiment Module
│   └── sentiment_analyzer.py             # VADER Sentiment Scoring & Daily Aggregator
│
├── module3(FeatureEngineering)/          # Feature Engineering Module
│   └── feature_engineer.py               # Technical Indicators & Feature Matrix Fusion
│
├── module4(ModelTraining)/               # Machine Learning Training Pipeline
│   └── train_model.py                    # Classifier Training, Evaluation & Serialization
│
├── module5(WebApp)/                      # Web Application & UI Dashboard
│   ├── app.py                            # Flask Web Server & REST API Endpoints
│   ├── templates/
│   │   └── index.html                    # Dashboard Layout Template
│   └── static/
│       ├── style.css                     # Glassmorphic Dark-Mode Styling
│       └── app.js                        # Plotly Charts & Dynamic Dashboard Engine
│
├── models/                               # Serialized Model Artifacts
│   ├── stock_predictor.joblib            # Trained Classifier Model
│   ├── scaler.joblib                     # Feature StandardScaler
│   └── metrics.json                      # Evaluation Metrics & Feature Importance
│
└── rawData/                              # Raw & Processed Datasets
    ├── numericalData/
    │   └── TCS_1month.csv                # Raw Stock OHLCV Price Data
    ├── sentimentData/
    │   ├── TCS_news_1month.csv           # Raw Financial News Articles
    │   └── TCS_daily_sentiment.csv       # Aggregated Daily Sentiment Metrics
    └── processed_feature_matrix.csv      # Consolidated Training Matrix
```

---

## 🔍 Module-by-Module Breakdown & Implementation Details

### 🔹 Module 1: Data Collection (`module1(DataCollection)`)

This module is responsible for gathering raw financial news text and stock trading price data.

#### 1. `script1.py` (Financial News Collector for NLP)
- **Objective**: Fetch recent financial news headlines and metadata for the targeted stock (`Tata Consultancy Services` / `TCS`).
- **Implementation**:
  - Uses `requests` to call the NewsAPI (`https://newsapi.org/v2/everything`).
  - Configured with `q='"Tata Consultancy Services"'`, `language="en"`, and `sortBy="publishedAt"`.
  - Parses JSON articles into structured fields: `published_at`, `title`, `description`, `content`, `source`, `author`, `url`.
  - **Robustness Layer**: Includes a supplementary news generator that ensures complete historical coverage across all stock trading days if API free-tier limits or quota bounds occur.
  - Formats date strings (`format="mixed", utc=True`) and drops duplicate articles based on URL.
  - **Output File**: `rawData/sentimentData/TCS_news_1month.csv`.

#### 2. `script2.py` (Stock Numerical Market Data Collector)
- **Objective**: Download historical Open, High, Low, Close, Volume (OHLCV) stock price data.
- **Implementation**:
  - Uses `yfinance` to fetch daily price bars for ticker `TCS.NS` (NSE India).
  - Configured for 6 months to 1 year of daily historical data (`period="6mo"`, `interval="1d"`).
  - Handles MultiIndex pandas column structures cleanly and resets date index into a standard column format.
  - **Output File**: `rawData/numericalData/TCS_1month.csv`.

---

### 🔹 Module 2: NLP Sentiment Analysis (`module2(SentimentAnalysis)`)

This module transforms unstructured news text into quantitative financial sentiment signals.

#### `sentiment_analyzer.py`
- **Objective**: Process news titles and descriptions to score financial sentiment polarity and aggregate results into a daily time series.
- **Implementation**:
  - Uses `vaderSentiment` (`SentimentIntensityAnalyzer`), a lexicon and rule-based sentiment analysis tool specifically attuned to sentiments expressed in social media and news.
  - Concatenates `title` and `description` to form full text context.
  - Computes four core sentiment scores per news item:
    - `sent_compound`: Normalized compound score bounded between `-1.0` (extremely negative) and `+1.0` (extremely positive).
    - `sent_pos`: Positive sentiment ratio.
    - `sent_neg`: Negative sentiment ratio.
    - `sent_neu`: Neutral sentiment ratio.
  - Groups articles by calendar date (`YYYY-MM-DD`) and computes daily aggregated metrics:
    - `sent_compound_mean`: Average compound sentiment per day.
    - `sent_pos_mean`: Average positive score.
    - `sent_neg_mean`: Average negative score.
    - `news_count`: Total number of published articles on that trading day.
    - `sent_polarity`: Net polarity (`sent_pos_mean - sent_neg_mean`).
  - **Output File**: `rawData/sentimentData/TCS_daily_sentiment.csv`.

---

### 🔹 Module 3: Technical Feature Engineering & Fusion (`module3(FeatureEngineering)`)

This module generates quantitative technical market indicators and fuses them with daily sentiment metrics into a unified machine learning feature matrix.

#### `feature_engineer.py`
- **Objective**: Calculate mathematical indicators from stock price series, generate target classification labels, and merge news sentiment features on trading dates.
- **Implementation**:
  - **Technical Indicators Created**:
    1. `Daily_Return`: Percentage change in close price `(Close_t - Close_{t-1}) / Close_{t-1} * 100`.
    2. `SMA_5`: 5-day Simple Moving Average.
    3. `SMA_20`: 20-day Simple Moving Average.
    4. `SMA_Ratio`: Moving average crossover ratio `SMA_5 / (SMA_20 + 1e-9)`.
    5. `Volatility_5d`: 5-day rolling standard deviation of daily returns.
    6. `RSI_14`: 14-day Relative Strength Index measuring speed and magnitude of recent price changes.
    7. `MACD`: Moving Average Convergence Divergence `EMA(Close, 12) - EMA(Close, 26)`.
    8. `MACD_Signal`: 9-day EMA of the MACD series.
    9. `Volume_Change`: Percentage change in daily trading volume.
  - **Target Variable Construction**:
    - `Next_Close`: Shifted close price `Close.shift(-1)`.
    - `Target_Up`: Binary target (`1` if `Next_Close > Close` else `0`).
  - **Multi-Modal Data Fusion**:
    - Performs a date-matched left merge between numerical stock features and daily sentiment metrics (`sent_compound_mean`, `sent_pos_mean`, `sent_neg_mean`, `sent_polarity`, `news_count`).
    - Fills missing non-news days with neutral defaults (`0.0`).
    - Uses `.bfill()` and `.fillna(0.0)` for full clean numeric compatibility with Scikit-Learn.
  - **Output File**: `rawData/processed_feature_matrix.csv`.

---

### 🔹 Module 4: Machine Learning Pipeline (`module4(ModelTraining)`)

This module trains, evaluates, and exports Machine Learning classification models.

#### `train_model.py`
- **Objective**: Train predictive models to classify next-day stock price movement and measure feature importance weights.
- **Implementation**:
  - **Feature Set**: Fuses 14 features across technical market data and NLP sentiment scores.
  - **Time-Series Split**: Performs an 80/20 sequential train-test split (avoiding data leakage across historical time steps).
  - **Feature Scaling**: Fits a `StandardScaler` on training data and transforms both train and test splits.
  - **Algorithms Evaluated**:
    - `RandomForestClassifier`: Ensemble of decision trees trained with bagging.
    - `GradientBoostingClassifier`: Boosting ensemble fitting sequential error residuals.
  - **Metrics Computed**: Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
  - **Feature Importance Analysis**: Extracts Gini feature importances to quantify the relative predictive weight of technical indicators vs NLP sentiment metrics.
  - **Artifact Serialization**: Saves the best-performing model, scaler, and metrics to disk using `joblib` and `json`.
  - **Output Files**: `models/stock_predictor.joblib`, `models/scaler.joblib`, `models/metrics.json`.

---

### 🔹 Module 5: Interactive Web Dashboard & Predictor UI (`module5(WebApp)`)

This module provides a responsive web user interface and REST API endpoints for real-time visualization and stock prediction.

#### 1. `app.py` (Flask Web Application Server)
- **Objective**: Serve the front-end dashboard and handle REST API requests.
- **API Endpoints**:
  - `GET /`: Renders `index.html`.
  - `GET /api/summary`: Returns latest market close, RSI, daily return, NLP sentiment, model accuracy, and AI stock directional prediction (`UP`/`DOWN` with confidence probability).
  - `GET /api/stock_data`: Returns complete historical price time series, moving averages, and sentiment scores for interactive plotting.
  - `GET /api/news`: Returns recent news articles with VADER sentiment scores.
  - `POST /api/predict`: Custom inference endpoint accepting custom technical/sentiment input vectors.

#### 2. Front-End Interface (`index.html`, `style.css`, `app.js`)
- **Design Aesthetics**: Dark obsidian canvas (`#0b0f19`), glassmorphism backdrop blurs, vibrant indigo/cyan/emerald accents, and custom typography (`Inter` & `Outfit`).
- **Dashboard Features**:
  - **AI Directional Forecast Banner**: Displays next-day directional prediction (📈 `BULLISH (UP)` / 📉 `BEARISH (DOWN)`) with model probability confidence %.
  - **Interactive Stock Price & SMA Chart**: Plotly.js line chart featuring close price overlaid with 5-day and 20-day Simple Moving Averages.
  - **Daily Sentiment Index Chart**: Bar chart illustrating daily positive/negative news sentiment scores.
  - **Feature Importance Breakdown**: Visual percentage bars comparing technical market indicators vs NLP sentiment features.
  - **Real-Time News Feed**: Structured list of financial news headlines with sentiment polarity tags.

---

## 🛠️ Installation & Setup Guide

### Prerequisites
- **Python 3.10+**
- **pip** and **virtualenv**

### 1. Clone & Set Up Environment
```bash
# Navigate to project directory
cd MARP

# Activate virtual environment
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# Or install directly:
pip install pandas numpy requests python-dotenv yfinance vaderSentiment scikit-learn plotly flask joblib
```

### 3. Configure API Keys (`.env`)
Create or verify your `.env` file in the root directory:
```env
NEWS_API_KEY=your_news_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
```

---

## 🚀 Execution Guide (Step-by-Step)

You can run the full pipeline step by step or start the web server directly:

### Step 1: Run Data Collection (Module 1)
```bash
# Fetch financial news data (NLP)
python "module1(DataCollection)/script1.py"

# Fetch stock price numerical data
python "module1(DataCollection)/script2.py"
```

### Step 2: Perform NLP Sentiment Analysis (Module 2)
```bash
python "module2(SentimentAnalysis)/sentiment_analyzer.py"
```

### Step 3: Feature Engineering & Data Merging (Module 3)
```bash
python "module3(FeatureEngineering)/feature_engineer.py"
```

### Step 4: Train Machine Learning Model (Module 4)
```bash
python "module4(ModelTraining)/train_model.py"
```

### Step 5: Launch Web Application Dashboard (Module 5)
```bash
python "module5(WebApp)/app.py"
```

Once started, open your web browser and go to:
👉 **`http://127.0.0.1:5000`**

---

## 📈 Sample Model Output & Results

```text
--- RandomForest Model Performance ---
Accuracy:  56.00%
F1 Score:  0.1538
ROC-AUC:   0.5962

Top Predictive Features:
 1. MACD Signal       (11.28%)
 2. Volatility 5d     (10.53%)
 3. Daily Return      (9.38%)
 4. SMA 20            (9.16%)
 5. MACD              (8.97%)
 6. RSI 14            (8.19%)
 7. sent_polarity     (5.40%) [NLP]
 8. sent_pos_mean     (5.08%) [NLP]
 9. sent_compound     (4.78%) [NLP]
```

---

## 🌐 Deployment Options & Hosting Guide

You can deploy the **MARP Stock Sentiment & ML App** to production using several free cloud platforms. Files [`requirements.txt`](file:///home/tannushree/Desktop/Desktop1/MARP/requirements.txt), [`Procfile`](file:///home/tannushree/Desktop/Desktop1/MARP/Procfile), and [`wsgi.py`](file:///home/tannushree/Desktop/Desktop1/MARP/wsgi.py) have been pre-configured for instant 1-click deployment.

### Option 1: Render.com (Recommended Free Web Service)
1. Push your repository to **GitHub**.
2. Go to [Render Dashboard](https://dashboard.render.com/) -> **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Set the build and start settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt && python precompute_stocks.py`
   - **Start Command**: `gunicorn wsgi:app`
5. Add Environment Variables under **Environment**:
   - `NEWS_API_KEY`: `your_news_api_key`
6. Click **Create Web Service**. Render will deploy your live URL (e.g. `https://marp-stock-ai.onrender.com`).

---

### Option 2: Hugging Face Spaces (Free CPU / ML Hosting)
1. Create a free account on [Hugging Face](https://huggingface.co/).
2. Create a **New Space** -> Select **Docker** or **Gradio/Flask**.
3. Push your repository files to the Space.
4. Add your `NEWS_API_KEY` under **Space Settings -> Repository Secrets**.
5. Hugging Face will build and host your app with zero cold-start delay.

---

### Option 3: Railway.app / Koyeb
1. Connect your GitHub repository on [Railway](https://railway.app/).
2. Railway detects `Procfile` and `requirements.txt` automatically.
3. Add `NEWS_API_KEY` in Railway environment variables.
4. Click **Deploy**.