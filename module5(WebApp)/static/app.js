let currentTicker = "TCS.NS";

document.addEventListener("DOMContentLoaded", () => {
    fetchTopStockPills();
    loadDashboard(currentTicker);
    
    // Bind search form submit
    const searchForm = document.getElementById("search-form");
    if (searchForm) {
        searchForm.addEventListener("submit", (e) => {
            e.preventDefault();
            const input = document.getElementById("stock-search-input");
            const val = input ? input.value.trim() : "";
            if (val) {
                loadDashboard(val);
                input.value = "";
            }
        });
    }
});

function showLoading(msg = "Fetching live data & running ML NLP pipeline...") {
    const overlay = document.getElementById("loading-overlay");
    const txt = document.getElementById("loading-text");
    if (txt) txt.innerText = msg;
    if (overlay) overlay.classList.remove("hidden");
}

function hideLoading() {
    const overlay = document.getElementById("loading-overlay");
    if (overlay) overlay.classList.add("hidden");
}

async function fetchTopStockPills() {
    try {
        const response = await fetch("/api/stocks");
        const data = await response.json();
        const container = document.getElementById("quick-stock-pills");
        if (!container || !data.stocks) return;
        
        container.innerHTML = "";
        data.stocks.forEach(st => {
            const btn = document.createElement("button");
            btn.className = `stock-pill ${st.ticker === currentTicker ? 'active' : ''}`;
            btn.innerText = st.ticker.replace(".NS", "");
            btn.title = `${st.name} (${st.region})`;
            btn.addEventListener("click", () => {
                loadDashboard(st.ticker);
            });
            container.appendChild(btn);
        });
    } catch (err) {
        console.error("Failed to load top stock pills:", err);
    }
}

async function loadDashboard(ticker) {
    showLoading(`Loading data & ML prediction for ${ticker.toUpperCase()}...`);
    currentTicker = ticker;
    
    // Highlight active pill
    document.querySelectorAll(".stock-pill").forEach(pill => {
        const pTxt = pill.innerText.toUpperCase();
        const tTxt = ticker.replace(".NS", "").toUpperCase();
        if (pTxt === tTxt) {
            pill.classList.add("active");
        } else {
            pill.classList.remove("active");
        }
    });
    
    try {
        await Promise.all([
            fetchSummary(ticker),
            fetchStockCharts(ticker),
            fetchNewsFeed(ticker)
        ]);
    } catch (err) {
        console.error("Error loading dashboard data:", err);
    } finally {
        hideLoading();
    }
}

async function fetchSummary(ticker) {
    try {
        const response = await fetch(`/api/summary?ticker=${encodeURIComponent(ticker)}`);
        const data = await response.json();
        
        if (data.error) {
            alert(data.error);
            return;
        }

        // Update Ticker Header Displays
        const tickDisp = document.getElementById("current-ticker-display");
        const compDisp = document.getElementById("current-company-display");
        if (tickDisp) tickDisp.innerText = data.ticker || ticker;
        if (compDisp) compDisp.innerText = data.latest.company_name || "";

        // Render Prediction Badge
        const directionEl = document.getElementById("direction-badge");
        const isUp = data.latest.prediction === 1;
        directionEl.className = `direction-badge ${isUp ? "up" : "down"}`;
        directionEl.innerHTML = isUp ? "📈 BULLISH (UP)" : "📉 BEARISH (DOWN)";
        
        // Confidence
        const confPct = (data.latest.confidence * 100).toFixed(1);
        document.getElementById("confidence-value").innerText = `${confPct}%`;
        
        // Latest Metrics
        const currencySymbol = (data.ticker && (data.ticker.endsWith(".NS") || data.ticker === "TCS" || data.ticker === "RELIANCE" || data.ticker === "INFY")) ? '₹' : '$';
        document.getElementById("val-close").innerText = `${currencySymbol}${data.latest.close.toFixed(2)}`;
        document.getElementById("val-return").innerText = `${data.latest.daily_return > 0 ? '+' : ''}${data.latest.daily_return.toFixed(2)}%`;
        document.getElementById("val-return").style.color = data.latest.daily_return >= 0 ? '#10b981' : '#f43f5e';
        document.getElementById("val-rsi").innerText = data.latest.rsi.toFixed(1);
        document.getElementById("val-sent").innerText = data.latest.sent_compound.toFixed(3);
        
        // Render Model Accuracy & Feature Importances
        if (data.metrics) {
            document.getElementById("model-name").innerText = data.metrics.model_name || "RandomForest";
            document.getElementById("model-acc").innerText = `${((data.metrics.accuracy || 0) * 100).toFixed(1)}%`;
            document.getElementById("model-f1").innerText = (data.metrics.f1_score || 0).toFixed(3);
            
            renderFeatureImportance(data.metrics.feature_importances);
        }
        
    } catch (err) {
        console.error("Failed to load summary:", err);
    }
}

async function fetchStockCharts(ticker) {
    try {
        const response = await fetch(`/api/stock_data?ticker=${encodeURIComponent(ticker)}`);
        const jsonRes = await response.json();
        const data = jsonRes.data || [];
        
        if (!Array.isArray(data) || data.length === 0) return;
        
        const dates = data.map(d => d.Date);
        const closes = data.map(d => d.Close);
        const sma5 = data.map(d => d.SMA_5);
        const sma20 = data.map(d => d.SMA_20);
        const sentiments = data.map(d => d.sent_compound_mean);
        
        // Price Chart with SMA overlays
        const traceClose = {
            x: dates,
            y: closes,
            type: 'scatter',
            mode: 'lines',
            name: `${jsonRes.ticker || ticker} Close`,
            line: { color: '#6366f1', width: 2.5 }
        };
        
        const traceSMA5 = {
            x: dates,
            y: sma5,
            type: 'scatter',
            mode: 'lines',
            name: 'SMA 5',
            line: { color: '#06b6d4', width: 1.5, dash: 'dot' }
        };
        
        const traceSMA20 = {
            x: dates,
            y: sma20,
            type: 'scatter',
            mode: 'lines',
            name: 'SMA 20',
            line: { color: '#f59e0b', width: 1.5, dash: 'dash' }
        };
        
        const priceLayout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: { t: 20, r: 20, l: 50, b: 40 },
            legend: { orientation: 'h', y: 1.1, font: { color: '#94a3b8' } },
            xaxis: { gridcolor: 'rgba(255,255,255,0.05)', tickfont: { color: '#94a3b8' } },
            yaxis: { gridcolor: 'rgba(255,255,255,0.05)', tickfont: { color: '#94a3b8' } }
        };
        
        Plotly.newPlot('price-chart', [traceClose, traceSMA5, traceSMA20], priceLayout, { responsive: true, displayModeBar: false });
        
        // Sentiment Chart
        const traceSent = {
            x: dates,
            y: sentiments,
            type: 'bar',
            name: 'Daily Sentiment Score',
            marker: {
                color: sentiments.map(s => s >= 0 ? 'rgba(16, 185, 129, 0.7)' : 'rgba(244, 63, 94, 0.7)')
            }
        };
        
        const sentLayout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: { t: 20, r: 20, l: 50, b: 40 },
            legend: { orientation: 'h', y: 1.1, font: { color: '#94a3b8' } },
            xaxis: { gridcolor: 'rgba(255,255,255,0.05)', tickfont: { color: '#94a3b8' } },
            yaxis: { gridcolor: 'rgba(255,255,255,0.05)', tickfont: { color: '#94a3b8' }, title: 'Sentiment (-1 to +1)' }
        };
        
        Plotly.newPlot('sentiment-chart', [traceSent], sentLayout, { responsive: true, displayModeBar: false });
        
    } catch (err) {
        console.error("Failed to load stock charts:", err);
    }
}

function renderFeatureImportance(importances) {
    if (!importances) return;
    const container = document.getElementById("feature-importance-list");
    container.innerHTML = "";
    
    const entries = Object.entries(importances).slice(0, 8);
    const maxVal = Math.max(...entries.map(e => e[1])) || 1;
    
    entries.forEach(([feat, val]) => {
        const pct = ((val / maxVal) * 100).toFixed(1);
        const isNLP = feat.startsWith("sent_") || feat === "news_count";
        
        const item = document.createElement("div");
        item.className = "feature-bar-item";
        item.innerHTML = `
            <div class="feature-info">
                <span>${feat} ${isNLP ? '(NLP)' : '(Tech)'}</span>
                <span>${(val * 100).toFixed(1)}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill ${isNLP ? 'nlp' : ''}" style="width: ${pct}%"></div>
            </div>
        `;
        container.appendChild(item);
    });
}

async function fetchNewsFeed(ticker) {
    try {
        const response = await fetch(`/api/news?ticker=${encodeURIComponent(ticker)}`);
        const jsonRes = await response.json();
        const news = jsonRes.news || [];
        
        const container = document.getElementById("news-feed-list");
        container.innerHTML = "";
        
        if (news.length === 0) {
            container.innerHTML = "<p style='color:#94a3b8;'>No news articles available.</p>";
            return;
        }
        
        news.forEach(item => {
            const newsEl = document.createElement("div");
            newsEl.className = "news-item";
            
            const dateStr = item.published_at ? item.published_at.substring(0, 10) : "Recent";
            
            newsEl.innerHTML = `
                <div class="news-header">
                    <span>${item.source || 'News'}</span>
                    <span>${dateStr}</span>
                </div>
                <div class="news-title">${item.title}</div>
            `;
            container.appendChild(newsEl);
        });
        
    } catch (err) {
        console.error("Failed to load news feed:", err);
    }
}
