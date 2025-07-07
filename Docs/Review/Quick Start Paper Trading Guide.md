# 🚀 Quick Start Paper Trading Guide

## Once Dashboard Shows "Healthy" Status:

### 1. **Start Your First Trading Cycle**
- Open dashboard: http://localhost:5010
- Click the **"Start Trading Cycle"** button
- The system will:
  - 📰 Collect news for catalyst detection
  - 🔍 Scan for stocks with momentum
  - 📊 Analyze patterns with news context
  - 📈 Generate trading signals
  - 💰 Execute paper trades via Alpaca

### 2. **Monitor in Real-Time**
Dashboard shows:
- **Trading Performance**: Win rate, P&L, Sharpe ratio
- **Portfolio Status**: Current positions, unrealized P&L
- **Recent Trades**: Last 5 executed trades
- **System Alerts**: Risk warnings, position limits

### 3. **View Detailed Logs**
```bash
# See what's happening in real-time
docker-compose logs -f

# Watch specific services
docker-compose logs -f trading-service   # Trade execution
docker-compose logs -f news-service      # News collection
docker-compose logs -f scanner-service   # Stock scanning
```

### 4. **Check Databa