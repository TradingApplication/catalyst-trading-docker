# Catalyst Trading System - Implementation Guide v2.1.0

## 🚀 Quick Start Deployment Guide

This guide will get your Catalyst Trading System operational on DigitalOcean in 30-45 minutes.

---

## Prerequisites

### DigitalOcean Resources
- **Droplet**: Ubuntu 22.04, 4GB RAM minimum
- **Managed PostgreSQL**: Basic tier ($15/month)
- **Region**: Singapore (for optimal market connectivity)

### Required API Keys
- NewsAPI key (https://newsapi.org)
- AlphaVantage key (https://www.alphavantage.co)
- Alpaca Markets API keys (https://alpaca.markets)

---

## Step 1: Initial Server Setup (5 minutes)

### 1.1 SSH into your droplet
```bash
ssh root@your-droplet-ip
```

### 1.2 Update system and install Docker
```bash
# Update package list
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
rm get-docker.sh

# Install Docker Compose
apt install docker-compose -y

# Verify installations
docker --version
docker-compose --version
```

### 1.3 Create project directory
```bash
mkdir -p /opt/catalyst-trading-system
cd /opt/catalyst-trading-system
```

---

## Step 2: Upload Project Files (10 minutes)

### 2.1 Upload all service files
Upload these files to `/opt/catalyst-trading-system/`:
- `coordination_service.py`
- `news_service.py`
- `scanner_service.py`
- `pattern_service.py`
- `technical_service.py`
- `trading_service.py`
- `reporting_service.py`
- `dashboard_service.py`
- `database_utils.py`
- `requirements.txt`
- `docker-compose.yml`
- `prometheus.yml`
- All Dockerfile.* files

### 2.2 Create the .env file
```bash
nano .env
```

Add this content (replace with your actual values):
```env
# Catalyst Trading System Environment Configuration
# Version: 2.1.0 Production

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# DigitalOcean Managed PostgreSQL
DATABASE_URL=postgresql://doadmin:YOUR_PASSWORD@YOUR_CLUSTER.db.ondigitalocean.com:25060/catalyst?sslmode=require

# Redis Configuration (Docker internal)
REDIS_URL=redis://redis:6379/0

# =============================================================================
# SERVICE URLS (Docker service names)
# =============================================================================
COORDINATION_URL=http://coordination-service:5000
NEWS_SERVICE_URL=http://news-service:5008
SCANNER_SERVICE_URL=http://scanner-service:5001
PATTERN_SERVICE_URL=http://pattern-service:5002
TECHNICAL_SERVICE_URL=http://technical-service:5003
TRADING_SERVICE_URL=http://trading-service:5005
REPORTING_SERVICE_URL=http://reporting-service:5009
DASHBOARD_SERVICE_URL=http://web-dashboard:5010

# =============================================================================
# API KEYS
# =============================================================================

# Alpaca Trading API (Paper Trading)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# News Data Sources
NEWSAPI_KEY=your_newsapi_key
ALPHAVANTAGE_KEY=your_alphavantage_key

# =============================================================================
# TRADING CONFIGURATION
# =============================================================================

# Risk Management
MAX_POSITIONS=5
POSITION_SIZE_PCT=20
STOP_LOSS_PCT=2
TAKE_PROFIT_PCT=5

# Trading Hours (EST)
MARKET_OPEN_HOUR=9
MARKET_OPEN_MINUTE=30
MARKET_CLOSE_HOUR=16
MARKET_CLOSE_MINUTE=0

# Pre-market Trading
PREMARKET_START_HOUR=4
PREMARKET_END_HOUR=9
PREMARKET_POSITION_PCT=10

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Service Configuration
FLASK_SECRET_KEY=your-secret-key-here
```

### 2.3 Create required directories
```bash
mkdir -p logs data models reports
chmod -R 755 logs data models reports
```

---

## Step 3: Database Initialization (5 minutes)

### 3.1 Connect to PostgreSQL and create database
```bash
# Connect to your DigitalOcean PostgreSQL cluster
psql $DATABASE_URL

# Create the catalyst database if it doesn't exist
CREATE DATABASE catalyst;
\q
```

### 3.2 Run database initialization
```bash
python init_database.py
```

### 3.3 Verify tables were created
```bash
psql $DATABASE_URL -c "\dt"
```

You should see all these tables:
- news_raw
- trading_candidates
- pattern_analysis
- trading_signals
- trade_records
- source_metrics
- narrative_clusters
- outcome_tracking
- trading_cycles
- service_health
- workflow_log
- configuration

---

## Step 4: Build and Deploy Services (20 minutes)

### 4.1 Build all Docker images
```bash
# This will take 10-15 minutes on first run
docker-compose build --no-cache
```

### 4.2 Start services in order
```bash
# Start Redis first
docker-compose up -d redis

# Wait for Redis to be ready
sleep 10

# Start all other services
docker-compose up -d

# Check service status
docker-compose ps
```

### 4.3 Verify all services are healthy
```bash
# Check each service health endpoint
curl http://localhost:5000/health  # Coordination
curl http://localhost:5008/health  # News
curl http://localhost:5001/health  # Scanner
curl http://localhost:5002/health  # Pattern
curl http://localhost:5003/health  # Technical
curl http://localhost:5005/health  # Trading
curl http://localhost:5009/health  # Reporting
curl http://localhost:5010/health  # Dashboard
```

---

## Step 5: Initial Configuration (5 minutes)

### 5.1 Access the web dashboard
Open in your browser:
```
http://your-droplet-ip:5010
```

### 5.2 Configure trading parameters
```bash
curl -X POST http://localhost:5000/workflow_config \
  -H "Content-Type: application/json" \
  -d '{
    "premarket_aggressive_mode": true,
    "scan_interval_minutes": 5,
    "min_catalyst_score": 30,
    "pattern_confidence_threshold": 70
  }'
```

### 5.3 Start your first trading cycle
```bash
# Manual cycle start
curl -X POST http://localhost:5000/start_trading_cycle

# Check cycle status
curl http://localhost:5000/current_cycle
```

---

## Step 6: Production Configuration (10 minutes)

### 6.1 Set up firewall rules
```bash
# Allow SSH, HTTP, and service ports
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 5010/tcp  # Dashboard
ufw enable
```

### 6.2 Configure automatic startup
```bash
# Enable Docker to start on boot
systemctl enable docker

# Create systemd service for Catalyst
cat > /etc/systemd/system/catalyst-trading.service << EOF
[Unit]
Description=Catalyst Trading System
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/catalyst-trading-system
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

# Enable the service
systemctl enable catalyst-trading
systemctl start catalyst-trading
```

### 6.3 Set up monitoring alerts (optional)
```bash
# Access Prometheus metrics
http://your-droplet-ip:9090

# Key metrics to monitor:
# - catalyst_trades_total
# - catalyst_positions_open
# - catalyst_pnl_total
# - catalyst_service_health
```

---

## 📊 Monitoring and Management

### Daily Operations

#### Check system status
```bash
# View all service logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f coordination-service
docker-compose logs -f trading-service

# Check resource usage
docker stats
```

#### Generate reports
```bash
# Daily trading summary
curl http://localhost:5009/daily_summary

# Pattern effectiveness
curl http://localhost:5009/pattern_effectiveness

# Source accuracy metrics
curl http://localhost:5009/source_accuracy
```

### Troubleshooting

#### Service not starting
```bash
# Check logs for specific service
docker-compose logs scanner-service

# Restart individual service
docker-compose restart scanner-service

# Check database connectivity
docker-compose exec coordination-service python -c "from database_utils import health_check; print(health_check())"
```

#### High memory usage
```bash
# Adjust memory limits in docker-compose.yml
# Current limits are optimized for 4GB droplet
# Reduce if needed:
# memory: 128M (instead of 256M)
```

---

## 🎯 Performance Tuning

### News Collection Frequency
```python
# Adjust in coordination_service.py schedule
'pre_market': '*/5 * * * *',    # Every 5 minutes
'market_hours': '*/15 * * * *',  # Every 15 minutes
'after_hours': '*/30 * * * *',   # Every 30 minutes
```

### Scanner Sensitivity
```python
# Modify catalyst scoring in scanner_service.py
'earnings': 40,    # Increase for more weight
'fda': 35,        # Decrease for less weight
'merger': 30,
```

### Pattern Confidence
```python
# Adjust in pattern_service.py
'min_confidence': 70,  # Higher = fewer but better signals
'catalyst_weight': 1.5,  # How much news affects patterns
```

---

## 🚨 Important Notes

1. **Paper Trading Only**: System is configured for paper trading. Do NOT use real money until thoroughly tested.

2. **API Rate Limits**: 
   - NewsAPI: 500 requests/day (free tier)
   - AlphaVantage: 5 requests/minute
   - Monitor usage in logs

3. **Database Backups**: Set up automated backups in DigitalOcean dashboard

4. **Security**: 
   - Change default passwords
   - Use SSH keys instead of passwords
   - Keep API keys secure

---

## 📈 Next Steps

1. **Monitor Initial Performance**
   - Let system run for 1-2 weeks
   - Review daily reports
   - Adjust parameters based on results

2. **Optimize Strategies**
   - Analyze pattern effectiveness reports
   - Tune catalyst scoring
   - Refine entry/exit rules

3. **Scale Infrastructure**
   - Add more droplets for redundancy
   - Implement load balancing
   - Set up disaster recovery

4. **ML Integration** (Phase 2)
   - Collect 3-6 months of data
   - Implement pattern recognition ML
   - Deploy GPU droplet for training

---

## 🆘 Support Resources

- **Logs**: `/opt/catalyst-trading-system/logs/`
- **Database**: Check `workflow_log` table for execution history
- **Metrics**: Prometheus at `http://your-ip:9090`
- **Dashboard**: `http://your-ip:5010`

---

## ✅ Deployment Verification Checklist

- [ ] All services show "healthy" status
- [ ] Database tables created successfully
- [ ] Web dashboard accessible
- [ ] First trading cycle completed
- [ ] News collection working (check logs)
- [ ] Scanner finding candidates
- [ ] Patterns being detected
- [ ] Paper trades executing
- [ ] Reports generating
- [ ] No errors in logs

---

**Congratulations! Your Catalyst Trading System is now operational!** 🎉

Monitor the dashboard and logs closely for the first few days to ensure smooth operation.