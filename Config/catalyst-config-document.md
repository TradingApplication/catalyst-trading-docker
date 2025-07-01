# Catalyst Trading System - Configuration Document
## Environment: Production
## Created: July 1, 2025

> **⚠️ SECURITY WARNING**: This document contains sensitive information. 
> - Store this file securely and encrypted
> - Never commit to public repositories
> - Use environment variables in production

---

## 1. DigitalOcean Infrastructure

### 1.1 Droplet Configuration
```yaml
Droplet Details:
  IP_Address: [FILL_IN_YOUR_DROPLET_IP]
  IPv6_Address: [FILL_IN_IF_ENABLED]
  Name: catalyst-trading-prod-01  # Set this after creation
  Region: [YOUR_REGION]  # e.g., nyc3, sfo3
  Size: s-2vcpu-4gb  # 4GB RAM, 2 vCPUs
  Image: ubuntu-22-04-x64
  Created: 2025-07-01
  Monthly_Cost: $24.00
  Backup_Enabled: Yes/No
  Monitoring_Enabled: Yes
```

### 1.2 SSH Access
```yaml
SSH Configuration:
  Host_Alias: catalyst-trading
  SSH_Port: 22
  SSH_User: root
  SSH_Key_Name: digitalocean_catalyst
  SSH_Key_Location: ~/.ssh/digitalocean_catalyst
  
# Add to ~/.ssh/config:
Host catalyst-trading
    HostName [YOUR_DROPLET_IP]
    User root
    Port 22
    IdentityFile ~/.ssh/digitalocean_catalyst
```

### 1.3 Domain Configuration (if applicable)
```yaml
Domain:
  Domain_Name: catalyst-trading.yourdomain.com
  DNS_Provider: DigitalOcean/Cloudflare/Other
  A_Record: [YOUR_DROPLET_IP]
  SSL_Certificate: Let's Encrypt
```

---

## 2. PostgreSQL Database Configuration

### 2.1 DigitalOcean Managed Database
```yaml
Database Cluster:
  Name: catalyst-trading-db
  Engine: PostgreSQL 15
  Plan: db-s-1vcpu-1gb  # Basic plan
  Region: [SAME_AS_DROPLET]
  Port: 25060
  SSL_Mode: require
  Connection_Pool_Size: 22
  Monthly_Cost: $15.00
  
Connection Details:
  Host: [FILL_IN_YOUR_DB_HOST].db.ondigitalocean.com
  Port: 25060
  Database: catalyst_trading
  Username: doadmin
  Password: [FILL_IN_YOUR_DB_PASSWORD]
  SSL_Required: true
  
  # Connection String Format:
  postgresql://doadmin:[PASSWORD]@[HOST]:25060/catalyst_trading?sslmode=require
  
Application User:
  Username: catalyst_app
  Password: [GENERATE_SECURE_PASSWORD]
  Permissions: ALL on catalyst_trading database
```

### 2.2 Database Schema
```yaml
Tables Created:
  - trading_cycles
  - service_health
  - workflow_log
  - news_raw
  - news_collection_stats
  - scanning_results_v2
  - pattern_analysis
  - pattern_statistics
  - trading_signals
  - signal_performance
  - trade_records
  - active_positions
  - trading_performance
  - daily_performance_summary
  - source_accuracy_report
  - pattern_performance_report
  - ml_data_quality_report
```

---

## 3. Service Configuration

### 3.1 Service Ports
```yaml
Microservices:
  coordination:
    Port: 5000
    Internal_URL: http://coordination:5000
    External_URL: http://[DROPLET_IP]:5000
    
  scanner:
    Port: 5001
    Internal_URL: http://scanner:5001
    External_URL: http://[DROPLET_IP]:5001
    
  pattern:
    Port: 5002
    Internal_URL: http://pattern:5002
    External_URL: http://[DROPLET_IP]:5002
    
  technical:
    Port: 5003
    Internal_URL: http://technical:5003
    External_URL: http://[DROPLET_IP]:5003
    
  trading:
    Port: 5005
    Internal_URL: http://trading:5005
    External_URL: http://[DROPLET_IP]:5005
    
  news:
    Port: 5008
    Internal_URL: http://news:5008
    External_URL: http://[DROPLET_IP]:5008
    
  reporting:
    Port: 5009
    Internal_URL: http://reporting:5009
    External_URL: http://[DROPLET_IP]:5009
    
  dashboard:
    Port: 5010
    Internal_URL: http://dashboard:5010
    External_URL: http://[DROPLET_IP]:5010
    Public_URL: http://[DROPLET_IP]  # Port 80
```

### 3.2 Docker Network
```yaml
Docker Configuration:
  Network_Name: catalyst-network
  Network_Driver: bridge
  Internal_DNS: automatic
  Container_Prefix: catalyst-
```

---

## 4. API Keys and External Services

### 4.1 Trading APIs
```yaml
Alpaca (Paper Trading):
  API_Key: [FILL_IN_YOUR_ALPACA_API_KEY]
  Secret_Key: [FILL_IN_YOUR_ALPACA_SECRET_KEY]
  Base_URL: https://paper-api.alpaca.markets
  Data_URL: https://data.alpaca.markets
  Account_Type: paper
```

### 4.2 News and Data APIs
```yaml
NewsAPI:
  API_Key: [FILL_IN_YOUR_NEWSAPI_KEY]
  Plan: Free/Developer/Professional
  Requests_Per_Day: 500/1000/unlimited
  
Alpha Vantage:
  API_Key: [FILL_IN_YOUR_ALPHAVANTAGE_KEY]
  Plan: Free/Premium
  Calls_Per_Minute: 5/75
  
Finnhub (Optional):
  API_Key: [FILL_IN_YOUR_FINNHUB_KEY]
  Plan: Free/Professional
```

---

## 5. Application Settings

### 5.1 Trading Parameters
```yaml
Risk Management:
  Max_Positions: 5
  Max_Position_Size_Pct: 20.0
  Max_Daily_Loss_Pct: 5.0
  Pre_Market_Position_Pct: 10.0
  Min_Price: 1.0
  Max_Price: 10000.0
  
Stop Loss Settings:
  Base_Stop_Loss_Pct: 2.0
  Strong_Catalyst_Stop: 1.5
  Moderate_Catalyst_Stop: 2.0
  Weak_Catalyst_Stop: 3.0
  
Position Sizing:
  Strong_Signal: 100%  # of allocated capital
  Normal_Signal: 50%
  Weak_Signal: 25%
```

### 5.2 Scanner Configuration
```yaml
Scanner Settings:
  Initial_Universe_Size: 100
  Catalyst_Filter_Size: 20
  Final_Selection_Size: 5
  Min_Volume: 500000
  Min_Relative_Volume: 1.5
  Min_Price_Change: 2.0
  PreMarket_Min_Volume: 50000
```

### 5.3 News Collection
```yaml
News Settings:
  Collection_Interval_Minutes: 5  # Pre-market
  Market_Hours_Interval: 30
  After_Hours_Interval: 60
  Max_Articles_Per_Source: 20
  Sources_Enabled:
    - NewsAPI
    - AlphaVantage
    - RSS_Feeds
```

---

## 6. File System Layout

### 6.1 Directory Structure
```bash
/opt/catalyst-trading-system/
├── config/
│   ├── .env                    # Environment variables
│   ├── config.json            # Application config
│   └── credentials/           # Encrypted credentials
├── services/
│   ├── coordination/
│   ├── scanner/
│   ├── pattern/
│   ├── technical/
│   ├── trading/
│   ├── news/
│   ├── reporting/
│   └── dashboard/
├── data/
│   ├── trading_system.db      # SQLite (if not using PostgreSQL)
│   └── cache/
├── logs/
│   ├── coordination.log
│   ├── scanner.log
│   ├── pattern.log
│   ├── technical.log
│   ├── trading.log
│   ├── news.log
│   ├── reporting.log
│   └── dashboard.log
├── scripts/
│   ├── deploy.sh
│   ├── backup.sh
│   ├── health-check.sh
│   └── restore.sh
├── backups/
├── docker-compose.yml
├── docker-compose.monitoring.yml
└── nginx/
    ├── nginx.conf
    └── ssl/
```

---

## 7. Security Configuration

### 7.1 Firewall Rules (UFW)
```bash
# Current firewall configuration
ufw allow 22/tcp     # SSH
ufw allow 80/tcp     # HTTP
ufw allow 443/tcp    # HTTPS
ufw allow 5432/tcp   # PostgreSQL (restrict to VPC later)
ufw allow 3000/tcp   # Grafana (restrict later)

# Service ports (internal only - Docker handles)
# 5000-5010 are not exposed externally
```

### 7.2 SSL/TLS Configuration
```yaml
SSL Certificate:
  Provider: Let's Encrypt
  Domain: [YOUR_DOMAIN]
  Auto_Renewal: Yes
  Certificate_Path: /etc/letsencrypt/live/[YOUR_DOMAIN]/
  Renewal_Command: certbot renew
```

---

## 8. Monitoring and Logging

### 8.1 Monitoring Stack
```yaml
Prometheus:
  Port: 9090
  Config: /opt/catalyst-trading-system/monitoring/prometheus.yml
  Retention: 15d
  
Grafana:
  Port: 3000
  Default_User: admin
  Default_Password: [CHANGE_ON_FIRST_LOGIN]
  Dashboards:
    - Trading Performance
    - Service Health
    - System Metrics
```

### 8.2 Backup Configuration
```yaml
Backup Settings:
  Schedule: Daily at 2:00 AM
  Retention: 7 days
  Backup_Location: /opt/catalyst-trading-system/backups/
  Include:
    - PostgreSQL database
    - Configuration files
    - Logs (compressed)
  Exclude:
    - Docker images
    - Temporary files
```

---

## 9. Environment Variables File

Create `/opt/catalyst-trading-system/config/.env`:
```bash
# Database Configuration
DATABASE_HOST=[YOUR_DB_HOST].db.ondigitalocean.com
DATABASE_PORT=25060
DATABASE_NAME=catalyst_trading
DATABASE_USER=catalyst_app
DATABASE_PASSWORD=[YOUR_SECURE_PASSWORD]
DATABASE_SSL_MODE=require

# Build connection URL
DATABASE_URL=postgresql://${DATABASE_USER}:${DATABASE_PASSWORD}@${DATABASE_HOST}:${DATABASE_PORT}/${DATABASE_NAME}?sslmode=${DATABASE_SSL_MODE}

# Application Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
SECRET_KEY=[GENERATE_RANDOM_KEY]

# Service Configuration
COORDINATION_PORT=5000
SCANNER_PORT=5001
PATTERN_PORT=5002
TECHNICAL_PORT=5003
TRADING_PORT=5005
NEWS_PORT=5008
REPORTING_PORT=5009
DASHBOARD_PORT=5010

# API Keys
ALPACA_API_KEY=[YOUR_KEY]
ALPACA_SECRET_KEY=[YOUR_SECRET]
NEWSAPI_KEY=[YOUR_KEY]
ALPHAVANTAGE_KEY=[YOUR_KEY]
FINNHUB_KEY=[YOUR_KEY]

# Feature Flags
AUTO_TRADING_ENABLED=false
PRE_MARKET_TRADING=true
AFTER_HOURS_TRADING=true
NEWS_COLLECTION_ENABLED=true
```

---

## 10. Quick Commands Reference

### 10.1 Droplet Management
```bash
# SSH to droplet
ssh catalyst-trading

# Rename droplet (via DigitalOcean API)
doctl compute droplet-action rename [DROPLET_ID] --droplet-name catalyst-trading-prod-01

# Check droplet status
doctl compute droplet get [DROPLET_ID]
```

### 10.2 Service Management
```bash
# Start all services
cd /opt/catalyst-trading-system && docker-compose up -d

# Check service health
./scripts/health-check.sh

# View logs
docker-compose logs -f [service_name]

# Restart a service
docker-compose restart [service_name]
```

### 10.3 Database Access
```bash
# Connect to PostgreSQL
psql "postgresql://catalyst_app:[PASSWORD]@[HOST]:25060/catalyst_trading?sslmode=require"

# Backup database
pg_dump "postgresql://..." > backup.sql

# Check database size
psql -c "SELECT pg_database_size('catalyst_trading');"
```

---

## 11. Verification Checklist

- [ ] Droplet IP recorded
- [ ] SSH access working
- [ ] Database connection string tested
- [ ] All API keys obtained
- [ ] Environment file created
- [ ] Firewall configured
- [ ] Backup script scheduled
- [ ] Monitoring accessible
- [ ] All services healthy

---

## 12. Support Information

### DigitalOcean Support
- Ticket System: https://cloud.digitalocean.com/support
- Status Page: https://status.digitalocean.com/
- Documentation: https://docs.digitalocean.com/

### Service Documentation
- GitHub Repo: https://github.com/TradingApplication/catalyst-trading-system
- Architecture Doc: See repository /Docs folder
- API Documentation: http://[DROPLET_IP]:5010/api/docs

---

**Document Version**: 1.0.0  
**Last Updated**: July 1, 2025  
**Next Review**: July 8, 2025

---

## Notes Section

Use this section to record any specific configuration decisions, issues encountered, or customizations made:

```
[DATE] - [NOTE]
Example:
2025-07-01 - Droplet created without name setting option, renamed after creation
2025-07-01 - Chose NYC3 region for lowest latency to trading APIs
```