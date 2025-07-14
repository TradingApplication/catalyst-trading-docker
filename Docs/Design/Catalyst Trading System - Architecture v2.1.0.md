# Catalyst Trading System - Architecture v2.1.0

**Repository**: catalyst-trading-system  
**Version**: 2.1.0  
**Date**: July 8, 2025  
**Status**: Production Architecture  
**Previous Version**: 2.0.0 (June 27, 2025)

## Revision History

### v2.1.0 (July 8, 2025)
- **Production Alignment**: Updated to reflect actual deployed architecture
- **Database Integration**: Aligned with Database Schema v2.1.0 and Services v2.1.0
- **Service Updates**: Reflects database_utils.py v2.3.5 integration
- **Performance Enhancements**: Added Redis caching layer details
- **DigitalOcean Specific**: Production deployment considerations
- **Implementation Reality**: Matches actual codebase structure

## Executive Summary

The Catalyst Trading System is a production-ready, news-driven algorithmic trading platform deployed on DigitalOcean infrastructure. The system identifies and executes day trading opportunities based on market catalysts, focusing exclusively on securities with news events that create tradeable momentum.

### Core Innovation

1. **News-Driven Selection**: Securities selected exclusively based on news catalysts
2. **Source Intelligence**: Tracks accuracy, alignment patterns, and narrative evolution
3. **Clean Data Architecture**: Raw data preserved for ML, processed data optimized for trading
4. **Social Mission**: Profits fund homeless shelter operations
5. **Production Ready**: Fully deployed on DigitalOcean with managed services

## System Architecture

### High-Level Production Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CATALYST TRADING SYSTEM v2.1.0                   │
│                    DigitalOcean Production Deployment               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Data Collection Layer                     │   │
│  │  ┌────────────────────┐      ┌────────────────────┐        │   │
│  │  │   News Collection   │      │   Market Data     │        │   │
│  │  │   Service (5008)    │      │   (yfinance)      │        │   │
│  │  │                     │      │                    │        │   │
│  │  │ • NewsAPI          │      │ • Price/Volume    │        │   │
│  │  │ • AlphaVantage      │      │ • Real-time       │        │   │
│  │  │ • Finnhub          │      │ • Historical      │        │   │
│  │  │ • Source tracking  │      │ • Pre-market      │        │   │
│  │  └──────────┬──────────┘      └─────────┬─────────┘        │   │
│  │             │                            │                   │   │
│  │             ▼                            ▼                   │   │
│  │  ┌──────────────────────────────────────────────────┐      │   │
│  │  │     PostgreSQL (Managed) + Redis Cache           │      │   │
│  │  │  • news_raw (preserved)                          │      │   │
│  │  │  • source_metrics                                │      │   │
│  │  │  • narrative_clusters                            │      │   │
│  │  └──────────────────────────────────────────────────┘      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                 Intelligence & Analysis Layer                │   │
│  │  ┌────────────────────┐      ┌────────────────────┐        │   │
│  │  │  Security Scanner  │      │ Pattern Analysis   │        │   │
│  │  │  Service (5001)    │      │ Service (5002)     │        │   │
│  │  │                    │      │                    │        │   │
│  │  │ • Catalyst score   │      │ • Candlesticks    │        │   │
│  │  │ • Dynamic universe │      │ • Context-aware   │        │   │
│  │  │ • 50→20→5 filter   │      │ • Catalyst-aligned│        │   │
│  │  └──────────┬─────────┘      └─────────┬─────────┘        │   │
│  │             │                           │                   │   │
│  │             ▼                           ▼                   │   │
│  │  ┌──────────────────────────────────────────────────┐      │   │
│  │  │     PostgreSQL Tables + Redis Caching            │      │   │
│  │  │  • trading_candidates                            │      │   │
│  │  │  • pattern_analysis                              │      │   │
│  │  │  • pattern_detections                            │      │   │
│  │  └──────────────────────────────────────────────────┘      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Decision Layer                            │   │
│  │  ┌────────────────────┐      ┌────────────────────┐        │   │
│  │  │ Technical Analysis │      │  Risk Management   │        │   │
│  │  │ Service (5003)     │      │  (Embedded)        │        │   │
│  │  │                    │      │                    │        │   │
│  │  │ • Multi-timeframe  │      │ • Position sizing │        │   │
│  │  │ • Signal generate  │      │ • Stop placement  │        │   │
│  │  │ • Confidence score │      │ • Risk per trade  │        │   │
│  │  └──────────┬─────────┘      └─────────┬─────────┘        │   │
│  │             │                           │                   │   │
│  │             ▼                           ▼                   │   │
│  │  ┌──────────────────────────────────────────────────┐      │   │
│  │  │  • trading_signals                               │      │   │
│  │  │  • technical_indicators                          │      │   │
│  │  └──────────────────────────────────────────────────┘      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Execution Layer                           │   │
│  │  ┌────────────────────┐      ┌────────────────────┐        │   │
│  │  │  Paper Trading     │      │  Live Trading      │        │   │
│  │  │  Service (5005)    │      │  (Phase 2)        │        │   │
│  │  │                    │      │                    │        │   │
│  │  │ • Alpaca Paper API │      │ • Real capital    │        │   │
│  │  │ • Order execution  │      │ • Compliance      │        │   │
│  │  │ • P&L tracking     │      │ • Audit ready     │        │   │
│  │  └──────────┬─────────┘      └─────────┬─────────┘        │   │
│  │             │                           │                   │   │
│  │             ▼                           ▼                   │   │
│  │  ┌──────────────────────────────────────────────────┐      │   │
│  │  │  • trade_records (with catalyst tracking)        │      │   │
│  │  │  • positions (real-time view)                    │      │   │
│  │  └──────────────────────────────────────────────────┘      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Orchestration & Support Layer                   │   │
│  │                                                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │   │
│  │  │ Coordination │  │  Dashboard   │  │  Reporting   │     │   │
│  │  │ Service      │  │  Service     │  │  Service     │     │   │
│  │  │ (5009)       │  │  (5006)      │  │  (5004)      │     │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘     │   │
│  │                                                              │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │           Database Services Layer (v2.1.0)           │   │   │
│  │  │  • database_utils.py v2.3.5 (connection pooling)    │   │   │
│  │  │  • Redis client (caching layer)                     │   │   │
│  │  │  • Migration management                             │   │   │
│  │  │  • Backup to DO Spaces                              │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Database Architecture (v2.1.0)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Storage Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              DigitalOcean Managed PostgreSQL              │   │
│  │                                                           │   │
│  │  News & Intelligence Tables:                              │   │
│  │  • news_raw (immutable raw data)                         │   │
│  │  • source_metrics (reliability tracking)                  │   │
│  │  • narrative_clusters (story grouping)                    │   │
│  │  • news_collection_stats                                  │   │
│  │                                                           │   │
│  │  Trading Operations Tables:                               │   │
│  │  • trading_candidates (scanner output)                    │   │
│  │  • trading_signals (actionable signals)                   │   │
│  │  • trade_records (execution & outcomes)                   │   │
│  │  • positions (view)                                       │   │
│  │                                                           │   │
│  │  Analysis Tables:                                         │   │
│  │  • pattern_analysis (detected patterns)                   │   │
│  │  • pattern_detections (service tracking)                  │   │
│  │  • technical_indicators (TA calculations)                 │   │
│  │                                                           │   │
│  │  System Tables:                                           │   │
│  │  • trading_cycles (workflow tracking)                     │   │
│  │  • service_health (monitoring)                            │   │
│  │  • workflow_log (detailed execution)                      │   │
│  │  • configuration (system settings)                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               DigitalOcean Managed Redis                  │   │
│  │                                                           │   │
│  │  • Pattern detection cache (5 min TTL)                    │   │
│  │  • Technical indicators cache (5 min TTL)                 │   │
│  │  • News sentiment cache (10 min TTL)                      │   │
│  │  • Active positions cache                                 │   │
│  │  • Service health status                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Service Architecture Details

### 1. News Collection Service (Port 5008)
**Purpose**: Collect raw news data from multiple sources  
**Database**: Writes to news_raw, news_collection_stats  
**Key Features**:
- Multi-source integration (NewsAPI, AlphaVantage, Finnhub)
- Source tier classification (1=Bloomberg/Reuters, 5=Unknown)
- Pre-market detection and flagging
- Keyword extraction and ticker mention tracking
- Deduplication via news_id hash

### 2. Security Scanner Service (Port 5001)
**Purpose**: Select top 5 trading candidates based on catalysts  
**Database**: Reads news_raw, writes trading_candidates  
**Key Features**:
- Dynamic universe (50-100 most active stocks)
- Multi-stage filtering (50 → 20 → 5)
- Catalyst scoring algorithm
- Pre-market priority during 4-9:30 AM
- Real-time narrowing throughout the day

### 3. Pattern Analysis Service (Port 5002)
**Purpose**: Detect technical patterns with catalyst context  
**Database**: Reads trading_candidates, writes pattern_analysis  
**Key Features**:
- 13 candlestick patterns
- Catalyst alignment checking
- Multi-timeframe analysis
- Redis caching for performance

### 4. Technical Analysis Service (Port 5003)
**Purpose**: Generate trading signals from patterns  
**Database**: Writes trading_signals, technical_indicators  
**Key Features**:
- Component scoring (catalyst, pattern, technical, volume)
- Entry/exit calculation
- Risk management parameters
- Confidence scoring

### 5. Trading Service (Port 5005)
**Purpose**: Execute trades via Alpaca API  
**Database**: Reads trading_signals, writes trade_records  
**Key Features**:
- Paper trading integration
- Position management
- P&L tracking
- Catalyst outcome tracking

### 6. Coordination Service (Port 5009)
**Purpose**: Orchestrate the complete workflow  
**Database**: Writes trading_cycles, workflow_log  
**Key Features**:
- Aggressive pre-market mode (4-9:30 AM)
- Normal market hours mode
- Light after-hours mode
- Service health monitoring

### 7. Dashboard Service (Port 5006)
**Purpose**: Real-time monitoring interface  
**Database**: Read-only access to all tables  
**Key Features**:
- WebSocket real-time updates
- Service health visualization
- Trading performance metrics
- System control interface

### 8. Reporting Service (Port 5004)
**Purpose**: Performance analytics and reporting  
**Database**: Read-only access via views  
**Key Features**:
- Trade history analysis
- Pattern success rates
- Source accuracy tracking
- Daily performance reports

## Data Flow Architecture

### Primary Trading Flow

```
1. News Collection (Continuous)
   ├── Fetch from APIs
   ├── Extract metadata
   ├── Store in news_raw
   └── → news_collection_stats

2. Security Selection (Every 5 min)
   ├── Query news_raw for catalysts
   ├── Score and rank symbols
   ├── Filter to top 5
   └── → trading_candidates

3. Pattern Analysis (On new candidates)
   ├── Fetch price data
   ├── Detect patterns
   ├── Check catalyst alignment
   └── → pattern_analysis

4. Signal Generation (On patterns)
   ├── Calculate indicators
   ├── Score components
   ├── Generate entry/exit
   └── → trading_signals

5. Trade Execution (On signals)
   ├── Check risk limits
   ├── Send to Alpaca
   ├── Track position
   └── → trade_records

6. Outcome Tracking (On exit)
   ├── Update news accuracy
   ├── Track pattern success
   ├── Update source metrics
   └── → Feedback loop
```

### Database Service Integration

All services interact with the database through:

```python
# database_utils.py v2.3.5 provides:
- get_db_connection()      # Pooled PostgreSQL connections
- get_redis()             # Redis client for caching
- health_check()          # Service health monitoring
- with_db_retry()         # Retry decorator

# Standard pattern for all services:
from database_utils import get_db_connection, get_redis

def service_operation():
    with get_db_connection() as conn:
        # Database operations
        pass
```

## Technology Stack

### Core Technologies
- **Language**: Python 3.10+
- **Web Framework**: Flask 2.3.2
- **Database**: PostgreSQL 14 (Managed by DigitalOcean)
- **Cache**: Redis 6 (Managed by DigitalOcean)
- **Container**: Docker 24.0
- **Orchestration**: Docker Compose 2.20

### Python Dependencies
```
# Core
flask==2.3.2
flask-socketio==5.3.4
psycopg2-binary==2.9.7
redis==4.6.0
python-dotenv==1.0.0

# Market Data
yfinance==0.2.28
alpha_vantage==2.3.1
finnhub-python==2.4.19

# Trading
alpaca-py==0.13.3

# Analysis
pandas==2.0.3
numpy==1.24.3
ta==0.10.2

# Utilities
requests==2.31.0
pytz==2023.3
schedule==1.2.0
```

### External Services
- **News APIs**: NewsAPI, AlphaVantage, Finnhub
- **Market Data**: Yahoo Finance, Alpha Vantage
- **Broker**: Alpaca Markets
- **Cloud**: DigitalOcean (US East region)
- **Monitoring**: DigitalOcean native tools

## Security Architecture

### API Security
- All credentials in environment variables
- No hardcoded secrets in code
- API key rotation every 90 days
- Rate limiting on all endpoints

### Database Security
- SSL/TLS required for all connections
- Private networking within DigitalOcean VPC
- Connection string includes SSL mode
- Regular security patches via managed service

### Application Security
- Internal services on Docker network
- Only dashboard/API exposed externally
- Environment-based configuration
- No PII storage

### Audit Trail
- Complete trade history with catalyst context
- All decisions logged with reasoning
- Service actions tracked in workflow_log
- Compliance-ready data structure

## Performance Architecture

### Database Performance
- Connection pooling (2-10 connections)
- Strategic indexes on all foreign keys
- JSONB for flexible metadata
- Partitioning ready for scale

### Caching Strategy
- Redis for hot data (5-10 min TTL)
- Pattern detection results cached
- Technical indicators cached
- Cache invalidation on updates

### Service Performance
- Async where possible
- Batch database operations
- Efficient query patterns
- Resource monitoring

### Performance Targets
- News collection: < 2 min per cycle
- Scanner: < 15 sec for top 5
- Pattern analysis: < 3 sec per symbol
- Signal generation: < 1 sec
- Trade execution: < 500 ms

## Deployment Architecture

### DigitalOcean Infrastructure
```yaml
# App Platform Configuration
services:
  - name: catalyst-trading-system
    github:
      repo: dijo/catalyst-trading-system
      branch: main
    docker:
      registry_type: DOCR
    instance_count: 1
    instance_size: professional-xs
    
# Managed Databases
databases:
  - name: catalyst-trading-db
    engine: PG
    version: "14"
    size: db-s-2vcpu-4gb
    
  - name: catalyst-trading-cache
    engine: REDIS
    version: "6"
    size: db-s-1vcpu-1gb

# Spaces for Backups
spaces:
  - name: catalyst-backups
    region: nyc3
```

### Container Architecture
- All services in single Docker Compose
- Shared network for internal communication
- Volume mounts for logs
- Health checks on all services

### Environment Management
```bash
# Production environment variables
DATABASE_URL=postgresql://user:pass@host:25060/catalyst_trading?sslmode=require
REDIS_URL=rediss://user:pass@host:25061/0
NEWS_API_KEY=xxx
ALPACA_API_KEY=xxx
ALPACA_SECRET_KEY=xxx
```

## Monitoring & Observability

### Service Health Monitoring
- Health endpoints on all services
- Status tracked in service_health table
- Dashboard real-time visualization
- Automated alerts on failures

### Business Metrics
- News collection rate and sources
- Catalyst hit rate (news → profit)
- Pattern success by type
- Trading performance (win rate, P&L)
- Source reliability scores

### Technical Metrics
- Service response times
- Database query performance
- Cache hit rates
- API rate limit usage
- Error rates by service

### Logging Strategy
- Structured JSON logging
- Service-specific log files
- Centralized in workflow_log table
- Log rotation and archival

## Scalability Considerations

### Horizontal Scaling Ready
- Stateless service design
- Database connection pooling
- Redis for shared state
- Load balancer compatible

### Data Growth Management
- Time-based partitioning ready
- Automated backup/archive
- Efficient data retention
- Index optimization

### Future Enhancements
- Message queue for async processing
- Read replicas for analytics
- Multi-region deployment
- Kubernetes migration path

## Machine Learning Preparation

### Data Collection Built-In
- Pattern outcomes tracked
- News accuracy recorded
- Source reliability metrics
- Full context preserved

### ML-Ready Schema
- Training data tables ready
- Feature engineering support
- Outcome labeling automated
- A/B testing framework

### Future ML Services
- Pattern recognition enhancement
- News sentiment analysis
- Optimal timing prediction
- Source credibility scoring

## Architecture Principles

1. **Data Integrity First**: Raw data immutable, processed data auditable
2. **Service Independence**: Each service owns its domain and data
3. **Fail Gracefully**: Degraded operation better than system failure
4. **Observable Everything**: Every decision traceable and measurable
5. **Security by Design**: Defense in depth, least privilege
6. **Performance Matters**: Speed directly impacts trading opportunity
7. **Cloud Native**: Built for DigitalOcean managed services
8. **Social Impact**: Architecture serves the mission of funding social good

## Migration & Evolution

### From v2.0.0 to v2.1.0
- Standardized table names across all services
- Added compatibility views for smooth transition
- Integrated database_utils.py for all DB operations
- Added Redis caching layer
- Aligned with production deployment

### Future Architecture Evolution
- Phase 2: ML service integration
- Phase 3: Multi-strategy support
- Phase 4: Global market expansion
- Phase 5: Institutional features

This architecture provides a robust, scalable foundation for news-driven algorithmic trading while maintaining flexibility for future enhancements and the ultimate goal of funding social good through profitable trading.