# Catalyst Trading System - Architecture v2.0.0

**Repository**: catalyst-trading-system  
**Version**: 2.0.0  
**Date**: June 27, 2025  
**Status**: Foundation Document  

## Executive Summary

The Catalyst Trading System is a news-driven algorithmic trading platform that identifies and executes day trading opportunities based on market catalysts. Unlike traditional technical analysis systems that scan all securities, this system focuses on securities with news events that create tradeable momentum.

### Core Innovation

1. **News-Driven Selection**: Securities are selected based on news catalysts, not random technical scans
2. **Source Intelligence**: Tracks not just accuracy but alignment and agenda patterns  
3. **Clean Data Architecture**: Raw data preserved for ML, trading data optimized for execution
4. **Social Mission**: Profits fund homeless shelter operations

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CATALYST TRADING SYSTEM                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Data Collection Layer                     │   │
│  │  ┌────────────────────┐      ┌────────────────────┐        │   │
│  │  │   News Collection   │      │   Market Data     │        │   │
│  │  │   Service (5008)    │      │   (yfinance)      │        │   │
│  │  │                     │      │                    │        │   │
│  │  │ • Multi-source      │      │ • Price/Volume    │        │   │
│  │  │ • Source tracking   │      │ • Real-time       │        │   │
│  │  │ • Alignment detect  │      │ • Historical      │        │   │
│  │  └──────────┬──────────┘      └─────────┬─────────┘        │   │
│  │             │                            │                   │   │
│  │             ▼                            ▼                   │   │
│  │        news_raw                    market_data              │   │
│  │      (preserved)                  (time-series)             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Intelligence Layer                        │   │
│  │  ┌────────────────────┐      ┌────────────────────┐        │   │
│  │  │  Security Scanner  │      │ Pattern Analysis   │        │   │
│  │  │  Service (5001)    │      │ Service (5002)     │        │   │
│  │  │                    │      │                    │        │   │
│  │  │ • News-driven      │      │ • Candlesticks    │        │   │
│  │  │ • Catalyst score   │      │ • Context-aware   │        │   │
│  │  │ • Top 5 picks      │      │ • News-weighted   │        │   │
│  │  └──────────┬─────────┘      └─────────┬─────────┘        │   │
│  │             │                           │                   │   │
│  │             ▼                           ▼                   │   │
│  │    trading_candidates           pattern_results             │   │
│  │       (processed)                 (analyzed)                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Decision Layer                            │   │
│  │  ┌────────────────────┐      ┌────────────────────┐        │   │
│  │  │ Technical Analysis │      │  Risk Management   │        │   │
│  │  │ Service (5003)     │      │  (Embedded)        │        │   │
│  │  │                    │      │                    │        │   │
│  │  │ • Signal generate  │      │ • Position size   │        │   │
│  │  │ • Entry/Exit       │      │ • Stop losses     │        │   │
│  │  │ • Confidence       │      │ • Risk limits     │        │   │
│  │  └──────────┬─────────┘      └─────────┬─────────┘        │   │
│  │             │                           │                   │   │
│  │             ▼                           ▼                   │   │
│  │       trading_signals              risk_params              │   │
│  │         (actionable)               (constraints)            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Execution Layer                           │   │
│  │  ┌────────────────────┐      ┌────────────────────┐        │   │
│  │  │  Paper Trading     │      │  Live Trading      │        │   │
│  │  │  Service (5005)    │      │  (Future)          │        │   │
│  │  │                    │      │                    │        │   │
│  │  │ • Alpaca API       │      │ • Real money      │        │   │
│  │  │ • Order execute    │      │ • Compliance      │        │   │
│  │  │ • P&L tracking     │      │ • Audit trail     │        │   │
│  │  └──────────┬─────────┘      └─────────┬─────────┘        │   │
│  │             │                           │                   │   │
│  │             ▼                           ▼                   │   │
│  │       trade_records               live_trades               │   │
│  │        (simulated)                  (real)                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                 Orchestration & Support                      │   │
│  │                                                              │   │
│  │  ┌─────────────────┐  ┌──────────────┐  ┌───────────────┐  │   │
│  │  │  Coordination    │  │  Reporting   │  │ Web Dashboard │  │   │
│  │  │  Service (5000)  │  │  (5009)      │  │   (5010)      │  │   │
│  │  │                  │  │              │  │               │  │   │
│  │  │ • Orchestrate    │  │ • Analytics  │  │ • Monitoring  │  │   │
│  │  │ • Health checks  │  │ • P&L report │  │ • Control     │  │   │
│  │  │ • Scheduling     │  │ • Tracking   │  │ • Visualize   │  │   │
│  │  └─────────────────┘  └──────────────┘  └───────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Service Architecture Details

| Service | Port | Purpose | Dependencies | Key Features |
|---------|------|---------|--------------|--------------|
| News Collection | 5008 | Gather market intelligence | External APIs | Multi-source, alignment tracking |
| Security Scanner | 5001 | Select trading candidates | news_raw | Catalyst scoring, top 5 selection |
| Pattern Analysis | 5002 | Detect technical patterns | trading_candidates | Context-aware, catalyst-weighted |
| Technical Analysis | 5003 | Generate signals | patterns + indicators | Entry/exit points, confidence |
| Paper Trading | 5005 | Execute trades | trading_signals | Alpaca integration, P&L tracking |
| Coordination | 5000 | Orchestrate workflow | All services | Health monitoring, scheduling |
| Reporting | 5009 | Analytics & metrics | Database | Performance tracking |
| Web Dashboard | 5010 | User interface | All services | Real-time monitoring |

## Data Flow Architecture

### Primary Trading Flow

```
1. News Collection (Continuous)
   ├── NewsAPI
   ├── AlphaVantage  
   ├── RSS Feeds
   └── → news_raw table (untouched)

2. Security Scanning (Every 5-30 min)
   ├── Read news_raw (last 24h)
   ├── Score by catalyst + recency
   ├── Validate with price/volume
   └── → trading_candidates (top 5)

3. Pattern Analysis (Per candidate)
   ├── Read price data
   ├── Detect candlestick patterns
   ├── Weight by news context
   └── → pattern_results

4. Signal Generation
   ├── Combine patterns + indicators + catalyst
   ├── Calculate entry/exit/stops
   ├── Assign confidence score
   └── → trading_signals

5. Trade Execution
   ├── Send to broker API
   ├── Monitor positions
   ├── Track P&L
   └── → trade_records

6. Outcome Tracking
   ├── Update news accuracy
   ├── Track beneficiaries
   ├── Feed ML training data
   └── → source_metrics
```

### Feedback Loops

```
Trade Outcomes → News Accuracy
   - Which sources were accurate?
   - Which catalysts led to profits?
   - Update source reliability scores

Price Movement → Source Alignment
   - Who published before moves?
   - Detect coordinated narratives
   - Track beneficiary patterns
```

## Technology Stack

### Core Technologies
- **Language**: Python 3.10+
- **Web Framework**: Flask
- **Database**: SQLite (dev) → PostgreSQL (prod)
- **Cache**: Redis
- **Message Queue**: Redis Pub/Sub (future)
- **Container**: Docker
- **Orchestration**: Docker Compose → Kubernetes

### External Services
- **News APIs**: NewsAPI, AlphaVantage, Finnhub
- **Market Data**: yfinance, Alpha Vantage
- **Broker**: Alpaca Markets
- **Cloud**: DigitalOcean (Singapore region)

### Libraries
- **Data Processing**: pandas, numpy
- **Technical Analysis**: ta-lib, custom implementations
- **Machine Learning**: scikit-learn (future)
- **Web Server**: gunicorn
- **Database ORM**: SQLAlchemy (future)

## Security Architecture

### API Security
- Environment variables for all credentials
- API key rotation schedule
- Rate limiting per endpoint
- Request signing for internal services

### Data Security
- No PII stored
- Trading credentials encrypted
- SSL/TLS for all communications
- VPN for production access

### Audit Trail
- All trades logged with full context
- Decision chain recorded
- Source tracking anonymized
- Compliance-ready architecture

## Scalability Design

### Horizontal Scaling
- Stateless services
- Database connection pooling
- Cache layer for hot data
- Load balancer ready

### Performance Targets
- News collection: < 5 min full cycle
- Scanner: < 30 sec for top 5
- Pattern analysis: < 5 sec per symbol
- Signal generation: < 2 sec
- Trade execution: < 1 sec

### Bottleneck Mitigation
- Parallel news collection
- Caching of technical data
- Database query optimization
- Async processing where possible

## Deployment Architecture

### Development Environment
- GitHub Codespaces
- SQLite database
- Local services
- Mock trading

### Production Environment (DigitalOcean)
- App Platform containers
- Managed PostgreSQL
- Redis cache
- Load balancer
- Singapore region (low latency to Perth)

### CI/CD Pipeline
- GitHub Actions
- Automated testing
- Container builds
- Blue-green deployment

## Monitoring Architecture

### Service Health
- Health endpoints on all services
- Heartbeat monitoring
- Dependency checking
- Automatic alerting

### Business Metrics
- News collection rate
- Catalyst hit rate
- Pattern accuracy
- Trading performance
- Source reliability trends

### Technical Metrics
- Response times
- Error rates
- Database performance
- API usage

## Future Architecture Considerations

### Machine Learning Integration
- Pattern recognition models
- Source reliability prediction
- Optimal timing models
- Sentiment analysis enhancement

### Scale Considerations
- Message queue for async processing
- Microservice mesh
- Event sourcing
- CQRS pattern

### Global Expansion
- Multi-region deployment
- Follow-the-sun trading
- Currency support
- Regulatory compliance

## Architecture Principles

1. **Data Integrity First**: Raw data never modified
2. **Service Independence**: Each service owns its domain
3. **Fail Gracefully**: Degraded service better than downtime
4. **Observable**: Every decision traceable
5. **Secure by Design**: Security not an afterthought
6. **Performance Matters**: Speed equals opportunity
7. **Social Impact**: Architecture serves the mission

This architecture provides a robust foundation for news-driven trading while maintaining flexibility for future enhancements and the ultimate goal of funding social good.