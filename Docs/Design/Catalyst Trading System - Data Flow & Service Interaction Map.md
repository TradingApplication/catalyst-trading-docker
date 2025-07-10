# Catalyst Trading System - Data Flow & Service Interaction Map

**Version**: 1.0.0  
**Last Updated**: 2025-07-10  
**Purpose**: Complete map of data flow through all services

## Overview

This document traces every piece of data from creation to consumption across all services.

## Data Flow Stages

### Stage 1: News Collection & Storage

#### News Service (Port 5008)
**Inputs:**
- External APIs: NewsAPI, AlphaVantage, Finnhub
- Scheduler triggers from Coordination Service
- Manual triggers via `/collect_news` endpoint

**Processing:**
- Fetches news from multiple sources
- Deduplicates articles
- Calculates source tier (1-5)
- Extracts sentiment and keywords
- Detects mentioned tickers

**Outputs:**
| Output | Type | Destination | Used By |
|--------|------|-------------|---------|
| news_raw records | Database | `news_raw` table | Scanner, Reporting |
| news_id | Database | Primary key | All services |
| source_metrics | Database | `source_metrics` table | News Service (feedback) |
| Collection stats | Redis | `news:stats:{date}` | Dashboard |
| `/search_news` | REST API | JSON response | External clients |

### Stage 2: Security Scanning & Selection

#### Scanner Service (Port 5001)
**Inputs:**
- `news_raw` table (last 24 hours)
- Market data from Yahoo Finance API
- Configuration from `configuration` table
- Trigger from Coordination Service

**Processing:**
- Groups news by symbol
- Calculates catalyst scores
- Fetches price/volume data
- Ranks securities
- Selects top 5 candidates

**Outputs:**
| Output | Type | Destination | Used By |
|--------|------|-------------|---------|
| trading_candidates | Database | `trading_candidates` table | Pattern, Technical |
| Scan results | Redis | `scan:latest:{scan_type}` | Dashboard |
| Selected symbols | Memory | Return to Coordination | Pattern Service |
| `/scan` response | REST API | JSON with top picks | Dashboard, Manual |

### Stage 3: Pattern Analysis

#### Pattern Service (Port 5002)
**Inputs:**
- Symbol list from Scanner Service
- Price data from yfinance
- Catalyst info from `trading_candidates` table
- Volume data for confirmation

**Processing:**
- Fetches OHLCV data (multiple timeframes)
- Detects candlestick patterns
- Checks catalyst alignment
- Calculates pattern strength
- Identifies support/resistance

**Outputs:**
| Output | Type | Destination | Used By |
|--------|------|-------------|---------|
| pattern_analysis | Database | `pattern_analysis` table | Technical Service |
| Pattern alerts | Redis | `patterns:{symbol}:latest` | Dashboard |
| Detection results | Memory | Return to Coordination | Technical Service |
| `/analyze_pattern` | REST API | Pattern details JSON | Manual analysis |

### Stage 4: Technical Analysis & Signal Generation

#### Technical Service (Port 5003)
**Inputs:**
- Symbols with patterns from Pattern Service
- OHLCV data from yfinance
- Pattern data from `pattern_analysis` table
- Catalyst data from `trading_candidates` table

**Processing:**
- Calculates 20+ indicators (RSI, MACD, etc.)
- Combines pattern + technical + catalyst scores
- Generates entry/exit/stop levels
- Calculates risk/reward ratios
- Creates trading signals

**Outputs:**
| Output | Type | Destination | Used By |
|--------|------|-------------|---------|
| technical_indicators | Database | `technical_indicators` table | Reporting |
| trading_signals | Database | `trading_signals` table | Trading Service |
| Signal alerts | Redis | `signals:pending` | Dashboard |
| `/generate_signal` | REST API | Signal JSON | Manual override |

### Stage 5: Trade Execution

#### Trading Service (Port 5005)
**Inputs:**
- Pending signals from `trading_signals` table
- Position limits from `configuration` table
- Account data from Alpaca API
- Risk parameters from configuration

**Processing:**
- Validates signals against risk rules
- Checks existing positions
- Calculates position sizes
- Places orders via Alpaca
- Monitors order fills

**Outputs:**
| Output | Type | Destination | Used By |
|--------|------|-------------|---------|
| trade_records | Database | `trade_records` table | Reporting |
| Order updates | Redis | `orders:{order_id}` | Dashboard |
| Position updates | Database | Update `trade_records` | Risk monitoring |
| `/execute_trade` | REST API | Execution confirmation | Manual trading |

### Stage 6: Coordination & Orchestration

#### Coordination Service (Port 5000)
**Inputs:**
- Schedule configuration
- Service health checks
- Market hours status

**Processing:**
- Manages trading cycles
- Orchestrates workflow
- Monitors service health
- Handles error recovery

**Outputs:**
| Output | Type | Destination | Used By |
|--------|------|-------------|---------|
| trading_cycles | Database | `trading_cycles` table | All services |
| workflow_log | Database | `workflow_log` table | Reporting |
| service_health | Database | `service_health` table | Dashboard |
| Cycle status | Redis | `cycle:current` | All services |

### Stage 7: Reporting & Analytics

#### Reporting Service (Port 5009)
**Inputs:**
- All database tables
- Redis cache data
- Completed trades from `trade_records`

**Processing:**
- Calculates performance metrics
- Generates reports
- Updates outcome tracking
- Creates visualizations

**Outputs:**
| Output | Type | Destination | Used By |
|--------|------|-------------|---------|
| performance_metrics | Database | `performance_metrics` table | Dashboard |
| outcome_tracking | Database | `outcome_tracking` table | News Service |
| Report files | Filesystem | `/app/reports/*.json` | Dashboard |
| `/generate_report` | REST API | Report data | External clients |

## Complete Data Flow Diagram

```
External APIs → News Service → news_raw table → Scanner Service
                                                       ↓
Dashboard ← Redis Cache ← All Services         trading_candidates
    ↑                                                 ↓
    |                                          Pattern Service
    |                                                 ↓
Reporting ← trade_records ← Trading Service ← Technical Service
              ↑                    ↑                  ↓
              |                    |           trading_signals
              |                    |
              └────── Alpaca API ──┘
```

## Redis Key Patterns

| Pattern | Purpose | TTL | Example |
|---------|---------|-----|---------|
| `news:stats:{date}` | Daily collection stats | 7 days | `news:stats:20250710` |
| `scan:latest:{type}` | Latest scan results | 1 hour | `scan:latest:premarket` |
| `patterns:{symbol}:latest` | Recent patterns | 2 hours | `patterns:AAPL:latest` |
| `signals:pending` | Unexecuted signals | 1 hour | `signals:pending` |
| `orders:{order_id}` | Order status | 24 hours | `orders:abc123` |
| `cycle:current` | Active cycle info | No expiry | `cycle:current` |
| `cache:{service}:{key}` | Generic cache | Varies | `cache:news:sources` |

## Database Table Dependencies

```
news_raw
    → trading_candidates (via symbol + catalyst score)
        → pattern_analysis (via symbol)
            → technical_indicators (via symbol + timeframe)
                → trading_signals (combines all above)
                    → trade_records (execution result)
                        → performance_metrics (aggregated)
                        → outcome_tracking (feedback loop)

configuration → All services (risk params, limits)
trading_cycles → workflow_log (execution tracking)
service_health → Dashboard (monitoring)
```

## Critical Data Paths

### 1. News to Trade Path (Happy Path)
```
1. News collected → news_raw
2. Scanner finds catalyst → trading_candidates  
3. Pattern detected → pattern_analysis
4. Signal generated → trading_signals
5. Trade executed → trade_records
6. Outcome tracked → outcome_tracking
```

### 2. Feedback Loop Path
```
1. Trade completed → trade_records
2. Performance calculated → performance_metrics
3. Source accuracy updated → source_metrics
4. Pattern success tracked → pattern_analysis (success field)
5. News service adjusts weights → Better future selections
```

### 3. Risk Management Path
```
1. Configuration loaded → All services
2. Position checked → trade_records (open positions)
3. Risk validated → Trading service logic
4. Order sized → Based on configuration limits
5. Stop loss set → From signal parameters
```

## Service Communication Matrix

| From Service | To Service | Method | Data |
|--------------|------------|---------|------|
| Coordination | News | HTTP POST | Trigger collection |
| Coordination | Scanner | HTTP POST | Start scan |
| Coordination | Pattern | HTTP POST | Analyze symbols |
| Coordination | Technical | HTTP POST | Generate signals |
| Coordination | Trading | HTTP POST | Execute trades |
| All Services | Database | SQL | CRUD operations |
| All Services | Redis | Redis Protocol | Cache/Queue |
| All Services | Coordination | HTTP GET | Health check |
| Trading | Alpaca | REST API | Orders |
| Dashboard | All Services | HTTP GET | Status/Data |

## Error Handling & Recovery

### Failed Data Flows
1. **News API Failure**: Cached data used, retry with exponential backoff
2. **Database Down**: Redis queue holds data, retry on reconnection  
3. **Service Down**: Coordination skips step, logs to workflow_log
4. **Trading Failure**: Signal marked failed, notification sent

### Data Consistency
- Database transactions for critical operations
- Redis used for speed, not source of truth
- All financial data in PostgreSQL with ACID guarantees
- Audit trail in workflow_log for every step

## Performance Considerations

### High-Frequency Data
- Market data: Cached in Redis (5-minute TTL)
- News data: Database indexed by symbol, timestamp
- Signals: In-memory processing, database for persistence

### Bottlenecks
1. News API rate limits → Managed by collection scheduling
2. Pattern analysis compute → Limited to top 5 symbols
3. Database writes → Pooled connections, batch where possible
4. Trading API limits → Queue with rate limiting

## Monitoring Points

| Metric | Location | Alert Threshold |
|--------|----------|----------------|
| News collection rate | workflow_log | < 100 articles/hour |
| Scan duration | workflow_log | > 30 seconds |
| Pattern detection rate | pattern_analysis | < 20% hit rate |
| Signal confidence | trading_signals | Average < 60% |
| Trade success rate | trade_records | < 40% profitable |
| Service uptime | service_health | < 95% per hour |