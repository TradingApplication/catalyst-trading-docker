# Catalyst Trading System - Functional Specification v2.1.0

**Version**: 2.1.0  
**Date**: July 9, 2025  
**Platform**: DigitalOcean  
**Status**: Implementation Ready  
**Previous Version**: 2.0.0 (June 28, 2025)

## Revision History

### v2.1.0 (July 9, 2025)
- **Market Data Collection**: Scanner now records ALL 50 securities per scan
- **Cumulative Dataset**: 50 potentially different securities each scan
- **Growth Projection**: Could track 500-1000+ unique securities annually  
- **Data Management**: Added automated aggregation service specification
- **Service Alignment**: Updated to match v2.1.1 architecture
- **ML Foundation**: Comprehensive data collection for future pattern discovery

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Business Logic](#2-core-business-logic)
3. [Service Specifications](#3-service-specifications)
4. [Data Flow Specifications](#4-data-flow-specifications)
5. [Data Management Specifications](#5-data-management-specifications) **[NEW]**
6. [Integration Points](#6-integration-points)
7. [Performance Requirements](#7-performance-requirements)
8. [Security Requirements](#8-security-requirements)
9. [Error Handling](#9-error-handling)

---

## 1. System Overview

### 1.1 Purpose
The Catalyst Trading System is a news-driven algorithmic trading platform that identifies and executes day trading opportunities based on market catalysts. The system **comprehensively records market data for all evaluated securities while focusing trading on the highest-conviction opportunities**.

### 1.2 Key Differentiators (Updated)
- **News-First Selection**: Only trade securities with news catalysts
- **Comprehensive Data Collection**: Record ALL 50 scanned securities per cycle **[NEW]**
- **Cumulative Learning**: Build dataset of hundreds/thousands of securities **[NEW]**
- **Focused Execution**: Only TOP 5 proceed to trading
- **Smart Storage**: Automatic data aggregation prevents exponential growth **[NEW]**
- **Source Intelligence**: Track source reliability and agenda patterns
- **Clean Architecture**: Raw data preserved, trading data optimized
- **ML-Ready**: Accumulating broad dataset for pattern discovery
- **Social Mission**: Profits fund homeless shelter operations

### 1.3 Operating Modes
- **Pre-Market Aggressive** (4:00 AM - 9:30 AM EST): 5-minute cycles
- **Market Hours Normal** (9:30 AM - 4:00 PM EST): 30-minute cycles  
- **After-Hours Light** (4:00 PM - 8:00 PM EST): 60-minute cycles
- **Weekend Maintenance**: Data cleanup and analysis
- **Nightly Aggregation** (2:00 AM daily): Data compression **[NEW]**

### 1.4 Data Growth Projections **[NEW SECTION]**
```
Unique Securities Growth:
- Day 1: ~50 unique symbols
- Week 1: ~150-200 unique symbols (with overlap)
- Month 1: ~300-500 unique symbols
- Year 1: ~1000-2000 unique symbols

Data Volume Growth (without compression):
- Month 1: ~120,000 records
- Month 6: ~700,000 records  
- Year 1: ~1,400,000 records

Data Volume (with compression):
- Month 1: ~120,000 records
- Month 6: ~180,000 records
- Year 1: ~200,000 records (86% reduction)
```

---

## 2. Core Business Logic

### 2.1 News-Driven Selection Algorithm

#### 2.1.1 Catalyst Scoring Formula
```
Catalyst Score = (Source Tier Weight × Recency Weight × Keyword Weight × Market State Multiplier)

Where:
- Source Tier Weight: Tier 1 = 1.0, Tier 2 = 0.8, Tier 3 = 0.6, Tier 4 = 0.4, Tier 5 = 0.2
- Recency Weight: exp(-hours_old / 4) 
- Keyword Weight: Earnings = 1.2, FDA = 1.5, M&A = 1.3, Default = 1.0
- Market State Multiplier: Pre-market = 2.0, Regular = 1.0, After-hours = 0.8
```

#### 2.1.2 Multi-Stage Filtering with Comprehensive Recording **[UPDATED]**
1. **Stage 1**: Collect all news (1000+ articles)
2. **Stage 2**: Identify 50 most active securities with catalysts
3. **Stage 3**: **RECORD market data for ALL 50** → market_data table **[NEW]**
4. **Stage 4**: Technical validation → 20 candidates
5. **Stage 5**: Final selection → TOP 5
6. **Stage 6**: Only TOP 5 → trading_candidates table

#### 2.1.3 Continuous Learning Dataset **[NEW]**
- Each scan may introduce new securities
- Previous securities remain in database
- Creates expanding universe for ML training
- No security is "forgotten" once scanned

---

## 3. Service Specifications

### 3.1 News Collection Service (Port 5008)
**Purpose**: Collect raw news from multiple sources  
**Changes**: None - continues as specified in v2.0.0

### 3.2 Security Scanner Service (Port 5001) **[UPDATED]**

#### Purpose
Select securities for comprehensive data collection and identify top trading candidates

#### Endpoints
- `GET /scan` - Regular market scan
- `GET /scan_premarket` - Aggressive pre-market scan  
- `POST /scan_symbols` - Scan specific symbols
- `GET /get_scan_results` - Retrieve latest results

#### Key Functions (Updated)
```python
def execute_dynamic_scan(self, mode='normal'):
    """Enhanced scan that records all evaluated securities"""
    
    # 1. Get universe of active stocks
    universe = self.get_active_universe()  # 100-200 stocks
    
    # 2. Filter by news catalysts
    candidates_with_news = self.filter_by_catalysts(universe)  # ~50-80
    
    # 3. Score all candidates
    scored_candidates = self.calculate_catalyst_scores(candidates_with_news)
    
    # 4. Take top 50 by score
    top_50 = scored_candidates[:50]
    
    # 5. SAVE ALL 50 TO MARKET DATA [NEW STEP]
    self.save_market_data_bulk(top_50)
    
    # 6. Technical validation  
    validated_20 = self.validate_with_technicals(top_50)[:20]
    
    # 7. Final selection
    top_5 = self.select_final_candidates(validated_20)[:5]
    
    # 8. Save only TOP 5 to trading_candidates
    self.save_trading_candidates(top_5)
    
    return {
        'scan_id': scan_id,
        'total_evaluated': len(top_50),
        'data_recorded': len(top_50),  # ALL 50
        'trading_candidates': len(top_5)  # Only 5
    }

def save_market_data_bulk(self, securities: List[Dict]):
    """Save market data for all scanned securities"""
    market_records = []
    
    for security in securities:
        ticker = yf.Ticker(security['symbol'])
        data = ticker.history(period='1d', interval='5m')
        
        if not data.empty:
            latest = data.iloc[-1]
            market_records.append({
                'symbol': security['symbol'],
                'timestamp': latest.name,
                'timeframe': '5min',
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'close': float(latest['Close']),
                'volume': int(latest['Volume']),
                'scan_id': self.current_scan_id,
                'is_trading_candidate': security in self.top_5,
                'has_news': True,
                'catalyst_score': security.get('catalyst_score', 0)
            })
    
    # Bulk insert for efficiency
    insert_market_data_bulk(market_records)
```

### 3.3 Pattern Analysis Service (Port 5002) **[CLARIFIED]**

#### Purpose
Detect technical patterns with catalyst context awareness

#### Scope Change
- **Only analyzes TOP 5 securities from trading_candidates**
- Does NOT process the other 45 securities
- Focused computational resources on highest-conviction trades

#### Endpoints
- `POST /analyze_pattern` - Analyze single symbol (must be in TOP 5)
- `POST /batch_analyze` - Analyze multiple symbols (filters to TOP 5)
- `GET /pattern_statistics` - Historical accuracy

### 3.4 Technical Analysis Service (Port 5003) **[CLARIFIED]**

#### Purpose  
Generate trading signals combining catalysts, patterns, and indicators

#### Scope Change
- **Only processes TOP 5 securities**
- Receives list from trading_candidates only
- No access to the broader 50 securities dataset

### 3.5 Paper Trading Service (Port 5005)
**Purpose**: Execute trades via Alpaca API  
**Changes**: None - continues trading only TOP 5

### 3.6 Coordination Service (Port 5009) **[UPDATED]**

#### Workflow Steps (Updated)
1. Trigger news collection
2. Wait for completion
3. Run security scanner
   - Scanner identifies 50 securities
   - **Saves ALL 50 to market_data** **[NEW]**
   - Returns TOP 5 as candidates
4. Analyze patterns for TOP 5 only
5. Generate signals for TOP 5 only
6. Execute trades on signals
7. Update outcomes

### 3.7 Market Data Aggregation Service **[NEW SERVICE]**

#### Purpose
Compress historical market data to manage exponential growth from tracking hundreds of securities

#### Type
Scheduled task (cron job), not a persistent service

#### Schedule
Daily at 2:00 AM local time

#### Key Functions
```python
class MarketDataAggregator:
    """Compress old market data to manage storage"""
    
    def aggregate_5min_to_15min(self):
        """Aggregate 7-day old 5-minute data to 15-minute"""
        query = """
            INSERT INTO market_data (
                symbol, timestamp, timeframe, 
                open, high, low, close, volume,
                aggregated_from
            )
            SELECT 
                symbol,
                date_trunc('hour', timestamp) + 
                    interval '15 minutes' * floor(extract(minute from timestamp)::int / 15),
                '15min',
                (array_agg(open ORDER BY timestamp))[1],
                MAX(high),
                MIN(low),
                (array_agg(close ORDER BY timestamp DESC))[1],
                SUM(volume),
                '5min'
            FROM market_data
            WHERE timeframe = '5min'
            AND created_at < NOW() - INTERVAL '7 days'
            AND created_at >= NOW() - INTERVAL '8 days'
            GROUP BY symbol, 
                     date_trunc('hour', timestamp) + 
                     interval '15 minutes' * floor(extract(minute from timestamp)::int / 15)
            ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
        """
        
        # Execute aggregation
        rows_created = execute_aggregation(query)
        
        # Delete source data
        delete_old_5min_data()
        
        # Log operation
        log_aggregation_operation('5min', '15min', rows_created)
    
    def run_all_aggregations(self):
        """Run all aggregation levels"""
        self.aggregate_5min_to_15min()      # 7 days old
        self.aggregate_15min_to_1hour()     # 30 days old  
        self.aggregate_1hour_to_daily()     # 90 days old
        
        # Report on data volume
        self.report_data_statistics()
```

---

## 4. Data Flow Specifications

### 4.1 Primary Trading Flow with Comprehensive Data Collection **[UPDATED]**

```
1. News Collection (Continuous)
   Input: API calls to news sources
   Output: news_raw records with metadata
   Frequency: Every 5 minutes (pre-market) to 60 minutes (after-hours)

2. Security Scanning (Scheduled)
   Input: news_raw from last 24 hours
   Process: Score → Filter → Validate
   Output: 
     - market_data: ALL 50 securities (different each scan)
     - trading_candidates: TOP 5 securities only
   Unique Securities: Accumulates over time (50 new/day average)

3. Pattern Analysis (Per TOP 5 Candidate Only)
   Input: Symbol + catalyst context (from trading_candidates)
   Process: Detect patterns + weight by catalyst
   Output: pattern_results with confidence
   Scope: LIMITED to 5 securities per cycle

4. Signal Generation (Per Pattern - TOP 5 Only)
   Input: Patterns + indicators + catalyst
   Process: Calculate confidence + entry/exit
   Output: trading_signals
   Scope: LIMITED to 5 securities per cycle

5. Trade Execution (Per Signal)
   Input: Signals above threshold
   Process: Risk checks + order placement
   Output: trade_records + positions

6. Data Aggregation (Nightly at 2 AM)
   Input: market_data table
   Process: Compress old data by timeframe
   Output: Aggregated records + cleanup
   Purpose: Manage growth from 100s of unique securities
```

### 4.2 Data Growth Management **[NEW SECTION]**

```
Without Aggregation (Exponential Growth):
- Month 1: 50 securities/day × 30 days × 78 candles = 117,000 records
- Month 6: 300 unique securities × 180 days × 78 candles = 4.2M records
- Year 1: 1000 unique securities × 365 days × 78 candles = 28.5M records

With Aggregation (Linear Growth):
- Recent (5min): 7 days × 50 securities × 78 candles = 27,300 records
- Medium (15min): 23 days × 200 securities × 26 candles = 119,600 records
- Older (1hour): 60 days × 500 securities × 6.5 candles = 195,000 records
- Archive (daily): 275 days × 1000 securities × 1 candle = 275,000 records
- Total: ~617,000 records (98% reduction)
```

---

## 5. Data Management Specifications **[NEW SECTION]**

### 5.1 Retention Policy

| Age Range | Original | Aggregated To | Compression | Purpose |
|-----------|----------|---------------|-------------|---------|
| 0-7 days | 5 min | Keep as-is | 1:1 | Full detail for analysis |
| 8-30 days | 5 min | 15 min | 3:1 | Reduced detail |
| 31-90 days | 15 min | 1 hour | 12:1 | Trend analysis |
| 91+ days | 1 hour | Daily | 78:1 | Long-term patterns |

### 5.2 Unique Security Tracking

```python
def get_unique_securities_count():
    """Track growth of unique securities over time"""
    query = """
    SELECT 
        DATE(created_at) as date,
        COUNT(DISTINCT symbol) as new_symbols,
        SUM(COUNT(DISTINCT symbol)) OVER (ORDER BY DATE(created_at)) as cumulative_symbols
    FROM market_data
    GROUP BY DATE(created_at)
    ORDER BY date DESC
    LIMIT 30
    """
    return execute_query(query)
```

### 5.3 Storage Projections

```
Year 1 Projections:
- Unique Securities: 1,000-2,000
- Total Records: ~600,000 (with aggregation)
- Storage Size: ~1-2 GB including indexes
- Query Performance: <100ms for recent data
- Aggregation Runtime: <5 minutes nightly
```

---

## 6. Integration Points

### 6.1 External APIs
- **NewsAPI**: General market news
- **AlphaVantage**: Financial news + sentiment
- **RSS Feeds**: Real-time updates
- **yfinance**: Price and volume data (expanded usage)
- **Alpaca Markets**: Trade execution

### 6.2 Internal Communication
- REST APIs between services
- Bulk data operations for efficiency
- JSON message format
- HTTP status codes for errors
- Timeout: 30 seconds default

---

## 7. Performance Requirements

### 7.1 Response Times (Updated)
- News collection: < 5 minutes full cycle
- Security scan: < 30 seconds for 50 securities
- Market data save: < 10 seconds for 50 records (bulk insert)
- Pattern analysis: < 5 seconds per symbol (5 symbols)
- Signal generation: < 2 seconds
- Trade execution: < 1 second
- Data aggregation: < 5 minutes (nightly)

### 7.2 Throughput (Updated)
- Handle 1000+ news articles per hour
- Process 100 securities in screening
- **Record 50 securities per scan cycle** **[NEW]**
- Analyze 5 candidates in detail
- Execute up to 50 trades per day
- **Aggregate up to 1M records nightly** **[NEW]**

### 7.3 Data Volume Handling **[NEW]**
- Ingest 3,900 new records daily
- Maintain sub-second queries on 600K+ records
- Support 1000+ unique securities
- Compress data without information loss

---

## 8. Security Requirements

### 8.1 API Security
- Environment variables for credentials
- No credentials in code
- API rate limiting
- Request validation
- Bulk operation authentication

### 8.2 Trading Security
- Alpaca paper trading only (initially)
- Position limits enforced
- Stop losses mandatory
- Manual override capability
- TOP 5 restriction enforced

### 8.3 Data Security
- No PII collected
- Secure credential storage
- Audit trail for all trades
- Encrypted API communications
- Secure bulk data transfers

---

## 9. Error Handling

### 9.1 Service Level
- Automatic retry with backoff
- Circuit breaker pattern
- Graceful degradation
- Error logging and alerting
- Bulk operation rollback

### 9.2 System Level
- Service health monitoring
- Automatic service restart
- Database lock handling
- Resource exhaustion prevention
- Aggregation failure recovery

### 9.3 Trading Level
- Order rejection handling
- Partial fill management
- Connection loss recovery
- Market halt detection
- TOP 5 validation

### 9.4 Data Management Level **[NEW]**
- Aggregation failure alerts
- Data integrity checks
- Compression verification
- Storage threshold warnings
- Backup before aggregation

---

## Implementation Priority

### Phase 1: Core Enhancement (Week 1)
1. Update scanner service for bulk market data recording
2. Deploy market_data table
3. Test with production data volumes
4. Verify TOP 5 trading still works

### Phase 2: Data Management (Week 2)
1. Deploy aggregation service
2. Set up cron job
3. Monitor compression effectiveness
4. Tune aggregation parameters

### Phase 3: Optimization (Week 3)
1. Optimize bulk insert performance
2. Add monitoring dashboards
3. Implement data quality checks
4. Document unique security growth

This specification provides the complete functional blueprint for implementing comprehensive data collection while maintaining focused trading execution in the Catalyst Trading System v2.1.0.