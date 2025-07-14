# Catalyst Trading System - Database Schema v2.1.0

**Version**: 2.1.0  
**Date**: July 8, 2025  
**Database**: PostgreSQL (Production)  
**Previous Version**: 2.0.0 (June 28, 2025)

## Revision History

### v2.1.0 (July 8, 2025)
- **Consistency Updates**: Aligned all table and field names with implementation
- **PostgreSQL Specific**: Removed SQLite syntax, PostgreSQL only
- **Type Corrections**: JSONB instead of JSON for PostgreSQL
- **Added Missing Tables**: Added news_collection_stats, pattern_detections
- **Service Alignment**: Ensured all tables match service expectations
- **Standardized Naming**: All timestamp fields use _timestamp suffix

## Table of Contents

1. [Schema Overview](#1-schema-overview)
2. [News & Intelligence Tables](#2-news--intelligence-tables)
3. [Trading Operations Tables](#3-trading-operations-tables)
4. [Analysis & Pattern Tables](#4-analysis--pattern-tables)
5. [System & Coordination Tables](#5-system--coordination-tables)
6. [Implementation Tables](#6-implementation-tables)
7. [Indexes & Performance](#7-indexes--performance)
8. [Data Relationships](#8-data-relationships)

---

## 1. Schema Overview

### 1.1 Database Design Principles
- **Raw Data Preservation**: news_raw never modified after insert
- **Clean Separation**: Raw data vs processed trading data
- **Audit Trail**: Complete history of all decisions
- **ML Readiness**: Outcome tracking built-in
- **Performance**: Strategic indexes for common queries
- **Service Alignment**: Tables match service expectations exactly

### 1.2 Table Categories
1. **News & Intelligence**: Raw news, source metrics, narratives
2. **Trading Operations**: Candidates, signals, trades, positions
3. **Analysis & Pattern**: Technical patterns, indicators
4. **System & Coordination**: Service health, workflow, configuration
5. **Implementation Specific**: Service-specific tracking tables

---

## 2. News & Intelligence Tables

### 2.1 news_raw
**Purpose**: Store all collected news without modification  
**Primary Key**: news_id (hash)

```sql
CREATE TABLE news_raw (
    id BIGSERIAL,
    news_id VARCHAR(100) PRIMARY KEY,      -- Hash of headline+source+timestamp
    
    -- Core news data
    symbol VARCHAR(10),                     -- Primary symbol (can be NULL)
    headline TEXT NOT NULL,
    source VARCHAR(200) NOT NULL,
    source_url TEXT,
    published_timestamp TIMESTAMPTZ NOT NULL,
    collected_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    content_snippet TEXT,                   -- First 500 chars
    full_url TEXT,
    
    -- Rich metadata
    metadata JSONB,                         -- All extra fields from APIs
    is_pre_market BOOLEAN DEFAULT FALSE,
    market_state VARCHAR(20),               -- pre-market, regular, after-hours, weekend
    headline_keywords JSONB,                -- ["earnings", "fda", "merger", etc]
    mentioned_tickers JSONB,                -- Other tickers in article
    article_length INTEGER,
    is_breaking_news BOOLEAN DEFAULT FALSE,
    update_count INTEGER DEFAULT 0,
    first_seen_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Source alignment tracking
    source_tier INTEGER DEFAULT 5,          -- 1=Bloomberg/Reuters, 5=Unknown
    confirmation_status VARCHAR(20) DEFAULT 'unconfirmed',
    confirmed_by VARCHAR(200),              -- Which tier 1-2 source confirmed
    confirmation_timestamp TIMESTAMPTZ,
    confirmation_delay_minutes INTEGER,
    was_accurate BOOLEAN,                   -- Did prediction come true?
    
    -- Clustering
    narrative_cluster_id VARCHAR(50)
);
```

### 2.2 source_metrics
**Purpose**: Track source reliability over time  
**Primary Key**: source_name

```sql
CREATE TABLE source_metrics (
    source_name VARCHAR(200) PRIMARY KEY,
    tier INTEGER DEFAULT 5,
    total_articles INTEGER DEFAULT 0,
    confirmed_articles INTEGER DEFAULT 0,
    accurate_predictions INTEGER DEFAULT 0,
    false_predictions INTEGER DEFAULT 0,
    
    -- Performance metrics
    avg_confirmation_delay_minutes DECIMAL(10,2),
    confirmation_rate DECIMAL(5,2),         -- % confirmed by tier 1-2
    accuracy_rate DECIMAL(5,2),             -- % predictions correct
    exclusive_scoops INTEGER DEFAULT 0,     -- Stories broken first
    
    -- Temporal patterns
    pre_market_articles INTEGER DEFAULT 0,
    avg_publish_time TIME,
    most_active_day VARCHAR(10),
    
    last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

### 2.3 narrative_clusters
**Purpose**: Group related news stories  
**Primary Key**: cluster_id

```sql
CREATE TABLE narrative_clusters (
    cluster_id VARCHAR(50) PRIMARY KEY,
    created_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Cluster properties
    primary_symbol VARCHAR(10),
    cluster_type VARCHAR(50),               -- earnings, merger, fda, etc
    headline_summary TEXT,
    article_count INTEGER DEFAULT 1,
    
    -- Evolution tracking
    first_source VARCHAR(200),
    first_source_tier INTEGER,
    peak_velocity_timestamp TIMESTAMPTZ,    -- When most articles published
    tier1_confirmation_timestamp TIMESTAMPTZ,
    
    -- Outcome
    market_impact_observed BOOLEAN,
    price_movement_pct DECIMAL(5,2),
    volume_spike_ratio DECIMAL(5,2),
    
    last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

### 2.4 news_collection_stats
**Purpose**: Track news collection performance  
**Primary Key**: id

```sql
CREATE TABLE news_collection_stats (
    id BIGSERIAL PRIMARY KEY,
    collection_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(100) NOT NULL,
    articles_collected INTEGER DEFAULT 0,
    articles_new INTEGER DEFAULT 0,
    articles_duplicate INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    collection_duration_seconds DECIMAL(10,3),
    metadata JSONB
);
```

---

## 3. Trading Operations Tables

### 3.1 trading_candidates
**Purpose**: Top securities selected for potential trading  
**Primary Key**: id

```sql
CREATE TABLE trading_candidates (
    id BIGSERIAL PRIMARY KEY,
    scan_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    selection_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Selection criteria
    catalyst_score DECIMAL(5,2) NOT NULL,
    confidence_score DECIMAL(5,2),          -- Added for service compatibility
    news_count INTEGER DEFAULT 0,
    primary_catalyst VARCHAR(50),           -- earnings, fda, merger, etc
    catalyst_keywords JSONB,
    
    -- Technical validation
    price DECIMAL(10,2),
    volume BIGINT,
    relative_volume DECIMAL(5,2),
    price_change_pct DECIMAL(5,2),
    
    -- Pre-market data (if applicable)
    pre_market_volume BIGINT,
    pre_market_change DECIMAL(5,2),
    has_pre_market_news BOOLEAN DEFAULT FALSE,
    
    -- Final scoring
    technical_score DECIMAL(5,2),
    combined_score DECIMAL(5,2),
    selection_rank INTEGER,                 -- 1-5 for top picks
    
    -- Status
    analyzed BOOLEAN DEFAULT FALSE,
    traded BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP  -- Added for compatibility
);
```

### 3.2 trading_signals
**Purpose**: Actionable trading signals generated  
**Primary Key**: signal_id

```sql
CREATE TABLE trading_signals (
    signal_id VARCHAR(50) PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    generated_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Signal details
    signal_type VARCHAR(10) NOT NULL,       -- BUY, SELL, HOLD
    confidence DECIMAL(5,2),                -- 0-100
    
    -- Component scores
    catalyst_score DECIMAL(5,2),
    pattern_score DECIMAL(5,2),
    technical_score DECIMAL(5,2),
    volume_score DECIMAL(5,2),
    
    -- Entry/Exit parameters
    recommended_entry DECIMAL(10,2),
    stop_loss DECIMAL(10,2),
    target_1 DECIMAL(10,2),
    target_2 DECIMAL(10,2),
    
    -- Context
    catalyst_type VARCHAR(50),
    detected_patterns JSONB,
    key_factors JSONB,                      -- Why this signal
    
    -- Risk parameters
    position_size_pct DECIMAL(5,2),
    risk_reward_ratio DECIMAL(5,2),
    
    -- Execution status
    executed BOOLEAN DEFAULT FALSE,
    execution_timestamp TIMESTAMPTZ,
    actual_entry DECIMAL(10,2)
);
```

### 3.3 trade_records
**Purpose**: Actual executed trades and outcomes  
**Primary Key**: trade_id

```sql
CREATE TABLE trade_records (
    trade_id VARCHAR(50) PRIMARY KEY,
    id SERIAL UNIQUE,                       -- For compatibility
    signal_id VARCHAR(50),
    symbol VARCHAR(10) NOT NULL,
    
    -- Execution details
    order_type VARCHAR(20),                 -- market, limit
    side VARCHAR(10) NOT NULL,              -- buy, sell
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(10,2),
    entry_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Exit details
    exit_price DECIMAL(10,2),
    exit_timestamp TIMESTAMPTZ,
    exit_reason VARCHAR(50),                -- stop_loss, take_profit, signal, manual
    
    -- Catalyst tracking
    entry_catalyst VARCHAR(50),
    entry_news_id VARCHAR(100),             -- Links to news_raw
    catalyst_score_at_entry DECIMAL(5,2),
    
    -- Performance
    pnl_amount DECIMAL(10,2),
    pnl_percentage DECIMAL(5,2),
    commission DECIMAL(10,2),
    holding_period_minutes INTEGER,
    
    -- Risk metrics
    max_drawdown DECIMAL(5,2),
    max_profit DECIMAL(5,2),
    
    -- Status
    status VARCHAR(20) DEFAULT 'open',      -- open, closed, cancelled
    
    -- Outcome tracking for ML
    outcome_category VARCHAR(20),           -- big_win, win, loss, big_loss
    catalyst_confirmed BOOLEAN,             -- Did catalyst play out?
    pattern_completed BOOLEAN,              -- Did pattern complete?
    
    -- Service compatibility fields
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (entry_news_id) REFERENCES news_raw(news_id),
    FOREIGN KEY (signal_id) REFERENCES trading_signals(signal_id)
);
```

### 3.4 positions (Real-time view)
**Purpose**: Current open positions  
**Implementation**: View over trade_records

```sql
CREATE VIEW positions AS
SELECT 
    t.id,
    t.symbol,
    t.quantity,
    t.entry_price AS avg_price,
    -- current_price would come from real-time data
    t.entry_price AS current_price,
    0.00 AS unrealized_pnl,                -- Would be calculated
    t.entry_timestamp AS created_at,
    t.entry_timestamp AS updated_at,
    true AS is_open,
    t.entry_timestamp AS entry_date
FROM trade_records t
WHERE t.status = 'open';
```

---

## 4. Analysis & Pattern Tables

### 4.1 pattern_analysis
**Purpose**: Detected chart patterns with catalyst context  
**Primary Key**: id

```sql
CREATE TABLE pattern_analysis (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    pattern_type VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,2) NOT NULL,
    detection_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Pattern details
    timeframe VARCHAR(10),                  -- 1min, 5min, 15min, etc
    start_price DECIMAL(10,2),
    current_price DECIMAL(10,2),
    pattern_data JSONB,                     -- Pattern-specific details
    
    -- Catalyst context
    has_catalyst BOOLEAN DEFAULT FALSE,
    catalyst_type VARCHAR(50),
    catalyst_aligned BOOLEAN,               -- Pattern supports catalyst?
    
    -- Pattern metrics
    pattern_strength DECIMAL(5,2),
    support_level DECIMAL(10,2),
    resistance_level DECIMAL(10,2),
    
    -- Validation
    volume_confirmation BOOLEAN,
    trend_confirmation BOOLEAN,
    
    -- Outcome tracking
    pattern_completed BOOLEAN,
    actual_move DECIMAL(5,2),
    success BOOLEAN,
    
    -- Service compatibility
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

### 4.2 pattern_detections
**Purpose**: Service-specific pattern tracking  
**Primary Key**: id

```sql
CREATE TABLE pattern_detections (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    pattern_type VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,2) NOT NULL,
    timeframe VARCHAR(10) DEFAULT '5min',
    catalyst_aligned BOOLEAN DEFAULT FALSE,
    detection_price DECIMAL(10,2),
    metadata JSONB,
    detection_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

### 4.3 technical_indicators
**Purpose**: Store calculated indicators for analysis  
**Primary Key**: id

```sql
CREATE TABLE technical_indicators (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    calculated_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    timeframe VARCHAR(10) DEFAULT '5min',
    
    -- Price action
    open_price DECIMAL(10,2),
    high_price DECIMAL(10,2),
    low_price DECIMAL(10,2),
    close_price DECIMAL(10,2),
    volume BIGINT,
    
    -- Indicators
    rsi DECIMAL(5,2),
    macd DECIMAL(10,4),
    macd_signal DECIMAL(10,4),
    sma_20 DECIMAL(10,2),
    sma_50 DECIMAL(10,2),
    ema_9 DECIMAL(10,2),
    
    -- Volatility
    atr DECIMAL(10,4),
    bollinger_upper DECIMAL(10,2),
    bollinger_lower DECIMAL(10,2),
    
    -- Volume analysis
    volume_sma DECIMAL(15,2),
    relative_volume DECIMAL(5,2)
);
```

---

## 5. System & Coordination Tables

### 5.1 trading_cycles
**Purpose**: Track complete workflow executions  
**Primary Key**: cycle_id

```sql
CREATE TABLE trading_cycles (
    cycle_id VARCHAR(50) PRIMARY KEY,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'running',   -- running, completed, failed
    mode VARCHAR(20),                       -- aggressive, normal, light
    
    -- Metrics from each stage
    news_collected INTEGER DEFAULT 0,
    securities_scanned INTEGER DEFAULT 0,
    candidates_selected INTEGER DEFAULT 0,
    patterns_analyzed INTEGER DEFAULT 0,
    signals_generated INTEGER DEFAULT 0,
    trades_executed INTEGER DEFAULT 0,
    
    -- Performance
    cycle_pnl DECIMAL(10,2),
    success_rate DECIMAL(5,2),
    
    -- Metadata
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

### 5.2 service_health
**Purpose**: Monitor service status and performance  
**Primary Key**: id

```sql
CREATE TABLE service_health (
    id BIGSERIAL PRIMARY KEY,
    service_name VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,            -- healthy, degraded, down
    last_check TIMESTAMPTZ NOT NULL,
    response_time_ms INTEGER,
    error_message TEXT,
    
    -- Performance metrics
    requests_processed INTEGER,
    errors_count INTEGER,
    avg_response_time_ms INTEGER,
    
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

### 5.3 workflow_log
**Purpose**: Detailed logging of workflow execution  
**Primary Key**: id

```sql
CREATE TABLE workflow_log (
    id BIGSERIAL PRIMARY KEY,
    cycle_id VARCHAR(50),
    step_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL,            -- started, completed, failed
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    duration_seconds DECIMAL(10,3),
    
    -- Results
    records_processed INTEGER,
    records_output INTEGER,
    result JSONB,
    error_message TEXT,
    
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (cycle_id) REFERENCES trading_cycles(cycle_id)
);
```

### 5.4 configuration
**Purpose**: Store system configuration  
**Primary Key**: key

```sql
CREATE TABLE configuration (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    data_type VARCHAR(20),                  -- int, float, string, json
    category VARCHAR(50),                   -- trading, risk, schedule, api
    description TEXT,
    last_modified TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    modified_by VARCHAR(100),
    active BOOLEAN DEFAULT TRUE
);
```

---

## 6. Implementation Tables

### 6.1 trades (Compatibility View)
**Purpose**: Backward compatibility for reporting service  
**Implementation**: View mapping to trade_records

```sql
CREATE VIEW trades AS 
SELECT 
    t.id,
    t.symbol,
    t.side,
    t.quantity,
    t.entry_price AS price,
    t.entry_timestamp AS executed_at,
    t.pnl_amount AS pnl,
    t.commission,
    t.status,
    t.trade_id AS order_id,
    NULL::INTEGER AS pattern_id,
    t.created_at
FROM trade_records t;
```

### 6.2 scan_results (Compatibility View)
**Purpose**: Scanner service compatibility  
**Implementation**: View over trading_candidates

```sql
CREATE VIEW scan_results AS
SELECT 
    id,
    symbol,
    primary_catalyst AS scan_type,
    catalyst_score AS score,
    jsonb_build_object(
        'catalyst_keywords', catalyst_keywords,
        'news_count', news_count,
        'confidence_score', confidence_score
    ) AS metadata,
    created_at
FROM trading_candidates;
```

---

## 7. Indexes & Performance

### 7.1 Critical Indexes

```sql
-- News performance
CREATE INDEX idx_news_symbol_time ON news_raw(symbol, published_timestamp DESC);
CREATE INDEX idx_news_premarket ON news_raw(is_pre_market, published_timestamp DESC);
CREATE INDEX idx_news_source_tier ON news_raw(source_tier, published_timestamp DESC);
CREATE INDEX idx_news_cluster ON news_raw(narrative_cluster_id);
CREATE INDEX idx_news_collected_time ON news_raw(collected_timestamp DESC);
CREATE INDEX idx_news_catalyst_keywords ON news_raw USING GIN(headline_keywords);
CREATE INDEX idx_news_mentioned_tickers ON news_raw USING GIN(mentioned_tickers);

-- Trading performance  
CREATE INDEX idx_candidates_score ON trading_candidates(catalyst_score DESC);
CREATE INDEX idx_candidates_scan ON trading_candidates(scan_id, selection_timestamp DESC);
CREATE INDEX idx_candidates_created ON trading_candidates(created_at DESC);
CREATE INDEX idx_signals_pending ON trading_signals(executed, confidence DESC);
CREATE INDEX idx_signals_generated ON trading_signals(generated_timestamp DESC);
CREATE INDEX idx_trades_symbol ON trade_records(symbol, entry_timestamp DESC);
CREATE INDEX idx_trades_active ON trade_records(exit_timestamp) WHERE exit_timestamp IS NULL;
CREATE INDEX idx_trades_pnl ON trade_records(pnl_percentage DESC);

-- Analysis performance
CREATE INDEX idx_patterns_recent ON pattern_analysis(detection_timestamp DESC);
CREATE INDEX idx_patterns_success ON pattern_analysis(symbol, success);
CREATE INDEX idx_patterns_type ON pattern_analysis(pattern_type, confidence DESC);
CREATE INDEX idx_pattern_detections_symbol ON pattern_detections(symbol, detection_timestamp DESC);
CREATE INDEX idx_indicators_symbol_time ON technical_indicators(symbol, calculated_timestamp DESC);

-- System monitoring
CREATE INDEX idx_service_health_status ON service_health(service_name, status);
CREATE INDEX idx_service_health_time ON service_health(created_at DESC);
CREATE INDEX idx_workflow_cycle ON workflow_log(cycle_id, step_name);
CREATE INDEX idx_collection_stats_time ON news_collection_stats(collection_timestamp DESC);
```

### 7.2 Database Configuration

```sql
-- PostgreSQL optimizations
-- shared_buffers = 256MB
-- work_mem = 16MB  
-- maintenance_work_mem = 128MB
-- effective_cache_size = 1GB
-- max_connections = 100
-- checkpoint_segments = 16
-- checkpoint_completion_target = 0.9
-- wal_buffers = 16MB
```

---

## 8. Data Relationships

### 8.1 Entity Relationship Overview

```
news_raw (1) ←→ (N) trading_candidates
    ↓                      ↓
source_metrics      pattern_analysis
    ↓                      ↓
narrative_clusters   trading_signals
                           ↓
                      trade_records
                           ↓
                    outcome_tracking → news_raw
```

### 8.2 Key Relationships

1. **News → Trading Flow**
   - news_raw.symbol → trading_candidates.symbol
   - trading_candidates → pattern_analysis
   - pattern_analysis + news → trading_signals
   - trading_signals → trade_records

2. **Feedback Loops**
   - trade_records.entry_news_id → news_raw.news_id
   - Update news_raw.was_accurate based on trades
   - Update source_metrics based on accuracy

3. **Coordination**
   - trading_cycles tracks complete workflows
   - workflow_log details each step
   - service_health monitors system status

### 8.3 Service Dependencies

| Service | Primary Tables | Secondary Tables |
|---------|---------------|------------------|
| News Service | news_raw, news_collection_stats | source_metrics |
| Scanner Service | trading_candidates | news_raw |
| Pattern Service | pattern_analysis, pattern_detections | trading_candidates |
| Technical Service | technical_indicators, trading_signals | pattern_analysis |
| Trading Service | trade_records | trading_signals |
| Reporting Service | All tables (read-only) | - |

---

## Implementation Notes

### Service Compatibility
1. All services expect specific table names - do not change
2. Timestamp fields must use TIMESTAMPTZ for timezone handling
3. JSONB is used instead of JSON for better PostgreSQL performance
4. Views provide backward compatibility during transition

### Migration from v2.0.0
1. Main change is ensuring table names match service expectations
2. Added compatibility views for smooth transition
3. All timestamp fields standardized to TIMESTAMPTZ
4. Added service-specific tracking tables

### Critical Constraints
1. news_raw.news_id must be unique (primary key)
2. trading_signals.signal_id must be unique (primary key)
3. trade_records maintains foreign keys to news and signals
4. All monetary values use DECIMAL(10,2) for precision

This schema provides the foundation for news-driven trading with comprehensive tracking of sources, outcomes, and patterns for future ML development while ensuring all services work correctly.