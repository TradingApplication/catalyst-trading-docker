# Catalyst Trading System - Database Schema v2.0.0

**Version**: 2.0.0  
**Date**: June 28, 2025  
**Database**: SQLite (Development) / PostgreSQL (Production)  

## Table of Contents

1. [Schema Overview](#1-schema-overview)
2. [News & Intelligence Tables](#2-news--intelligence-tables)
3. [Trading Operations Tables](#3-trading-operations-tables)
4. [Analysis & Pattern Tables](#4-analysis--pattern-tables)
5. [System & Coordination Tables](#5-system--coordination-tables)
6. [Indexes & Performance](#6-indexes--performance)
7. [Data Relationships](#7-data-relationships)

---

## 1. Schema Overview

### 1.1 Database Design Principles
- **Raw Data Preservation**: news_raw never modified after insert
- **Clean Separation**: Raw data vs processed trading data
- **Audit Trail**: Complete history of all decisions
- **ML Readiness**: Outcome tracking built-in
- **Performance**: Strategic indexes for common queries

### 1.2 Table Categories
1. **News & Intelligence**: Raw news, source metrics, narratives
2. **Trading Operations**: Candidates, signals, trades, positions
3. **Analysis & Pattern**: Technical patterns, indicators
4. **System & Coordination**: Service health, workflow, configuration

---

## 2. News & Intelligence Tables

### 2.1 news_raw
**Purpose**: Store all collected news without modification

```sql
CREATE TABLE news_raw (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    news_id TEXT UNIQUE NOT NULL,  -- Hash of headline+source+timestamp
    
    -- Core news data
    symbol TEXT,                   -- Primary symbol (can be NULL)
    headline TEXT NOT NULL,
    source TEXT NOT NULL,
    source_url TEXT,
    published_timestamp TIMESTAMP NOT NULL,
    collected_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    content_snippet TEXT,          -- First 500 chars
    full_url TEXT,
    
    -- Rich metadata
    metadata JSON,                 -- All extra fields from APIs
    is_pre_market BOOLEAN DEFAULT FALSE,
    market_state TEXT,             -- pre-market, regular, after-hours, weekend
    headline_keywords JSON,        -- ["earnings", "fda", "merger", etc]
    mentioned_tickers JSON,        -- Other tickers in article
    article_length INTEGER,
    is_breaking_news BOOLEAN DEFAULT FALSE,
    update_count INTEGER DEFAULT 0,
    first_seen_timestamp TIMESTAMP,
    
    -- Source alignment tracking
    source_tier INTEGER DEFAULT 5,  -- 1=Bloomberg/Reuters, 5=Unknown
    confirmation_status TEXT DEFAULT 'unconfirmed',
    confirmed_by TEXT,             -- Which tier 1-2 source confirmed
    confirmation_timestamp TIMESTAMP,
    confirmation_delay_minutes INTEGER,
    was_accurate BOOLEAN,          -- Did prediction come true?
    
    -- Narrative tracking
    narrative_cluster_id TEXT,     -- Groups similar stories
    sentiment_keywords JSON,       -- ["breakthrough", "concerns", etc]
    similar_stories_count INTEGER DEFAULT 0,
    
    -- Outcome tracking (updated post-trade)
    price_move_1h DECIMAL(5,2),   -- % move 1 hour after news
    price_move_24h DECIMAL(5,2),  -- % move 24 hours after
    volume_surge_ratio DECIMAL(5,2),
    subsequent_actions JSON,       -- Corporate actions, trades
    beneficiaries JSON,           -- Who profited from this news
    
    UNIQUE(headline, source, published_timestamp)
);
```

### 2.2 source_metrics
**Purpose**: Track source reliability and patterns over time

```sql
CREATE TABLE source_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    source_tier INTEGER NOT NULL,
    
    -- Accuracy metrics
    total_articles INTEGER DEFAULT 0,
    confirmed_articles INTEGER DEFAULT 0,
    accurate_predictions INTEGER DEFAULT 0,
    false_predictions INTEGER DEFAULT 0,
    accuracy_rate DECIMAL(5,2),
    
    -- Timing patterns
    avg_early_minutes INTEGER,     -- How early they publish
    breaking_news_count INTEGER,
    exclusive_story_count INTEGER,
    
    -- Narrative patterns
    pump_keywords_count INTEGER,
    dump_keywords_count INTEGER,
    narrative_clusters JSON,       -- Common themes
    
    -- Beneficiary tracking
    frequent_beneficiaries JSON,   -- Entities that profit
    agenda_indicators JSON,        -- Detected biases
    
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(source_name)
);
```

### 2.3 narrative_clusters
**Purpose**: Group coordinated messaging patterns

```sql
CREATE TABLE narrative_clusters (
    cluster_id TEXT PRIMARY KEY,   -- symbol_date_keywords
    symbol TEXT NOT NULL,
    cluster_date DATE NOT NULL,
    
    -- Cluster characteristics
    primary_keywords JSON NOT NULL,
    source_count INTEGER,
    article_count INTEGER,
    sources_involved JSON,
    
    -- Timing analysis
    first_published TIMESTAMP,
    last_published TIMESTAMP,
    time_spread_hours DECIMAL(5,2),
    
    -- Coordination scoring
    coordination_score DECIMAL(5,2),  -- Higher = more suspicious
    avg_source_tier DECIMAL(3,2),
    
    -- Outcome
    resulted_in_movement BOOLEAN,
    price_impact DECIMAL(5,2),
    volume_impact DECIMAL(5,2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 3. Trading Operations Tables

### 3.1 trading_candidates
**Purpose**: Top securities selected for potential trading

```sql
CREATE TABLE trading_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT NOT NULL,         -- Links to scanning_results_v2
    symbol TEXT NOT NULL,
    selection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Selection criteria
    catalyst_score DECIMAL(5,2) NOT NULL,
    news_count INTEGER,
    primary_catalyst TEXT,         -- earnings, fda, merger, etc
    catalyst_keywords JSON,
    
    -- Technical validation
    price DECIMAL(10,2),
    volume INTEGER,
    relative_volume DECIMAL(5,2),
    price_change_pct DECIMAL(5,2),
    
    -- Pre-market data (if applicable)
    pre_market_volume INTEGER,
    pre_market_change DECIMAL(5,2),
    has_pre_market_news BOOLEAN DEFAULT FALSE,
    
    -- Final scoring
    technical_score DECIMAL(5,2),
    combined_score DECIMAL(5,2),
    selection_rank INTEGER,        -- 1-5 for top picks
    
    -- Status
    analyzed BOOLEAN DEFAULT FALSE,
    traded BOOLEAN DEFAULT FALSE,
    
    INDEX idx_scan_symbol (scan_id, symbol),
    INDEX idx_selection_time (selection_timestamp)
);
```

### 3.2 trading_signals
**Purpose**: Actionable trading signals generated

```sql
CREATE TABLE trading_signals (
    signal_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    generated_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Signal details
    signal_type TEXT NOT NULL,     -- BUY, SELL, HOLD
    confidence DECIMAL(5,2),       -- 0-100
    
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
    catalyst_type TEXT,
    detected_patterns JSON,
    key_factors JSON,              -- Why this signal
    
    -- Risk parameters
    position_size_pct DECIMAL(5,2),
    risk_reward_ratio DECIMAL(5,2),
    
    -- Execution status
    executed BOOLEAN DEFAULT FALSE,
    execution_timestamp TIMESTAMP,
    actual_entry DECIMAL(10,2),
    
    INDEX idx_symbol_time (symbol, generated_timestamp),
    INDEX idx_confidence (confidence DESC)
);
```

### 3.3 trade_records
**Purpose**: Actual executed trades and their outcomes

```sql
CREATE TABLE trade_records (
    trade_id TEXT PRIMARY KEY,
    signal_id TEXT,
    symbol TEXT NOT NULL,
    
    -- Execution details
    order_type TEXT,               -- market, limit
    side TEXT NOT NULL,            -- buy, sell
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(10,2),
    entry_timestamp TIMESTAMP,
    
    -- Exit details
    exit_price DECIMAL(10,2),
    exit_timestamp TIMESTAMP,
    exit_reason TEXT,              -- stop_loss, target, time_stop, signal
    
    -- Performance
    pnl_amount DECIMAL(10,2),
    pnl_percentage DECIMAL(5,2),
    commission DECIMAL(10,2),
    
    -- Catalyst tracking
    entry_catalyst TEXT,
    entry_news_id TEXT,            -- Link to news_raw
    catalyst_score_at_entry DECIMAL(5,2),
    
    -- Outcome analysis
    max_profit DECIMAL(10,2),
    max_loss DECIMAL(10,2),
    time_to_target_minutes INTEGER,
    
    -- ML features
    pattern_confirmed BOOLEAN,
    catalyst_confirmed BOOLEAN,
    
    FOREIGN KEY (signal_id) REFERENCES trading_signals(signal_id),
    FOREIGN KEY (entry_news_id) REFERENCES news_raw(news_id),
    INDEX idx_symbol_time (symbol, entry_timestamp)
);
```

---

## 4. Analysis & Pattern Tables

### 4.1 pattern_analysis
**Purpose**: Store detected technical patterns

```sql
CREATE TABLE pattern_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    timeframe TEXT DEFAULT '5min',
    
    -- Pattern details
    pattern_type TEXT NOT NULL,    -- hammer, doji, engulfing, etc
    pattern_direction TEXT,         -- bullish, bearish, neutral
    confidence DECIMAL(5,2),
    
    -- Context awareness
    has_catalyst BOOLEAN DEFAULT FALSE,
    catalyst_type TEXT,
    catalyst_alignment BOOLEAN,     -- Does pattern align with news?
    
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
    
    INDEX idx_symbol_pattern (symbol, pattern_type),
    INDEX idx_timestamp (detection_timestamp)
);
```

### 4.2 technical_indicators
**Purpose**: Store calculated indicators for analysis

```sql
CREATE TABLE technical_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    calculated_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    timeframe TEXT DEFAULT '5min',
    
    -- Price action
    open_price DECIMAL(10,2),
    high_price DECIMAL(10,2),
    low_price DECIMAL(10,2),
    close_price DECIMAL(10,2),
    volume INTEGER,
    
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
    relative_volume DECIMAL(5,2),
    
    INDEX idx_symbol_time (symbol, calculated_timestamp)
);
```

---

## 5. System & Coordination Tables

### 5.1 trading_cycles
**Purpose**: Track complete workflow executions

```sql
CREATE TABLE trading_cycles (
    cycle_id TEXT PRIMARY KEY,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status TEXT DEFAULT 'running',  -- running, completed, failed
    mode TEXT,                      -- aggressive, normal, light
    
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
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 5.2 service_health
**Purpose**: Monitor service status and performance

```sql
CREATE TABLE service_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service_name TEXT NOT NULL,
    status TEXT NOT NULL,           -- healthy, degraded, down
    last_check TIMESTAMP NOT NULL,
    response_time_ms INTEGER,
    error_message TEXT,
    
    -- Performance metrics
    requests_processed INTEGER,
    errors_count INTEGER,
    avg_response_time_ms INTEGER,
    
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_service_status (service_name, status)
);
```

### 5.3 workflow_log
**Purpose**: Detailed logging of workflow execution

```sql
CREATE TABLE workflow_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT,
    step_name TEXT NOT NULL,
    status TEXT NOT NULL,           -- started, completed, failed
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    duration_seconds DECIMAL(10,3),
    
    -- Results
    records_processed INTEGER,
    records_output INTEGER,
    result JSON,
    error_message TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (cycle_id) REFERENCES trading_cycles(cycle_id),
    INDEX idx_cycle_step (cycle_id, step_name)
);
```

### 5.4 configuration
**Purpose**: Store system configuration

```sql
CREATE TABLE configuration (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    data_type TEXT,                 -- int, float, string, json
    category TEXT,                  -- trading, risk, schedule, api
    description TEXT,
    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_by TEXT
);

-- Insert default configuration
INSERT INTO configuration VALUES
('max_positions', '5', 'int', 'risk', 'Maximum concurrent positions', CURRENT_TIMESTAMP, 'system'),
('position_size_pct', '20', 'float', 'risk', 'Max position size as % of capital', CURRENT_TIMESTAMP, 'system'),
('premarket_position_pct', '10', 'float', 'risk', 'Pre-market position size limit', CURRENT_TIMESTAMP, 'system'),
('min_catalyst_score', '30', 'float', 'trading', 'Minimum score to consider', CURRENT_TIMESTAMP, 'system'),
('stop_loss_pct', '2', 'float', 'risk', 'Default stop loss percentage', CURRENT_TIMESTAMP, 'system');
```

---

## 6. Indexes & Performance

### 6.1 Critical Indexes

```sql
-- News performance
CREATE INDEX idx_news_symbol_time ON news_raw(symbol, published_timestamp DESC);
CREATE INDEX idx_news_premarket ON news_raw(is_pre_market, published_timestamp DESC);
CREATE INDEX idx_news_source_tier ON news_raw(source_tier, published_timestamp DESC);
CREATE INDEX idx_news_cluster ON news_raw(narrative_cluster_id);

-- Trading performance  
CREATE INDEX idx_candidates_score ON trading_candidates(catalyst_score DESC);
CREATE INDEX idx_signals_pending ON trading_signals(executed, confidence DESC);
CREATE INDEX idx_trades_active ON trade_records(exit_timestamp) WHERE exit_timestamp IS NULL;

-- Analysis performance
CREATE INDEX idx_patterns_recent ON pattern_analysis(detection_timestamp DESC);
CREATE INDEX idx_patterns_success ON pattern_analysis(symbol, success);
```

### 6.2 Database Configuration

```sql
-- SQLite optimizations
PRAGMA journal_mode = WAL;          -- Better concurrency
PRAGMA synchronous = NORMAL;        -- Balance safety/speed
PRAGMA cache_size = -64000;         -- 64MB cache
PRAGMA temp_store = MEMORY;         -- Temp tables in RAM
PRAGMA mmap_size = 268435456;      -- 256MB memory map

-- PostgreSQL optimizations (production)
-- shared_buffers = 256MB
-- work_mem = 16MB
-- maintenance_work_mem = 128MB
-- effective_cache_size = 1GB
```

---

## 7. Data Relationships

### 7.1 Entity Relationship Overview

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

### 7.2 Key Relationships

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

---

## Migration Notes

### From v1.0 to v2.0
1. Add all source alignment fields to news_raw
2. Create new tables: source_metrics, narrative_clusters
3. Add catalyst fields to trading tables
4. Migrate existing data with default values
5. Rebuild indexes for performance

### Production Considerations
1. Use PostgreSQL for better concurrency
2. Implement table partitioning for news_raw
3. Add read replicas for analytics
4. Regular VACUUM and ANALYZE
5. Monitor slow queries

This schema provides the foundation for news-driven trading with comprehensive tracking of sources, outcomes, and patterns for future ML development.