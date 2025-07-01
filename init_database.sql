-- =============================================================================
-- CATALYST TRADING SYSTEM - POSTGRESQL DATABASE INITIALIZATION
-- DigitalOcean Managed Database Schema v2.0.0
-- =============================================================================

-- Create database if it doesn't exist (run manually on DigitalOcean)
-- CREATE DATABASE catalyst_trading;

-- Connect to the database
\c catalyst_trading;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Set timezone for consistency
SET timezone = 'UTC';

-- =============================================================================
-- NEWS & INTELLIGENCE TABLES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Raw news data (never modified after insert)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_raw (
    id BIGSERIAL PRIMARY KEY,
    news_id VARCHAR(64) UNIQUE NOT NULL,  -- Hash of headline+source+timestamp
    
    -- Core news data
    symbol VARCHAR(10),                    -- Primary symbol (can be NULL)
    headline TEXT NOT NULL,
    source VARCHAR(100) NOT NULL,
    source_url TEXT,
    published_timestamp TIMESTAMPTZ NOT NULL,
    collected_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    content_snippet TEXT,                  -- First 500 chars
    full_url TEXT,
    
    -- Rich metadata
    metadata JSONB,                        -- All extra fields from APIs
    is_pre_market BOOLEAN DEFAULT FALSE,
    market_state VARCHAR(20),              -- pre-market, regular, after-hours, weekend
    headline_keywords JSONB,               -- ["earnings", "fda", "merger", etc]
    mentioned_tickers JSONB,               -- Other tickers in article
    article_length INTEGER,
    is_breaking_news BOOLEAN DEFAULT FALSE,
    update_count INTEGER DEFAULT 0,
    first_seen_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Source alignment tracking
    source_tier INTEGER DEFAULT 5,         -- 1=Bloomberg/Reuters, 5=Unknown
    confirmation_status VARCHAR(20) DEFAULT 'unconfirmed',
    confirmed_by VARCHAR(100),             -- Which tier 1-2 source confirmed
    confirmation_timestamp TIMESTAMPTZ,
    confirmation_delay_minutes INTEGER,
    was_accurate BOOLEAN,                  -- Did prediction come true?
    
    -- Narrative tracking
    narrative_cluster_id VARCHAR(100),     -- Groups similar stories
    sentiment_keywords JSONB,              -- ["breakthrough", "concerns", etc]
    similar_stories_count INTEGER DEFAULT 0,
    
    -- Outcome tracking (updated post-trade)
    price_move_1h DECIMAL(5,2),           -- % move 1 hour after news
    price_move_24h DECIMAL(5,2),          -- % move 24 hours after
    volume_surge_ratio DECIMAL(5,2),
    subsequent_actions JSONB,              -- Corporate actions, trades
    beneficiaries JSONB,                   -- Who profited from this news
    
    CONSTRAINT unique_news_item UNIQUE(headline, source, published_timestamp)
);

-- -----------------------------------------------------------------------------
-- Source reliability metrics
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS source_metrics (
    id SERIAL PRIMARY KEY,
    source_name VARCHAR(100) UNIQUE NOT NULL,
    source_tier INTEGER NOT NULL,
    
    -- Accuracy metrics
    total_articles INTEGER DEFAULT 0,
    confirmed_articles INTEGER DEFAULT 0,
    accurate_predictions INTEGER DEFAULT 0,
    false_predictions INTEGER DEFAULT 0,
    accuracy_rate DECIMAL(5,2),
    
    -- Timing patterns
    avg_early_minutes INTEGER,             -- How early they publish
    breaking_news_count INTEGER DEFAULT 0,
    exclusive_story_count INTEGER DEFAULT 0,
    
    -- Narrative patterns
    pump_keywords_count INTEGER DEFAULT 0,
    dump_keywords_count INTEGER DEFAULT 0,
    narrative_clusters JSONB,              -- Common themes
    
    -- Beneficiary tracking
    frequent_beneficiaries JSONB,          -- Entities that profit
    agenda_indicators JSONB,               -- Detected biases
    
    last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- Narrative cluster analysis
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS narrative_clusters (
    cluster_id VARCHAR(100) PRIMARY KEY,   -- symbol_date_keywords
    symbol VARCHAR(10) NOT NULL,
    cluster_date DATE NOT NULL,
    
    -- Cluster characteristics
    primary_keywords JSONB NOT NULL,
    source_count INTEGER,
    article_count INTEGER,
    sources_involved JSONB,
    
    -- Timing analysis
    first_published TIMESTAMPTZ,
    last_published TIMESTAMPTZ,
    time_spread_hours DECIMAL(5,2),
    
    -- Coordination scoring
    coordination_score DECIMAL(5,2),       -- Higher = more suspicious
    avg_source_tier DECIMAL(3,2),
    
    -- Outcome
    resulted_in_movement BOOLEAN,
    price_impact DECIMAL(5,2),
    volume_impact DECIMAL(5,2),
    
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- TRADING OPERATIONS TABLES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Selected trading candidates
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trading_candidates (
    id BIGSERIAL PRIMARY KEY,
    scan_id VARCHAR(50) NOT NULL,          -- Links to scanning_results
    symbol VARCHAR(10) NOT NULL,
    selection_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Selection criteria
    catalyst_score DECIMAL(5,2) NOT NULL,
    news_count INTEGER,
    primary_catalyst VARCHAR(50),          -- earnings, fda, merger, etc
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
    selection_rank INTEGER,                -- 1-5 for top picks
    
    -- Status
    analyzed BOOLEAN DEFAULT FALSE,
    traded BOOLEAN DEFAULT FALSE
);

-- -----------------------------------------------------------------------------
-- Trading signals generated
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trading_signals (
    signal_id VARCHAR(50) PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    generated_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Signal details
    signal_type VARCHAR(10) NOT NULL,      -- BUY, SELL, HOLD
    confidence DECIMAL(5,2),               -- 0-100
    
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
    key_factors JSONB,                     -- Why this signal
    
    -- Risk parameters
    position_size_pct DECIMAL(5,2),
    risk_reward_ratio DECIMAL(5,2),
    
    -- Execution status
    executed BOOLEAN DEFAULT FALSE,
    execution_timestamp TIMESTAMPTZ,
    actual_entry DECIMAL(10,2)
);

-- -----------------------------------------------------------------------------
-- Executed trades and outcomes
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trade_records (
    trade_id VARCHAR(50) PRIMARY KEY,
    signal_id VARCHAR(50),
    symbol VARCHAR(10) NOT NULL,
    
    -- Execution details
    order_type VARCHAR(20),                -- market, limit
    side VARCHAR(10) NOT NULL,             -- buy, sell
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(10,2),
    entry_timestamp TIMESTAMPTZ,
    
    -- Exit details
    exit_price DECIMAL(10,2),
    exit_timestamp TIMESTAMPTZ,
    exit_reason VARCHAR(50),               -- stop_loss, target, time_stop, signal
    
    -- Performance
    pnl_amount DECIMAL(10,2),
    pnl_percentage DECIMAL(5,2),
    commission DECIMAL(10,2),
    
    -- Catalyst tracking
    entry_catalyst VARCHAR(50),
    entry_news_id VARCHAR(64),             -- Link to news_raw
    catalyst_score_at_entry DECIMAL(5,2),
    
    -- Outcome analysis
    max_profit DECIMAL(10,2),
    max_loss DECIMAL(10,2),
    time_to_target_minutes INTEGER,
    
    -- ML features
    pattern_confirmed BOOLEAN,
    catalyst_confirmed BOOLEAN,
    
    FOREIGN KEY (signal_id) REFERENCES trading_signals(signal_id),
    FOREIGN KEY (entry_news_id) REFERENCES news_raw(news_id)
);

-- =============================================================================
-- ANALYSIS & PATTERN TABLES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Technical pattern detection results
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pattern_analysis (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    detection_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    timeframe VARCHAR(10) DEFAULT '5min',
    
    -- Pattern details
    pattern_type VARCHAR(50) NOT NULL,     -- hammer, doji, engulfing, etc
    pattern_direction VARCHAR(10),         -- bullish, bearish, neutral
    confidence DECIMAL(5,2),
    
    -- Context awareness
    has_catalyst BOOLEAN DEFAULT FALSE,
    catalyst_type VARCHAR(50),
    catalyst_alignment BOOLEAN,            -- Does pattern align with news?
    
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
    success BOOLEAN
);

-- -----------------------------------------------------------------------------
-- Technical indicators calculated
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS technical_indicators (
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

-- =============================================================================
-- SYSTEM & COORDINATION TABLES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Trading cycle tracking
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trading_cycles (
    cycle_id VARCHAR(50) PRIMARY KEY,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'running',  -- running, completed, failed
    mode VARCHAR(20),                      -- aggressive, normal, light
    
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

-- -----------------------------------------------------------------------------
-- Service health monitoring
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS service_health (
    id BIGSERIAL PRIMARY KEY,
    service_name VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,           -- healthy, degraded, down
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

-- -----------------------------------------------------------------------------
-- Workflow execution logging
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS workflow_log (
    id BIGSERIAL PRIMARY KEY,
    cycle_id VARCHAR(50),
    step_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL,           -- started, completed, failed
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

-- -----------------------------------------------------------------------------
-- System configuration
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS configuration (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    data_type VARCHAR(20),                 -- int, float, string, json
    category VARCHAR(50),                  -- trading, risk, schedule, api
    description TEXT,
    last_modified TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    modified_by VARCHAR(100)
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- News performance indexes
CREATE INDEX IF NOT EXISTS idx_news_symbol_time ON news_raw(symbol, published_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_news_premarket ON news_raw(is_pre_market, published_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_news_source_tier ON news_raw(source_tier, published_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_news_cluster ON news_raw(narrative_cluster_id);
CREATE INDEX IF NOT EXISTS idx_news_collected_time ON news_raw(collected_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_news_catalyst_keywords ON news_raw USING GIN(headline_keywords);
CREATE INDEX IF NOT EXISTS idx_news_mentioned_tickers ON news_raw USING GIN(mentioned_tickers);

-- Trading performance indexes
CREATE INDEX IF NOT EXISTS idx_candidates_score ON trading_candidates(catalyst_score DESC);
CREATE INDEX IF NOT EXISTS idx_candidates_scan ON trading_candidates(scan_id, selection_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_pending ON trading_signals(executed, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_signals_generated ON trading_signals(generated_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trade_records(symbol, entry_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_active ON trade_records(exit_timestamp) WHERE exit_timestamp IS NULL;
CREATE INDEX IF NOT EXISTS idx_trades_pnl ON trade_records(pnl_percentage DESC);

-- Analysis performance indexes
CREATE INDEX IF NOT EXISTS idx_patterns_recent ON pattern_analysis(detection_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_success ON pattern_analysis(symbol, success);
CREATE INDEX IF NOT EXISTS idx_patterns_type ON pattern_analysis(pattern_type, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_indicators_symbol_time ON technical_indicators(symbol, calculated_timestamp DESC);

-- System monitoring indexes
CREATE INDEX IF NOT EXISTS idx_service_health_status ON service_health(service_name, status);
CREATE INDEX IF NOT EXISTS idx_service_health_time ON service_health(last_check DESC);
CREATE INDEX IF NOT EXISTS idx_workflow_cycle ON workflow_log(cycle_id, step_name);
CREATE INDEX IF NOT EXISTS idx_workflow_status ON workflow_log(status, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_trading_cycles_time ON trading_cycles(start_time DESC);

-- =============================================================================
-- DEFAULT CONFIGURATION VALUES
-- =============================================================================

INSERT INTO configuration (key, value, data_type, category, description) VALUES
-- Risk Management
('max_positions', '5', 'int', 'risk', 'Maximum concurrent positions'),
('position_size_pct', '20', 'float', 'risk', 'Max position size as % of capital'),
('premarket_position_pct', '10', 'float', 'risk', 'Pre-market position size limit'),
('stop_loss_pct', '2', 'float', 'risk', 'Default stop loss percentage'),
('max_daily_trades', '20', 'int', 'risk', 'Maximum trades per day'),

-- Trading Parameters
('min_catalyst_score', '30', 'float', 'trading', 'Minimum score to consider'),
('min_price', '1.0', 'float', 'trading', 'Minimum stock price'),
('max_price', '500.0', 'float', 'trading', 'Maximum stock price'),
('min_volume', '500000', 'int', 'trading', 'Minimum volume threshold'),
('min_relative_volume', '1.5', 'float', 'trading', 'Minimum relative volume'),

-- Schedule Configuration
('premarket_start', '04:00', 'string', 'schedule', 'Pre-market start time EST'),
('premarket_end', '09:30', 'string', 'schedule', 'Pre-market end time EST'),
('premarket_interval', '5', 'int', 'schedule', 'Pre-market scan interval minutes'),
('market_interval', '30', 'int', 'schedule', 'Regular market scan interval minutes'),
('afterhours_interval', '60', 'int', 'schedule', 'After-hours scan interval minutes'),

-- API Configuration
('news_cache_ttl', '3600', 'int', 'api', 'News cache TTL in seconds'),
('api_timeout', '30', 'int', 'api', 'Default API timeout in seconds'),
('max_news_age_hours', '24', 'int', 'api', 'Maximum news age for consideration'),

-- Source Weights
('tier_1_weight', '1.0', 'float', 'sources', 'Tier 1 source weight'),
('tier_2_weight', '0.8', 'float', 'sources', 'Tier 2 source weight'),
('tier_3_weight', '0.6', 'float', 'sources', 'Tier 3 source weight'),
('tier_4_weight', '0.4', 'float', 'sources', 'Tier 4 source weight'),
('tier_5_weight', '0.2', 'float', 'sources', 'Tier 5 source weight'),

-- Pattern Analysis
('pattern_confidence_threshold', '70', 'float', 'patterns', 'Minimum pattern confidence'),
('technical_signal_threshold', '50', 'float', 'patterns', 'Minimum technical signal strength'),
('catalyst_alignment_boost', '1.5', 'float', 'patterns', 'Catalyst alignment multiplier')

ON CONFLICT (key) DO NOTHING;

-- =============================================================================
-- DATABASE OPTIMIZATION
-- =============================================================================

-- Optimize PostgreSQL for time-series data
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET work_mem = '16MB';
ALTER SYSTEM SET maintenance_work_mem = '128MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Enable automatic vacuuming
ALTER SYSTEM SET autovacuum = on;
ALTER SYSTEM SET autovacuum_max_workers = 3;
ALTER SYSTEM SET autovacuum_naptime = '1min';

-- =============================================================================
-- PARTITIONING SETUP (for high volume)
-- =============================================================================

-- Partition news_raw by month for better performance
-- (Enable if news volume becomes very high)

-- CREATE TABLE news_raw_2025_01 PARTITION OF news_raw 
-- FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- =============================================================================
-- SECURITY SETUP
-- =============================================================================

-- Create application user with restricted permissions
CREATE USER catalyst_app WITH PASSWORD 'secure_password_here';

-- Grant necessary permissions
GRANT CONNECT ON DATABASE catalyst_trading TO catalyst_app;
GRANT USAGE ON SCHEMA public TO catalyst_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO catalyst_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO catalyst_app;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO catalyst_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO catalyst_app;

-- =============================================================================
-- INITIAL DATA SETUP
-- =============================================================================

-- Insert known source tier classifications
INSERT INTO source_metrics (source_name, source_tier, total_articles, accuracy_rate) VALUES
-- Tier 1 - Institutional Grade
('Bloomberg', 1, 0, 95.0),
('Reuters', 1, 0, 94.0),
('Dow Jones', 1, 0, 93.0),
('Associated Press', 1, 0, 92.0),

-- Tier 2 - Major Financial Media
('Wall Street Journal', 2, 0, 88.0),
('Financial Times', 2, 0, 87.0),
('CNBC', 2, 0, 85.0),
('MarketWatch', 2, 0, 83.0),

-- Tier 3 - Established Financial Sites
('Yahoo Finance', 3, 0, 75.0),
('Seeking Alpha', 3, 0, 70.0),
('Benzinga', 3, 0, 72.0),
('The Motley Fool', 3, 0, 68.0),

-- Tier 4 - Mixed/Blog Sources
('Zacks', 4, 0, 60.0),
('TipRanks', 4, 0, 58.0),
('Stocktwits', 4, 0, 45.0)

ON CONFLICT (source_name) DO NOTHING;

-- =============================================================================
-- BACKUP AND MAINTENANCE
-- =============================================================================

-- Set up automated backups (managed by DigitalOcean)
-- Point-in-time recovery enabled
-- Daily backups retained for 7 days
-- Weekly backups retained for 4 weeks

-- Maintenance window: Sundays 2-4 AM UTC (off-market hours)

-- =============================================================================
-- MONITORING SETUP
-- =============================================================================

-- Enable query statistics
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
ALTER SYSTEM SET pg_stat_statements.track = 'all';
ALTER SYSTEM SET pg_stat_statements.max = 10000;

-- Enable slow query logging
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- Log queries > 1 second
ALTER SYSTEM SET log_statement = 'none';
ALTER SYSTEM SET log_duration = off;

-- =============================================================================
-- SCHEMA VERSION TRACKING
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    applied_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO schema_migrations (version, name) VALUES 
(1, 'initial_schema_v2.0.0')
ON CONFLICT (version) DO NOTHING;

-- =============================================================================
-- COMPLETION
-- =============================================================================

-- Refresh statistics
ANALYZE;

-- Create a view for quick system overview
CREATE OR REPLACE VIEW system_overview AS
SELECT 
    'news_articles' as metric, COUNT(*)::text as value 
    FROM news_raw WHERE collected_timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours'
UNION ALL
SELECT 
    'active_candidates' as metric, COUNT(*)::text as value 
    FROM trading_candidates WHERE NOT traded AND selection_timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
UNION ALL
SELECT 
    'open_positions' as metric, COUNT(*)::text as value 
    FROM trade_records WHERE exit_timestamp IS NULL
UNION ALL
SELECT 
    'today_pnl' as metric, COALESCE(SUM(pnl_amount), 0)::text as value 
    FROM trade_records WHERE entry_timestamp::date = CURRENT_DATE;

-- Grant view access
GRANT SELECT ON system_overview TO catalyst_app;

COMMIT;

-- =============================================================================
-- DEPLOYMENT VERIFICATION
-- =============================================================================

-- Run these queries to verify setup:
-- SELECT version(), current_database(), current_user;
-- SELECT COUNT(*) FROM configuration;
-- SELECT COUNT(*) FROM source_metrics;
-- SELECT * FROM system_overview;

-- =============================================================================
-- COMPLETION MESSAGE
-- =============================================================================

\echo 'Catalyst Trading System database initialization completed successfully!'
\echo 'Database: catalyst_trading'
\echo 'Schema version: 2.0.0'
\echo 'Tables created: 12'
\echo 'Indexes created: 20+'
\echo 'Configuration entries: 20+'
\echo 'Ready for DigitalOcean deployment!'