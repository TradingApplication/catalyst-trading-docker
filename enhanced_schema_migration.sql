-- ============================================================================
-- Name of Application: Catalyst Trading System
-- Name of file: enhanced_schema_migration.sql
-- Version: 1.0.0
-- Last Updated: 2025-01-27
-- Purpose: Migration script to add enhanced data collection tables for top 100 securities tracking
-- 
-- REVISION HISTORY:
-- v1.0.0 (2025-01-27) - Initial migration for enhanced data collection
-- - Adds high-frequency, hourly, and daily data tables
-- - Implements tracking state and ML pattern discovery tables
-- - Creates performance monitoring infrastructure
-- 
-- Description of Service:
-- This migration safely adds new tables for tracking top 100 securities with:
-- 1. Multi-frequency data storage (15min, hourly, daily)
-- 2. Intelligent aging and state tracking
-- 3. Pattern discovery and correlation analysis
-- 4. Performance metrics monitoring
-- ============================================================================

-- Start transaction for safe migration
BEGIN;

-- ============================================================================
-- STEP 1: Create high-frequency data table (15-minute intervals)
-- ============================================================================
CREATE TABLE IF NOT EXISTS security_data_high_freq (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Price data
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    volume BIGINT,
    
    -- Microstructure
    bid_ask_spread DECIMAL(8,4),
    order_imbalance DECIMAL(8,4),
    
    -- Technical indicators
    rsi_14 DECIMAL(5,2),
    macd DECIMAL(8,4),
    macd_signal DECIMAL(8,4),
    bb_upper DECIMAL(10,2),
    bb_lower DECIMAL(10,2),
    vwap DECIMAL(10,2),
    
    -- Context
    news_count INTEGER DEFAULT 0,
    catalyst_active BOOLEAN DEFAULT FALSE,
    catalyst_score DECIMAL(5,2),
    
    -- Performance indexes
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_hf_symbol_time ON security_data_high_freq(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_hf_timestamp ON security_data_high_freq(timestamp);
CREATE INDEX IF NOT EXISTS idx_hf_catalyst ON security_data_high_freq(catalyst_active) WHERE catalyst_active = true;

-- ============================================================================
-- STEP 2: Create hourly aggregated data table
-- ============================================================================
CREATE TABLE IF NOT EXISTS security_data_hourly (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- OHLCV for the hour
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    volume BIGINT,
    vwap DECIMAL(10,2),
    
    -- Aggregated metrics
    volatility DECIMAL(8,4),
    volume_ratio DECIMAL(8,4),  -- vs 20-period average
    price_range DECIMAL(8,4),   -- (high-low)/close
    
    -- Technical summary
    rsi_14 DECIMAL(5,2),
    trend_strength DECIMAL(5,2),
    
    -- News/Catalyst summary
    news_count INTEGER DEFAULT 0,
    max_catalyst_score DECIMAL(5,2),
    
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_hourly_symbol_time ON security_data_hourly(symbol, timestamp DESC);

-- ============================================================================
-- STEP 3: Create daily aggregated data table
-- ============================================================================
CREATE TABLE IF NOT EXISTS security_data_daily (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    
    -- Daily OHLCV
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    volume BIGINT,
    
    -- Daily metrics
    daily_return DECIMAL(8,4),
    volatility DECIMAL(8,4),
    dollar_volume DECIMAL(15,2),
    
    -- Pattern detection readiness
    pattern_readiness_score DECIMAL(5,2),
    
    -- News summary
    total_news_count INTEGER DEFAULT 0,
    catalyst_events JSONB,
    
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_daily_symbol_date ON security_data_daily(symbol, date DESC);

-- ============================================================================
-- STEP 4: Create security tracking state table
-- ============================================================================
CREATE TABLE IF NOT EXISTS security_tracking_state (
    symbol VARCHAR(10) PRIMARY KEY,
    first_seen TIMESTAMPTZ NOT NULL,
    last_updated TIMESTAMPTZ NOT NULL,
    
    -- Collection settings
    collection_frequency VARCHAR(20) NOT NULL,
    data_points_collected INTEGER DEFAULT 0,
    
    -- State tracking
    last_price DECIMAL(10,2),
    last_volume BIGINT,
    flatline_periods INTEGER DEFAULT 0,
    
    -- Catalyst tracking
    last_catalyst_score DECIMAL(5,2),
    catalyst_events JSONB,
    
    -- Aging metrics
    hours_since_catalyst DECIMAL(8,2),
    activity_score DECIMAL(5,2),
    
    -- Metadata
    metadata JSONB
);

-- ============================================================================
-- STEP 5: Create ML pattern discoveries table
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_pattern_discoveries (
    id SERIAL PRIMARY KEY,
    discovery_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    pattern_type VARCHAR(100) NOT NULL,
    
    -- Pattern details
    securities_involved JSONB NOT NULL,
    pattern_confidence DECIMAL(5,2),
    
    -- What triggered this pattern
    trigger_conditions JSONB,
    news_correlation JSONB,
    market_conditions JSONB,
    
    -- Predictive value
    predicted_outcome JSONB,
    actual_outcome JSONB,
    pattern_success BOOLEAN,
    
    -- For ML training
    feature_vector JSONB,
    model_version VARCHAR(50),
    
    -- Tracking
    last_verified TIMESTAMPTZ,
    verification_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_patterns_type ON ml_pattern_discoveries(pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_date ON ml_pattern_discoveries(discovery_date DESC);

-- ============================================================================
-- STEP 6: Create correlation tracking table
-- ============================================================================
CREATE TABLE IF NOT EXISTS security_correlations (
    id SERIAL PRIMARY KEY,
    symbol1 VARCHAR(10) NOT NULL,
    symbol2 VARCHAR(10) NOT NULL,
    correlation_period VARCHAR(20),  -- '1h', '1d', '5d'
    
    -- Correlation metrics
    correlation_coefficient DECIMAL(5,4),
    cointegration_score DECIMAL(5,4),
    
    -- Context
    catalyst_similarity DECIMAL(5,2),
    sector_match BOOLEAN,
    
    calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(symbol1, symbol2, correlation_period, calculated_at)
);

CREATE INDEX IF NOT EXISTS idx_corr_symbols ON security_correlations(symbol1, symbol2);
CREATE INDEX IF NOT EXISTS idx_corr_strength ON security_correlations(correlation_coefficient);

-- ============================================================================
-- STEP 7: Create performance metrics table
-- ============================================================================
CREATE TABLE IF NOT EXISTS tracking_performance_metrics (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    
    -- Tracking metrics
    total_securities_tracked INTEGER,
    active_high_freq INTEGER,
    active_medium_freq INTEGER,
    archived_count INTEGER,
    
    -- Data collection metrics
    total_data_points INTEGER,
    storage_gb_used DECIMAL(8,2),
    
    -- Pattern discovery metrics
    patterns_discovered INTEGER,
    patterns_validated INTEGER,
    pattern_success_rate DECIMAL(5,2),
    
    -- Value metrics
    profitable_patterns INTEGER,
    estimated_pattern_value DECIMAL(10,2),
    
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- STEP 8: Create analysis views
-- ============================================================================

-- View for active securities with recent data
CREATE OR REPLACE VIEW v_active_securities AS
SELECT 
    t.symbol,
    t.collection_frequency,
    t.last_catalyst_score,
    t.data_points_collected,
    t.last_updated,
    EXTRACT(HOURS FROM (NOW() - t.last_updated)) as hours_since_update,
    s.close as last_price,
    s.volume as last_volume,
    s.rsi_14,
    s.catalyst_active
FROM security_tracking_state t
LEFT JOIN LATERAL (
    SELECT * FROM security_data_high_freq 
    WHERE symbol = t.symbol 
    ORDER BY timestamp DESC 
    LIMIT 1
) s ON true
WHERE t.collection_frequency != 'archive';

-- View for pattern discovery candidates
CREATE OR REPLACE VIEW v_pattern_candidates AS
WITH recent_movers AS (
    SELECT 
        symbol,
        AVG(close) as avg_price,
        STDDEV(close) as price_volatility,
        AVG(volume) as avg_volume,
        COUNT(*) as data_points
    FROM security_data_high_freq
    WHERE timestamp > NOW() - INTERVAL '24 hours'
    GROUP BY symbol
    HAVING COUNT(*) > 10
)
SELECT 
    rm.*,
    t.last_catalyst_score,
    t.collection_frequency
FROM recent_movers rm
JOIN security_tracking_state t ON rm.symbol = t.symbol
WHERE rm.price_volatility > 0.02
ORDER BY rm.price_volatility DESC;

-- ============================================================================
-- STEP 9: Log migration completion
-- ============================================================================
INSERT INTO tracking_performance_metrics (
    date,
    total_securities_tracked,
    active_high_freq,
    active_medium_freq,
    archived_count,
    total_data_points,
    storage_gb_used,
    patterns_discovered,
    patterns_validated,
    pattern_success_rate,
    profitable_patterns,
    estimated_pattern_value
) VALUES (
    CURRENT_DATE,
    0,  -- Will be updated by first scan
    0,
    0,
    0,
    0,
    0.0,
    0,
    0,
    0.0,
    0,
    0.0
);

-- Commit transaction
COMMIT;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Check that all tables were created
SELECT 
    table_name,
    pg_size_pretty(pg_total_relation_size(table_schema||'.'||table_name)) as size
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN (
    'security_data_high_freq',
    'security_data_hourly',
    'security_data_daily',
    'security_tracking_state',
    'ml_pattern_discoveries',
    'security_correlations',
    'tracking_performance_metrics'
)
ORDER BY table_name;

-- Check indexes
SELECT 
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) as index_size
FROM pg_indexes
WHERE schemaname = 'public'
AND tablename LIKE 'security_%' OR tablename LIKE 'ml_%' OR tablename LIKE 'tracking_%'
ORDER BY tablename, indexname;