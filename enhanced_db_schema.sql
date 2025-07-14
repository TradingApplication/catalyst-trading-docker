-- ============================================================================
-- CATALYST TRADING SYSTEM - ENHANCED DATABASE SCHEMA
-- Version 2.2.0 - Top 100 Securities Tracking
-- ============================================================================

-- High-frequency data table (15-minute intervals)
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
CREATE INDEX idx_hf_symbol_time ON security_data_high_freq(symbol, timestamp DESC);
CREATE INDEX idx_hf_timestamp ON security_data_high_freq(timestamp);
CREATE INDEX idx_hf_catalyst ON security_data_high_freq(catalyst_active) WHERE catalyst_active = true;

-- Hourly aggregated data
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

CREATE INDEX idx_hourly_symbol_time ON security_data_hourly(symbol, timestamp DESC);

-- Daily aggregated data
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

CREATE INDEX idx_daily_symbol_date ON security_data_daily(symbol, date DESC);

-- Tracking state table
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

-- Pattern discoveries table
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

CREATE INDEX idx_patterns_type ON ml_pattern_discoveries(pattern_type);
CREATE INDEX idx_patterns_date ON ml_pattern_discoveries(discovery_date DESC);

-- Correlation tracking table
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

CREATE INDEX idx_corr_symbols ON security_correlations(symbol1, symbol2);
CREATE INDEX idx_corr_strength ON security_correlations(correlation_coefficient);

-- Performance metrics for tracked securities
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

-- Create partitions for time-series data (optional but recommended)
-- This helps with query performance and data management

-- Example: Create monthly partitions for high-frequency data
CREATE TABLE security_data_high_freq_2025_07
    PARTITION OF security_data_high_freq
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');

-- Function to automatically create monthly partitions
CREATE OR REPLACE FUNCTION create_monthly_partitions()
RETURNS void AS $$
DECLARE
    start_date date;
    end_date date;
    partition_name text;
BEGIN
    -- Create partitions for next 3 months
    FOR i IN 0..2 LOOP
        start_date := date_trunc('month', CURRENT_DATE + (i || ' months')::interval);
        end_date := start_date + '1 month'::interval;
        partition_name := 'security_data_high_freq_' || to_char(start_date, 'YYYY_MM');
        
        -- Check if partition exists
        IF NOT EXISTS (
            SELECT 1 FROM pg_tables 
            WHERE tablename = partition_name
        ) THEN
            EXECUTE format(
                'CREATE TABLE %I PARTITION OF security_data_high_freq FOR VALUES FROM (%L) TO (%L)',
                partition_name, start_date, end_date
            );
            RAISE NOTICE 'Created partition %', partition_name;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Run the partition creation
SELECT create_monthly_partitions();

-- Create a scheduled job to create partitions (if pg_cron is available)
-- SELECT cron.schedule('create-partitions', '0 0 1 * *', 'SELECT create_monthly_partitions()');

-- ============================================================================
-- VIEWS FOR EASY ANALYSIS
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
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to get correlation matrix for a set of symbols
CREATE OR REPLACE FUNCTION get_correlation_matrix(
    p_symbols text[],
    p_period interval DEFAULT '1 day'
)
RETURNS TABLE(symbol1 text, symbol2 text, correlation numeric)
AS $$
BEGIN
    -- This is a placeholder - actual implementation would calculate
    -- correlations using window functions and statistical formulas
    RETURN QUERY
    SELECT 
        a.symbol::text as symbol1,
        b.symbol::text as symbol2,
        0.0::numeric as correlation  -- Placeholder
    FROM security_data_high_freq a
    CROSS JOIN security_data_high_freq b
    WHERE a.symbol = ANY(p_symbols)
    AND b.symbol = ANY(p_symbols)
    AND a.symbol < b.symbol
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;