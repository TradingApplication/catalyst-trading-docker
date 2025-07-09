-- Add market data tables to existing database
-- Safe to run multiple times (IF NOT EXISTS)

CREATE TABLE IF NOT EXISTS market_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open DECIMAL(10,2) NOT NULL,
    high DECIMAL(10,2) NOT NULL,
    low DECIMAL(10,2) NOT NULL,
    close DECIMAL(10,2) NOT NULL,
    volume BIGINT NOT NULL,
    scan_id VARCHAR(50),
    is_trading_candidate BOOLEAN DEFAULT FALSE,
    has_news BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    aggregated_from VARCHAR(10),
    UNIQUE(symbol, timeframe, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_scan ON market_data(scan_id);
CREATE INDEX IF NOT EXISTS idx_market_data_created ON market_data(created_at);
CREATE INDEX IF NOT EXISTS idx_market_data_timeframe ON market_data(timeframe, created_at);

CREATE TABLE IF NOT EXISTS data_aggregation_log (
    id BIGSERIAL PRIMARY KEY,
    aggregation_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    source_timeframe VARCHAR(10),
    target_timeframe VARCHAR(10),
    date_range_start TIMESTAMPTZ,
    date_range_end TIMESTAMPTZ,
    records_processed INTEGER,
    records_created INTEGER,
    records_deleted INTEGER,
    duration_seconds DECIMAL(10,3),
    status VARCHAR(20),
    error_message TEXT
);

-- Verify tables created
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('market_data', 'data_aggregation_log');