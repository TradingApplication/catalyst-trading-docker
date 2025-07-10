-- Create new table for comprehensive scanner market data
-- This is separate from the existing market_data table which has a different structure

CREATE TABLE IF NOT EXISTS scan_market_data (
    id BIGSERIAL PRIMARY KEY,
    scan_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    scan_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Price data
    price DECIMAL(10,2) NOT NULL,
    open_price DECIMAL(10,2),
    high_price DECIMAL(10,2),
    low_price DECIMAL(10,2),
    previous_close DECIMAL(10,2),
    
    -- Volume data
    volume BIGINT NOT NULL,
    average_volume BIGINT,
    relative_volume DECIMAL(5,2),
    dollar_volume DECIMAL(15,2),
    
    -- Change metrics
    price_change DECIMAL(10,2),
    price_change_pct DECIMAL(5,2),
    gap_pct DECIMAL(5,2),
    day_range_pct DECIMAL(5,2),
    
    -- Technical indicators (basic set)
    rsi_14 DECIMAL(5,2),
    sma_20 DECIMAL(10,2),
    sma_50 DECIMAL(10,2),
    vwap DECIMAL(10,2),
    
    -- Catalyst data
    has_news BOOLEAN DEFAULT FALSE,
    news_count INTEGER DEFAULT 0,
    catalyst_score DECIMAL(5,2),
    primary_catalyst VARCHAR(50),
    news_recency_hours DECIMAL(5,2),
    
    -- Ranking data
    scan_rank INTEGER CHECK (scan_rank BETWEEN 1 AND 100),
    made_top_20 BOOLEAN DEFAULT FALSE,
    made_top_5 BOOLEAN DEFAULT FALSE,
    selected_for_trading BOOLEAN DEFAULT FALSE,
    
    -- Market context
    market_cap BIGINT,
    sector VARCHAR(50),
    industry VARCHAR(100),
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Prevent duplicate entries for same symbol in same scan
    UNIQUE(scan_id, symbol)
);

-- Create indexes for performance
CREATE INDEX idx_scan_market_data_scan_id ON scan_market_data(scan_id);
CREATE INDEX idx_scan_market_data_symbol ON scan_market_data(symbol);
CREATE INDEX idx_scan_market_data_timestamp ON scan_market_data(scan_timestamp);
CREATE INDEX idx_scan_market_data_catalyst ON scan_market_data(has_news, catalyst_score DESC);
CREATE INDEX idx_scan_market_data_rank ON scan_market_data(scan_rank);
CREATE INDEX idx_scan_market_data_symbol_time ON scan_market_data(symbol, scan_timestamp DESC);

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON scan_market_data TO catalyst_app;
GRANT USAGE, SELECT ON SEQUENCE scan_market_data_id_seq TO catalyst_app;

-- Update the daily aggregation function to use the new table
CREATE OR REPLACE FUNCTION update_scan_market_data_daily()
RETURNS void AS $$
BEGIN
    INSERT INTO market_data_daily (
        symbol, date, scan_count, times_in_top_50, times_in_top_20, 
        times_in_top_5, times_traded, avg_catalyst_score, avg_relative_volume,
        avg_price_change_pct, day_open, day_high, day_low, day_close,
        total_news_count
    )
    SELECT 
        symbol,
        DATE(scan_timestamp) as date,
        COUNT(*) as scan_count,
        SUM(CASE WHEN scan_rank <= 50 THEN 1 ELSE 0 END) as times_in_top_50,
        SUM(CASE WHEN made_top_20 THEN 1 ELSE 0 END) as times_in_top_20,
        SUM(CASE WHEN made_top_5 THEN 1 ELSE 0 END) as times_in_top_5,
        SUM(CASE WHEN selected_for_trading THEN 1 ELSE 0 END) as times_traded,
        AVG(catalyst_score) as avg_catalyst_score,
        AVG(relative_volume) as avg_relative_volume,
        AVG(price_change_pct) as avg_price_change_pct,
        (array_agg(price ORDER BY scan_timestamp ASC))[1] as day_open,
        MAX(high_price) as day_high,
        MIN(low_price) as day_low,
        (array_agg(price ORDER BY scan_timestamp DESC))[1] as day_close,
        SUM(news_count) as total_news_count
    FROM scan_market_data
    WHERE DATE(scan_timestamp) = CURRENT_DATE
    GROUP BY symbol, DATE(scan_timestamp)
    ON CONFLICT (symbol, date) DO UPDATE SET
        scan_count = EXCLUDED.scan_count,
        times_in_top_50 = EXCLUDED.times_in_top_50,
        times_in_top_20 = EXCLUDED.times_in_top_20,
        times_in_top_5 = EXCLUDED.times_in_top_5,
        times_traded = EXCLUDED.times_traded,
        avg_catalyst_score = EXCLUDED.avg_catalyst_score,
        avg_relative_volume = EXCLUDED.avg_relative_volume,
        avg_price_change_pct = EXCLUDED.avg_price_change_pct,
        day_high = GREATEST(market_data_daily.day_high, EXCLUDED.day_high),
        day_low = LEAST(market_data_daily.day_low, EXCLUDED.day_low),
        day_close = EXCLUDED.day_close,
        total_news_count = EXCLUDED.total_news_count,
        updated_at = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

SELECT 'scan_market_data table created' as status;
SELECT 'Indexes created successfully' as status;
SELECT 'Aggregation function updated' as status;