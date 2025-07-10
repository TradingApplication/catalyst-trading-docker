-- Add the missing market_data_daily table and aggregation function

-- Create aggregated daily summary table
CREATE TABLE IF NOT EXISTS market_data_daily (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    
    -- Aggregated metrics
    scan_count INTEGER DEFAULT 0,
    times_in_top_50 INTEGER DEFAULT 0,
    times_in_top_20 INTEGER DEFAULT 0,
    times_in_top_5 INTEGER DEFAULT 0,
    times_traded INTEGER DEFAULT 0,
    
    -- Average scores
    avg_catalyst_score DECIMAL(5,2),
    avg_relative_volume DECIMAL(5,2),
    avg_price_change_pct DECIMAL(5,2),
    
    -- Price range
    day_open DECIMAL(10,2),
    day_high DECIMAL(10,2),
    day_low DECIMAL(10,2),
    day_close DECIMAL(10,2),
    
    -- News metrics
    total_news_count INTEGER DEFAULT 0,
    unique_catalysts JSONB,
    
    -- Performance if traded
    trade_count INTEGER DEFAULT 0,
    total_pnl DECIMAL(10,2),
    win_rate DECIMAL(5,2),
    
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_daily_symbol_date ON market_data_daily(symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_date ON market_data_daily(date DESC);

-- Function to update daily aggregates
CREATE OR REPLACE FUNCTION update_market_data_daily()
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
        COUNT(*) as times_in_top_50,
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
    FROM market_data
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

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON market_data_daily TO catalyst_app;
GRANT USAGE, SELECT ON SEQUENCE market_data_daily_id_seq TO catalyst_app;

-- Test the function exists
SELECT 'market_data_daily table created' as status;
SELECT 'update_market_data_daily function created' as status;