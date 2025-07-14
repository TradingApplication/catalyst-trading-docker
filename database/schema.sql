-- Catalyst Trading System PostgreSQL Schema
-- Designed for ML-ready data collection and analysis

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb"; -- If available on DO

-- Create schema
CREATE SCHEMA IF NOT EXISTS trading;

-- =====================================================
-- CORE TRADING TABLES
-- =====================================================

-- Trades table (executed trades)
CREATE TABLE trading.trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    trade_type VARCHAR(10) NOT NULL CHECK (trade_type IN ('BUY', 'SELL', 'SHORT', 'COVER')),
    quantity INTEGER NOT NULL,
    price DECIMAL(10,4) NOT NULL,
    total_value DECIMAL(12,4) NOT NULL,
    commission DECIMAL(8,4) DEFAULT 0,
    alpaca_order_id VARCHAR(100) UNIQUE,
    executed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    strategy_name VARCHAR(50),
    pattern_detected VARCHAR(50),
    confidence_score DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Positions table (current holdings)
CREATE TABLE trading.positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL UNIQUE,
    quantity INTEGER NOT NULL,
    avg_entry_price DECIMAL(10,4) NOT NULL,
    current_price DECIMAL(10,4),
    market_value DECIMAL(12,4),
    unrealized_pnl DECIMAL(12,4),
    unrealized_pnl_pct DECIMAL(8,4),
    entry_date TIMESTAMP WITH TIME ZONE NOT NULL,
    entry_pattern VARCHAR(50),
    stop_loss DECIMAL(10,4),
    take_profit DECIMAL(10,4),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Scanner results (stocks meeting criteria)
CREATE TABLE trading.scanner_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    scan_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    price DECIMAL(10,4) NOT NULL,
    volume BIGINT NOT NULL,
    relative_volume DECIMAL(6,2),
    price_change_pct DECIMAL(8,4),
    
    -- Technical indicators
    rsi_14 DECIMAL(5,2),
    bb_position DECIMAL(5,4), -- Position within Bollinger Bands (0-1)
    macd_signal VARCHAR(10), -- 'BULLISH', 'BEARISH', 'NEUTRAL'
    sma_20 DECIMAL(10,4),
    sma_50 DECIMAL(10,4),
    
    -- Pattern detection
    pattern_detected VARCHAR(50),
    pattern_confidence DECIMAL(5,4),
    
    -- Market context
    market_cap BIGINT,
    sector VARCHAR(50),
    
    -- Scoring
    overall_score DECIMAL(5,2),
    meets_criteria BOOLEAN DEFAULT FALSE,
    
    INDEX idx_scanner_timestamp (scan_timestamp DESC),
    INDEX idx_scanner_symbol (symbol, scan_timestamp DESC)
);

-- =====================================================
-- ML TRAINING DATA TABLES
-- =====================================================

-- Pattern training data (for ML model development)
CREATE TABLE trading.pattern_training_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    pattern_type VARCHAR(50) NOT NULL,
    
    -- Pattern features (candlestick characteristics)
    open_price DECIMAL(10,4) NOT NULL,
    high_price DECIMAL(10,4) NOT NULL,
    low_price DECIMAL(10,4) NOT NULL,
    close_price DECIMAL(10,4) NOT NULL,
    volume BIGINT NOT NULL,
    
    body_to_range_ratio DECIMAL(5,4),
    upper_shadow_ratio DECIMAL(5,4),
    lower_shadow_ratio DECIMAL(5,4),
    volume_surge DECIMAL(6,2), -- Volume vs 20-day average
    
    -- Technical context at pattern detection
    rsi_14 DECIMAL(5,2),
    rsi_divergence BOOLEAN DEFAULT FALSE,
    bb_position DECIMAL(5,4),
    bb_width DECIMAL(5,4),
    macd_value DECIMAL(10,4),
    macd_signal DECIMAL(10,4),
    macd_histogram DECIMAL(10,4),
    
    -- Market context
    market_trend VARCHAR(20), -- 'UPTREND', 'DOWNTREND', 'SIDEWAYS'
    vix_level DECIMAL(5,2),
    spy_correlation DECIMAL(5,4),
    sector_strength DECIMAL(5,2),
    
    -- Catalyst information
    news_catalyst BOOLEAN DEFAULT FALSE,
    news_sentiment DECIMAL(3,2), -- -1 to 1
    news_relevance DECIMAL(3,2), -- 0 to 1
    earnings_proximity INTEGER, -- Days to earnings
    
    -- Outcome tracking (filled after pattern completion)
    pattern_completed BOOLEAN DEFAULT FALSE,
    pattern_success BOOLEAN,
    max_gain DECIMAL(6,4), -- Maximum % gain after pattern
    max_loss DECIMAL(6,4), -- Maximum % loss after pattern
    time_to_max_gain INTEGER, -- Minutes to max gain
    time_to_max_loss INTEGER, -- Minutes to max loss
    optimal_hold_time INTEGER, -- Minutes for best risk/reward
    final_outcome_pct DECIMAL(6,4), -- % change at pattern completion
    
    INDEX idx_pattern_timestamp (timestamp DESC),
    INDEX idx_pattern_symbol (symbol, timestamp DESC),
    INDEX idx_pattern_type (pattern_type, pattern_success)
);

-- Feature engineering aggregates (pre-computed for ML)
CREATE TABLE trading.ml_features_5min (
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Price features
    price_sma_20 DECIMAL(10,4),
    price_sma_50 DECIMAL(10,4),
    price_ema_12 DECIMAL(10,4),
    price_ema_26 DECIMAL(10,4),
    
    -- Volume features
    volume_sma_20 BIGINT,
    volume_ratio DECIMAL(6,2),
    
    -- Volatility features
    atr_14 DECIMAL(10,4),
    bb_upper DECIMAL(10,4),
    bb_lower DECIMAL(10,4),
    bb_width_pct DECIMAL(5,2),
    
    -- Momentum features
    rsi_14 DECIMAL(5,2),
    stoch_k DECIMAL(5,2),
    stoch_d DECIMAL(5,2),
    williams_r DECIMAL(5,2),
    
    -- Trend features
    adx_14 DECIMAL(5,2),
    plus_di DECIMAL(5,2),
    minus_di DECIMAL(5,2),
    
    PRIMARY KEY (symbol, timestamp),
    INDEX idx_ml_features_timestamp (timestamp DESC)
);

-- =====================================================
-- NEWS & CATALYST TABLES
-- =====================================================

-- News articles
CREATE TABLE trading.news_articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    headline TEXT NOT NULL,
    summary TEXT,
    source VARCHAR(100),
    url TEXT,
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Sentiment analysis
    sentiment_score DECIMAL(3,2), -- -1 to 1
    sentiment_confidence DECIMAL(3,2), -- 0 to 1
    
    -- Impact assessment
    relevance_score DECIMAL(3,2), -- 0 to 1
    potential_impact VARCHAR(20), -- 'HIGH', 'MEDIUM', 'LOW'
    
    -- ML features
    headline_entities JSONB, -- Extracted entities
    key_phrases JSONB, -- Important phrases
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    INDEX idx_news_symbol (symbol, published_at DESC),
    INDEX idx_news_published (published_at DESC)
);

-- =====================================================
-- PERFORMANCE & ANALYTICS TABLES
-- =====================================================

-- Daily performance metrics
CREATE TABLE trading.daily_performance (
    date DATE PRIMARY KEY,
    starting_balance DECIMAL(12,2),
    ending_balance DECIMAL(12,2),
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    gross_profit DECIMAL(10,2) DEFAULT 0,
    gross_loss DECIMAL(10,2) DEFAULT 0,
    net_profit DECIMAL(10,2) DEFAULT 0,
    win_rate DECIMAL(5,2),
    profit_factor DECIMAL(5,2),
    largest_win DECIMAL(10,2),
    largest_loss DECIMAL(10,2),
    commissions DECIMAL(8,2) DEFAULT 0,
    
    -- Pattern performance
    best_pattern VARCHAR(50),
    worst_pattern VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ML model performance tracking
CREATE TABLE trading.ml_model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50), -- 'PATTERN_DETECTION', 'OUTCOME_PREDICTION', etc.
    
    -- Training metrics
    training_date TIMESTAMP WITH TIME ZONE NOT NULL,
    training_samples INTEGER,
    feature_count INTEGER,
    
    -- Performance metrics
    accuracy DECIMAL(5,4),
    precision DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    auc_roc DECIMAL(5,4),
    
    -- Trading performance when live
    live_trades INTEGER DEFAULT 0,
    live_win_rate DECIMAL(5,2),
    live_profit_factor DECIMAL(5,2),
    
    -- Model metadata
    parameters JSONB,
    feature_importance JSONB,
    
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- SYSTEM & CONFIGURATION TABLES
-- =====================================================

-- Service health monitoring
CREATE TABLE trading.service_health (
    service_name VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    status VARCHAR(20) NOT NULL, -- 'HEALTHY', 'DEGRADED', 'DOWN'
    response_time_ms INTEGER,
    memory_usage_mb INTEGER,
    cpu_usage_pct DECIMAL(5,2),
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    
    PRIMARY KEY (service_name, timestamp),
    INDEX idx_health_timestamp (timestamp DESC)
);

-- Configuration parameters
CREATE TABLE trading.system_config (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_by VARCHAR(50) DEFAULT 'system'
);

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- Current portfolio view
CREATE VIEW trading.portfolio_summary AS
SELECT 
    p.symbol,
    p.quantity,
    p.avg_entry_price,
    p.current_price,
    p.market_value,
    p.unrealized_pnl,
    p.unrealized_pnl_pct,
    p.entry_date,
    p.entry_pattern,
    EXTRACT(EPOCH FROM (NOW() - p.entry_date))/3600 as hours_held
FROM trading.positions p
WHERE p.quantity > 0
ORDER BY p.unrealized_pnl_pct DESC;

-- Pattern success rates
CREATE VIEW trading.pattern_success_rates AS
SELECT 
    pattern_type,
    COUNT(*) as total_patterns,
    COUNT(*) FILTER (WHERE pattern_success = true) as successful_patterns,
    ROUND(COUNT(*) FILTER (WHERE pattern_success = true)::DECIMAL / COUNT(*) * 100, 2) as success_rate,
    AVG(max_gain) as avg_max_gain,
    AVG(max_loss) as avg_max_loss,
    AVG(optimal_hold_time) as avg_hold_minutes
FROM trading.pattern_training_data
WHERE pattern_completed = true
GROUP BY pattern_type
ORDER BY success_rate DESC;

-- =====================================================
-- INITIAL DATA & CONFIGURATION
-- =====================================================

-- Insert default configuration
INSERT INTO trading.system_config (key, value, description) VALUES
('max_position_size', '0.02', 'Maximum position size as fraction of portfolio'),
('max_daily_trades', '10', 'Maximum trades allowed per day'),
('min_pattern_confidence', '0.65', 'Minimum confidence score for pattern trading'),
('stop_loss_pct', '0.02', 'Default stop loss percentage'),
('take_profit_pct', '0.04', 'Default take profit percentage'),
('ml_model_version', '1.0', 'Current ML model version'),
('scanner_interval_seconds', '300', 'How often to run scanner'),
('news_check_interval_seconds', '600', 'How often to check news')
ON CONFLICT (key) DO NOTHING;

-- Create indexes for performance
CREATE INDEX idx_trades_symbol_time ON trading.trades(symbol, executed_at DESC);
CREATE INDEX idx_pattern_training_incomplete ON trading.pattern_training_data(pattern_completed) WHERE pattern_completed = false;
CREATE INDEX idx_scanner_results_recent ON trading.scanner_results(scan_timestamp) WHERE scan_timestamp > NOW() - INTERVAL '1 day';

-- Grant permissions (adjust based on your user setup)
GRANT ALL PRIVILEGES ON SCHEMA trading TO catalyst_app;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO catalyst_app;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO catalyst_app;