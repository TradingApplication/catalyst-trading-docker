-- Name of Application: Catalyst Trading System
-- Name of file: init.sql
-- Version: 1.0.0
-- Last Updated: 2025-01-03
-- Purpose: Initialize database tables for Catalyst Trading System

-- REVISION HISTORY:
-- v1.0.0 (2025-01-03) - Initial database schema
-- - Core trading tables
-- - Position tracking
-- - Scan results and patterns
-- - Performance indexes

-- Description of Service:
-- Database initialization script that creates all required tables
-- and indexes for the Catalyst Trading System

-- Create tables
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10,2),
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    pnl DECIMAL(10,2),
    commission DECIMAL(10,2),
    status VARCHAR(20),
    order_id VARCHAR(100),
    pattern_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    avg_price DECIMAL(10,2),
    current_price DECIMAL(10,2),
    unrealized_pnl DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scan_results (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    scan_type VARCHAR(50),
    score DECIMAL(5,2),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patterns (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    pattern_type VARCHAR(50),
    confidence DECIMAL(5,2),
    metadata JSONB,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(100) UNIQUE,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    order_type VARCHAR(20),
    limit_price DECIMAL(10,2),
    stop_price DECIMAL(10,2),
    status VARCHAR(20),
    filled_qty INTEGER DEFAULT 0,
    avg_fill_price DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trade_signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    signal_type VARCHAR(50),
    strength DECIMAL(5,2),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_date DATE NOT NULL,
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    total_pnl DECIMAL(10,2),
    win_rate DECIMAL(5,2),
    profit_factor DECIMAL(5,2),
    sharpe_ratio DECIMAL(5,2),
    max_drawdown DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_executed_at ON trades(executed_at);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_scan_results_created_at ON scan_results(created_at);
CREATE INDEX idx_patterns_symbol ON patterns(symbol);
CREATE INDEX idx_patterns_detected_at ON patterns(detected_at);
CREATE INDEX idx_orders_symbol ON orders(symbol);
CREATE INDEX idx_orders_status ON orders(status);

-- Grant permissions (adjust if needed)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ${DATABASE_USER};
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ${DATABASE_USER};