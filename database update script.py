#!/usr/bin/env python3
"""
Database Update Script for Catalyst Trading System
Version: 2.1.3
Last Updated: 2025-07-12
Purpose: Create and update database tables for trading signal support

CRITICAL: This script ensures all necessary tables exist with proper structure
for the trading system to function correctly.
"""

import os
import sys
import json
import psycopg2
import psycopg2.extras
from datetime import datetime
from typing import Dict, List

# Database configuration from environment
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'database': os.getenv('DB_NAME', 'catalyst_trading'),
    'user': os.getenv('DB_USER', 'catalyst_user'),
    'password': os.getenv('DB_PASSWORD', 'catalyst_password'),
    'sslmode': os.getenv('DB_SSLMODE', 'prefer')
}

def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(**DATABASE_CONFIG)
        conn.cursor_factory = psycopg2.extras.RealDictCursor
        return conn
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        sys.exit(1)

def execute_sql(conn, sql: str, description: str):
    """Execute SQL with error handling"""
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
            print(f"✅ {description}")
    except Exception as e:
        conn.rollback()
        print(f"❌ Failed to {description}: {e}")
        raise

def create_trading_signals_table(conn):
    """Create or update trading_signals table - CRITICAL FOR TRADING"""
    sql = """
    CREATE TABLE IF NOT EXISTS trading_signals (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(10) NOT NULL,
        signal_type VARCHAR(20) NOT NULL,
        action VARCHAR(10) NOT NULL,
        entry_price DECIMAL(10,2),
        stop_loss DECIMAL(10,2),
        take_profit DECIMAL(10,2),
        confidence DECIMAL(5,2),
        risk_reward_ratio DECIMAL(5,2),
        catalyst_info JSONB,
        technical_info JSONB,
        executed BOOLEAN DEFAULT FALSE,
        execution_time TIMESTAMP,
        trade_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP,
        metadata JSONB
    );
    
    -- Create indexes for performance
    CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_executed 
    ON trading_signals(symbol, executed, created_at);
    
    CREATE INDEX IF NOT EXISTS idx_trading_signals_created_at 
    ON trading_signals(created_at);
    
    CREATE INDEX IF NOT EXISTS idx_trading_signals_expires_at 
    ON trading_signals(expires_at) WHERE expires_at IS NOT NULL;
    """
    
    execute_sql(conn, sql, "create trading_signals table with indexes")

def create_trade_records_table(conn):
    """Create or update trade_records table"""
    sql = """
    CREATE TABLE IF NOT EXISTS trade_records (
        id SERIAL PRIMARY KEY,
        signal_id INTEGER REFERENCES trading_signals(id),
        symbol VARCHAR(10) NOT NULL,
        order_id VARCHAR(50),
        side VARCHAR(10) NOT NULL,
        order_type VARCHAR(20) NOT NULL,
        quantity INTEGER NOT NULL,
        entry_price DECIMAL(10,2),
        stop_loss DECIMAL(10,2),
        take_profit DECIMAL(10,2),
        status VARCHAR(20) DEFAULT 'open',
        entry_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        exit_timestamp TIMESTAMP,
        exit_price DECIMAL(10,2),
        pnl DECIMAL(10,2),
        entry_reason TEXT,
        exit_reason TEXT,
        metadata JSONB
    );
    
    -- Create indexes for performance
    CREATE INDEX IF NOT EXISTS idx_trade_records_symbol_status 
    ON trade_records(symbol, status);
    
    CREATE INDEX IF NOT EXISTS idx_trade_records_entry_timestamp 
    ON trade_records(entry_timestamp);
    
    CREATE INDEX IF NOT EXISTS idx_trade_records_signal_id 
    ON trade_records(signal_id);
    """
    
    execute_sql(conn, sql, "create trade_records table with indexes")

def create_news_raw_table(conn):
    """Create or update news_raw table"""
    sql = """
    CREATE TABLE IF NOT EXISTS news_raw (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        content TEXT,
        source VARCHAR(100),
        url TEXT UNIQUE,
        published_timestamp TIMESTAMP,
        collected_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        sentiment_score DECIMAL(3,2),
        relevance_score DECIMAL(3,2),
        is_pre_market BOOLEAN DEFAULT FALSE,
        source_tier INTEGER DEFAULT 5,
        mentioned_tickers TEXT[],
        headline_keywords TEXT[],
        metadata JSONB
    );
    
    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_news_raw_collected_timestamp 
    ON news_raw(collected_timestamp);
    
    CREATE INDEX IF NOT EXISTS idx_news_raw_mentioned_tickers 
    ON news_raw USING GIN(mentioned_tickers);
    
    CREATE INDEX IF NOT EXISTS idx_news_raw_source_tier 
    ON news_raw(source_tier, collected_timestamp);
    """
    
    execute_sql(conn, sql, "create news_raw table with indexes")

def create_trading_candidates_table(conn):
    """Create or update trading_candidates table"""
    sql = """
    CREATE TABLE IF NOT EXISTS trading_candidates (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(10) NOT NULL,
        scan_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        catalyst_score DECIMAL(5,2),
        catalyst_type VARCHAR(50),
        catalyst_summary TEXT,
        price_data JSONB,
        volume_data JSONB,
        news_count INTEGER DEFAULT 0,
        sentiment_avg DECIMAL(3,2),
        relevance_avg DECIMAL(3,2),
        metadata JSONB
    );
    
    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_trading_candidates_scan_timestamp 
    ON trading_candidates(scan_timestamp);
    
    CREATE INDEX IF NOT EXISTS idx_trading_candidates_catalyst_score 
    ON trading_candidates(catalyst_score DESC);
    
    CREATE INDEX IF NOT EXISTS idx_trading_candidates_symbol 
    ON trading_candidates(symbol, scan_timestamp);
    """
    
    execute_sql(conn, sql, "create trading_candidates table with indexes")

def create_pattern_analysis_table(conn):
    """Create or update pattern_analysis table"""
    sql = """
    CREATE TABLE IF NOT EXISTS pattern_analysis (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(10) NOT NULL,
        analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        timeframe VARCHAR(10),
        pattern_name VARCHAR(50),
        pattern_confidence DECIMAL(5,2),
        pattern_signal VARCHAR(20),
        support_level DECIMAL(10,2),
        resistance_level DECIMAL(10,2),
        catalyst_alignment BOOLEAN DEFAULT FALSE,
        technical_confluence JSONB,
        metadata JSONB
    );
    
    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_pattern_analysis_symbol_timestamp 
    ON pattern_analysis(symbol, analysis_timestamp);
    
    CREATE INDEX IF NOT EXISTS idx_pattern_analysis_pattern_confidence 
    ON pattern_analysis(pattern_confidence DESC);
    """
    
    execute_sql(conn, sql, "create pattern_analysis table with indexes")

def create_workflow_log_table(conn):
    """Create workflow log table for tracking system operations"""
    sql = """
    CREATE TABLE IF NOT EXISTS workflow_log (
        id SERIAL PRIMARY KEY,
        cycle_id VARCHAR(50),
        step_name VARCHAR(50) NOT NULL,
        status VARCHAR(20) NOT NULL,
        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        end_time TIMESTAMP,
        duration_seconds INTEGER,
        input_data JSONB,
        output_data JSONB,
        error_message TEXT,
        metadata JSONB
    );
    
    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_workflow_log_cycle_id 
    ON workflow_log(cycle_id);
    
    CREATE INDEX IF NOT EXISTS idx_workflow_log_start_time 
    ON workflow_log(start_time);
    """
    
    execute_sql(conn, sql, "create workflow_log table with indexes")

def create_configuration_table(conn):
    """Create configuration table for system settings"""
    sql = """
    CREATE TABLE IF NOT EXISTS configuration (
        id SERIAL PRIMARY KEY,
        config_key VARCHAR(100) UNIQUE NOT NULL,
        config_value TEXT NOT NULL,
        config_type VARCHAR(20) DEFAULT 'string',
        description TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_by VARCHAR(50) DEFAULT 'system'
    );
    
    -- Insert default configurations
    INSERT INTO configuration (config_key, config_value, config_type, description) 
    VALUES 
        ('trading_enabled', 'true', 'boolean', 'Master switch for trading execution'),
        ('max_daily_trades', '10', 'integer', 'Maximum trades per day'),
        ('min_confidence', '60.0', 'float', 'Minimum signal confidence for trading'),
        ('max_position_size', '1000.0', 'float', 'Maximum position size in dollars'),
        ('risk_per_trade', '2.0', 'float', 'Maximum risk per trade as percentage')
    ON CONFLICT (config_key) DO NOTHING;
    """
    
    execute_sql(conn, sql, "create configuration table with defaults")

def verify_tables(conn):
    """Verify all tables exist and have expected structure"""
    required_tables = [
        'trading_signals',
        'trade_records', 
        'news_raw',
        'trading_candidates',
        'pattern_analysis',
        'workflow_log',
        'configuration'
    ]
    
    print("\n🔍 Verifying table structure...")
    
    with conn.cursor() as cur:
        # Check table existence
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """)
        
        existing_tables = [row['table_name'] for row in cur.fetchall()]
        
        missing_tables = set(required_tables) - set(existing_tables)
        if missing_tables:
            print(f"❌ Missing tables: {missing_tables}")
            return False
        
        # Check trading_signals structure (most critical)
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'trading_signals'
            ORDER BY ordinal_position
        """)
        
        columns = cur.fetchall()
        required_columns = [
            'id', 'symbol', 'signal_type', 'action', 'entry_price',
            'stop_loss', 'take_profit', 'confidence', 'executed'
        ]
        
        existing_columns = [col['column_name'] for col in columns]
        missing_columns = set(required_columns) - set(existing_columns)
        
        if missing_columns:
            print(f"❌ Missing columns in trading_signals: {missing_columns}")
            return False
        
        # Check for indexes
        cur.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename = 'trading_signals'
        """)
        
        indexes = [row['indexname'] for row in cur.fetchall()]
        if 'idx_trading_signals_symbol_executed' not in indexes:
            print("❌ Missing critical index: idx_trading_signals_symbol_executed")
            return False
    
    print("✅ All tables and indexes verified successfully")
    return True

def insert_test_data(conn):
    """Insert test data for verification"""
    print("\n🧪 Inserting test data...")
    
    # Test trading signal
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO trading_signals (
                symbol, signal_type, action, entry_price, stop_loss, 
                take_profit, confidence, risk_reward_ratio, 
                catalyst_info, technical_info, metadata
            ) VALUES (
                'TEST', 'LONG', 'BUY', 100.00, 98.00, 
                104.00, 75.50, 2.0,
                '{"type": "test", "score": 80}',
                '{"rsi": 45, "macd": 0.5}',
                '{"test": true, "created_by": "update_script"}'
            )
            ON CONFLICT DO NOTHING
            RETURNING id
        """)
        
        result = cur.fetchone()
        if result:
            signal_id = result['id']
            print(f"✅ Test trading signal created with ID: {signal_id}")
        else:
            print("ℹ️ Test signal already exists")
        
        # Verify we can retrieve pending signals
        cur.execute("""
            SELECT COUNT(*) as count 
            FROM trading_signals 
            WHERE executed = FALSE
        """)
        
        count = cur.fetchone()['count']
        print(f"✅ Found {count} pending signals")
        
        conn.commit()

def cleanup_test_data(conn):
    """Remove test data"""
    with conn.cursor() as cur:
        cur.execute("""
            DELETE FROM trading_signals 
            WHERE symbol = 'TEST' 
            AND metadata->>'test' = 'true'
        """)
        
        deleted = cur.rowcount
        if deleted > 0:
            print(f"🧹 Cleaned up {deleted} test records")
        
        conn.commit()

def main():
    """Main function to run database updates"""
    print("🚀 Catalyst Trading System - Database Update Script v2.1.3")
    print("=" * 60)
    
    # Test database connection
    print("📡 Connecting to database...")
    conn = get_db_connection()
    print(f"✅ Connected to {DATABASE_CONFIG['database']} at {DATABASE_CONFIG['host']}")
    
    try:
        # Create all tables
        print("\n📊 Creating/updating database tables...")
        create_trading_signals_table(conn)  # MOST CRITICAL
        create_trade_records_table(conn)
        create_news_raw_table(conn)
        create_trading_candidates_table(conn)
        create_pattern_analysis_table(conn)
        create_workflow_log_table(conn)
        create_configuration_table(conn)
        
        # Verify everything is correct
        if not verify_tables(conn):
            print("❌ Table verification failed!")
            sys.exit(1)
        
        # Test with sample data
        insert_test_data(conn)
        
        # Clean up test data
        cleanup_test_data(conn)
        
        print("\n" + "=" * 60)
        print("🎉 Database update completed successfully!")
        print("\n📋 Summary:")
        print("✅ trading_signals table ready for signal storage")
        print("✅ trade_records table ready for trade tracking")
        print("✅ All indexes created for optimal performance")
        print("✅ Configuration table populated with defaults")
        print("\n🔄 Next steps:")
        print("1. Update your .env file with Alpaca credentials")
        print("2. Set TRADING_ENABLED=true in docker-compose.yml")
        print("3. Restart services: docker-compose restart")
        print("4. Test trading endpoint: curl -X POST http://localhost:5005/execute_signals")
        
    except Exception as e:
        print(f"\n❌ Database update failed: {e}")
        sys.exit(1)
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()