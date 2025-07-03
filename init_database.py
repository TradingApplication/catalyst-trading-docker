#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: init_database.py
Version: 1.0.0
Last Updated: 2025-01-03
Purpose: Initialize database tables for Catalyst Trading System

REVISION HISTORY:
v1.0.0 (2025-01-03) - Initial database initialization script
- Creates all required tables
- Sets up indexes
- Can be run multiple times safely (idempotent)

Description of Service:
Standalone script to initialize the Catalyst Trading System database tables
in the DigitalOcean managed PostgreSQL instance.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def init_database():
    """Initialize all database tables"""
    
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        print("ERROR: DATABASE_URL not found in environment variables!")
        return False
    
    print(f"Connecting to database...")
    
    try:
        # Connect to database
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        print("Creating tables...")
        
        # Create tables
        sql_commands = [
            """
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
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS positions (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                quantity INTEGER NOT NULL,
                avg_price DECIMAL(10,2),
                current_price DECIMAL(10,2),
                unrealized_pnl DECIMAL(10,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS scan_results (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                scan_type VARCHAR(50),
                score DECIMAL(5,2),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS patterns (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                pattern_type VARCHAR(50),
                confidence DECIMAL(5,2),
                metadata JSONB,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            """
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
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS trade_signals (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                signal_type VARCHAR(50),
                strength DECIMAL(5,2),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            """
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
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS news_events (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10),
                headline TEXT,
                source VARCHAR(100),
                url TEXT,
                sentiment_score DECIMAL(3,2),
                published_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        # Execute each CREATE TABLE command
        for sql in sql_commands:
            cur.execute(sql)
            table_name = sql.split('EXISTS')[1].split('(')[0].strip()
            print(f"  ✓ Created table: {table_name}")
        
        print("\nCreating indexes...")
        
        # Create indexes
        index_commands = [
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_trades_executed_at ON trades(executed_at)",
            "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)",
            "CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_scan_results_created_at ON scan_results(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_patterns_symbol ON patterns(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_patterns_detected_at ON patterns(detected_at)",
            "CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
            "CREATE INDEX IF NOT EXISTS idx_news_events_symbol ON news_events(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_news_events_published_at ON news_events(published_at)"
        ]
        
        for sql in index_commands:
            cur.execute(sql)
            index_name = sql.split('INDEX')[1].split('ON')[0].strip()
            print(f"  ✓ Created index: {index_name}")
        
        # Commit changes
        conn.commit()
        
        print("\n✅ Database initialization completed successfully!")
        
        # List all tables
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """)
        
        tables = cur.fetchall()
        print(f"\nTables in database ({len(tables)} total):")
        for table in tables:
            print(f"  - {table[0]}")
        
        cur.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return False


if __name__ == "__main__":
    print("Catalyst Trading System - Database Initialization")
    print("=" * 50)
    
    # Check for DATABASE_URL
    if not os.getenv('DATABASE_URL'):
        print("\nERROR: DATABASE_URL environment variable not set!")
        print("Please ensure your .env file is in the current directory")
        exit(1)
    
    # Initialize database
    success = init_database()
    
    if success:
        print("\nDatabase is ready for use!")
    else:
        print("\nDatabase initialization failed!")
        exit(1)