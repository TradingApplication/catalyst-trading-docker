#!/usr/bin/env python3
"""
Catalyst Trading System - Data Migration Script
Migrates data from current app to new PostgreSQL database
"""

import os
import sys
import json
import psycopg2
from datetime import datetime
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataMigration:
    def __init__(self, source_data_path: str, db_url: str):
        """
        Initialize migration with source data path and target database URL
        """
        self.source_data_path = source_data_path
        self.db_url = db_url
        self.conn = None
        self.cursor = None
        
    def connect_db(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(self.db_url)
            self.cursor = self.conn.cursor()
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            sys.exit(1)
            
    def close_db(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            
    def migrate_trades(self, trades_file: str):
        """Migrate historical trades"""
        logger.info("Migrating trades...")
        
        try:
            with open(trades_file, 'r') as f:
                trades = json.load(f)
                
            for trade in trades:
                self.cursor.execute("""
                    INSERT INTO trading.trades (
                        symbol, trade_type, quantity, price, 
                        total_value, commission, alpaca_order_id,
                        executed_at, strategy_name, pattern_detected,
                        confidence_score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (alpaca_order_id) DO NOTHING
                """, (
                    trade['symbol'],
                    trade['trade_type'],
                    trade['quantity'],
                    trade['price'],
                    trade['total_value'],
                    trade.get('commission', 0),
                    trade.get('order_id'),
                    trade['executed_at'],
                    trade.get('strategy', 'unknown'),
                    trade.get('pattern'),
                    trade.get('confidence', 0.5)
                ))
                
            self.conn.commit()
            logger.info(f"Migrated {len(trades)} trades")
            
        except Exception as e:
            logger.error(f"Failed to migrate trades: {e}")
            self.conn.rollback()
            
    def migrate_positions(self, positions_file: str):
        """Migrate current positions"""
        logger.info("Migrating positions...")
        
        try:
            with open(positions_file, 'r') as f:
                positions = json.load(f)
                
            # Clear existing positions
            self.cursor.execute("DELETE FROM trading.positions")
            
            for position in positions:
                self.cursor.execute("""
                    INSERT INTO trading.positions (
                        symbol, quantity, avg_entry_price,
                        current_price, market_value,
                        unrealized_pnl, unrealized_pnl_pct,
                        entry_date, entry_pattern,
                        stop_loss, take_profit
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    position['symbol'],
                    position['quantity'],
                    position['avg_entry_price'],
                    position.get('current_price'),
                    position.get('market_value'),
                    position.get('unrealized_pnl', 0),
                    position.get('unrealized_pnl_pct', 0),
                    position['entry_date'],
                    position.get('pattern'),
                    position.get('stop_loss'),
                    position.get('take_profit')
                ))
                
            self.conn.commit()
            logger.info(f"Migrated {len(positions)} positions")
            
        except Exception as e:
            logger.error(f"Failed to migrate positions: {e}")
            self.conn.rollback()
            
    def migrate_scanner_results(self, scanner_file: str):
        """Migrate scanner history"""
        logger.info("Migrating scanner results...")
        
        try:
            with open(scanner_file, 'r') as f:
                scanner_results = json.load(f)
                
            for result in scanner_results:
                self.cursor.execute("""
                    INSERT INTO trading.scanner_results (
                        symbol, scan_timestamp, price, volume,
                        relative_volume, price_change_pct,
                        rsi_14, pattern_detected, pattern_confidence,
                        overall_score, meets_criteria
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    result['symbol'],
                    result['timestamp'],
                    result['price'],
                    result['volume'],
                    result.get('relative_volume'),
                    result.get('price_change_pct'),
                    result.get('rsi'),
                    result.get('pattern'),
                    result.get('confidence', 0.5),
                    result.get('score', 0),
                    result.get('meets_criteria', False)
                ))
                
            self.conn.commit()
            logger.info(f"Migrated {len(scanner_results)} scanner results")
            
        except Exception as e:
            logger.error(f"Failed to migrate scanner results: {e}")
            self.conn.rollback()
            
    def migrate_performance_data(self, performance_file: str):
        """Migrate performance history"""
        logger.info("Migrating performance data...")
        
        try:
            with open(performance_file, 'r') as f:
                performance_data = json.load(f)
                
            for day_data in performance_data:
                self.cursor.execute("""
                    INSERT INTO trading.daily_performance (
                        date, starting_balance, ending_balance,
                        total_trades, winning_trades, losing_trades,
                        gross_profit, gross_loss, net_profit,
                        win_rate, profit_factor
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date) DO UPDATE SET
                        ending_balance = EXCLUDED.ending_balance,
                        total_trades = EXCLUDED.total_trades,
                        net_profit = EXCLUDED.net_profit
                """, (
                    day_data['date'],
                    day_data['starting_balance'],
                    day_data['ending_balance'],
                    day_data.get('total_trades', 0),
                    day_data.get('winning_trades', 0),
                    day_data.get('losing_trades', 0),
                    day_data.get('gross_profit', 0),
                    day_data.get('gross_loss', 0),
                    day_data.get('net_profit', 0),
                    day_data.get('win_rate', 0),
                    day_data.get('profit_factor', 0)
                ))
                
            self.conn.commit()
            logger.info(f"Migrated {len(performance_data)} days of performance data")
            
        except Exception as e:
            logger.error(f"Failed to migrate performance data: {e}")
            self.conn.rollback()
            
    def verify_migration(self):
        """Verify data migration was successful"""
        logger.info("Verifying migration...")
        
        queries = [
            ("Trades", "SELECT COUNT(*) FROM trading.trades"),
            ("Positions", "SELECT COUNT(*) FROM trading.positions"),
            ("Scanner Results", "SELECT COUNT(*) FROM trading.scanner_results"),
            ("Performance Days", "SELECT COUNT(*) FROM trading.daily_performance")
        ]
        
        for name, query in queries:
            self.cursor.execute(query)
            count = self.cursor.fetchone()[0]
            logger.info(f"{name}: {count} records")
            
    def run_migration(self):
        """Run full migration"""
        logger.info("Starting data migration...")
        
        self.connect_db()
        
        try:
            # Migrate each data type
            if os.path.exists(f"{self.source_data_path}/trades.json"):
                self.migrate_trades(f"{self.source_data_path}/trades.json")
                
            if os.path.exists(f"{self.source_data_path}/positions.json"):
                self.migrate_positions(f"{self.source_data_path}/positions.json")
                
            if os.path.exists(f"{self.source_data_path}/scanner_results.json"):
                self.migrate_scanner_results(f"{self.source_data_path}/scanner_results.json")
                
            if os.path.exists(f"{self.source_data_path}/performance.json"):
                self.migrate_performance_data(f"{self.source_data_path}/performance.json")
                
            # Verify migration
            self.verify_migration()
            
            logger.info("Migration completed successfully!")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            
        finally:
            self.close_db()

if __name__ == "__main__":
    # Configuration
    SOURCE_DATA_PATH = "./data_export"  # Path to exported data
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    if not DATABASE_URL:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)
        
    # Run migration
    migrator = DataMigration(SOURCE_DATA_PATH, DATABASE_URL)
    migrator.run_migration()