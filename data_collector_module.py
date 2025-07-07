# ============================================================================
# Name of Application: Catalyst Trading System
# Name of file: data_collection_manager.py
# Version: 1.0.0
# Last Updated: 2025-01-27
# Purpose: Manage data collection and aging for top 100 securities
# 
# REVISION HISTORY:
# v1.0.0 (2025-01-27) - Initial implementation
# - Intelligent data collection frequencies
# - Data aging logic
# - Storage management
# - Pattern discovery preparation
# 
# Description of Service:
# This module manages the collection and storage of data for the top 100
# tracked securities. It implements intelligent aging to manage storage
# while maximizing data available for pattern discovery.
# ============================================================================

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor, execute_batch
import pandas as pd
import numpy as np
from database_utils import get_logger


class DataCollectionManager:
    """Manages data collection and aging for tracked securities"""
    
    def __init__(self, db_pool: ThreadedConnectionPool):
        self.db_pool = db_pool
        self.logger = get_logger().bind(module='data_collection_manager')
        
        # Collection intervals (minutes)
        self.collection_intervals = {
            'ultra_high': 1,    # Top 5
            'high_freq': 15,    # Top 20
            'medium_freq': 60,  # Top 50
            'low_freq': 360,    # Top 100
            'archive': 1440     # Daily
        }
        
        # Aging thresholds
        self.aging_criteria = {
            'volatility_threshold': 0.02,      # 2% movement
            'volume_threshold': 2.0,           # 2x average volume
            'news_threshold': 5,               # 5+ news mentions
            'flatline_periods': 4,             # Consecutive low activity
            'catalyst_decay_hours': 48,        # Catalyst effect duration
            'archive_after_days': 7            # Move to archive
        }
        
    def should_collect_data(self, symbol: str, tracking_info: Dict) -> bool:
        """Determine if it's time to collect data for a security"""
        last_updated = tracking_info.get('last_updated')
        if not last_updated:
            return True
            
        frequency = tracking_info.get('collection_frequency', 'low_freq')
        interval_minutes = self.collection_intervals.get(frequency, 360)
        
        time_since_update = (datetime.now() - last_updated).total_seconds() / 60
        
        return time_since_update >= interval_minutes
        
    def store_high_frequency_data(self, symbol: str, data: Dict):
        """Store data in high-frequency table"""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO security_data_high_freq (
                        symbol, timestamp, 
                        open, high, low, close, volume,
                        bid_ask_spread, order_imbalance,
                        rsi_14, macd, macd_signal, 
                        bb_upper, bb_lower, vwap,
                        news_count, catalyst_active, catalyst_score
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    symbol, data.get('timestamp', datetime.now()),
                    data.get('open'), data.get('high'), 
                    data.get('low'), data.get('close'),
                    data.get('volume'),
                    data.get('bid_ask_spread'), data.get('order_imbalance'),
                    data.get('rsi'), data.get('macd'), data.get('macd_signal'),
                    data.get('bb_upper'), data.get('bb_lower'), data.get('vwap'),
                    data.get('news_count', 0),
                    data.get('catalyst_score', 0) > 20,
                    data.get('catalyst_score', 0)
                ))
                conn.commit()
                self.logger.debug(f"Stored high-freq data for {symbol}")
        except Exception as e:
            self.logger.error(f"Error storing data for {symbol}: {e}")
            conn.rollback()
        finally:
            self.db_pool.putconn(conn)
            
    def aggregate_to_hourly(self, symbol: str, start_time: datetime):
        """Aggregate high-frequency data to hourly"""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Aggregate last hour of data
                cursor.execute("""
                    INSERT INTO security_data_hourly (
                        symbol, timestamp,
                        open, high, low, close, volume, vwap,
                        volatility, volume_ratio, price_range,
                        rsi_14, trend_strength,
                        news_count, max_catalyst_score
                    )
                    SELECT 
                        symbol,
                        date_trunc('hour', timestamp) as hour,
                        (array_agg(open ORDER BY timestamp))[1] as open,
                        MAX(high) as high,
                        MIN(low) as low,
                        (array_agg(close ORDER BY timestamp DESC))[1] as close,
                        SUM(volume) as volume,
                        AVG(vwap) as vwap,
                        STDDEV(close) as volatility,
                        AVG(volume) / NULLIF((
                            SELECT AVG(volume) 
                            FROM security_data_high_freq 
                            WHERE symbol = %s 
                            AND timestamp > NOW() - INTERVAL '20 hours'
                        ), 0) as volume_ratio,
                        (MAX(high) - MIN(low)) / NULLIF(AVG(close), 0) as price_range,
                        AVG(rsi_14) as rsi_14,
                        NULL as trend_strength,  -- Calculate separately
                        SUM(news_count) as news_count,
                        MAX(catalyst_score) as max_catalyst_score
                    FROM security_data_high_freq
                    WHERE symbol = %s
                    AND timestamp >= %s
                    AND timestamp < %s + INTERVAL '1 hour'
                    GROUP BY symbol, date_trunc('hour', timestamp)
                    ON CONFLICT (symbol, timestamp) DO NOTHING
                """, (symbol, symbol, start_time, start_time))
                
                conn.commit()
                self.logger.debug(f"Aggregated hourly data for {symbol}")
        except Exception as e:
            self.logger.error(f"Error aggregating hourly data for {symbol}: {e}")
            conn.rollback()
        finally:
            self.db_pool.putconn(conn)
            
    def aggregate_to_daily(self, symbol: str, date: datetime):
        """Aggregate hourly data to daily"""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO security_data_daily (
                        symbol, date,
                        open, high, low, close, volume,
                        daily_return, volatility, dollar_volume,
                        pattern_readiness_score,
                        total_news_count, catalyst_events
                    )
                    SELECT 
                        symbol,
                        DATE(%s) as date,
                        (array_agg(open ORDER BY timestamp))[1] as open,
                        MAX(high) as high,
                        MIN(low) as low,
                        (array_agg(close ORDER BY timestamp DESC))[1] as close,
                        SUM(volume) as volume,
                        ((array_agg(close ORDER BY timestamp DESC))[1] - 
                         (array_agg(open ORDER BY timestamp))[1]) / 
                         NULLIF((array_agg(open ORDER BY timestamp))[1], 0) as daily_return,
                        STDDEV(close) as volatility,
                        SUM(volume * close) as dollar_volume,
                        NULL as pattern_readiness_score,  -- Calculate separately
                        SUM(news_count) as total_news_count,
                        '[]'::jsonb as catalyst_events  -- Aggregate separately
                    FROM security_data_hourly
                    WHERE symbol = %s
                    AND timestamp >= %s
                    AND timestamp < %s + INTERVAL '1 day'
                    GROUP BY symbol
                    ON CONFLICT (symbol, date) DO UPDATE SET
                        high = GREATEST(security_data_daily.high, EXCLUDED.high),
                        low = LEAST(security_data_daily.low, EXCLUDED.low),
                        close = EXCLUDED.close,
                        volume = security_data_daily.volume + EXCLUDED.volume
                """, (date, symbol, date, date))
                
                conn.commit()
                self.logger.debug(f"Aggregated daily data for {symbol}")
        except Exception as e:
            self.logger.error(f"Error aggregating daily data for {symbol}: {e}")
            conn.rollback()
        finally:
            self.db_pool.putconn(conn)
            
    def calculate_aging_metrics(self, symbol: str, tracking_info: Dict) -> Dict:
        """Calculate metrics to determine if security should be aged"""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get recent activity metrics
                cursor.execute("""
                    WITH recent_data AS (
                        SELECT 
                            symbol,
                            close,
                            volume,
                            news_count,
                            catalyst_score,
                            timestamp
                        FROM security_data_high_freq
                        WHERE symbol = %s
                        AND timestamp > NOW() - INTERVAL '24 hours'
                        ORDER BY timestamp DESC
                    ),
                    metrics AS (
                        SELECT 
                            STDDEV(close) / NULLIF(AVG(close), 0) as volatility,
                            AVG(volume) as avg_volume,
                            MAX(volume) / NULLIF(AVG(volume), 0) as max_volume_ratio,
                            SUM(news_count) as total_news,
                            MAX(catalyst_score) as max_catalyst_score,
                            COUNT(DISTINCT DATE(timestamp)) as active_days
                        FROM recent_data
                    )
                    SELECT * FROM metrics
                """, (symbol,))
                
                metrics = cursor.fetchone()
                
                if not metrics:
                    return {
                        'volatility': 0,
                        'relative_volume': 1.0,
                        'recent_news_count': 0,
                        'technical_signal_strength': 0,
                        'flatline_score': 10,
                        'hours_since_last_catalyst': 999
                    }
                    
                # Calculate flatline score
                cursor.execute("""
                    SELECT COUNT(*) as flatline_count
                    FROM (
                        SELECT 
                            timestamp,
                            ABS(close - LAG(close) OVER (ORDER BY timestamp)) / 
                            NULLIF(close, 0) as price_change
                        FROM security_data_high_freq
                        WHERE symbol = %s
                        AND timestamp > NOW() - INTERVAL '4 hours'
                    ) t
                    WHERE price_change < 0.001
                """, (symbol,))
                
                flatline_result = cursor.fetchone()
                
                return {
                    'volatility': float(metrics['volatility'] or 0),
                    'relative_volume': float(metrics['max_volume_ratio'] or 1.0),
                    'recent_news_count': int(metrics['total_news'] or 0),
                    'technical_signal_strength': 0.5,  # Placeholder
                    'flatline_score': int(flatline_result['flatline_count'] or 0),
                    'hours_since_last_catalyst': self._calculate_hours_since_catalyst(symbol)
                }
                
        finally:
            self.db_pool.putconn(conn)
            
    def _calculate_hours_since_catalyst(self, symbol: str) -> float:
        """Calculate hours since last significant catalyst"""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT MAX(timestamp) as last_catalyst_time
                    FROM security_data_high_freq
                    WHERE symbol = %s
                    AND catalyst_score > 50
                    AND timestamp > NOW() - INTERVAL '7 days'
                """, (symbol,))
                
                result = cursor.fetchone()
                if result and result[0]:
                    return (datetime.now() - result[0]).total_seconds() / 3600
                    
                return 999  # No recent catalyst
                
        finally:
            self.db_pool.putconn(conn)
            
    def should_promote_frequency(self, metrics: Dict) -> bool:
        """Check if security needs more frequent monitoring"""
        # High volatility
        if metrics['volatility'] > self.aging_criteria['volatility_threshold']:
            return True
            
        # Volume surge
        if metrics['relative_volume'] > self.aging_criteria['volume_threshold']:
            return True
            
        # News catalyst
        if metrics['recent_news_count'] > self.aging_criteria['news_threshold']:
            return True
            
        # Technical breakout
        if metrics.get('technical_signal_strength', 0) > 0.7:
            return True
            
        return False
        
    def should_demote_frequency(self, metrics: Dict) -> bool:
        """Check if security can be monitored less frequently"""
        # Flatlined price action
        if metrics['flatline_score'] > self.aging_criteria['flatline_periods']:
            return True
            
        # Low volume
        if metrics['relative_volume'] < 0.5:
            return True
            
        # Old catalyst
        if metrics['hours_since_last_catalyst'] > self.aging_criteria['catalyst_decay_hours']:
            return True
            
        return False
        
    def update_collection_frequency(self, symbol: str, current_freq: str, 
                                  metrics: Dict) -> Optional[str]:
        """Determine new collection frequency based on metrics"""
        if self.should_promote_frequency(metrics):
            # Promote to higher frequency
            if current_freq == 'low_freq':
                return 'medium_freq'
            elif current_freq == 'medium_freq':
                return 'high_freq'
            elif current_freq == 'high_freq':
                return 'ultra_high'
                
        elif self.should_demote_frequency(metrics):
            # Demote to lower frequency
            if current_freq == 'ultra_high':
                return 'high_freq'
            elif current_freq == 'high_freq':
                return 'medium_freq'
            elif current_freq == 'medium_freq':
                return 'low_freq'
            elif current_freq == 'low_freq':
                # Check if should archive
                if metrics['hours_since_last_catalyst'] > 168:  # 7 days
                    return 'archive'
                    
        return None  # No change
        
    def cleanup_old_data(self, retention_days: Dict[str, int]):
        """Clean up old data based on retention policies"""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Clean high-frequency data older than retention
                cursor.execute("""
                    DELETE FROM security_data_high_freq
                    WHERE timestamp < NOW() - INTERVAL '%s days'
                """, (retention_days.get('high_freq', 7),))
                
                high_freq_deleted = cursor.rowcount
                
                # Clean hourly data
                cursor.execute("""
                    DELETE FROM security_data_hourly
                    WHERE timestamp < NOW() - INTERVAL '%s days'
                """, (retention_days.get('hourly', 30),))
                
                hourly_deleted = cursor.rowcount
                
                # Daily data typically kept longer
                cursor.execute("""
                    DELETE FROM security_data_daily
                    WHERE date < NOW() - INTERVAL '%s days'
                """, (retention_days.get('daily', 365),))
                
                daily_deleted = cursor.rowcount
                
                conn.commit()
                
                self.logger.info(
                    "Data cleanup complete",
                    high_freq_deleted=high_freq_deleted,
                    hourly_deleted=hourly_deleted,
                    daily_deleted=daily_deleted
                )
                
        except Exception as e:
            self.logger.error(f"Error cleaning up data: {e}")
            conn.rollback()
        finally:
            self.db_pool.putconn(conn)
            
    def get_storage_metrics(self) -> Dict:
        """Get current storage usage metrics"""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        'security_data_high_freq' as table_name,
                        pg_size_pretty(pg_total_relation_size('security_data_high_freq')) as size,
                        COUNT(*) as row_count
                    FROM security_data_high_freq
                    UNION ALL
                    SELECT 
                        'security_data_hourly' as table_name,
                        pg_size_pretty(pg_total_relation_size('security_data_hourly')) as size,
                        COUNT(*) as row_count
                    FROM security_data_hourly
                    UNION ALL
                    SELECT 
                        'security_data_daily' as table_name,
                        pg_size_pretty(pg_total_relation_size('security_data_daily')) as size,
                        COUNT(*) as row_count
                    FROM security_data_daily
                """)
                
                results = cursor.fetchall()
                
                return {
                    'tables': results,
                    'total_size': self._get_total_size(),
                    'timestamp': datetime.now().isoformat()
                }
                
        finally:
            self.db_pool.putconn(conn)
            
    def _get_total_size(self) -> str:
        """Get total database size"""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT pg_size_pretty(
                        pg_database_size(current_database())
                    ) as total_size
                """)
                result = cursor.fetchone()
                return result[0] if result else "Unknown"
        finally:
            self.db_pool.putconn(conn)