#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: database_utils.py
Version: 2.3.6
Last Updated: 2025-07-12
Purpose: Database utility functions with complete trading signal support

REVISION HISTORY:
v2.3.6 (2025-07-12) - CRITICAL FIX: Complete trading signal support
- Added complete insert_trading_signal function with proper error handling
- Added get_pending_signals function with comprehensive filtering
- Added mark_signal_executed function for signal lifecycle management
- Enhanced error handling and logging throughout
- Fixed all import issues for trading services

v2.3.5 (2025-07-11) - Enhanced health check formatting
- Updated health_check to return v2.3.1 compatible format
- Added postgresql and redis specific status reporting
- Improved connection pool management

Description:
Complete database utility functions for the Catalyst Trading System.
Provides all database operations needed by all services with proper
error handling, connection pooling, and transaction management.

KEY FEATURES:
- Connection pooling for all services
- Complete trading signal lifecycle management
- Comprehensive error handling
- Transaction safety
- Redis caching support
- Health monitoring
"""

import os
import json
import time
import psycopg2
import psycopg2.extras
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from contextlib import contextmanager
from structlog import get_logger
import threading

# Initialize logger
logger = get_logger()

# Database configuration
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'database': os.getenv('DB_NAME', 'catalyst_trading'),
    'user': os.getenv('DB_USER', 'catalyst_user'),
    'password': os.getenv('DB_PASSWORD', 'catalyst_password'),
    'sslmode': os.getenv('DB_SSLMODE', 'prefer'),
    'connect_timeout': int(os.getenv('DB_CONNECT_TIMEOUT', '10')),
    'command_timeout': int(os.getenv('DB_COMMAND_TIMEOUT', '30'))
}

# Redis configuration
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', '6379')),
    'db': int(os.getenv('REDIS_DB', '0')),
    'password': os.getenv('REDIS_PASSWORD'),
    'socket_timeout': int(os.getenv('REDIS_TIMEOUT', '5')),
    'socket_connect_timeout': int(os.getenv('REDIS_CONNECT_TIMEOUT', '5'))
}

# Connection pools
_db_pool = None
_redis_client = None
_pool_lock = threading.Lock()

# Custom exceptions
class DatabaseError(Exception):
    """Custom database error"""
    pass

class ConnectionPoolError(Exception):
    """Connection pool error"""
    pass

def _create_connection_pool():
    """Create database connection pool"""
    global _db_pool
    
    if _db_pool is not None:
        return _db_pool
        
    try:
        import psycopg2.pool
        
        _db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            host=DATABASE_CONFIG['host'],
            port=DATABASE_CONFIG['port'],
            database=DATABASE_CONFIG['database'],
            user=DATABASE_CONFIG['user'],
            password=DATABASE_CONFIG['password'],
            sslmode=DATABASE_CONFIG['sslmode'],
            connect_timeout=DATABASE_CONFIG['connect_timeout']
        )
        
        logger.info("Database connection pool created")
        return _db_pool
        
    except Exception as e:
        logger.error("Failed to create connection pool", error=str(e))
        raise ConnectionPoolError(f"Failed to create connection pool: {str(e)}")

def _create_redis_client():
    """Create Redis client"""
    global _redis_client
    
    if _redis_client is not None:
        return _redis_client
        
    try:
        _redis_client = redis.Redis(
            host=REDIS_CONFIG['host'],
            port=REDIS_CONFIG['port'],
            db=REDIS_CONFIG['db'],
            password=REDIS_CONFIG['password'],
            socket_timeout=REDIS_CONFIG['socket_timeout'],
            socket_connect_timeout=REDIS_CONFIG['socket_connect_timeout'],
            decode_responses=True
        )
        
        # Test connection
        _redis_client.ping()
        
        logger.info("Redis client created")
        return _redis_client
        
    except Exception as e:
        logger.error("Failed to create Redis client", error=str(e))
        # Don't raise error for Redis, as it's not critical
        return None

@contextmanager
def get_db_connection():
    """Get database connection with proper error handling"""
    with _pool_lock:
        if _db_pool is None:
            _create_connection_pool()
    
    connection = None
    try:
        connection = _db_pool.getconn()
        if connection:
            # Set cursor factory for named results
            connection.cursor_factory = psycopg2.extras.RealDictCursor
            yield connection
        else:
            raise DatabaseError("Could not get connection from pool")
            
    except Exception as e:
        if connection:
            connection.rollback()
        logger.error("Database connection error", error=str(e))
        raise DatabaseError(f"Database operation failed: {str(e)}")
        
    finally:
        if connection and _db_pool:
            _db_pool.putconn(connection)

def get_redis():
    """Get Redis client"""
    with _pool_lock:
        if _redis_client is None:
            _create_redis_client()
    
    return _redis_client

def health_check() -> Dict:
    """Comprehensive health check - v2.3.1 compatible format"""
    health_status = {
        'postgresql': {'status': 'unhealthy', 'error': None},
        'redis': {'status': 'unhealthy', 'error': None},
        'timestamp': datetime.now().isoformat()
    }
    
    # Test PostgreSQL
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT 1')
                result = cur.fetchone()
                if result:
                    health_status['postgresql']['status'] = 'healthy'
                    
    except Exception as e:
        health_status['postgresql']['error'] = str(e)
        logger.error("PostgreSQL health check failed", error=str(e))
    
    # Test Redis
    try:
        redis_client = get_redis()
        if redis_client:
            redis_client.ping()
            health_status['redis']['status'] = 'healthy'
        else:
            health_status['redis']['error'] = 'Redis client not available'
            
    except Exception as e:
        health_status['redis']['error'] = str(e)
        logger.error("Redis health check failed", error=str(e))
    
    return health_status

# =============================================================================
# TRADING SIGNAL FUNCTIONS - CRITICAL FIX
# =============================================================================

def insert_trading_signal(signal_data: Dict) -> int:
    """
    Insert a trading signal into the database
    
    Args:
        signal_data: Dictionary with signal information
        
    Returns:
        ID of the inserted signal
        
    Raises:
        DatabaseError: If insertion fails
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Ensure table exists with all required columns
                cur.execute("""
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
                    )
                """)
                
                # Create index for performance
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_executed 
                    ON trading_signals(symbol, executed, created_at)
                """)
                
                # Insert signal
                cur.execute("""
                    INSERT INTO trading_signals (
                        symbol, signal_type, action, entry_price,
                        stop_loss, take_profit, confidence, risk_reward_ratio,
                        catalyst_info, technical_info, expires_at, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    signal_data.get('symbol'),
                    signal_data.get('signal_type'),
                    signal_data.get('action'),
                    signal_data.get('entry_price'),
                    signal_data.get('stop_loss'),
                    signal_data.get('take_profit'),
                    signal_data.get('confidence'),
                    signal_data.get('risk_reward_ratio'),
                    json.dumps(signal_data.get('catalyst_info', {})),
                    json.dumps(signal_data.get('technical_info', {})),
                    signal_data.get('expires_at'),
                    json.dumps(signal_data.get('metadata', {}))
                ))
                
                signal_id = cur.fetchone()['id']
                conn.commit()
                
                logger.info("Trading signal inserted", 
                           signal_id=signal_id, 
                           symbol=signal_data.get('symbol'),
                           action=signal_data.get('action'),
                           confidence=signal_data.get('confidence'))
                
                return signal_id
                
    except Exception as e:
        logger.error("Failed to insert trading signal", 
                    symbol=signal_data.get('symbol'), 
                    error=str(e))
        raise DatabaseError(f"Failed to insert trading signal: {str(e)}")

def get_pending_signals(limit: int = 10) -> List[Dict]:
    """
    Get pending trading signals that haven't been executed
    
    Args:
        limit: Maximum number of signals to return
        
    Returns:
        List of pending signal dictionaries
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Ensure table exists
                cur.execute("""
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
                    )
                """)
                
                cur.execute("""
                    SELECT 
                        id, symbol, signal_type, action, entry_price,
                        stop_loss, take_profit, confidence, risk_reward_ratio,
                        catalyst_info, technical_info, created_at, expires_at, metadata
                    FROM trading_signals 
                    WHERE executed = FALSE 
                    AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                    AND created_at > CURRENT_TIMESTAMP - INTERVAL '2 hours'
                    ORDER BY confidence DESC, created_at DESC
                    LIMIT %s
                """, (limit,))
                
                signals = []
                for row in cur.fetchall():
                    signal = dict(row)
                    # Convert timestamps to ISO format
                    if signal.get('created_at'):
                        signal['created_at'] = signal['created_at'].isoformat()
                    if signal.get('expires_at'):
                        signal['expires_at'] = signal['expires_at'].isoformat()
                    signals.append(signal)
                
                logger.info(f"Retrieved {len(signals)} pending signals")
                return signals
                
    except Exception as e:
        logger.error("Failed to get pending signals", error=str(e))
        return []

def mark_signal_executed(signal_id: int, trade_id: int) -> bool:
    """
    Mark a trading signal as executed
    
    Args:
        signal_id: ID of the signal
        trade_id: ID of the resulting trade
        
    Returns:
        True if updated successfully
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE trading_signals
                    SET executed = TRUE,
                        execution_time = CURRENT_TIMESTAMP,
                        trade_id = %s
                    WHERE id = %s
                """, (trade_id, signal_id))
                
                updated = cur.rowcount > 0
                conn.commit()
                
                if updated:
                    logger.info("Signal marked as executed", 
                               signal_id=signal_id, 
                               trade_id=trade_id)
                
                return updated
                
    except Exception as e:
        logger.error("Failed to mark signal executed", 
                    signal_id=signal_id, 
                    error=str(e))
        return False

# =============================================================================
# TRADE RECORD FUNCTIONS
# =============================================================================

def insert_trade_record(trade_data: Dict) -> int:
    """
    Insert a trade record into the database
    
    Args:
        trade_data: Dictionary with trade information
        
    Returns:
        ID of the inserted trade record
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Ensure table exists
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trade_records (
                        id SERIAL PRIMARY KEY,
                        signal_id INTEGER,
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
                    )
                """)
                
                # Insert trade record
                cur.execute("""
                    INSERT INTO trade_records (
                        signal_id, symbol, order_id, side, order_type,
                        quantity, entry_price, stop_loss, take_profit,
                        entry_reason, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    trade_data.get('signal_id'),
                    trade_data.get('symbol'),
                    trade_data.get('order_id'),
                    trade_data.get('side'),
                    trade_data.get('order_type'),
                    trade_data.get('quantity'),
                    trade_data.get('entry_price'),
                    trade_data.get('stop_loss'),
                    trade_data.get('take_profit'),
                    trade_data.get('entry_reason'),
                    json.dumps(trade_data.get('metadata', {}))
                ))
                
                trade_id = cur.fetchone()['id']
                conn.commit()
                
                logger.info("Trade record inserted", 
                           trade_id=trade_id,
                           symbol=trade_data.get('symbol'),
                           side=trade_data.get('side'))
                
                return trade_id
                
    except Exception as e:
        logger.error("Failed to insert trade record", 
                    symbol=trade_data.get('symbol'),
                    error=str(e))
        raise DatabaseError(f"Failed to insert trade record: {str(e)}")

def get_open_positions() -> List[Dict]:
    """
    Get all currently open trading positions
    
    Returns:
        List of open position dictionaries
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        id, symbol, signal_id, order_id, side, order_type,
                        quantity, entry_price, stop_loss, take_profit,
                        entry_timestamp, entry_reason, metadata,
                        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - entry_timestamp)) as position_age_seconds
                    FROM trade_records
                    WHERE status = 'open'
                    ORDER BY entry_timestamp DESC
                """)
                
                positions = []
                for row in cur.fetchall():
                    position = dict(row)
                    # Convert timestamps to ISO format
                    if position.get('entry_timestamp'):
                        position['entry_timestamp'] = position['entry_timestamp'].isoformat()
                    positions.append(position)
                
                logger.info(f"Retrieved {len(positions)} open positions")
                return positions
                
    except Exception as e:
        logger.error("Failed to get open positions", error=str(e))
        return []

def update_trade_exit(trade_id: int, exit_price: float, exit_reason: str) -> bool:
    """
    Update trade record with exit information
    
    Args:
        trade_id: ID of the trade record
        exit_price: Price at which position was closed
        exit_reason: Reason for closing position
        
    Returns:
        True if updated successfully
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get trade info for PnL calculation
                cur.execute("""
                    SELECT quantity, entry_price, side
                    FROM trade_records
                    WHERE id = %s AND status = 'open'
                """, (trade_id,))
                
                trade_info = cur.fetchone()
                if not trade_info:
                    logger.warning("Trade not found or already closed", trade_id=trade_id)
                    return False
                
                # Calculate PnL
                quantity = trade_info['quantity']
                entry_price = float(trade_info['entry_price'])
                side = trade_info['side']
                
                if side.lower() == 'buy':
                    pnl = (exit_price - entry_price) * quantity
                else:  # sell/short
                    pnl = (entry_price - exit_price) * quantity
                
                # Update trade record
                cur.execute("""
                    UPDATE trade_records
                    SET status = 'closed',
                        exit_timestamp = CURRENT_TIMESTAMP,
                        exit_price = %s,
                        exit_reason = %s,
                        pnl = %s
                    WHERE id = %s
                """, (exit_price, exit_reason, pnl, trade_id))
                
                updated = cur.rowcount > 0
                conn.commit()
                
                if updated:
                    logger.info("Trade updated with exit info", 
                               trade_id=trade_id,
                               exit_price=exit_price,
                               pnl=pnl)
                
                return updated
                
    except Exception as e:
        logger.error("Failed to update trade exit", 
                    trade_id=trade_id,
                    error=str(e))
        return False

# =============================================================================
# NEWS AND SCANNING FUNCTIONS
# =============================================================================

def insert_news_article(article_data: Dict) -> int:
    """Insert news article into database"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Ensure table exists
                cur.execute("""
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
                    )
                """)
                
                cur.execute("""
                    INSERT INTO news_raw (
                        title, content, source, url, published_timestamp,
                        sentiment_score, relevance_score, is_pre_market,
                        source_tier, mentioned_tickers, headline_keywords, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO NOTHING
                    RETURNING id
                """, (
                    article_data.get('title'),
                    article_data.get('content'),
                    article_data.get('source'),
                    article_data.get('url'),
                    article_data.get('published_timestamp'),
                    article_data.get('sentiment_score'),
                    article_data.get('relevance_score'),
                    article_data.get('is_pre_market', False),
                    article_data.get('source_tier', 5),
                    article_data.get('mentioned_tickers', []),
                    article_data.get('headline_keywords', []),
                    json.dumps(article_data.get('metadata', {}))
                ))
                
                result = cur.fetchone()
                if result:
                    news_id = result['id']
                    conn.commit()
                    logger.info("News article inserted", news_id=news_id)
                    return news_id
                else:
                    # Article already exists (conflict on URL)
                    logger.debug("News article already exists", url=article_data.get('url'))
                    return 0
                
    except Exception as e:
        logger.error("Failed to insert news article", error=str(e))
        raise DatabaseError(f"Failed to insert news article: {str(e)}")

def insert_trading_candidate(candidate_data: Dict) -> int:
    """Insert trading candidate into database"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Ensure table exists
                cur.execute("""
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
                    )
                """)
                
                cur.execute("""
                    INSERT INTO trading_candidates (
                        symbol, catalyst_score, catalyst_type, catalyst_summary,
                        price_data, volume_data, news_count, sentiment_avg,
                        relevance_avg, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    candidate_data.get('symbol'),
                    candidate_data.get('catalyst_score'),
                    candidate_data.get('catalyst_type'),
                    candidate_data.get('catalyst_summary'),
                    json.dumps(candidate_data.get('price_data', {})),
                    json.dumps(candidate_data.get('volume_data', {})),
                    candidate_data.get('news_count', 0),
                    candidate_data.get('sentiment_avg'),
                    candidate_data.get('relevance_avg'),
                    json.dumps(candidate_data.get('metadata', {}))
                ))
                
                candidate_id = cur.fetchone()['id']
                conn.commit()
                
                logger.info("Trading candidate inserted", 
                           candidate_id=candidate_id,
                           symbol=candidate_data.get('symbol'))
                
                return candidate_id
                
    except Exception as e:
        logger.error("Failed to insert trading candidate", error=str(e))
        raise DatabaseError(f"Failed to insert trading candidate: {str(e)}")

# =============================================================================
# PATTERN ANALYSIS FUNCTIONS
# =============================================================================

def insert_pattern_detection(pattern_data: Dict) -> int:
    """Insert pattern detection result into database"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Ensure table exists
                cur.execute("""
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
                    )
                """)
                
                cur.execute("""
                    INSERT INTO pattern_analysis (
                        symbol, timeframe, pattern_name, pattern_confidence,
                        pattern_signal, support_level, resistance_level,
                        catalyst_alignment, technical_confluence, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    pattern_data.get('symbol'),
                    pattern_data.get('timeframe'),
                    pattern_data.get('pattern_name'),
                    pattern_data.get('pattern_confidence'),
                    pattern_data.get('pattern_signal'),
                    pattern_data.get('support_level'),
                    pattern_data.get('resistance_level'),
                    pattern_data.get('catalyst_alignment', False),
                    json.dumps(pattern_data.get('technical_confluence', {})),
                    json.dumps(pattern_data.get('metadata', {}))
                ))
                
                pattern_id = cur.fetchone()['id']
                conn.commit()
                
                logger.info("Pattern detection inserted", 
                           pattern_id=pattern_id,
                           symbol=pattern_data.get('symbol'),
                           pattern=pattern_data.get('pattern_name'))
                
                return pattern_id
                
    except Exception as e:
        logger.error("Failed to insert pattern detection", error=str(e))
        raise DatabaseError(f"Failed to insert pattern detection: {str(e)}")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def execute_query(query: str, params: tuple = None) -> List[Dict]:
    """Execute a SELECT query and return results"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()
                
    except Exception as e:
        logger.error("Query execution failed", query=query[:50], error=str(e))
        raise DatabaseError(f"Query execution failed: {str(e)}")

def execute_update(query: str, params: tuple = None) -> int:
    """Execute an UPDATE/INSERT/DELETE query and return affected rows"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                affected_rows = cur.rowcount
                conn.commit()
                return affected_rows
                
    except Exception as e:
        logger.error("Update execution failed", query=query[:50], error=str(e))
        raise DatabaseError(f"Update execution failed: {str(e)}")

def close_connections():
    """Close all database connections and cleanup"""
    global _db_pool, _redis_client
    
    try:
        if _db_pool:
            _db_pool.closeall()
            _db_pool = None
            logger.info("Database connection pool closed")
            
        if _redis_client:
            _redis_client.close()
            _redis_client = None
            logger.info("Redis client closed")
            
    except Exception as e:
        logger.error("Error closing connections", error=str(e))

# Initialize pools on import
try:
    _create_connection_pool()
    _create_redis_client()
except Exception as e:
    logger.warning("Failed to initialize connections on import", error=str(e))