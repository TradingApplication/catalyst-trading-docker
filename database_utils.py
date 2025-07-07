#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: database_utils.py
Version: 2.32
Last Updated: 2025-07-07
Purpose: Centralized database connection management for all services

REVISION HISTORY:
v2.32 (2025-07-07) - Fixed health check and added missing functions
- Changed health_check to use 'database' key instead of 'postgresql'
- Added insert_trading_candidates function
- Added get_active_candidates function
- Added update_candidate_status function
- Added insert_pattern_detection function

v2.31 (2025-07-01) - Production optimizations
- Enhanced connection pooling
- Better error handling
- Performance improvements

v2.2.0 (2025-01-03) - Fixed for external database connections
- Use DATABASE_URL directly from environment
- Proper SSL mode handling for DigitalOcean
- Better error messages for debugging
- Connection pooling with proper parameters

v2.1.0 (2025-07-01) - Production-ready database utilities
- PostgreSQL connection pooling
- Redis connection management
- Retry logic for resilient connections
- Comprehensive error handling
- Performance monitoring hooks

Description of Service:
Provides centralized database connection management for the Catalyst Trading System.
Handles both PostgreSQL and Redis connections with proper pooling, retry logic,
and error handling. All services use this module for consistent database access.
"""

import os
import time
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import redis
from redis.exceptions import ConnectionError as RedisConnectionError
import structlog
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = structlog.get_logger(__name__)

# Global connection pools
_db_pool: Optional[pool.SimpleConnectionPool] = None
_redis_client: Optional[redis.Redis] = None


class DatabaseError(Exception):
    """Custom exception for database-related errors"""
    pass


def init_db_pool(min_connections: int = 2, max_connections: int = 10) -> pool.SimpleConnectionPool:
    """Initialize PostgreSQL connection pool using DATABASE_URL from environment"""
    global _db_pool
    
    if _db_pool is not None:
        return _db_pool
    
    try:
        # Get DATABASE_URL from environment
        database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            # Fallback: construct from individual components
            host = os.getenv('DATABASE_HOST')
            port = os.getenv('DATABASE_PORT', '5432')
            database = os.getenv('DATABASE_NAME', 'catalyst_trading')
            user = os.getenv('DATABASE_USER', 'doadmin')
            password = os.getenv('DATABASE_PASSWORD')
            
            if not all([host, user, password]):
                raise DatabaseError(
                    "Missing database configuration. Please set DATABASE_URL or "
                    "DATABASE_HOST, DATABASE_USER, and DATABASE_PASSWORD environment variables."
                )
            
            # Construct DATABASE_URL with SSL mode for DigitalOcean
            database_url = f"postgresql://{user}:{password}@{host}:{port}/{database}?sslmode=require"
        
        logger.info("Initializing database connection pool", 
                   host=database_url.split('@')[1].split('/')[0] if '@' in database_url else 'unknown')
        
        _db_pool = pool.SimpleConnectionPool(
            min_connections,
            max_connections,
            database_url,
            cursor_factory=RealDictCursor
        )
        
        # Test the connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                if result:
                    logger.info("Database connection pool initialized",
                               min_connections=min_connections,
                               max_connections=max_connections)
        
        return _db_pool
        
    except Exception as e:
        logger.error("Failed to initialize database pool", error=str(e))
        raise DatabaseError(f"Database pool initialization failed: {str(e)}")


def init_redis_client() -> redis.Redis:
    """Initialize Redis client using REDIS_URL from environment"""
    global _redis_client
    
    if _redis_client is not None:
        return _redis_client
    
    try:
        # Get Redis URL from environment
        redis_url = os.getenv('REDIS_URL')
        
        if not redis_url:
            # Fallback: construct from individual components
            host = os.getenv('REDIS_HOST', 'redis')
            port = os.getenv('REDIS_PORT', '6379')
            password = os.getenv('REDIS_PASSWORD')
            
            if password:
                redis_url = f"redis://:{password}@{host}:{port}/0"
            else:
                redis_url = f"redis://{host}:{port}/0"
        
        logger.info("Initializing Redis client", 
                   url=redis_url.replace(password, '***') if 'password' in locals() and password else redis_url)
        
        _redis_client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # Test the connection
        _redis_client.ping()
        logger.info("Redis client initialized", url=redis_url.split('@')[-1] if '@' in redis_url else redis_url)
        
        return _redis_client
        
    except Exception as e:
        logger.error("Failed to initialize Redis client", error=str(e))
        raise DatabaseError(f"Redis initialization failed: {str(e)}")


@contextmanager
def get_db_connection():
    """Get a database connection from the pool"""
    if _db_pool is None:
        init_db_pool()
    
    conn = None
    try:
        conn = _db_pool.getconn()
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error("Database operation failed", error=str(e))
        raise
    finally:
        if conn and _db_pool:
            _db_pool.putconn(conn)


def get_redis() -> redis.Redis:
    """Get Redis client instance"""
    if _redis_client is None:
        return init_redis_client()
    return _redis_client


def close_connections():
    """Close all database connections"""
    global _db_pool, _redis_client
    
    if _db_pool:
        _db_pool.closeall()
        _db_pool = None
        logger.info("Database connection pool closed")
    
    if _redis_client:
        _redis_client.close()
        _redis_client = None
        logger.info("Redis connection closed")


def health_check() -> Dict[str, Any]:
    """Perform health check on database connections"""
    health_status = {
        'database': {'status': 'unknown', 'error': None},  # Changed from 'postgresql' to 'database'
        'redis': {'status': 'unknown', 'error': None}
    }
    
    # Check PostgreSQL
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                health_status['database']['status'] = 'healthy'
    except Exception as e:
        health_status['database']['status'] = 'unhealthy'
        health_status['database']['error'] = str(e)
    
    # Check Redis
    try:
        redis_client = get_redis()
        redis_client.ping()
        health_status['redis']['status'] = 'healthy'
    except Exception as e:
        health_status['redis']['status'] = 'unhealthy'
        health_status['redis']['error'] = str(e)
    
    return health_status


# Retry decorator for database operations
def with_db_retry(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry database operations on failure"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (psycopg2.OperationalError, RedisConnectionError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Database operation failed, retrying in {wait_time}s",
                                     attempt=attempt + 1, error=str(e))
                        time.sleep(wait_time)
                    else:
                        logger.error("Database operation failed after all retries",
                                   max_retries=max_retries, error=str(e))
            raise last_error
        return wrapper
    return decorator


# =============================================================================
# TRADING CANDIDATE FUNCTIONS
# =============================================================================

def insert_trading_candidates(candidates: List[Dict], scan_id: str) -> int:
    """
    Insert trading candidates into the database
    
    Args:
        candidates: List of candidate dictionaries with symbol, scores, etc.
        scan_id: Unique identifier for this scan
        
    Returns:
        Number of candidates inserted
    """
    inserted_count = 0
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trading_candidates (
                        id SERIAL PRIMARY KEY,
                        scan_id VARCHAR(50) NOT NULL,
                        symbol VARCHAR(10) NOT NULL,
                        price DECIMAL(10,2),
                        volume BIGINT,
                        relative_volume DECIMAL(5,2),
                        price_change_pct DECIMAL(5,2),
                        catalyst_score INTEGER,
                        technical_score INTEGER,
                        news_count INTEGER,
                        catalysts JSONB,
                        primary_catalyst VARCHAR(100),
                        sector VARCHAR(50),
                        market_cap BIGINT,
                        status VARCHAR(20) DEFAULT 'active',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes if they don't exist
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_candidates_scan_id 
                    ON trading_candidates(scan_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_candidates_symbol 
                    ON trading_candidates(symbol)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_candidates_status 
                    ON trading_candidates(status)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_candidates_created 
                    ON trading_candidates(created_at DESC)
                """)
                
                # Insert candidates
                for candidate in candidates:
                    cur.execute("""
                        INSERT INTO trading_candidates (
                            scan_id, symbol, price, volume, relative_volume,
                            price_change_pct, catalyst_score, technical_score,
                            news_count, catalysts, primary_catalyst, sector, 
                            market_cap, status
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                    """, (
                        scan_id,
                        candidate.get('symbol'),
                        candidate.get('price'),
                        candidate.get('volume'),
                        candidate.get('relative_volume'),
                        candidate.get('price_change_pct'),
                        candidate.get('catalyst_score', 0),
                        candidate.get('technical_score', 0),
                        candidate.get('news_count', 0),
                        json.dumps(candidate.get('catalysts', [])),
                        candidate.get('primary_catalyst'),
                        candidate.get('sector'),
                        candidate.get('market_cap'),
                        'active'
                    ))
                    inserted_count += 1
                
                conn.commit()
                logger.info(f"Inserted {inserted_count} trading candidates", scan_id=scan_id)
                
    except Exception as e:
        logger.error("Failed to insert trading candidates", error=str(e))
        raise DatabaseError(f"Failed to insert candidates: {str(e)}")
    
    return inserted_count


def get_active_candidates(limit: int = 10) -> List[Dict]:
    """
    Get active trading candidates
    
    Args:
        limit: Maximum number of candidates to return
        
    Returns:
        List of candidate dictionaries
    """
    candidates = []
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get most recent active candidates
                cur.execute("""
                    SELECT 
                        scan_id, symbol, price, volume, relative_volume,
                        price_change_pct, catalyst_score, technical_score,
                        news_count, catalysts, primary_catalyst, sector,
                        market_cap, status, created_at
                    FROM trading_candidates
                    WHERE status = 'active'
                    AND created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
                    ORDER BY created_at DESC, catalyst_score DESC
                    LIMIT %s
                """, (limit,))
                
                rows = cur.fetchall()
                
                for row in rows:
                    candidate = dict(row)
                    # Convert datetime to ISO format string
                    if candidate.get('created_at'):
                        candidate['created_at'] = candidate['created_at'].isoformat()
                    candidates.append(candidate)
                
                logger.info(f"Retrieved {len(candidates)} active candidates")
                
    except Exception as e:
        logger.error("Failed to get active candidates", error=str(e))
        # Return empty list instead of raising to avoid breaking the service
        return []
    
    return candidates


def update_candidate_status(symbol: str, status: str, scan_id: Optional[str] = None) -> bool:
    """
    Update the status of a trading candidate
    
    Args:
        symbol: Stock symbol
        status: New status (e.g., 'traded', 'expired', 'cancelled')
        scan_id: Optional scan ID to update specific scan
        
    Returns:
        True if updated successfully, False otherwise
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if scan_id:
                    cur.execute("""
                        UPDATE trading_candidates
                        SET status = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = %s AND scan_id = %s
                    """, (status, symbol, scan_id))
                else:
                    # Update most recent entry for this symbol
                    cur.execute("""
                        UPDATE trading_candidates
                        SET status = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = %s AND status = 'active'
                        ORDER BY created_at DESC
                        LIMIT 1
                    """, (status, symbol))
                
                updated = cur.rowcount > 0
                conn.commit()
                
                if updated:
                    logger.info(f"Updated candidate status", symbol=symbol, status=status)
                
                return updated
                
    except Exception as e:
        logger.error("Failed to update candidate status", error=str(e))
        return False


def insert_pattern_detection(pattern_data: Dict) -> bool:
    """
    Insert pattern detection results into the database
    
    Args:
        pattern_data: Dictionary containing pattern detection results
        
    Returns:
        True if inserted successfully, False otherwise
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS pattern_detections (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        pattern_type VARCHAR(50),
                        pattern_name VARCHAR(100),
                        confidence DECIMAL(5,2),
                        catalyst_alignment VARCHAR(20),
                        news_sentiment VARCHAR(20),
                        detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB
                    )
                """)
                
                # Insert pattern detection
                cur.execute("""
                    INSERT INTO pattern_detections (
                        symbol, pattern_type, pattern_name, confidence,
                        catalyst_alignment, news_sentiment, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    pattern_data.get('symbol'),
                    pattern_data.get('pattern_type'),
                    pattern_data.get('pattern_name'),
                    pattern_data.get('confidence'),
                    pattern_data.get('catalyst_alignment'),
                    pattern_data.get('news_sentiment'),
                    json.dumps(pattern_data.get('metadata', {}))
                ))
                
                conn.commit()
                logger.info("Inserted pattern detection", symbol=pattern_data.get('symbol'))
                return True
                
    except Exception as e:
        logger.error("Failed to insert pattern detection", error=str(e))
        return False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def execute_query(query: str, params: Optional[tuple] = None) -> List[Dict]:
    """
    Execute a SELECT query and return results
    
    Args:
        query: SQL query to execute
        params: Optional query parameters
        
    Returns:
        List of dictionaries representing rows
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()
    except Exception as e:
        logger.error("Query execution failed", query=query, error=str(e))
        raise DatabaseError(f"Query execution failed: {str(e)}")


def execute_update(query: str, params: Optional[tuple] = None) -> int:
    """
    Execute an INSERT/UPDATE/DELETE query
    
    Args:
        query: SQL query to execute
        params: Optional query parameters
        
    Returns:
        Number of affected rows
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                affected_rows = cur.rowcount
                conn.commit()
                return affected_rows
    except Exception as e:
        logger.error("Update execution failed", query=query, error=str(e))
        raise DatabaseError(f"Update execution failed: {str(e)}")


def create_tables():
    """Create all required tables for the Catalyst Trading System"""
    tables = [
        # Trading candidates table
        """
        CREATE TABLE IF NOT EXISTS trading_candidates (
            id SERIAL PRIMARY KEY,
            scan_id VARCHAR(50) NOT NULL,
            symbol VARCHAR(10) NOT NULL,
            price DECIMAL(10,2),
            volume BIGINT,
            relative_volume DECIMAL(5,2),
            price_change_pct DECIMAL(5,2),
            catalyst_score INTEGER,
            technical_score INTEGER,
            news_count INTEGER,
            catalysts JSONB,
            primary_catalyst VARCHAR(100),
            sector VARCHAR(50),
            market_cap BIGINT,
            status VARCHAR(20) DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        
        # Pattern detections table
        """
        CREATE TABLE IF NOT EXISTS pattern_detections (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            pattern_type VARCHAR(50),
            pattern_name VARCHAR(100),
            confidence DECIMAL(5,2),
            catalyst_alignment VARCHAR(20),
            news_sentiment VARCHAR(20),
            detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB
        )
        """,
        
        # Trade signals table
        """
        CREATE TABLE IF NOT EXISTS trade_signals (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            signal_type VARCHAR(20),
            entry_price DECIMAL(10,2),
            stop_loss DECIMAL(10,2),
            target_price DECIMAL(10,2),
            confidence DECIMAL(5,2),
            risk_reward_ratio DECIMAL(5,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(20) DEFAULT 'active',
            metadata JSONB
        )
        """,
        
        # Trade executions table
        """
        CREATE TABLE IF NOT EXISTS trade_executions (
            id SERIAL PRIMARY KEY,
            signal_id INTEGER REFERENCES trade_signals(id),
            symbol VARCHAR(10) NOT NULL,
            order_type VARCHAR(20),
            quantity INTEGER,
            entry_price DECIMAL(10,2),
            exit_price DECIMAL(10,2),
            profit_loss DECIMAL(10,2),
            status VARCHAR(20),
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            metadata JSONB
        )
        """
    ]
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for table_sql in tables:
                    cur.execute(table_sql)
                conn.commit()
                logger.info("All tables created successfully")
    except Exception as e:
        logger.error("Failed to create tables", error=str(e))
        raise DatabaseError(f"Failed to create tables: {str(e)}")


# Initialize connections on module import
try:
    init_db_pool()
    init_redis_client()
except Exception as e:
    logger.warning("Failed to initialize database utilities on import", error=str(e))
    # Don't raise here - let services handle initialization in their __init__
