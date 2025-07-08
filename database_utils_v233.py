#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: database_utils.py
Version: 2.3.3
Last Updated: 2025-07-07
Purpose: Centralized database connection management for all services

REVISION HISTORY:
v2.3.3 (2025-07-08) - New trading functions added
    - update_service_health, get_configuration, get_active_trading_cycle

v2.3.2 (2025-07-07) - Fixed health check and added missing functions
- Changed health_check to use 'database' key instead of 'postgresql'
- Added insert_trading_candidates function
- Added get_active_candidates function
- Added update_candidate_status function
- Added insert_pattern_detection function

v2.3.1 (2025-07-01) - Production optimizations
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
        """,
        
        # News tables
        """
        CREATE TABLE IF NOT EXISTS news_raw (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10),
            headline TEXT,
            description TEXT,
            summary TEXT,
            url TEXT,
            source VARCHAR(100),
            author VARCHAR(100),
            published_at TIMESTAMP,
            collected_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sentiment_score FLOAT,
            relevance_score FLOAT,
            keywords TEXT,
            category VARCHAR(50)
        )
        """,
        
        """
        CREATE TABLE IF NOT EXISTS news_collection_stats (
            id SERIAL PRIMARY KEY,
            source VARCHAR(100),
            collection_date DATE DEFAULT CURRENT_DATE,
            symbol VARCHAR(10),
            articles_collected INTEGER DEFAULT 0,
            articles_processed INTEGER DEFAULT 0,
            last_collection TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    ]
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for table_sql in tables:
                    cur.execute(table_sql)
                
                # Create indexes
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_candidates_scan_id ON trading_candidates(scan_id)",
                    "CREATE INDEX IF NOT EXISTS idx_candidates_symbol ON trading_candidates(symbol)",
                    "CREATE INDEX IF NOT EXISTS idx_candidates_status ON trading_candidates(status)",
                    "CREATE INDEX IF NOT EXISTS idx_candidates_created ON trading_candidates(created_at DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_raw(symbol)",
                    "CREATE INDEX IF NOT EXISTS idx_news_published ON news_raw(published_at DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_news_collected ON news_raw(collected_timestamp DESC)"
                ]
                
                for index_sql in indexes:
                    cur.execute(index_sql)
                
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
#!/usr/bin/env python3
"""
Trading Cycle Functions to add to database_utils.py
These functions manage trading workflow cycles for the coordination service
"""

# =============================================================================
# TRADING CYCLE FUNCTIONS - Add these to database_utils.py
# =============================================================================

def create_trading_cycle(cycle_data: Dict) -> int:
    """
    Create a new trading cycle entry
    
    Args:
        cycle_data: Dictionary containing cycle information
        
    Returns:
        ID of the created cycle
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trading_cycles (
                        id SERIAL PRIMARY KEY,
                        cycle_id VARCHAR(50) UNIQUE NOT NULL,
                        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        end_time TIMESTAMP,
                        mode VARCHAR(20),
                        status VARCHAR(20) DEFAULT 'active',
                        news_collected INTEGER DEFAULT 0,
                        securities_scanned INTEGER DEFAULT 0,
                        patterns_detected INTEGER DEFAULT 0,
                        signals_generated INTEGER DEFAULT 0,
                        trades_executed INTEGER DEFAULT 0,
                        metadata JSONB
                    )
                """)
                
                # Insert new cycle
                cur.execute("""
                    INSERT INTO trading_cycles (
                        cycle_id, mode, metadata
                    ) VALUES (%s, %s, %s)
                    RETURNING id
                """, (
                    cycle_data.get('cycle_id'),
                    cycle_data.get('mode', 'normal'),
                    json.dumps(cycle_data.get('metadata', {}))
                ))
                
                cycle_id = cur.fetchone()['id']
                conn.commit()
                
                logger.info(f"Created trading cycle", cycle_id=cycle_id)
                return cycle_id
                
    except Exception as e:
        logger.error("Failed to create trading cycle", error=str(e))
        raise DatabaseError(f"Failed to create trading cycle: {str(e)}")


def update_trading_cycle(cycle_id: str, updates: Dict) -> bool:
    """
    Update trading cycle information
    
    Args:
        cycle_id: Cycle identifier
        updates: Dictionary of fields to update
        
    Returns:
        True if updated successfully
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Build update query dynamically
                update_fields = []
                values = []
                
                for field, value in updates.items():
                    if field in ['news_collected', 'securities_scanned', 'patterns_detected', 
                               'signals_generated', 'trades_executed', 'status']:
                        update_fields.append(f"{field} = %s")
                        values.append(value)
                
                if not update_fields:
                    return False
                
                values.append(cycle_id)
                query = f"""
                    UPDATE trading_cycles 
                    SET {', '.join(update_fields)}
                    WHERE cycle_id = %s
                """
                
                cur.execute(query, values)
                updated = cur.rowcount > 0
                conn.commit()
                
                return updated
                
    except Exception as e:
        logger.error("Failed to update trading cycle", error=str(e))
        return False


def get_active_trading_cycle() -> Optional[Dict]:
    """
    Get the current active trading cycle
    
    Returns:
        Dictionary with cycle information or None
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM trading_cycles 
                    WHERE status = 'active' 
                    ORDER BY start_time DESC 
                    LIMIT 1
                """)
                
                result = cur.fetchone()
                if result:
                    cycle = dict(result)
                    # Convert timestamps to ISO format
                    if cycle.get('start_time'):
                        cycle['start_time'] = cycle['start_time'].isoformat()
                    if cycle.get('end_time'):
                        cycle['end_time'] = cycle['end_time'].isoformat()
                    return cycle
                    
                return None
                
    except Exception as e:
        logger.error("Failed to get active trading cycle", error=str(e))
        return None


def complete_trading_cycle(cycle_id: str) -> bool:
    """
    Mark a trading cycle as complete
    
    Args:
        cycle_id: Cycle identifier
        
    Returns:
        True if completed successfully
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE trading_cycles 
                    SET status = 'completed', 
                        end_time = CURRENT_TIMESTAMP
                    WHERE cycle_id = %s AND status = 'active'
                """, (cycle_id,))
                
                updated = cur.rowcount > 0
                conn.commit()
                
                if updated:
                    logger.info(f"Completed trading cycle", cycle_id=cycle_id)
                
                return updated
                
    except Exception as e:
        logger.error("Failed to complete trading cycle", error=str(e))
        return False


def log_workflow_step(cycle_id: str, step: str, status: str, details: Optional[Dict] = None) -> bool:
    """
    Log a workflow step execution
    
    Args:
        cycle_id: Trading cycle ID
        step: Step name (e.g., 'news_collection', 'security_scan')
        status: Status of the step (e.g., 'started', 'completed', 'failed')
        details: Optional details about the step
        
    Returns:
        True if logged successfully
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create workflow_log table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS workflow_log (
                        id SERIAL PRIMARY KEY,
                        cycle_id VARCHAR(50),
                        step VARCHAR(50),
                        status VARCHAR(20),
                        details JSONB,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Insert log entry
                cur.execute("""
                    INSERT INTO workflow_log (cycle_id, step, status, details)
                    VALUES (%s, %s, %s, %s)
                """, (
                    cycle_id,
                    step,
                    status,
                    json.dumps(details) if details else None
                ))
                
                conn.commit()
                return True
                
    except Exception as e:
        logger.error("Failed to log workflow step", error=str(e))
        return False


def get_workflow_status(cycle_id: str) -> List[Dict]:
    """
    Get workflow status for a trading cycle
    
    Args:
        cycle_id: Trading cycle ID
        
    Returns:
        List of workflow steps with their status
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT step, status, details, timestamp
                    FROM workflow_log
                    WHERE cycle_id = %s
                    ORDER BY timestamp DESC
                """, (cycle_id,))
                
                results = []
                for row in cur.fetchall():
                    step_info = dict(row)
                    if step_info.get('timestamp'):
                        step_info['timestamp'] = step_info['timestamp'].isoformat()
                    results.append(step_info)
                
                return results
                
    except Exception as e:
        logger.error("Failed to get workflow status", error=str(e))
        return []

# =============================================================================
# v2.3.3 additional trading functions
# =============================================================================

def update_service_health(service_name: str, status: str, metrics: Optional[Dict] = None) -> bool:
    """
    Update service health status
    
    Args:
        service_name: Name of the service
        status: Status (healthy, degraded, down)
        metrics: Optional performance metrics
        
    Returns:
        True if updated successfully
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS service_health (
                        id SERIAL PRIMARY KEY,
                        service_name VARCHAR(50) NOT NULL,
                        status VARCHAR(20) NOT NULL,
                        last_check TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        response_time_ms INTEGER,
                        error_message TEXT,
                        requests_processed INTEGER,
                        errors_count INTEGER,
                        avg_response_time_ms INTEGER,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Update or insert service health
                cur.execute("""
                    INSERT INTO service_health (
                        service_name, status, last_check, response_time_ms,
                        requests_processed, errors_count, avg_response_time_ms, metadata
                    ) VALUES (%s, %s, CURRENT_TIMESTAMP, %s, %s, %s, %s, %s)
                    ON CONFLICT (service_name) DO UPDATE SET
                        status = EXCLUDED.status,
                        last_check = EXCLUDED.last_check,
                        response_time_ms = EXCLUDED.response_time_ms,
                        requests_processed = EXCLUDED.requests_processed,
                        errors_count = EXCLUDED.errors_count,
                        avg_response_time_ms = EXCLUDED.avg_response_time_ms,
                        metadata = EXCLUDED.metadata
                """, (
                    service_name,
                    status,
                    metrics.get('response_time_ms') if metrics else None,
                    metrics.get('requests_processed') if metrics else None,
                    metrics.get('errors_count') if metrics else None,
                    metrics.get('avg_response_time_ms') if metrics else None,
                    json.dumps(metrics) if metrics else None
                ))
                
                conn.commit()
                return True
                
    except Exception as e:
        logger.error("Failed to update service health", error=str(e))
        return False


def get_configuration(key: str, default: Any = None) -> Any:
    """
    Get configuration value from database
    
    Args:
        key: Configuration key
        default: Default value if not found
        
    Returns:
        Configuration value or default
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS configuration (
                        key VARCHAR(100) PRIMARY KEY,
                        value TEXT NOT NULL,
                        data_type VARCHAR(20),
                        category VARCHAR(50),
                        description TEXT,
                        last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        modified_by VARCHAR(50)
                    )
                """)
                
                # Insert default configurations if not exist
                default_configs = [
                    ('max_positions', '5', 'int', 'risk', 'Maximum concurrent positions', 'system'),
                    ('position_size_pct', '20', 'float', 'risk', 'Max position size as % of capital', 'system'),
                    ('premarket_position_pct', '10', 'float', 'risk', 'Pre-market position size limit', 'system'),
                    ('min_catalyst_score', '30', 'float', 'trading', 'Minimum score to consider', 'system'),
                    ('stop_loss_pct', '2', 'float', 'risk', 'Default stop loss percentage', 'system'),
                    ('scan_interval_premarket', '300', 'int', 'schedule', 'Pre-market scan interval seconds', 'system'),
                    ('scan_interval_regular', '1800', 'int', 'schedule', 'Regular hours scan interval', 'system'),
                    ('scan_interval_afterhours', '3600', 'int', 'schedule', 'After-hours scan interval', 'system')
                ]
                
                for config in default_configs:
                    cur.execute("""
                        INSERT INTO configuration (key, value, data_type, category, description, modified_by)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (key) DO NOTHING
                    """, config)
                
                # Get the requested configuration
                cur.execute("SELECT value, data_type FROM configuration WHERE key = %s", (key,))
                result = cur.fetchone()
                
                if result:
                    value, data_type = result['value'], result['data_type']
                    # Convert to appropriate type
                    if data_type == 'int':
                        return int(value)
                    elif data_type == 'float':
                        return float(value)
                    elif data_type == 'bool':
                        return value.lower() in ('true', '1', 'yes')
                    elif data_type == 'json':
                        return json.loads(value)
                    else:
                        return value
                else:
                    return default
                    
    except Exception as e:
        logger.error("Failed to get configuration", key=key, error=str(e))
        return default


def get_active_trading_cycle() -> Optional[Dict]:
    """
    Get the current active trading cycle
    
    Returns:
        Dictionary with cycle information or None
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM trading_cycles 
                    WHERE status = 'active' 
                    ORDER BY start_time DESC 
                    LIMIT 1
                """)
                
                result = cur.fetchone()
                if result:
                    cycle = dict(result)
                    # Convert timestamps to ISO format
                    if cycle.get('start_time'):
                        cycle['start_time'] = cycle['start_time'].isoformat()
                    if cycle.get('end_time'):
                        cycle['end_time'] = cycle['end_time'].isoformat()
                    return cycle
                    
                return None
                
    except Exception as e:
        logger.error("Failed to get active trading cycle", error=str(e))
        return None


def complete_trading_cycle(cycle_id: str) -> bool:
    """
    Mark a trading cycle as complete
    
    Args:
        cycle_id: Cycle identifier
        
    Returns:
        True if completed successfully
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE trading_cycles 
                    SET status = 'completed', 
                        end_time = CURRENT_TIMESTAMP
                    WHERE cycle_id = %s AND status = 'active'
                """, (cycle_id,))
                
                updated = cur.rowcount > 0
                conn.commit()
                
                if updated:
                    logger.info(f"Completed trading cycle", cycle_id=cycle_id)
                
                return updated
                
    except Exception as e:
        logger.error("Failed to complete trading cycle", error=str(e))
        return False


def get_workflow_status(cycle_id: str) -> List[Dict]:
    """
    Get workflow status for a trading cycle
    
    Args:
        cycle_id: Trading cycle ID
        
    Returns:
        List of workflow steps with their status
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT step, status, details, timestamp
                    FROM workflow_log
                    WHERE cycle_id = %s
                    ORDER BY timestamp DESC
                """, (cycle_id,))
                
                results = []
                for row in cur.fetchall():
                    step_info = dict(row)
                    if step_info.get('timestamp'):
                        step_info['timestamp'] = step_info['timestamp'].isoformat()
                    results.append(step_info)
                
                return results
                
    except Exception as e:
        logger.error("Failed to get workflow status", error=str(e))
        return []


# Also need to ensure service_health table has unique constraint
def ensure_service_health_constraint():
    """
    Ensure service_health table has proper constraints
    This should be called once during initialization
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Add unique constraint if it doesn't exist
                cur.execute("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_constraint 
                            WHERE conname = 'unique_service_name'
                        ) THEN
                            ALTER TABLE service_health 
                            ADD CONSTRAINT unique_service_name UNIQUE (service_name);
                        END IF;
                    END$$;
                """)
                conn.commit()
    except Exception as e:
        logger.warning("Could not add unique constraint", error=str(e))

# =============================================================================
# v2.3.3 NEWS SERVICE FUNCTIONS
# =============================================================================

def insert_news_article(article_data: Dict) -> bool:
    """
    Insert a news article into the database
    
    Args:
        article_data: Dictionary containing article information
        
    Returns:
        True if inserted successfully, False otherwise
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS news_raw (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10),
                        headline TEXT,
                        description TEXT,
                        summary TEXT,
                        url TEXT,
                        source VARCHAR(100),
                        author VARCHAR(100),
                        published_at TIMESTAMP,
                        collected_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        sentiment_score FLOAT,
                        relevance_score FLOAT,
                        keywords TEXT,
                        category VARCHAR(50),
                        source_tier INTEGER,
                        is_pre_market BOOLEAN DEFAULT FALSE,
                        UNIQUE(headline, source, published_at)
                    )
                """)
                
                # Insert article
                cur.execute("""
                    INSERT INTO news_raw (
                        symbol, headline, description, summary, url, source,
                        author, published_at, sentiment_score, relevance_score,
                        keywords, category, source_tier, is_pre_market
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (headline, source, published_at) DO NOTHING
                """, (
                    article_data.get('symbol'),
                    article_data.get('headline'),
                    article_data.get('description'),
                    article_data.get('summary'),
                    article_data.get('url'),
                    article_data.get('source'),
                    article_data.get('author'),
                    article_data.get('published_at'),
                    article_data.get('sentiment_score'),
                    article_data.get('relevance_score'),
                    article_data.get('keywords'),
                    article_data.get('category'),
                    article_data.get('source_tier', 3),
                    article_data.get('is_pre_market', False)
                ))
                
                # Update collection stats
                if cur.rowcount > 0:
                    update_collection_stats(
                        article_data.get('source', 'unknown'),
                        article_data.get('symbol')
                    )
                
                conn.commit()
                return True
                
    except Exception as e:
        logger.error("Failed to insert news article", error=str(e))
        return False


def get_recent_news(symbol: Optional[str] = None, hours: int = 24) -> List[Dict]:
    """
    Get recent news articles
    
    Args:
        symbol: Optional symbol to filter by
        hours: How many hours back to look
        
    Returns:
        List of news articles
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if symbol:
                    cur.execute("""
                        SELECT * FROM news_raw
                        WHERE symbol = %s 
                        AND collected_timestamp > NOW() - INTERVAL '%s hours'
                        ORDER BY published_at DESC
                    """, (symbol, hours))
                else:
                    cur.execute("""
                        SELECT * FROM news_raw
                        WHERE collected_timestamp > NOW() - INTERVAL '%s hours'
                        ORDER BY published_at DESC
                        LIMIT 100
                    """, (hours,))
                
                articles = []
                for row in cur.fetchall():
                    article = dict(row)
                    # Convert timestamps to ISO format
                    if article.get('published_at'):
                        article['published_at'] = article['published_at'].isoformat()
                    if article.get('collected_timestamp'):
                        article['collected_timestamp'] = article['collected_timestamp'].isoformat()
                    articles.append(article)
                
                return articles
                
    except Exception as e:
        logger.error("Failed to get recent news", error=str(e))
        return []


def update_collection_stats(source: str, symbol: Optional[str] = None) -> bool:
    """
    Update news collection statistics
    
    Args:
        source: News source name
        symbol: Optional symbol
        
    Returns:
        True if updated successfully
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS news_collection_stats (
                        id SERIAL PRIMARY KEY,
                        source VARCHAR(100),
                        collection_date DATE DEFAULT CURRENT_DATE,
                        symbol VARCHAR(10),
                        articles_collected INTEGER DEFAULT 0,
                        articles_processed INTEGER DEFAULT 0,
                        last_collection TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(source, collection_date, symbol)
                    )
                """)
                
                # Update or insert stats
                cur.execute("""
                    INSERT INTO news_collection_stats 
                    (source, symbol, articles_collected, last_collection)
                    VALUES (%s, %s, 1, CURRENT_TIMESTAMP)
                    ON CONFLICT (source, collection_date, symbol) 
                    DO UPDATE SET 
                        articles_collected = news_collection_stats.articles_collected + 1,
                        last_collection = CURRENT_TIMESTAMP
                """, (source, symbol))
                
                conn.commit()
                return True
                
    except Exception as e:
        logger.error("Failed to update collection stats", error=str(e))
        return False


def get_news_stats() -> Dict:
    """
    Get news collection statistics
    
    Returns:
        Dictionary with statistics
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Overall stats
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_articles,
                        COUNT(DISTINCT symbol) as unique_symbols,
                        COUNT(DISTINCT source) as unique_sources,
                        MIN(published_at) as oldest_article,
                        MAX(published_at) as newest_article
                    FROM news_raw
                    WHERE collected_timestamp > NOW() - INTERVAL '24 hours'
                """)
                
                overall = cur.fetchone()
                
                # By source
                cur.execute("""
                    SELECT source, COUNT(*) as count
                    FROM news_raw
                    WHERE collected_timestamp > NOW() - INTERVAL '24 hours'
                    GROUP BY source
                    ORDER BY count DESC
                    LIMIT 10
                """)
                
                by_source = [dict(row) for row in cur.fetchall()]
                
                # By symbol
                cur.execute("""
                    SELECT symbol, COUNT(*) as count
                    FROM news_raw
                    WHERE collected_timestamp > NOW() - INTERVAL '24 hours'
                    GROUP BY symbol
                    ORDER BY count DESC
                    LIMIT 10
                """)
                
                by_symbol = [dict(row) for row in cur.fetchall()]
                
                return {
                    'overall': dict(overall) if overall else {},
                    'by_source': by_source,
                    'by_symbol': by_symbol
                }
                
    except Exception as e:
        logger.error("Failed to get news stats", error=str(e))
        return {'error': str(e)}


def search_news(query: str, symbol: Optional[str] = None) -> List[Dict]:
    """
    Search news articles
    
    Args:
        query: Search query
        symbol: Optional symbol filter
        
    Returns:
        List of matching articles
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if symbol:
                    cur.execute("""
                        SELECT * FROM news_raw
                        WHERE symbol = %s
                        AND (headline ILIKE %s OR description ILIKE %s)
                        ORDER BY published_at DESC
                        LIMIT 50
                    """, (symbol, f'%{query}%', f'%{query}%'))
                else:
                    cur.execute("""
                        SELECT * FROM news_raw
                        WHERE headline ILIKE %s OR description ILIKE %s
                        ORDER BY published_at DESC
                        LIMIT 50
                    """, (f'%{query}%', f'%{query}%'))
                
                return [dict(row) for row in cur.fetchall()]
                
    except Exception as e:
        logger.error("Failed to search news", error=str(e))
        return []

