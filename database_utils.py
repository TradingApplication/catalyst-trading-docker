#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: database_utils.py
Version: 2.3.9
Last Updated: 2025-07-14
Purpose: COMPLETE database utilities with thread-safe pooling and all functions

REVISION HISTORY:
v2.3.9 (2025-07-14) - Merged v236 and v238 improvements
- Restored ThreadedConnectionPool for thread safety (from v236)
- Kept all workflow and configuration functions (from v238)
- Maintained critical trading signal fixes (from v236)
- Combined best practices from both versions
- Fixed connection pooling thread safety issues

v2.3.8 (2025-07-10) - Fixed SQL syntax error
- Fixed log_workflow_step UPDATE query for PostgreSQL compatibility
- Changed to use subquery instead of ORDER BY in UPDATE

v2.3.7 (2025-07-10) - FINAL COMPLETE VERSION
- Added log_workflow_step function
- Added update_service_health function
- Added get_configuration function
- Added all helper functions

v2.3.6 (2025-07-12) - CRITICAL FIX: Complete trading signal support
- Added complete insert_trading_signal function with proper error handling
- Added get_pending_signals function with comprehensive filtering
- Added mark_signal_executed function for signal lifecycle management
- Enhanced error handling and logging throughout

Description of Service:
COMPLETE database utilities for the Catalyst Trading System.
This version includes EVERY function needed by ALL services with
thread-safe connection pooling.
"""

import os
import time
import json
import logging
import uuid
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import psycopg2
import psycopg2.pool
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
_db_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
_redis_client: Optional[redis.Redis] = None
_pool_lock = threading.Lock()


class DatabaseError(Exception):
    """Custom exception for database-related errors"""
    pass


class ConnectionPoolError(Exception):
    """Connection pool error"""
    pass


def init_db_pool(min_connections: int = 2, max_connections: int = 10) -> psycopg2.pool.ThreadedConnectionPool:
    """Initialize PostgreSQL connection pool with thread safety"""
    global _db_pool
    
    with _pool_lock:
        if _db_pool is not None:
            return _db_pool
        
        try:
            # Get DATABASE_URL from environment
            database_url = os.getenv('DATABASE_URL')
            
            if not database_url:
                # Fallback: construct from individual components
                host = os.getenv('DATABASE_HOST', os.getenv('DB_HOST', 'localhost'))
                port = os.getenv('DATABASE_PORT', os.getenv('DB_PORT', '5432'))
                database = os.getenv('DATABASE_NAME', os.getenv('DB_NAME', 'catalyst_trading'))
                user = os.getenv('DATABASE_USER', os.getenv('DB_USER', 'doadmin'))
                password = os.getenv('DATABASE_PASSWORD', os.getenv('DB_PASSWORD'))
                sslmode = os.getenv('DB_SSLMODE', 'prefer')
                
                if not all([host, user, password]):
                    raise DatabaseError(
                        "Missing database configuration. Please set DATABASE_URL or "
                        "DATABASE_HOST, DATABASE_USER, and DATABASE_PASSWORD environment variables."
                    )
                
                # Construct DATABASE_URL with SSL mode for DigitalOcean
                database_url = f"postgresql://{user}:{password}@{host}:{port}/{database}?sslmode={sslmode}"
            
            logger.info("Initializing thread-safe database connection pool", 
                       host=database_url.split('@')[1].split('/')[0] if '@' in database_url else 'unknown')
            
            # Use ThreadedConnectionPool for thread safety
            _db_pool = psycopg2.pool.ThreadedConnectionPool(
                min_connections,
                max_connections,
                database_url,
                cursor_factory=RealDictCursor
            )
            
            # Test the connection
            conn = _db_pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    result = cur.fetchone()
                    if result:
                        logger.info("Database connection pool initialized",
                                   min_connections=min_connections,
                                   max_connections=max_connections)
            finally:
                _db_pool.putconn(conn)
            
            return _db_pool
            
        except Exception as e:
            logger.error("Failed to initialize database pool", error=str(e))
            raise DatabaseError(f"Database pool initialization failed: {str(e)}")


def init_redis_client() -> redis.Redis:
    """Initialize Redis client using REDIS_URL from environment"""
    global _redis_client
    
    with _pool_lock:
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
            # Don't raise error for Redis, as it's not critical
            return None


@contextmanager
def get_db_connection():
    """Get a database connection from the pool with thread safety"""
    if _db_pool is None:
        init_db_pool()
    
    conn = None
    try:
        conn = _db_pool.getconn()
        if conn:
            # Ensure cursor factory is set
            conn.cursor_factory = RealDictCursor
            yield conn
        else:
            raise DatabaseError("Could not get connection from pool")
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
    
    with _pool_lock:
        if _db_pool:
            _db_pool.closeall()
            _db_pool = None
            logger.info("Database connection pool closed")
        
        if _redis_client:
            _redis_client.close()
            _redis_client = None
            logger.info("Redis connection closed")


def health_check() -> Dict[str, Any]:
    """Perform health check on database connections - v2.3.1 compatible format"""
    health_status = {
        'postgresql': {'status': 'unhealthy', 'error': None},
        'redis': {'status': 'unhealthy', 'error': None},
        'timestamp': datetime.now().isoformat()
    }
    
    # Check PostgreSQL
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                if result:
                    health_status['postgresql']['status'] = 'healthy'
    except Exception as e:
        health_status['postgresql']['status'] = 'unhealthy'
        health_status['postgresql']['error'] = str(e)
        logger.error("PostgreSQL health check failed", error=str(e))
    
    # Check Redis
    try:
        redis_client = get_redis()
        if redis_client:
            redis_client.ping()
            health_status['redis']['status'] = 'healthy'
        else:
            health_status['redis']['error'] = 'Redis client not available'
    except Exception as e:
        health_status['redis']['status'] = 'unhealthy'
        health_status['redis']['error'] = str(e)
        logger.error("Redis health check failed", error=str(e))
    
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
# TRADING CYCLE MANAGEMENT FUNCTIONS
# =============================================================================

def create_trading_cycle(mode: str = 'normal') -> str:
    """
    Create a new trading cycle for workflow tracking
    
    Args:
        mode: Trading mode ('aggressive', 'normal', 'light')
        
    Returns:
        cycle_id: Unique identifier for the trading cycle
    """
    try:
        cycle_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trading_cycles (
                        cycle_id VARCHAR(50) PRIMARY KEY,
                        start_time TIMESTAMPTZ NOT NULL,
                        end_time TIMESTAMPTZ,
                        status VARCHAR(20) DEFAULT 'running',
                        mode VARCHAR(20),
                        news_collected INTEGER DEFAULT 0,
                        securities_scanned INTEGER DEFAULT 0,
                        candidates_selected INTEGER DEFAULT 0,
                        patterns_analyzed INTEGER DEFAULT 0,
                        signals_generated INTEGER DEFAULT 0,
                        trades_executed INTEGER DEFAULT 0,
                        cycle_pnl DECIMAL(10,2),
                        success_rate DECIMAL(5,2),
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Insert new cycle
                cur.execute("""
                    INSERT INTO trading_cycles (cycle_id, start_time, mode)
                    VALUES (%s, %s, %s)
                    RETURNING cycle_id
                """, (cycle_id, datetime.now(), mode))
                
                result = cur.fetchone()
                conn.commit()
                
                logger.info(f"Created trading cycle", cycle_id=cycle_id, mode=mode)
                return result['cycle_id']
                
    except Exception as e:
        logger.error("Failed to create trading cycle", error=str(e))
        raise DatabaseError(f"Failed to create trading cycle: {str(e)}")


def update_trading_cycle(cycle_id: str, updates: Dict) -> bool:
    """
    Update trading cycle metrics
    
    Args:
        cycle_id: Trading cycle identifier
        updates: Dictionary of fields to update
        
    Returns:
        True if updated successfully
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Build update query dynamically
                set_clauses = []
                values = []
                for key, value in updates.items():
                    if key not in ['cycle_id', 'created_at']:  # Prevent updating primary key
                        if key == 'metadata' and isinstance(value, dict):
                            set_clauses.append(f"{key} = %s::jsonb")
                            values.append(json.dumps(value))
                        else:
                            set_clauses.append(f"{key} = %s")
                            values.append(value)
                
                if not set_clauses:
                    return False
                
                values.append(cycle_id)
                query = f"""
                    UPDATE trading_cycles 
                    SET {', '.join(set_clauses)}
                    WHERE cycle_id = %s
                """
                
                cur.execute(query, values)
                updated = cur.rowcount > 0
                conn.commit()
                
                if updated:
                    logger.info(f"Updated trading cycle", cycle_id=cycle_id, updates=updates)
                
                return updated
                
    except Exception as e:
        logger.error("Failed to update trading cycle", error=str(e))
        return False


def complete_trading_cycle(cycle_id: str, metrics: Optional[Dict] = None) -> bool:
    """
    Mark a trading cycle as completed with final metrics
    
    Args:
        cycle_id: Trading cycle identifier
        metrics: Optional final metrics to update
        
    Returns:
        True if completed successfully
    """
    updates = {
        'status': 'completed',
        'end_time': datetime.now()
    }
    
    if metrics:
        updates.update(metrics)
    
    return update_trading_cycle(cycle_id, updates)


def get_current_cycle() -> Optional[Dict]:
    """
    Get the currently running trading cycle
    
    Returns:
        Current cycle dictionary or None
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM trading_cycles
                    WHERE status = 'running'
                    ORDER BY start_time DESC
                    LIMIT 1
                """)
                
                result = cur.fetchone()
                if result:
                    cycle = dict(result)
                    # Convert timestamps
                    for field in ['start_time', 'end_time', 'created_at']:
                        if cycle.get(field):
                            cycle[field] = cycle[field].isoformat()
                    return cycle
                
                return None
                
    except Exception as e:
        logger.error("Failed to get current cycle", error=str(e))
        return None


# =============================================================================
# WORKFLOW LOGGING FUNCTIONS - WITH POSTGRESQL FIX
# =============================================================================

def log_workflow_step(cycle_id: str, step_name: str, status: str, 
                     result: Optional[Dict] = None,
                     records_processed: Optional[int] = None,
                     records_output: Optional[int] = None,
                     error_message: Optional[str] = None) -> int:
    """
    Log a workflow step execution
    
    Args:
        cycle_id: Trading cycle identifier
        step_name: Name of the workflow step
        status: Status of the step ('started', 'completed', 'failed')
        result: Optional result data
        records_processed: Number of records processed
        records_output: Number of records output
        error_message: Error message if failed
        
    Returns:
        ID of the inserted/updated log entry
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS workflow_log (
                        id SERIAL PRIMARY KEY,
                        cycle_id VARCHAR(50),
                        step_name VARCHAR(100) NOT NULL,
                        status VARCHAR(20) NOT NULL,
                        start_time TIMESTAMPTZ NOT NULL,
                        end_time TIMESTAMPTZ,
                        duration_seconds DECIMAL(10,3),
                        records_processed INTEGER,
                        records_output INTEGER,
                        result JSONB,
                        error_message TEXT,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_workflow_cycle 
                    ON workflow_log(cycle_id, step_name)
                """)
                
                if status == 'started':
                    # Starting a new step
                    cur.execute("""
                        INSERT INTO workflow_log (
                            cycle_id, step_name, status, start_time
                        ) VALUES (%s, %s, %s, %s)
                        RETURNING id
                    """, (cycle_id, step_name, status, datetime.now()))
                    
                    result = cur.fetchone()
                    conn.commit()
                    return result['id']
                    
                else:
                    # Completing or failing a step - update existing record
                    # Use a subquery to find the ID of the record to update
                    cur.execute("""
                        UPDATE workflow_log 
                        SET status = %s,
                            end_time = %s,
                            duration_seconds = EXTRACT(EPOCH FROM (%s - start_time)),
                            records_processed = %s,
                            records_output = %s,
                            result = %s,
                            error_message = %s
                        WHERE id = (
                            SELECT id FROM workflow_log
                            WHERE cycle_id = %s 
                            AND step_name = %s
                            AND status = 'started'
                            ORDER BY start_time DESC
                            LIMIT 1
                        )
                        RETURNING id
                    """, (
                        status,
                        datetime.now(),
                        datetime.now(),
                        records_processed,
                        records_output,
                        json.dumps(result) if result else None,
                        error_message,
                        cycle_id,
                        step_name
                    ))
                    
                    if cur.rowcount > 0:
                        result = cur.fetchone()
                        conn.commit()
                        return result['id']
                    else:
                        # No existing record to update, insert new completed record
                        cur.execute("""
                            INSERT INTO workflow_log (
                                cycle_id, step_name, status, start_time, end_time,
                                records_processed, records_output, result, error_message
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING id
                        """, (
                            cycle_id, step_name, status, datetime.now(), datetime.now(),
                            records_processed, records_output,
                            json.dumps(result) if result else None,
                            error_message
                        ))
                        
                        result = cur.fetchone()
                        conn.commit()
                        return result['id']
                
    except Exception as e:
        logger.error("Failed to log workflow step", error=str(e))
        raise DatabaseError(f"Failed to log workflow step: {str(e)}")


# =============================================================================
# SERVICE HEALTH MONITORING FUNCTIONS
# =============================================================================

def update_service_health(service_name: str, status: str, 
                         response_time_ms: Optional[int] = None,
                         error_message: Optional[str] = None,
                         metrics: Optional[Dict] = None) -> bool:
    """
    Update service health status
    
    Args:
        service_name: Name of the service
        status: Health status ('healthy', 'degraded', 'down')
        response_time_ms: Response time in milliseconds
        error_message: Error message if unhealthy
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
                        last_check TIMESTAMPTZ NOT NULL,
                        response_time_ms INTEGER,
                        error_message TEXT,
                        requests_processed INTEGER,
                        errors_count INTEGER,
                        avg_response_time_ms INTEGER,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_service_status 
                    ON service_health(service_name, status)
                """)
                
                # Extract metrics if provided
                requests_processed = metrics.get('requests_processed') if metrics else None
                errors_count = metrics.get('errors_count') if metrics else None
                avg_response_time_ms = metrics.get('avg_response_time_ms') if metrics else None
                
                # Insert health check record
                cur.execute("""
                    INSERT INTO service_health (
                        service_name, status, last_check, response_time_ms,
                        error_message, requests_processed, errors_count,
                        avg_response_time_ms, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    service_name,
                    status,
                    datetime.now(),
                    response_time_ms,
                    error_message,
                    requests_processed,
                    errors_count,
                    avg_response_time_ms,
                    json.dumps(metrics) if metrics else None
                ))
                
                conn.commit()
                return True
                
    except Exception as e:
        logger.error("Failed to update service health", error=str(e))
        return False


def get_service_health(service_name: Optional[str] = None, 
                      hours: int = 1) -> List[Dict]:
    """
    Get service health history
    
    Args:
        service_name: Optional service name filter
        hours: Number of hours to look back
        
    Returns:
        List of health check records
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if service_name:
                    cur.execute("""
                        SELECT * FROM service_health
                        WHERE service_name = %s
                        AND last_check > CURRENT_TIMESTAMP - INTERVAL '%s hours'
                        ORDER BY last_check DESC
                        LIMIT 100
                    """, (service_name, hours))
                else:
                    cur.execute("""
                        SELECT * FROM service_health
                        WHERE last_check > CURRENT_TIMESTAMP - INTERVAL '%s hours'
                        ORDER BY last_check DESC
                        LIMIT 500
                    """, (hours,))
                
                health_records = []
                for row in cur.fetchall():
                    record = dict(row)
                    if record.get('last_check'):
                        record['last_check'] = record['last_check'].isoformat()
                    if record.get('created_at'):
                        record['created_at'] = record['created_at'].isoformat()
                    health_records.append(record)
                
                return health_records
                
    except Exception as e:
        logger.error("Failed to get service health", error=str(e))
        return []


# =============================================================================
# CONFIGURATION MANAGEMENT FUNCTIONS
# =============================================================================

def get_configuration(key: Optional[str] = None, 
                     category: Optional[str] = None) -> Dict[str, Any]:
    """
    Get system configuration values
    
    Args:
        key: Optional specific configuration key
        category: Optional category filter
        
    Returns:
        Dictionary of configuration values
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
                        last_modified TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        modified_by VARCHAR(100),
                        active BOOLEAN DEFAULT TRUE
                    )
                """)
                
                # Insert default values if table is empty
                cur.execute("SELECT COUNT(*) as count FROM configuration")
                if cur.fetchone()['count'] == 0:
                    _insert_default_configuration(cur)
                    conn.commit()
                
                # Query configuration
                if key:
                    cur.execute("""
                        SELECT * FROM configuration
                        WHERE key = %s AND active = TRUE
                    """, (key,))
                    result = cur.fetchone()
                    if result:
                        return _parse_config_value(dict(result))
                    return {}
                    
                elif category:
                    cur.execute("""
                        SELECT * FROM configuration
                        WHERE category = %s AND active = TRUE
                    """, (category,))
                else:
                    cur.execute("""
                        SELECT * FROM configuration
                        WHERE active = TRUE
                    """)
                
                # Build configuration dictionary
                config = {}
                for row in cur.fetchall():
                    parsed = _parse_config_value(dict(row))
                    config[row['key']] = parsed
                
                return config
                
    except Exception as e:
        logger.error("Failed to get configuration", error=str(e))
        return {}


def update_configuration(key: str, value: Any, 
                        modified_by: str = 'system') -> bool:
    """
    Update a configuration value
    
    Args:
        key: Configuration key
        value: New value
        modified_by: Who is making the change
        
    Returns:
        True if updated successfully
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Determine data type
                if isinstance(value, bool):
                    data_type = 'boolean'
                    value_str = 'true' if value else 'false'
                elif isinstance(value, int):
                    data_type = 'int'
                    value_str = str(value)
                elif isinstance(value, float):
                    data_type = 'float'
                    value_str = str(value)
                elif isinstance(value, dict) or isinstance(value, list):
                    data_type = 'json'
                    value_str = json.dumps(value)
                else:
                    data_type = 'string'
                    value_str = str(value)
                
                # Update configuration
                cur.execute("""
                    UPDATE configuration
                    SET value = %s,
                        data_type = %s,
                        last_modified = CURRENT_TIMESTAMP,
                        modified_by = %s
                    WHERE key = %s
                """, (value_str, data_type, modified_by, key))
                
                updated = cur.rowcount > 0
                conn.commit()
                
                if updated:
                    logger.info(f"Updated configuration", 
                               key=key, value=value_str, 
                               modified_by=modified_by)
                
                return updated
                
    except Exception as e:
        logger.error("Failed to update configuration", error=str(e))
        return False


def _parse_config_value(config_row: Dict) -> Any:
    """Parse configuration value based on data type"""
    value = config_row['value']
    data_type = config_row.get('data_type', 'string')
    
    try:
        if data_type == 'int':
            return int(value)
        elif data_type == 'float':
            return float(value)
        elif data_type == 'boolean':
            return value.lower() in ('true', '1', 'yes', 'on')
        elif data_type == 'json':
            return json.loads(value)
        else:
            return value
    except:
        return value


def _insert_default_configuration(cursor):
    """Insert default configuration values"""
    defaults = [
        # Risk Management
        ('max_positions', '5', 'int', 'risk', 'Maximum concurrent positions'),
        ('position_size_pct', '20', 'float', 'risk', 'Max position size as % of capital'),
        ('premarket_position_pct', '10', 'float', 'risk', 'Pre-market position size limit'),
        ('stop_loss_pct', '2', 'float', 'risk', 'Default stop loss percentage'),
        ('max_daily_trades', '20', 'int', 'risk', 'Maximum trades per day'),
        
        # Trading Parameters
        ('min_catalyst_score', '30', 'float', 'trading', 'Minimum score to consider'),
        ('min_price', '1.0', 'float', 'trading', 'Minimum stock price'),
        ('max_price', '500.0', 'float', 'trading', 'Maximum stock price'),
        ('min_volume', '500000', 'int', 'trading', 'Minimum volume threshold'),
        ('min_relative_volume', '1.5', 'float', 'trading', 'Minimum relative volume'),
        
        # Schedule Configuration
        ('premarket_start', '04:00', 'string', 'schedule', 'Pre-market start time EST'),
        ('premarket_end', '09:30', 'string', 'schedule', 'Pre-market end time EST'),
        ('premarket_interval', '5', 'int', 'schedule', 'Pre-market scan interval minutes'),
        ('market_interval', '30', 'int', 'schedule', 'Regular market scan interval minutes'),
        ('afterhours_interval', '60', 'int', 'schedule', 'After-hours scan interval minutes'),
        
        # API Configuration
        ('news_cache_ttl', '3600', 'int', 'api', 'News cache TTL in seconds'),
        ('api_timeout', '30', 'int', 'api', 'Default API timeout in seconds'),
        ('max_news_age_hours', '24', 'int', 'api', 'Maximum news age for consideration'),
        
        # Source Weights
        ('tier_1_weight', '1.0', 'float', 'sources', 'Tier 1 source weight'),
        ('tier_2_weight', '0.8', 'float', 'sources', 'Tier 2 source weight'),
        ('tier_3_weight', '0.5', 'float', 'sources', 'Tier 3 source weight'),
        ('tier_4_weight', '0.3', 'float', 'sources', 'Tier 4 source weight'),
        ('tier_5_weight', '0.1', 'float', 'sources', 'Tier 5 source weight'),
    ]
    
    for key, value, data_type, category, description in defaults:
        cursor.execute("""
            INSERT INTO configuration (key, value, data_type, category, description, modified_by)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (key) DO NOTHING
        """, (key, value, data_type, category, description, 'system'))


# =============================================================================
# TRADING SIGNAL FUNCTIONS - CRITICAL FIX FROM v236
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
# NEWS DATABASE FUNCTIONS
# =============================================================================

def insert_news_article(news_data: Dict) -> str:
    """
    Insert a news article with deduplication
    
    Args:
        news_data: Dictionary containing news information
        
    Returns:
        news_id of the inserted or existing article
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS news_raw (
                        news_id VARCHAR(255) PRIMARY KEY,
                        symbol VARCHAR(10),
                        headline TEXT NOT NULL,
                        source VARCHAR(100) NOT NULL,
                        published_timestamp TIMESTAMPTZ NOT NULL,
                        url TEXT,
                        content TEXT,
                        sentiment_score DECIMAL(5,2),
                        is_pre_market BOOLEAN DEFAULT FALSE,
                        market_state VARCHAR(20),
                        source_tier INTEGER DEFAULT 5,
                        confirmation_status VARCHAR(50),
                        narrative_cluster_id VARCHAR(100),
                        headline_keywords TEXT[],
                        mentioned_tickers TEXT[],
                        metadata JSONB,
                        collected_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_news_symbol 
                    ON news_raw(symbol)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_news_published 
                    ON news_raw(published_timestamp DESC)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_news_source_tier 
                    ON news_raw(source_tier)
                """)
                
                # Try to insert, handling duplicates
                cur.execute("""
                    INSERT INTO news_raw (
                        news_id, symbol, headline, source, published_timestamp,
                        url, content, sentiment_score, is_pre_market, market_state,
                        source_tier, confirmation_status, narrative_cluster_id,
                        headline_keywords, mentioned_tickers, metadata
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (news_id) DO UPDATE 
                    SET collected_at = CURRENT_TIMESTAMP
                    RETURNING news_id
                """, (
                    news_data.get('news_id'),
                    news_data.get('symbol'),
                    news_data.get('headline'),
                    news_data.get('source'),
                    news_data.get('published_timestamp'),
                    news_data.get('url'),
                    news_data.get('content'),
                    news_data.get('sentiment_score'),
                    news_data.get('is_pre_market', False),
                    news_data.get('market_state'),
                    news_data.get('source_tier', 5),
                    news_data.get('confirmation_status'),
                    news_data.get('narrative_cluster_id'),
                    news_data.get('headline_keywords'),
                    news_data.get('mentioned_tickers'),
                    json.dumps(news_data.get('metadata', {}))
                ))
                
                result = cur.fetchone()
                conn.commit()
                
                return result['news_id']
                
    except Exception as e:
        logger.error("Failed to insert news article", error=str(e))
        raise DatabaseError(f"Failed to insert news article: {str(e)}")


def get_recent_news(symbol: Optional[str] = None, hours: int = 24, limit: int = 100) -> List[Dict]:
    """
    Get recent news articles
    
    Args:
        symbol: Optional symbol to filter by
        hours: Number of hours to look back
        limit: Maximum number of articles to return
        
    Returns:
        List of news article dictionaries
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if symbol:
                    cur.execute("""
                        SELECT * FROM news_raw
                        WHERE symbol = %s
                        AND published_timestamp > CURRENT_TIMESTAMP - INTERVAL '%s hours'
                        ORDER BY published_timestamp DESC
                        LIMIT %s
                    """, (symbol, hours, limit))
                else:
                    cur.execute("""
                        SELECT * FROM news_raw
                        WHERE published_timestamp > CURRENT_TIMESTAMP - INTERVAL '%s hours'
                        ORDER BY published_timestamp DESC
                        LIMIT %s
                    """, (hours, limit))
                
                articles = []
                for row in cur.fetchall():
                    article = dict(row)
                    # Convert timestamps
                    for field in ['published_timestamp', 'collected_at']:
                        if article.get(field):
                            article[field] = article[field].isoformat()
                    articles.append(article)
                
                return articles
                
    except Exception as e:
        logger.error("Failed to get recent news", error=str(e))
        return []


# =============================================================================
# SCANNER SERVICE FUNCTIONS
# =============================================================================

def insert_trading_candidate(candidate_data: Dict) -> int:
    """
    Insert a trading candidate from scanner results
    
    Args:
        candidate_data: Dictionary with candidate information
        
    Returns:
        ID of the inserted candidate
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trading_candidates (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        scan_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        scan_type VARCHAR(20),
                        catalyst_score DECIMAL(5,2),
                        catalyst_type VARCHAR(50),
                        primary_news_id VARCHAR(255),
                        volume_ratio DECIMAL(5,2),
                        price_change_pct DECIMAL(5,2),
                        market_cap BIGINT,
                        selection_rank INTEGER,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Insert candidate
                cur.execute("""
                    INSERT INTO trading_candidates (
                        symbol, scan_type, catalyst_score, catalyst_type,
                        primary_news_id, volume_ratio, price_change_pct,
                        market_cap, selection_rank, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    candidate_data.get('symbol'),
                    candidate_data.get('scan_type', 'regular'),
                    candidate_data.get('catalyst_score'),
                    candidate_data.get('catalyst_type'),
                    candidate_data.get('primary_news_id'),
                    candidate_data.get('volume_ratio'),
                    candidate_data.get('price_change_pct'),
                    candidate_data.get('market_cap'),
                    candidate_data.get('selection_rank'),
                    json.dumps(candidate_data.get('metadata', {}))
                ))
                
                candidate_id = cur.fetchone()['id']
                conn.commit()
                
                logger.info(f"Inserted trading candidate", 
                           candidate_id=candidate_id, 
                           symbol=candidate_data.get('symbol'))
                
                return candidate_id
                
    except Exception as e:
        logger.error("Failed to insert trading candidate", error=str(e))
        raise DatabaseError(f"Failed to insert trading candidate: {str(e)}")


def get_active_candidates(scan_type: Optional[str] = None) -> List[Dict]:
    """
    Get active trading candidates from recent scans
    
    Args:
        scan_type: Optional scan type filter ('premarket', 'regular', etc.)
        
    Returns:
        List of candidate dictionaries
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create default query
                base_query = """
                    SELECT 
                        tc.*,
                        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - tc.created_at)) as age_seconds
                    FROM trading_candidates tc
                """
                
                # Add WHERE conditions
                conditions = ["created_at > CURRENT_TIMESTAMP - INTERVAL '1 hour'"]
                params = []
                
                if scan_type:
                    conditions.append("scan_type = %s")
                    params.append(scan_type)
                
                query = base_query + " WHERE " + " AND ".join(conditions)
                query += " ORDER BY catalyst_score DESC, selection_rank ASC LIMIT 10"
                
                cur.execute(query, params)
                
                candidates = []
                for row in cur.fetchall():
                    candidate = dict(row)
                    # Convert timestamps
                    for field in ['scan_timestamp', 'created_at']:
                        if candidate.get(field):
                            candidate[field] = candidate[field].isoformat()
                    candidates.append(candidate)
                
                return candidates
                
    except Exception as e:
        logger.error("Failed to get active candidates", error=str(e))
        return []


# =============================================================================
# PATTERN ANALYSIS FUNCTIONS  
# =============================================================================

def insert_pattern_detection(pattern_data: Dict) -> int:
    """
    Insert a detected pattern
    
    Args:
        pattern_data: Dictionary with pattern information
        
    Returns:
        ID of the inserted pattern
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS pattern_analysis (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        pattern_type VARCHAR(50) NOT NULL,
                        timeframe VARCHAR(10) DEFAULT '5min',
                        detection_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        pattern_strength DECIMAL(5,2),
                        support_level DECIMAL(10,2),
                        resistance_level DECIMAL(10,2),
                        volume_confirmation BOOLEAN,
                        trend_confirmation BOOLEAN,
                        pattern_completed BOOLEAN,
                        actual_move DECIMAL(5,2),
                        success BOOLEAN,
                        metadata JSONB
                    )
                """)
                
                # Insert pattern
                cur.execute("""
                    INSERT INTO pattern_analysis (
                        symbol, pattern_type, timeframe, pattern_strength,
                        support_level, resistance_level, volume_confirmation,
                        trend_confirmation, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    pattern_data.get('symbol'),
                    pattern_data.get('pattern_type'),
                    pattern_data.get('timeframe', '5min'),
                    pattern_data.get('pattern_strength'),
                    pattern_data.get('support_level'),
                    pattern_data.get('resistance_level'),
                    pattern_data.get('volume_confirmation'),
                    pattern_data.get('trend_confirmation'),
                    json.dumps(pattern_data.get('metadata', {}))
                ))
                
                pattern_id = cur.fetchone()['id']
                conn.commit()
                
                logger.info(f"Inserted pattern detection", 
                           pattern_id=pattern_id, 
                           symbol=pattern_data.get('symbol'),
                           pattern=pattern_data.get('pattern_type'))
                
                return pattern_id
                
    except Exception as e:
        logger.error("Failed to insert pattern detection", error=str(e))
        raise DatabaseError(f"Failed to insert pattern detection: {str(e)}")


# =============================================================================
# TECHNICAL INDICATOR FUNCTIONS
# =============================================================================

def insert_technical_indicators(indicator_data: Dict) -> int:
    """
    Insert calculated technical indicators
    
    Args:
        indicator_data: Dictionary with technical indicator values
        
    Returns:
        ID of the inserted record
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS technical_indicators (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        calculated_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        timeframe VARCHAR(10) DEFAULT '5min',
                        open_price DECIMAL(10,2),
                        high_price DECIMAL(10,2),
                        low_price DECIMAL(10,2),
                        close_price DECIMAL(10,2),
                        volume BIGINT,
                        rsi DECIMAL(5,2),
                        macd DECIMAL(10,4),
                        macd_signal DECIMAL(10,4),
                        sma_20 DECIMAL(10,2),
                        sma_50 DECIMAL(10,2),
                        ema_9 DECIMAL(10,2),
                        atr DECIMAL(10,4),
                        bollinger_upper DECIMAL(10,2),
                        bollinger_lower DECIMAL(10,2),
                        volume_sma DECIMAL(15,2),
                        relative_volume DECIMAL(5,2),
                        metadata JSONB
                    )
                """)
                
                # Insert indicators
                cur.execute("""
                    INSERT INTO technical_indicators (
                        symbol, timeframe, open_price, high_price, low_price,
                        close_price, volume, rsi, macd, macd_signal,
                        sma_20, sma_50, ema_9, atr, bollinger_upper,
                        bollinger_lower, volume_sma, relative_volume, metadata
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    RETURNING id
                """, (
                    indicator_data.get('symbol'),
                    indicator_data.get('timeframe', '5min'),
                    indicator_data.get('open_price'),
                    indicator_data.get('high_price'),
                    indicator_data.get('low_price'),
                    indicator_data.get('close_price'),
                    indicator_data.get('volume'),
                    indicator_data.get('rsi'),
                    indicator_data.get('macd'),
                    indicator_data.get('macd_signal'),
                    indicator_data.get('sma_20'),
                    indicator_data.get('sma_50'),
                    indicator_data.get('ema_9'),
                    indicator_data.get('atr'),
                    indicator_data.get('bollinger_upper'),
                    indicator_data.get('bollinger_lower'),
                    indicator_data.get('volume_sma'),
                    indicator_data.get('relative_volume'),
                    json.dumps(indicator_data.get('metadata', {}))
                ))
                
                indicator_id = cur.fetchone()['id']
                conn.commit()
                
                return indicator_id
                
    except Exception as e:
        logger.error("Failed to insert technical indicators", error=str(e))
        raise DatabaseError(f"Failed to insert technical indicators: {str(e)}")


# =============================================================================
# TRADING SERVICE FUNCTIONS
# =============================================================================

def insert_trade_record(trade_data: Dict) -> int:
    """
    Insert a new trade record into the database
    
    Args:
        trade_data: Dictionary containing trade information including:
            - symbol: Stock symbol
            - signal_id: ID of the signal that triggered this trade
            - entry_price: Entry price
            - quantity: Number of shares
            - order_type: Type of order (market, limit, etc.)
            - side: buy/sell
            - stop_loss: Stop loss price
            - take_profit: Take profit price
            
    Returns:
        ID of the inserted trade record
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trade_records (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        signal_id INTEGER,
                        order_id VARCHAR(100),
                        side VARCHAR(10) NOT NULL,
                        order_type VARCHAR(20),
                        quantity INTEGER NOT NULL,
                        entry_price DECIMAL(10,2),
                        exit_price DECIMAL(10,2),
                        stop_loss DECIMAL(10,2),
                        take_profit DECIMAL(10,2),
                        entry_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        exit_timestamp TIMESTAMP,
                        status VARCHAR(20) DEFAULT 'open',
                        pnl_amount DECIMAL(10,2),
                        pnl_percentage DECIMAL(5,2),
                        commission DECIMAL(10,2),
                        entry_reason TEXT,
                        exit_reason TEXT,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trades_symbol 
                    ON trade_records(symbol)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trades_status 
                    ON trade_records(status)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trades_entry_timestamp 
                    ON trade_records(entry_timestamp DESC)
                """)
                
                # Insert trade record
                cur.execute("""
                    INSERT INTO trade_records (
                        symbol, signal_id, order_id, side, order_type,
                        quantity, entry_price, stop_loss, take_profit,
                        entry_reason, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    trade_data.get('symbol'),
                    trade_data.get('signal_id'),
                    trade_data.get('order_id'),
                    trade_data.get('side', 'buy'),
                    trade_data.get('order_type', 'market'),
                    trade_data.get('quantity'),
                    trade_data.get('entry_price'),
                    trade_data.get('stop_loss'),
                    trade_data.get('take_profit'),
                    trade_data.get('entry_reason'),
                    json.dumps(trade_data.get('metadata', {}))
                ))
                
                trade_id = cur.fetchone()['id']
                conn.commit()
                
                logger.info(f"Inserted trade record", 
                           trade_id=trade_id, 
                           symbol=trade_data.get('symbol'))
                
                return trade_id
                
    except Exception as e:
        logger.error("Failed to insert trade record", error=str(e))
        raise DatabaseError(f"Failed to insert trade record: {str(e)}")


def update_trade_exit(trade_id: int, exit_data: Dict) -> bool:
    """
    Update trade record with exit information
    
    Args:
        trade_id: ID of the trade to update
        exit_data: Dictionary containing exit information:
            - exit_price: Exit price
            - exit_reason: Reason for exit (stop_loss, take_profit, manual, etc.)
            - exit_timestamp: Optional exit timestamp
            
    Returns:
        True if updated successfully, False otherwise
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get current trade info for P&L calculation
                cur.execute("""
                    SELECT entry_price, quantity, side 
                    FROM trade_records 
                    WHERE id = %s AND status = 'open'
                """, (trade_id,))
                
                trade = cur.fetchone()
                if not trade:
                    logger.warning(f"Trade not found or already closed", trade_id=trade_id)
                    return False
                
                # Calculate P&L
                entry_price = float(trade['entry_price'])
                exit_price = float(exit_data.get('exit_price'))
                quantity = int(trade['quantity'])
                side = trade['side']
                
                if side == 'buy':
                    pnl_amount = (exit_price - entry_price) * quantity
                else:  # sell/short
                    pnl_amount = (entry_price - exit_price) * quantity
                
                pnl_percentage = (pnl_amount / (entry_price * quantity)) * 100
                
                # Update trade record
                cur.execute("""
                    UPDATE trade_records 
                    SET exit_price = %s,
                        exit_timestamp = %s,
                        exit_reason = %s,
                        pnl_amount = %s,
                        pnl_percentage = %s,
                        status = 'closed'
                    WHERE id = %s
                """, (
                    exit_data.get('exit_price'),
                    exit_data.get('exit_timestamp') or datetime.now(),
                    exit_data.get('exit_reason'),
                    pnl_amount,
                    pnl_percentage,
                    trade_id
                ))
                
                updated = cur.rowcount > 0
                conn.commit()
                
                if updated:
                    logger.info(f"Updated trade exit", 
                               trade_id=trade_id, 
                               pnl=pnl_amount)
                
                return updated
                
    except Exception as e:
        logger.error("Failed to update trade exit", error=str(e))
        return False


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
                        CURRENT_TIMESTAMP - entry_timestamp as position_age
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
                    # Convert timedelta to seconds
                    if position.get('position_age'):
                        position['position_age_seconds'] = position['position_age'].total_seconds()
                        del position['position_age']
                    positions.append(position)
                
                logger.info(f"Retrieved {len(positions)} open positions")
                return positions
                
    except Exception as e:
        logger.error("Failed to get open positions", error=str(e))
        return []


def get_position_by_symbol(symbol: str) -> Optional[Dict]:
    """
    Get open position for a specific symbol
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Position dictionary or None if no open position
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM trade_records
                    WHERE symbol = %s AND status = 'open'
                    ORDER BY entry_timestamp DESC
                    LIMIT 1
                """, (symbol,))
                
                result = cur.fetchone()
                if result:
                    position = dict(result)
                    if position.get('entry_timestamp'):
                        position['entry_timestamp'] = position['entry_timestamp'].isoformat()
                    return position
                
                return None
                
    except Exception as e:
        logger.error("Failed to get position by symbol", error=str(e))
        return None


def get_trade_history(symbol: Optional[str] = None, days: int = 30) -> List[Dict]:
    """
    Get historical trades
    
    Args:
        symbol: Optional symbol to filter by
        days: Number of days to look back
        
    Returns:
        List of historical trades
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if symbol:
                    cur.execute("""
                        SELECT * FROM trade_records
                        WHERE symbol = %s
                        AND entry_timestamp > CURRENT_TIMESTAMP - INTERVAL '%s days'
                        ORDER BY entry_timestamp DESC
                    """, (symbol, days))
                else:
                    cur.execute("""
                        SELECT * FROM trade_records
                        WHERE entry_timestamp > CURRENT_TIMESTAMP - INTERVAL '%s days'
                        ORDER BY entry_timestamp DESC
                        LIMIT 100
                    """, (days,))
                
                trades = []
                for row in cur.fetchall():
                    trade = dict(row)
                    # Convert timestamps
                    for field in ['entry_timestamp', 'exit_timestamp', 'created_at']:
                        if trade.get(field):
                            trade[field] = trade[field].isoformat()
                    trades.append(trade)
                
                return trades
                
    except Exception as e:
        logger.error("Failed to get trade history", error=str(e))
        return []


def calculate_portfolio_metrics() -> Dict:
    """
    Calculate portfolio performance metrics
    
    Returns:
        Dictionary with portfolio metrics
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get closed trades for metrics
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN pnl_amount > 0 THEN 1 END) as winning_trades,
                        COUNT(CASE WHEN pnl_amount < 0 THEN 1 END) as losing_trades,
                        SUM(pnl_amount) as total_pnl,
                        AVG(pnl_percentage) as avg_pnl_pct,
                        MAX(pnl_amount) as best_trade,
                        MIN(pnl_amount) as worst_trade
                    FROM trade_records
                    WHERE status = 'closed'
                    AND exit_timestamp > CURRENT_TIMESTAMP - INTERVAL '30 days'
                """)
                
                metrics = dict(cur.fetchone())
                
                # Calculate win rate
                if metrics['total_trades'] > 0:
                    metrics['win_rate'] = (metrics['winning_trades'] / metrics['total_trades']) * 100
                else:
                    metrics['win_rate'] = 0
                
                # Get open positions value
                cur.execute("""
                    SELECT 
                        COUNT(*) as open_positions,
                        SUM(quantity * entry_price) as open_value
                    FROM trade_records
                    WHERE status = 'open'
                """)
                
                open_info = dict(cur.fetchone())
                metrics.update(open_info)
                
                return metrics
                
    except Exception as e:
        logger.error("Failed to calculate portfolio metrics", error=str(e))
        return {}


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


# Initialize connections on module import
try:
    init_db_pool()
    init_redis_client()
except Exception as e:
    logger.warning("Failed to initialize database utilities on import", error=str(e))
    # Don't raise here - let services handle initialization in their __init__
