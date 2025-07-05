#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: database_utils.py
Version: 2.3.0
Last Updated: 2025-07-04
Purpose: Centralized database connection management for all services

REVISION HISTORY:
v2.3.0 (2025-07-04) - Added missing functions for coordination service
- Added create_trading_cycle() and update_trading_cycle()
- Added log_workflow_step() and update_workflow_step()
- Added update_service_health()
- Added get_configuration() and set_configuration()
- Added insert_news_article() and related functions
- Added insert_trading_signal() and get_pending_signals()
- Added pattern and technical indicator functions
- Maintained backward compatibility with v2.2.0

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
import uuid
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
            
            # Build connection string
            database_url = f"postgresql://{user}:{password}@{host}:{port}/{database}?sslmode=require"
        
        logger.info("Initializing database connection pool", 
                   host=database_url.split('@')[1].split('/')[0])
        
        _db_pool = pool.SimpleConnectionPool(
            min_connections, 
            max_connections,
            database_url
        )
        
        logger.info("Database connection pool initialized", 
                   min_connections=min_connections,
                   max_connections=max_connections)
        
        return _db_pool
        
    except Exception as e:
        logger.error("Failed to initialize database pool", error=str(e))
        raise DatabaseError(f"Database pool initialization failed: {str(e)}")


def init_redis_client() -> redis.Redis:
    """Initialize Redis client with retry logic"""
    global _redis_client
    
    if _redis_client is not None:
        return _redis_client
    
    redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/0')
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            logger.info("Initializing Redis client", url=redis_url)
            
            _redis_client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_keepalive=True,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            _redis_client.ping()
            
            logger.info("Redis client initialized", 
                       url=redis_url.split('@')[-1].split('/')[0])
            
            return _redis_client
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Redis connection attempt {attempt + 1} failed, retrying...",
                             error=str(e))
                time.sleep(retry_delay * (attempt + 1))
            else:
                logger.error("Failed to connect to Redis after all retries", error=str(e))
                # Don't raise - Redis is optional for most operations
                _redis_client = None
                return None


@contextmanager
def get_db_connection():
    """Get a database connection from the pool with automatic cleanup"""
    conn = None
    pool = init_db_pool()
    
    try:
        conn = pool.getconn()
        conn.set_session(autocommit=False)
        conn.cursor_factory = RealDictCursor
        yield conn
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
        'postgresql': {'status': 'unknown', 'error': None},
        'redis': {'status': 'unknown', 'error': None}
    }
    
    # Check PostgreSQL
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                health_status['postgresql']['status'] = 'healthy'
    except Exception as e:
        health_status['postgresql']['status'] = 'unhealthy'
        health_status['postgresql']['error'] = str(e)
    
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
# TRADING CYCLE FUNCTIONS (Added in v2.3.0)
# =============================================================================

def create_trading_cycle(mode: str = 'normal', metadata: Dict = None) -> str:
    """Create a new trading cycle"""
    cycle_id = f"cycle_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trading_cycles (cycle_id, start_time, status, mode, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING cycle_id
            """, (cycle_id, datetime.utcnow(), 'running', mode, json.dumps(metadata or {})))
            
            conn.commit()
            result = cur.fetchone()
            return result['cycle_id'] if result else cycle_id
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to create trading cycle: {e}")
        raise
    finally:
        if conn:
            conn.close()

def update_trading_cycle(cycle_id: str, updates: Dict) -> bool:
    """Update trading cycle with metrics"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Build dynamic update query
            set_clauses = []
            values = []
            
            for key, value in updates.items():
                if key in ['news_collected', 'securities_scanned', 'candidates_selected',
                          'patterns_analyzed', 'signals_generated', 'trades_executed',
                          'cycle_pnl', 'success_rate', 'status']:
                    set_clauses.append(f"{key} = %s")
                    values.append(value)
            
            if 'status' in updates and updates['status'] in ['completed', 'failed']:
                set_clauses.append("end_time = %s")
                values.append(datetime.utcnow())
            
            if set_clauses:
                values.append(cycle_id)
                query = f"UPDATE trading_cycles SET {', '.join(set_clauses)} WHERE cycle_id = %s"
                cur.execute(query, values)
                conn.commit()
                return True
                
        return False
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to update trading cycle: {e}")
        return False
    finally:
        if conn:
            conn.close()

# =============================================================================
# WORKFLOW LOGGING FUNCTIONS (Added in v2.3.0)
# =============================================================================

def log_workflow_step(cycle_id: str, step_name: str, status: str, 
                     result: Any = None, error: str = None) -> int:
    """Log a workflow step execution"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO workflow_log 
                (cycle_id, step_name, status, start_time, result, error_message)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                cycle_id, 
                step_name, 
                status, 
                datetime.utcnow(),
                json.dumps(result) if result else None,
                error
            ))
            
            conn.commit()
            row = cur.fetchone()
            return row['id'] if row else 0
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to log workflow step: {e}")
        return 0
    finally:
        if conn:
            conn.close()

def update_workflow_step(workflow_id: int, status: str, 
                        records_processed: int = None,
                        records_output: int = None,
                        result: Any = None, 
                        error: str = None):
    """Update a workflow step completion"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            end_time = datetime.utcnow()
            
            # Get start time to calculate duration
            cur.execute("SELECT start_time FROM workflow_log WHERE id = %s", (workflow_id,))
            row = cur.fetchone()
            if row:
                duration = (end_time - row['start_time']).total_seconds()
            else:
                duration = None
            
            cur.execute("""
                UPDATE workflow_log 
                SET status = %s, end_time = %s, duration_seconds = %s,
                    records_processed = %s, records_output = %s,
                    result = %s, error_message = %s
                WHERE id = %s
            """, (
                status, end_time, duration,
                records_processed, records_output,
                json.dumps(result) if result else None,
                error, workflow_id
            ))
            
            conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to update workflow step: {e}")
    finally:
        if conn:
            conn.close()

# =============================================================================
# SERVICE HEALTH FUNCTIONS (Added in v2.3.0)
# =============================================================================

def update_service_health(service_name: str, status: str, 
                         response_time_ms: int = None,
                         error_message: str = None,
                         metadata: Dict = None):
    """Update service health status"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Insert new health record
            cur.execute("""
                INSERT INTO service_health 
                (service_name, status, last_check, response_time_ms, 
                 error_message, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (service_name, status, datetime.utcnow(), response_time_ms,
                 error_message, json.dumps(metadata) if metadata else None))
            
            conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to update service health: {e}")
    finally:
        if conn:
            conn.close()

# =============================================================================
# CONFIGURATION FUNCTIONS (Added in v2.3.0)
# =============================================================================

def get_configuration(key: str = None, category: str = None) -> Any:
    """Get configuration values"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            if key:
                cur.execute("SELECT value, data_type FROM configuration WHERE key = %s", (key,))
                row = cur.fetchone()
                if row:
                    return _cast_config_value(row['value'], row['data_type'])
                return None
            elif category:
                cur.execute("SELECT key, value, data_type FROM configuration WHERE category = %s", (category,))
                rows = cur.fetchall()
                return {row['key']: _cast_config_value(row['value'], row['data_type']) for row in rows}
            else:
                cur.execute("SELECT key, value, data_type FROM configuration")
                rows = cur.fetchall()
                return {row['key']: _cast_config_value(row['value'], row['data_type']) for row in rows}
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        return None
    finally:
        if conn:
            conn.close()

def set_configuration(key: str, value: Any, data_type: str = 'string',
                     category: str = None, description: str = None):
    """Set configuration value"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO configuration (key, value, data_type, category, description)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, 
                    data_type = EXCLUDED.data_type,
                    last_modified = CURRENT_TIMESTAMP
            """, (key, str(value), data_type, category, description))
            
            conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to set configuration: {e}")
    finally:
        if conn:
            conn.close()

def _cast_config_value(value: str, data_type: str) -> Any:
    """Cast configuration value to appropriate type"""
    try:
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
    except:
        return value

# =============================================================================
# NEWS FUNCTIONS (Added in v2.3.0)
# =============================================================================

def insert_news_article(article_data: Dict) -> int:
    """Insert news article into news_raw table"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO news_raw (
                    symbol, headline, description, url, source, 
                    published_timestamp, collected_timestamp,
                    sentiment_score, sentiment_label, relevance_score,
                    is_pre_market, source_tier, mentioned_tickers,
                    headline_keywords, metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id
            """, (
                article_data.get('symbol'),
                article_data.get('headline'),
                article_data.get('description'),
                article_data.get('url'),
                article_data.get('source'),
                article_data.get('published_timestamp'),
                article_data.get('collected_timestamp', datetime.utcnow()),
                article_data.get('sentiment_score'),
                article_data.get('sentiment_label'),
                article_data.get('relevance_score'),
                article_data.get('is_pre_market', False),
                article_data.get('source_tier', 5),
                article_data.get('mentioned_tickers', []),
                article_data.get('headline_keywords', []),
                json.dumps(article_data.get('metadata', {}))
            ))
            
            conn.commit()
            row = cur.fetchone()
            return row['id'] if row else 0
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to insert news article: {e}")
        return 0
    finally:
        if conn:
            conn.close()

# =============================================================================
# TRADING FUNCTIONS (Added in v2.3.0)
# =============================================================================

def insert_trading_signal(signal_data: Dict) -> int:
    """Insert a trading signal"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trading_signals (
                    symbol, signal_timestamp, signal_type, confidence_score,
                    entry_price, stop_loss, take_profit, position_size,
                    pattern_basis, catalyst_basis, technical_basis,
                    news_id, metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id
            """, (
                signal_data.get('symbol'),
                signal_data.get('signal_timestamp', datetime.utcnow()),
                signal_data.get('signal_type', 'BUY'),
                signal_data.get('confidence_score'),
                signal_data.get('entry_price'),
                signal_data.get('stop_loss'),
                signal_data.get('take_profit'),
                signal_data.get('position_size'),
                signal_data.get('pattern_basis'),
                signal_data.get('catalyst_basis'),
                signal_data.get('technical_basis'),
                signal_data.get('news_id'),
                json.dumps(signal_data.get('metadata', {}))
            ))
            
            conn.commit()
            row = cur.fetchone()
            return row['id'] if row else 0
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to insert trading signal: {e}")
        return 0
    finally:
        if conn:
            conn.close()

def get_pending_signals() -> List[Dict]:
    """Get pending trading signals"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM trading_signals 
                WHERE executed = FALSE 
                AND signal_timestamp > NOW() - INTERVAL '1 hour'
                ORDER BY confidence_score DESC
            """)
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Failed to get pending signals: {e}")
        return []
    finally:
        if conn:
            conn.close()

# =============================================================================
# PATTERN ANALYSIS FUNCTIONS (Added in v2.3.0)
# =============================================================================

def insert_pattern_analysis(pattern_data: Dict) -> int:
    """Insert pattern analysis result"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO pattern_analysis (
                    symbol, detection_timestamp, pattern_type, pattern_name,
                    confidence, timeframe, catalyst_present, catalyst_type,
                    pattern_strength, support_level, resistance_level,
                    volume_confirmation, trend_confirmation
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id
            """, (
                pattern_data.get('symbol'),
                pattern_data.get('detection_timestamp', datetime.utcnow()),
                pattern_data.get('pattern_type'),
                pattern_data.get('pattern_name'),
                pattern_data.get('confidence'),
                pattern_data.get('timeframe', '5min'),
                pattern_data.get('catalyst_present', False),
                pattern_data.get('catalyst_type'),
                pattern_data.get('pattern_strength'),
                pattern_data.get('support_level'),
                pattern_data.get('resistance_level'),
                pattern_data.get('volume_confirmation', False),
                pattern_data.get('trend_confirmation', False)
            ))
            
            conn.commit()
            row = cur.fetchone()
            return row['id'] if row else 0
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to insert pattern analysis: {e}")
        return 0
    finally:
        if conn:
            conn.close()

# =============================================================================
# TECHNICAL INDICATORS FUNCTIONS (Added in v2.3.0)
# =============================================================================

def insert_technical_indicators(indicator_data: Dict) -> int:
    """Insert technical indicators"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO technical_indicators (
                    symbol, calculated_timestamp, timeframe,
                    open_price, high_price, low_price, close_price, volume,
                    rsi, macd, macd_signal, sma_20, sma_50, ema_9,
                    atr, bollinger_upper, bollinger_lower,
                    volume_sma, relative_volume
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id
            """, (
                indicator_data.get('symbol'),
                indicator_data.get('calculated_timestamp', datetime.utcnow()),
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
                indicator_data.get('relative_volume')
            ))
            
            conn.commit()
            row = cur.fetchone()
            return row['id'] if row else 0
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to insert technical indicators: {e}")
        return 0
    finally:
        if conn:
            conn.close()

# =============================================================================
# Initialize default configuration if needed
# =============================================================================

def init_default_configuration():
    """Initialize default configuration values"""
    defaults = [
        ('max_positions', '5', 'int', 'risk', 'Maximum concurrent positions'),
        ('position_size_pct', '20', 'float', 'risk', 'Max position size as % of capital'),
        ('premarket_position_pct', '10', 'float', 'risk', 'Pre-market position size limit'),
        ('min_catalyst_score', '30', 'float', 'trading', 'Minimum score to consider'),
        ('stop_loss_pct', '2', 'float', 'risk', 'Default stop loss percentage'),
        ('take_profit_pct', '5', 'float', 'risk', 'Default take profit percentage'),
        ('scan_interval_minutes', '5', 'int', 'schedule', 'Scanner interval in minutes'),
        ('pattern_confidence_threshold', '70', 'float', 'trading', 'Minimum pattern confidence'),
    ]
    
    for key, value, data_type, category, description in defaults:
        try:
            set_configuration(key, value, data_type, category, description)
        except Exception as e:
            logger.debug(f"Config {key} may already exist: {e}")

# Initialize connections on module import
try:
    init_db_pool()
    init_redis_client()
    # Only try to init configuration if we have a good database connection
    if _db_pool:
        init_default_configuration()
except Exception as e:
    logger.warning("Failed to initialize database utilities on import", error=str(e))
    # Don't raise here - let services handle initialization in their __init__