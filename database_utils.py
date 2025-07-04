"""
Name of Application: Catalyst Trading System
Name of file: database_utils.py
Version: 2.1.0
Last Updated: 2025-07-04
Purpose: Shared database utilities and helper functions for all services

REVISION HISTORY:
v2.1.0 (2025-07-04) - Complete implementation with all required functions
- Database connection pooling
- Redis connection management
- Trading cycle management
- Workflow logging
- Service health tracking
- Configuration management

Description of Service:
Provides centralized database operations for all microservices
"""

import os
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
import redis
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid

logger = logging.getLogger(__name__)

# Global connection pool
_db_pool = None
_redis_client = None

def init_db_pool():
    """Initialize the database connection pool"""
    global _db_pool
    if _db_pool is None:
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        _db_pool = SimpleConnectionPool(
            1, 20,  # min and max connections
            database_url
        )
        logger.info("Database connection pool initialized")
    return _db_pool

def get_db_connection():
    """Get a connection from the pool"""
    pool = init_db_pool()
    return pool.getconn()

def return_db_connection(conn):
    """Return a connection to the pool"""
    if _db_pool:
        _db_pool.putconn(conn)

def get_redis():
    """Get Redis connection"""
    global _redis_client
    if _redis_client is None:
        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/0')
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        logger.info("Redis client initialized")
    return _redis_client

def get_redis_connection():
    """Alias for get_redis for compatibility"""
    return get_redis()

def health_check():
    """Check database and Redis health"""
    status = {'database': 'unhealthy', 'redis': 'unhealthy'}
    
    # Check database
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return_db_connection(conn)
        status['database'] = 'healthy'
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
    
    # Check Redis
    try:
        r = get_redis()
        if r and r.ping():
            status['redis'] = 'healthy'
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
    
    return status

# =============================================================================
# TRADING CYCLE FUNCTIONS
# =============================================================================

def create_trading_cycle(mode: str = 'normal', metadata: Dict = None) -> str:
    """Create a new trading cycle"""
    cycle_id = f"cycle_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trading_cycles (cycle_id, start_time, status, mode, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING cycle_id
            """, (cycle_id, datetime.utcnow(), 'running', mode, json.dumps(metadata or {})))
            
            conn.commit()
            result = cur.fetchone()
            return result['cycle_id']
    finally:
        return_db_connection(conn)

def update_trading_cycle(cycle_id: str, updates: Dict) -> bool:
    """Update trading cycle with metrics"""
    conn = get_db_connection()
    try:
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
    finally:
        return_db_connection(conn)

# =============================================================================
# WORKFLOW LOGGING FUNCTIONS
# =============================================================================

def log_workflow_step(cycle_id: str, step_name: str, status: str, 
                     result: Any = None, error: str = None) -> int:
    """Log a workflow step execution"""
    conn = get_db_connection()
    try:
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
            return cur.fetchone()['id']
    finally:
        return_db_connection(conn)

def update_workflow_step(workflow_id: int, status: str, 
                        records_processed: int = None,
                        records_output: int = None,
                        result: Any = None, 
                        error: str = None):
    """Update a workflow step completion"""
    conn = get_db_connection()
    try:
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
    finally:
        return_db_connection(conn)

# =============================================================================
# SERVICE HEALTH FUNCTIONS
# =============================================================================

def update_service_health(service_name: str, status: str, 
                         response_time_ms: int = None,
                         error_message: str = None,
                         metadata: Dict = None):
    """Update service health status"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Check if record exists
            cur.execute("""
                SELECT id FROM service_health 
                WHERE service_name = %s 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (service_name,))
            
            existing = cur.fetchone()
            
            if existing and (datetime.utcnow() - existing.get('last_check', datetime.utcnow())).seconds < 300:
                # Update existing record if less than 5 minutes old
                cur.execute("""
                    UPDATE service_health 
                    SET status = %s, last_check = %s, response_time_ms = %s,
                        error_message = %s, metadata = %s
                    WHERE id = %s
                """, (status, datetime.utcnow(), response_time_ms, 
                     error_message, json.dumps(metadata), existing['id']))
            else:
                # Insert new record
                cur.execute("""
                    INSERT INTO service_health 
                    (service_name, status, last_check, response_time_ms, 
                     error_message, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (service_name, status, datetime.utcnow(), response_time_ms,
                     error_message, json.dumps(metadata)))
            
            conn.commit()
    finally:
        return_db_connection(conn)

# =============================================================================
# CONFIGURATION FUNCTIONS
# =============================================================================

def get_configuration(key: str = None, category: str = None) -> Any:
    """Get configuration values"""
    conn = get_db_connection()
    try:
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
    finally:
        return_db_connection(conn)

def set_configuration(key: str, value: Any, data_type: str = 'string',
                     category: str = None, description: str = None):
    """Set configuration value"""
    conn = get_db_connection()
    try:
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
    finally:
        return_db_connection(conn)

def _cast_config_value(value: str, data_type: str) -> Any:
    """Cast configuration value to appropriate type"""
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

# =============================================================================
# NEWS FUNCTIONS
# =============================================================================

def insert_news_article(article_data: Dict) -> int:
    """Insert news article into news_raw table"""
    conn = get_db_connection()
    try:
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
            return cur.fetchone()['id']
    finally:
        return_db_connection(conn)

# =============================================================================
# TRADING FUNCTIONS
# =============================================================================

def insert_trading_signal(signal_data: Dict) -> int:
    """Insert a trading signal"""
    conn = get_db_connection()
    try:
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
            return cur.fetchone()['id']
    finally:
        return_db_connection(conn)

def get_pending_signals() -> List[Dict]:
    """Get pending trading signals"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM trading_signals 
                WHERE executed = FALSE 
                AND signal_timestamp > NOW() - INTERVAL '1 hour'
                ORDER BY confidence_score DESC
            """)
            return cur.fetchall()
    finally:
        return_db_connection(conn)

# =============================================================================
# PATTERN ANALYSIS FUNCTIONS
# =============================================================================

def insert_pattern_analysis(pattern_data: Dict) -> int:
    """Insert pattern analysis result"""
    conn = get_db_connection()
    try:
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
            return cur.fetchone()['id']
    finally:
        return_db_connection(conn)

# =============================================================================
# TECHNICAL INDICATORS FUNCTIONS
# =============================================================================

def insert_technical_indicators(indicator_data: Dict) -> int:
    """Insert technical indicators"""
    conn = get_db_connection()
    try:
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
            return cur.fetchone()['id']
    finally:
        return_db_connection(conn)

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

# Initialize configuration on module load
try:
    init_default_configuration()
except Exception as e:
    logger.warning(f"Could not initialize default configuration: {e}")