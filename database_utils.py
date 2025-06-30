#!/usr/bin/env python3
"""
Database Utility Functions for Catalyst Trading System
Provides connection pooling, common queries, and error handling
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
from decimal import Decimal

import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import ThreadedConnectionPool
import redis
from structlog import get_logger

# Initialize logger
logger = get_logger()

# Database connection pool
_db_pool = None
_redis_client = None


class DatabaseError(Exception):
    """Custom exception for database operations"""
    pass


def init_db_pool(min_conn: int = 2, max_conn: int = 10) -> ThreadedConnectionPool:
    """Initialize database connection pool"""
    global _db_pool
    
    if _db_pool is None:
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                raise DatabaseError("DATABASE_URL environment variable not set")
            
            _db_pool = ThreadedConnectionPool(
                min_conn,
                max_conn,
                database_url,
                cursor_factory=RealDictCursor
            )
            logger.info("Database connection pool initialized", 
                       min_connections=min_conn, 
                       max_connections=max_conn)
        except Exception as e:
            logger.error("Failed to initialize database pool", error=str(e))
            raise DatabaseError(f"Database pool initialization failed: {e}")
    
    return _db_pool


def init_redis_client() -> redis.Redis:
    """Initialize Redis client"""
    global _redis_client
    
    if _redis_client is None:
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            _redis_client = redis.from_url(redis_url, decode_responses=True)
            _redis_client.ping()
            logger.info("Redis client initialized", url=redis_url)
        except Exception as e:
            logger.error("Failed to initialize Redis client", error=str(e))
            raise DatabaseError(f"Redis initialization failed: {e}")
    
    return _redis_client


@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    pool = init_db_pool()
    conn = None
    try:
        conn = pool.getconn()
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error("Database operation failed", error=str(e))
        raise
    finally:
        if conn:
            pool.putconn(conn)


def get_redis() -> redis.Redis:
    """Get Redis client instance"""
    return init_redis_client()


# =============================================================================
# NEWS OPERATIONS
# =============================================================================

def insert_news_article(article: Dict[str, Any]) -> Optional[str]:
    """Insert a news article into the database"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    INSERT INTO news_raw (
                        news_id, symbol, headline, source, source_url,
                        published_timestamp, content_snippet, full_url,
                        metadata, is_pre_market, market_state,
                        headline_keywords, mentioned_tickers, source_tier
                    ) VALUES (
                        %(news_id)s, %(symbol)s, %(headline)s, %(source)s, %(source_url)s,
                        %(published_timestamp)s, %(content_snippet)s, %(full_url)s,
                        %(metadata)s, %(is_pre_market)s, %(market_state)s,
                        %(headline_keywords)s, %(mentioned_tickers)s, %(source_tier)s
                    )
                    ON CONFLICT (news_id) DO NOTHING
                    RETURNING news_id
                """, {
                    'news_id': article['news_id'],
                    'symbol': article.get('symbol'),
                    'headline': article['headline'],
                    'source': article['source'],
                    'source_url': article.get('source_url'),
                    'published_timestamp': article['published_timestamp'],
                    'content_snippet': article.get('content_snippet'),
                    'full_url': article.get('full_url'),
                    'metadata': Json(article.get('metadata', {})),
                    'is_pre_market': article.get('is_pre_market', False),
                    'market_state': article.get('market_state'),
                    'headline_keywords': Json(article.get('headline_keywords', [])),
                    'mentioned_tickers': Json(article.get('mentioned_tickers', [])),
                    'source_tier': article.get('source_tier', 5)
                })
                
                result = cur.fetchone()
                return result['news_id'] if result else None
                
            except Exception as e:
                logger.error("Failed to insert news article", 
                           headline=article.get('headline'), 
                           error=str(e))
                raise


def get_recent_news(symbol: Optional[str] = None, 
                   hours: int = 24, 
                   limit: int = 100) -> List[Dict]:
    """Get recent news articles"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT * FROM news_raw
                WHERE published_timestamp > NOW() - INTERVAL '%s hours'
            """
            params = [hours]
            
            if symbol:
                query += " AND (symbol = %s OR %s = ANY(mentioned_tickers))"
                params.extend([symbol, symbol])
            
            query += " ORDER BY published_timestamp DESC LIMIT %s"
            params.append(limit)
            
            cur.execute(query, params)
            return cur.fetchall()


# =============================================================================
# TRADING CANDIDATES OPERATIONS
# =============================================================================

def insert_trading_candidates(candidates: List[Dict[str, Any]], scan_id: str) -> int:
    """Insert multiple trading candidates"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            inserted = 0
            for candidate in candidates:
                try:
                    cur.execute("""
                        INSERT INTO trading_candidates (
                            scan_id, symbol, catalyst_score, news_count,
                            primary_catalyst, catalyst_keywords, price,
                            volume, relative_volume, price_change_pct,
                            pre_market_volume, pre_market_change,
                            has_pre_market_news, technical_score,
                            combined_score, selection_rank
                        ) VALUES (
                            %(scan_id)s, %(symbol)s, %(catalyst_score)s, %(news_count)s,
                            %(primary_catalyst)s, %(catalyst_keywords)s, %(price)s,
                            %(volume)s, %(relative_volume)s, %(price_change_pct)s,
                            %(pre_market_volume)s, %(pre_market_change)s,
                            %(has_pre_market_news)s, %(technical_score)s,
                            %(combined_score)s, %(selection_rank)s
                        )
                    """, {
                        'scan_id': scan_id,
                        'symbol': candidate['symbol'],
                        'catalyst_score': candidate['catalyst_score'],
                        'news_count': candidate.get('news_count', 0),
                        'primary_catalyst': candidate.get('primary_catalyst'),
                        'catalyst_keywords': Json(candidate.get('catalyst_keywords', [])),
                        'price': candidate.get('price'),
                        'volume': candidate.get('volume'),
                        'relative_volume': candidate.get('relative_volume'),
                        'price_change_pct': candidate.get('price_change_pct'),
                        'pre_market_volume': candidate.get('pre_market_volume'),
                        'pre_market_change': candidate.get('pre_market_change'),
                        'has_pre_market_news': candidate.get('has_pre_market_news', False),
                        'technical_score': candidate.get('technical_score'),
                        'combined_score': candidate.get('combined_score'),
                        'selection_rank': candidate.get('selection_rank')
                    })
                    inserted += 1
                except Exception as e:
                    logger.error("Failed to insert candidate", 
                               symbol=candidate.get('symbol'), 
                               error=str(e))
            
            return inserted


def get_active_candidates(limit: int = 10) -> List[Dict]:
    """Get active trading candidates that haven't been traded"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM trading_candidates
                WHERE NOT traded
                AND selection_timestamp > NOW() - INTERVAL '2 hours'
                ORDER BY combined_score DESC, selection_timestamp DESC
                LIMIT %s
            """, [limit])
            return cur.fetchall()


# =============================================================================
# TRADING SIGNALS OPERATIONS
# =============================================================================

def insert_trading_signal(signal: Dict[str, Any]) -> str:
    """Insert a trading signal"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trading_signals (
                    signal_id, symbol, signal_type, confidence,
                    catalyst_score, pattern_score, technical_score, volume_score,
                    recommended_entry, stop_loss, target_1, target_2,
                    catalyst_type, detected_patterns, key_factors,
                    position_size_pct, risk_reward_ratio
                ) VALUES (
                    %(signal_id)s, %(symbol)s, %(signal_type)s, %(confidence)s,
                    %(catalyst_score)s, %(pattern_score)s, %(technical_score)s, %(volume_score)s,
                    %(recommended_entry)s, %(stop_loss)s, %(target_1)s, %(target_2)s,
                    %(catalyst_type)s, %(detected_patterns)s, %(key_factors)s,
                    %(position_size_pct)s, %(risk_reward_ratio)s
                )
                ON CONFLICT (signal_id) DO NOTHING
                RETURNING signal_id
            """, {
                'signal_id': signal['signal_id'],
                'symbol': signal['symbol'],
                'signal_type': signal['signal_type'],
                'confidence': signal['confidence'],
                'catalyst_score': signal.get('catalyst_score'),
                'pattern_score': signal.get('pattern_score'),
                'technical_score': signal.get('technical_score'),
                'volume_score': signal.get('volume_score'),
                'recommended_entry': signal.get('recommended_entry'),
                'stop_loss': signal.get('stop_loss'),
                'target_1': signal.get('target_1'),
                'target_2': signal.get('target_2'),
                'catalyst_type': signal.get('catalyst_type'),
                'detected_patterns': Json(signal.get('detected_patterns', [])),
                'key_factors': Json(signal.get('key_factors', [])),
                'position_size_pct': signal.get('position_size_pct'),
                'risk_reward_ratio': signal.get('risk_reward_ratio')
            })
            
            result = cur.fetchone()
            return result['signal_id'] if result else signal['signal_id']


def get_pending_signals() -> List[Dict]:
    """Get trading signals that haven't been executed"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM trading_signals
                WHERE NOT executed
                AND generated_timestamp > NOW() - INTERVAL '1 hour'
                AND confidence >= 50
                ORDER BY confidence DESC, generated_timestamp DESC
            """)
            return cur.fetchall()


# =============================================================================
# TRADE RECORDS OPERATIONS
# =============================================================================

def insert_trade_record(trade: Dict[str, Any]) -> str:
    """Insert a trade record"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trade_records (
                    trade_id, signal_id, symbol, order_type, side,
                    quantity, entry_price, entry_timestamp,
                    entry_catalyst, entry_news_id, catalyst_score_at_entry
                ) VALUES (
                    %(trade_id)s, %(signal_id)s, %(symbol)s, %(order_type)s, %(side)s,
                    %(quantity)s, %(entry_price)s, %(entry_timestamp)s,
                    %(entry_catalyst)s, %(entry_news_id)s, %(catalyst_score_at_entry)s
                )
                RETURNING trade_id
            """, trade)
            
            return cur.fetchone()['trade_id']


def update_trade_exit(trade_id: str, exit_data: Dict[str, Any]) -> bool:
    """Update trade with exit information"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE trade_records
                SET exit_price = %(exit_price)s,
                    exit_timestamp = %(exit_timestamp)s,
                    exit_reason = %(exit_reason)s,
                    pnl_amount = %(pnl_amount)s,
                    pnl_percentage = %(pnl_percentage)s,
                    commission = %(commission)s,
                    max_profit = %(max_profit)s,
                    max_loss = %(max_loss)s
                WHERE trade_id = %(trade_id)s
            """, {
                'trade_id': trade_id,
                **exit_data
            })
            
            return cur.rowcount > 0


def get_open_positions() -> List[Dict]:
    """Get all open trading positions"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM trade_records
                WHERE exit_timestamp IS NULL
                ORDER BY entry_timestamp DESC
            """)
            return cur.fetchall()


# =============================================================================
# PATTERN ANALYSIS OPERATIONS
# =============================================================================

def insert_pattern_detection(pattern: Dict[str, Any]) -> int:
    """Insert a detected pattern"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO pattern_analysis (
                    symbol, timeframe, pattern_type, pattern_direction,
                    confidence, has_catalyst, catalyst_type, catalyst_alignment,
                    pattern_strength, support_level, resistance_level,
                    volume_confirmation, trend_confirmation
                ) VALUES (
                    %(symbol)s, %(timeframe)s, %(pattern_type)s, %(pattern_direction)s,
                    %(confidence)s, %(has_catalyst)s, %(catalyst_type)s, %(catalyst_alignment)s,
                    %(pattern_strength)s, %(support_level)s, %(resistance_level)s,
                    %(volume_confirmation)s, %(trend_confirmation)s
                )
                RETURNING id
            """, pattern)
            
            return cur.fetchone()['id']


# =============================================================================
# TECHNICAL INDICATORS OPERATIONS
# =============================================================================

def insert_technical_indicators(indicators: Dict[str, Any]) -> int:
    """Insert technical indicators"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO technical_indicators (
                    symbol, timeframe, open_price, high_price, low_price, close_price,
                    volume, rsi, macd, macd_signal, sma_20, sma_50, ema_9,
                    atr, bollinger_upper, bollinger_lower,
                    volume_sma, relative_volume
                ) VALUES (
                    %(symbol)s, %(timeframe)s, %(open_price)s, %(high_price)s, 
                    %(low_price)s, %(close_price)s, %(volume)s, %(rsi)s, %(macd)s, 
                    %(macd_signal)s, %(sma_20)s, %(sma_50)s, %(ema_9)s,
                    %(atr)s, %(bollinger_upper)s, %(bollinger_lower)s,
                    %(volume_sma)s, %(relative_volume)s
                )
                RETURNING id
            """, indicators)
            
            return cur.fetchone()['id']


# =============================================================================
# SYSTEM OPERATIONS
# =============================================================================

def create_trading_cycle(mode: str = 'normal') -> str:
    """Create a new trading cycle"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cycle_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            cur.execute("""
                INSERT INTO trading_cycles (cycle_id, start_time, mode)
                VALUES (%s, NOW(), %s)
                RETURNING cycle_id
            """, [cycle_id, mode])
            
            return cur.fetchone()['cycle_id']


def update_trading_cycle(cycle_id: str, updates: Dict[str, Any]) -> bool:
    """Update trading cycle metrics"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            set_clause = ", ".join([f"{k} = %({k})s" for k in updates.keys()])
            query = f"""
                UPDATE trading_cycles
                SET {set_clause}
                WHERE cycle_id = %(cycle_id)s
            """
            
            cur.execute(query, {'cycle_id': cycle_id, **updates})
            return cur.rowcount > 0


def log_workflow_step(cycle_id: str, step_name: str, 
                     status: str, result: Optional[Dict] = None) -> int:
    """Log a workflow step execution"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO workflow_log (
                    cycle_id, step_name, status, start_time, result
                ) VALUES (
                    %s, %s, %s, NOW(), %s
                )
                RETURNING id
            """, [cycle_id, step_name, status, Json(result or {})])
            
            return cur.fetchone()['id']


def update_service_health(service_name: str, status: str, 
                         response_time_ms: Optional[int] = None,
                         error_message: Optional[str] = None) -> None:
    """Update service health status"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO service_health (
                    service_name, status, last_check, response_time_ms, error_message
                ) VALUES (
                    %s, %s, NOW(), %s, %s
                )
            """, [service_name, status, response_time_ms, error_message])


def get_configuration(key: str) -> Optional[str]:
    """Get configuration value"""
    # Try Redis cache first
    redis_client = get_redis()
    cache_key = f"config:{key}"
    
    try:
        value = redis_client.get(cache_key)
        if value:
            return value
    except Exception as e:
        logger.warning("Redis cache error", error=str(e))
    
    # Fallback to database
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT value FROM configuration WHERE key = %s
            """, [key])
            
            result = cur.fetchone()
            if result:
                value = result['value']
                # Cache in Redis for 5 minutes
                try:
                    redis_client.setex(cache_key, 300, value)
                except:
                    pass
                return value
    
    return None


def get_system_overview() -> Dict[str, Any]:
    """Get system overview metrics"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM system_overview")
            results = cur.fetchall()
            
            return {row['metric']: row['value'] for row in results}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clean_old_data(days: int = 30) -> Dict[str, int]:
    """Clean old data from various tables"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cleaned = {}
            
            # Clean old news
            cur.execute("""
                DELETE FROM news_raw
                WHERE collected_timestamp < NOW() - INTERVAL '%s days'
                AND news_id NOT IN (SELECT entry_news_id FROM trade_records WHERE entry_news_id IS NOT NULL)
            """, [days])
            cleaned['news'] = cur.rowcount
            
            # Clean old workflow logs
            cur.execute("""
                DELETE FROM workflow_log
                WHERE created_at < NOW() - INTERVAL '%s days'
            """, [days * 2])
            cleaned['workflow_logs'] = cur.rowcount
            
            # Clean old service health records
            cur.execute("""
                DELETE FROM service_health
                WHERE created_at < NOW() - INTERVAL '7 days'
            """, [])
            cleaned['service_health'] = cur.rowcount
            
            return cleaned


def health_check() -> Dict[str, Any]:
    """Perform database health check"""
    try:
        # Check database connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                db_ok = cur.fetchone() is not None
        
        # Check Redis connection
        redis_client = get_redis()
        redis_ok = redis_client.ping()
        
        return {
            'database': 'healthy' if db_ok else 'unhealthy',
            'redis': 'healthy' if redis_ok else 'unhealthy',
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            'database': 'unhealthy',
            'redis': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


# Initialize on module load
if __name__ != "__main__":
    try:
        init_db_pool()
        init_redis_client()
    except Exception as e:
        logger.warning("Failed to initialize database utilities on import", error=str(e))