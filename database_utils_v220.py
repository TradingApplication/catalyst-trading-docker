#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: database_utils.py
Version: 2.2.0
Last Updated: 2025-01-03
Purpose: Centralized database connection management for all services

REVISION HISTORY:
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
from typing import Optional, Dict, Any
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


# Initialize connections on module import
try:
    init_db_pool()
    init_redis_client()
except Exception as e:
    logger.warning("Failed to initialize database utilities on import", error=str(e))
    # Don't raise here - let services handle initialization in their __init__