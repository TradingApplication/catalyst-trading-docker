# Catalyst Trading System - Database Services & Data Management v2.1.0

**Version**: 2.1.0  
**Date**: July 8, 2025  
**Platform**: DigitalOcean  
**Purpose**: Define database services and data persistence management aligned with Schema v2.1.0

## Revision History

### v2.1.0 (July 8, 2025)
- **Schema Alignment**: Updated to match Database Schema v2.1.0
- **Service Integration**: Aligned with actual database_utils.py v2.3.5
- **Table References**: Updated all table names to match implementation
- **Connection Management**: Reflects current pooling implementation
- **Performance Updates**: Added specific query patterns for new schema

## Table of Contents

1. [Overview](#1-overview)
2. [Database Service Architecture](#2-database-service-architecture)
3. [Connection Management Service](#3-connection-management-service)
4. [Data Persistence Service](#4-data-persistence-service)
5. [Cache Management Service](#5-cache-management-service)
6. [Database Migration Service](#6-database-migration-service)
7. [Backup & Recovery Service](#7-backup--recovery-service)
8. [Data Synchronization Service](#8-data-synchronization-service)
9. [DigitalOcean Integration](#9-digitalocean-integration)
10. [Performance & Monitoring](#10-performance--monitoring)

---

## 1. Overview

### 1.1 Database Services Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                    DigitalOcean Infrastructure                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Application Services Layer                │   │
│  │  (News, Scanner, Pattern, Technical, Trading, etc.)      │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                             │                                   │
│  ┌─────────────────────────┴───────────────────────────────┐   │
│  │               Database Services Layer (v2.1.0)            │   │
│  │                                                           │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ database_    │  │    Redis     │  │   Migration   │  │   │
│  │  │ utils.py     │  │   Client     │  │   Service     │  │   │
│  │  │ (v2.3.5)     │  │              │  │               │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  │                                                           │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ Persistence  │  │  Monitoring  │  │ Backup/Restore│  │   │
│  │  │  Functions   │  │   Service    │  │   Service     │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Storage Layer                          │   │
│  │                                                           │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ PostgreSQL   │  │    Redis     │  │  DO Spaces    │  │   │
│  │  │  (Managed)   │  │   (Cache)    │  │  (Backups)    │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  └───────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

| Component | Implementation | Purpose |
|-----------|----------------|---------|
| database_utils.py | v2.3.5 | Connection pooling, all DB operations |
| Redis Client | Integrated | Caching, real-time data |
| Migration Scripts | SQL files | Schema version control |
| Monitoring | health_check() | Service health tracking |
| Backup Service | Cron/DO | Automated backups |

### 1.3 Key Tables (Schema v2.1.0)

| Table | Purpose | Primary Key |
|-------|---------|-------------|
| news_raw | Raw news storage | news_id |
| trading_candidates | Scanner results | id |
| trading_signals | Generated signals | signal_id |
| trade_records | Executed trades | trade_id |
| pattern_analysis | Pattern detection | id |
| technical_indicators | TA calculations | id |

---

## 2. Database Service Architecture

### 2.1 Service Integration Map

```python
# Service to Database Function Mapping
SERVICE_DB_FUNCTIONS = {
    'news_service': [
        'insert_news_article',
        'get_recent_news',
        'news_collection_stats'
    ],
    'scanner_service': [
        'insert_trading_candidate',
        'get_candidates_for_analysis'
    ],
    'pattern_service': [
        'insert_pattern_detection',
        'get_pattern_history'
    ],
    'technical_service': [
        'insert_trading_signal',
        'insert_technical_indicators',
        'get_latest_indicators',
        'get_signal_history'
    ],
    'trading_service': [
        'insert_trade_record',
        'update_trade_exit',
        'get_open_positions',
        'get_pending_signals'
    ],
    'reporting_service': [
        # Read-only access to all tables
        'get_trade_history',
        'calculate_portfolio_metrics'
    ]
}
```

### 2.2 Connection Flow

```python
# Actual implementation from database_utils.py v2.3.5
from database_utils import (
    get_db_connection,     # Context manager for connections
    get_redis,            # Redis client instance
    health_check,         # Service health monitoring
    with_db_retry        # Retry decorator
)

# Example service usage
class TradingService:
    def execute_trade(self, signal):
        trade_data = {
            'signal_id': signal['signal_id'],
            'symbol': signal['symbol'],
            'entry_price': signal['recommended_entry'],
            # ... other fields
        }
        
        # Uses connection pool automatically
        trade_id = insert_trade_record(trade_data)
        return trade_id
```

---

## 3. Connection Management Service

### 3.1 PostgreSQL Connection Pooling

```python
# From database_utils.py v2.3.5
class ConnectionManager:
    """Actual implementation in use"""
    
    def __init__(self):
        self.pool = psycopg2.pool.SimpleConnectionPool(
            minconn=2,
            maxconn=10,
            dsn=os.getenv('DATABASE_URL'),
            cursor_factory=RealDictCursor
        )
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with automatic cleanup"""
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
```

### 3.2 Redis Connection Management

```python
# Redis client initialization
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'redis'),
    'port': int(os.getenv('REDIS_PORT', '6379')),
    'password': os.getenv('REDIS_PASSWORD', ''),
    'db': int(os.getenv('REDIS_DB', '0')),
    'decode_responses': True,
    'socket_connect_timeout': 5,
    'socket_timeout': 5,
    'retry_on_timeout': True,
    'health_check_interval': 30
}
```

### 3.3 Health Monitoring

```python
def health_check() -> Dict[str, Any]:
    """Current health check implementation"""
    return {
        'postgresql': {
            'status': 'healthy|unhealthy',
            'error': None  # or error message
        },
        'redis': {
            'status': 'healthy|unhealthy', 
            'error': None  # or error message
        }
    }
```

---

## 4. Data Persistence Service

### 4.1 News Data Persistence

```python
def persist_news_catalyst_data(news_item: Dict) -> str:
    """Persist news with catalyst tracking"""
    
    # Extract catalyst information
    news_data = {
        'news_id': generate_news_id(
            news_item['headline'],
            news_item['source'],
            news_item['published_timestamp']
        ),
        'symbol': news_item.get('symbol'),
        'headline': news_item['headline'],
        'source': news_item['source'],
        'published_timestamp': news_item['published_timestamp'],
        'is_pre_market': is_pre_market_news(news_item['published_timestamp']),
        'market_state': get_market_state(news_item['published_timestamp']),
        'headline_keywords': extract_keywords(news_item['headline']),
        'mentioned_tickers': extract_tickers(news_item['content']),
        'source_tier': determine_source_tier(news_item['source']),
        'metadata': news_item.get('metadata', {})
    }
    
    # Persist with deduplication
    news_id = insert_news_article(news_data)
    
    # Update source metrics
    update_source_metrics(news_item['source'])
    
    return news_id
```

### 4.2 Trading Signal Persistence

```python
def persist_trading_signal(signal: Dict) -> bool:
    """Persist signal with full catalyst context"""
    
    signal_data = {
        'signal_id': f"SIG_{signal['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'symbol': signal['symbol'],
        'signal_type': signal['signal_type'],
        'confidence': signal['confidence'],
        
        # Component scores (v2.1.0 schema)
        'catalyst_score': signal.get('catalyst_score', 0),
        'pattern_score': signal.get('pattern_score', 0),
        'technical_score': signal.get('technical_score', 0),
        'volume_score': signal.get('volume_score', 0),
        
        # Trading parameters
        'recommended_entry': signal['recommended_entry'],
        'stop_loss': signal['stop_loss'],
        'target_1': signal['target_1'],
        'target_2': signal['target_2'],
        
        # Context
        'catalyst_type': signal.get('catalyst_type'),
        'detected_patterns': signal.get('detected_patterns', []),
        'key_factors': signal.get('key_factors', []),
        
        # Risk management
        'position_size_pct': signal['position_size_pct'],
        'risk_reward_ratio': signal['risk_reward_ratio']
    }
    
    return insert_trading_signal(signal_data)
```

### 4.3 Trade Execution Persistence

```python
def persist_trade_execution(trade: Dict) -> int:
    """Persist trade with full catalyst tracking"""
    
    trade_data = {
        'trade_id': f"TRD_{trade['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'signal_id': trade.get('signal_id'),
        'symbol': trade['symbol'],
        'order_type': trade['order_type'],
        'side': trade['side'],
        'quantity': trade['quantity'],
        'entry_price': trade['entry_price'],
        
        # Catalyst tracking (v2.1.0 schema)
        'entry_catalyst': trade.get('catalyst_type'),
        'entry_news_id': trade.get('news_id'),
        'catalyst_score_at_entry': trade.get('catalyst_score'),
        
        # Risk parameters
        'stop_loss': trade.get('stop_loss'),
        'take_profit': trade.get('take_profit')
    }
    
    trade_id = insert_trade_record(trade_data)
    
    # Mark signal as executed
    if trade.get('signal_id'):
        mark_signal_executed(trade['signal_id'], trade_id)
    
    return trade_id
```

---

## 5. Cache Management Service

### 5.1 Caching Strategy

```python
class CacheService:
    """Redis caching for hot data"""
    
    def __init__(self):
        self.redis = get_redis()
        self.default_ttl = 300  # 5 minutes
        
    def cache_pattern_detection(self, symbol: str, pattern: Dict):
        """Cache recent pattern detections"""
        key = f"pattern:{symbol}:{pattern['pattern_type']}"
        self.redis.setex(
            key,
            self.default_ttl,
            json.dumps(pattern)
        )
    
    def cache_technical_indicators(self, symbol: str, indicators: Dict):
        """Cache latest indicators"""
        key = f"indicators:{symbol}:{indicators['timeframe']}"
        self.redis.setex(
            key,
            self.default_ttl,
            json.dumps(indicators)
        )
    
    def cache_news_sentiment(self, symbol: str, sentiment: Dict):
        """Cache news sentiment analysis"""
        key = f"sentiment:{symbol}"
        self.redis.setex(
            key,
            600,  # 10 minutes for news
            json.dumps(sentiment)
        )
```

### 5.2 Cache Invalidation

```python
class CacheInvalidator:
    """Handle cache invalidation on updates"""
    
    def invalidate_symbol_cache(self, symbol: str):
        """Clear all cached data for a symbol"""
        pattern = f"*:{symbol}:*"
        
        # Scan and delete matching keys
        for key in self.redis.scan_iter(match=pattern):
            self.redis.delete(key)
    
    def invalidate_on_trade(self, symbol: str):
        """Invalidate caches when trade executed"""
        # Clear technical indicators (prices changed)
        self.invalidate_pattern(f"indicators:{symbol}:*")
        
        # Clear pattern caches (market state changed)
        self.invalidate_pattern(f"pattern:{symbol}:*")
```

---

## 6. Database Migration Service

### 6.1 Migration Management

```python
class MigrationService:
    """Database schema migration management"""
    
    def __init__(self):
        self.migrations_path = '/app/migrations'
        self.applied_table = 'schema_migrations'
        
    def run_pending_migrations(self):
        """Execute all pending migrations"""
        applied = self.get_applied_migrations()
        pending = self.get_pending_migrations(applied)
        
        for migration in pending:
            self.apply_migration(migration)
    
    def apply_migration(self, migration_file: str):
        """Apply a single migration"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Read migration SQL
                with open(migration_file, 'r') as f:
                    sql = f.read()
                
                # Execute migration
                cur.execute(sql)
                
                # Record migration
                version = self.extract_version(migration_file)
                cur.execute("""
                    INSERT INTO schema_migrations (version, name, applied_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                """, (version, migration_file))
```

### 6.2 Schema Version Tracking

```sql
-- Migration tracking table
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    applied_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Current schema version
INSERT INTO schema_migrations (version, name) VALUES 
(210, 'schema_v2.1.0_complete')
ON CONFLICT (version) DO NOTHING;
```

---

## 7. Backup & Recovery Service

### 7.1 Automated Backup Strategy

```python
class BackupService:
    """Automated database backup to DO Spaces"""
    
    def __init__(self):
        self.spaces_client = self._init_spaces_client()
        self.backup_bucket = os.getenv('BACKUP_BUCKET', 'catalyst-backups')
        
    def create_backup(self, backup_type='incremental'):
        """Create database backup"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if backup_type == 'full':
            backup_file = f"catalyst_full_{timestamp}.sql"
            self._full_backup(backup_file)
        else:
            backup_file = f"catalyst_incr_{timestamp}.sql"
            self._incremental_backup(backup_file)
        
        # Upload to DO Spaces
        self._upload_to_spaces(backup_file)
        
        # Clean up old backups
        self._cleanup_old_backups()
        
    def _full_backup(self, filename: str):
        """Create full database backup"""
        db_url = os.getenv('DATABASE_URL')
        cmd = f"pg_dump {db_url} > /tmp/{filename}"
        os.system(cmd)
        
    def _upload_to_spaces(self, filename: str):
        """Upload backup to DigitalOcean Spaces"""
        self.spaces_client.upload_file(
            f"/tmp/{filename}",
            self.backup_bucket,
            filename
        )
```

### 7.2 Recovery Procedures

```python
class RecoveryService:
    """Database recovery from backups"""
    
    def restore_from_backup(self, backup_date: str):
        """Restore database from specific backup"""
        
        # Download backup from Spaces
        backup_file = self._download_backup(backup_date)
        
        # Create restore point
        self._create_restore_point()
        
        # Restore database
        db_url = os.getenv('DATABASE_URL')
        cmd = f"psql {db_url} < {backup_file}"
        
        result = os.system(cmd)
        
        if result == 0:
            logger.info(f"Successfully restored from {backup_date}")
            
            # Run any post-restore migrations
            migration_service = MigrationService()
            migration_service.run_pending_migrations()
        else:
            logger.error(f"Restore failed from {backup_date}")
            self._rollback_to_restore_point()
```

---

## 8. Data Synchronization Service

### 8.1 Cross-Service Data Sync

```python
class DataSyncService:
    """Ensure data consistency across services"""
    
    def sync_catalyst_outcomes(self):
        """Update news accuracy based on trade outcomes"""
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Find completed trades with catalysts
                cur.execute("""
                    SELECT 
                        t.trade_id,
                        t.entry_news_id,
                        t.outcome_category,
                        t.pnl_percentage
                    FROM trade_records t
                    WHERE t.entry_news_id IS NOT NULL
                    AND t.status = 'closed'
                    AND t.catalyst_confirmed IS NULL
                """)
                
                for trade in cur.fetchall():
                    # Update news accuracy
                    was_accurate = trade['pnl_percentage'] > 0
                    
                    cur.execute("""
                        UPDATE news_raw
                        SET was_accurate = %s
                        WHERE news_id = %s
                    """, (was_accurate, trade['entry_news_id']))
                    
                    # Update trade catalyst confirmation
                    cur.execute("""
                        UPDATE trade_records
                        SET catalyst_confirmed = %s
                        WHERE trade_id = %s
                    """, (was_accurate, trade['trade_id']))
    
    def sync_source_metrics(self):
        """Update source reliability metrics"""
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE source_metrics sm
                    SET 
                        accurate_predictions = (
                            SELECT COUNT(*) FROM news_raw nr
                            WHERE nr.source = sm.source_name
                            AND nr.was_accurate = true
                        ),
                        false_predictions = (
                            SELECT COUNT(*) FROM news_raw nr
                            WHERE nr.source = sm.source_name
                            AND nr.was_accurate = false
                        ),
                        accuracy_rate = CASE
                            WHEN accurate_predictions + false_predictions > 0
                            THEN (accurate_predictions::float / (accurate_predictions + false_predictions)) * 100
                            ELSE 0
                        END,
                        last_updated = CURRENT_TIMESTAMP
                """)
```

---

## 9. DigitalOcean Integration

### 9.1 Managed Database Configuration

```yaml
# DigitalOcean PostgreSQL Settings
postgresql:
  cluster_name: catalyst-trading-db
  engine: pg
  version: "14"
  size: db-s-2vcpu-4gb
  region: nyc3
  node_count: 1
  
  # Connection pooling
  pool_mode: transaction
  pool_size: 25
  
  # Performance settings
  shared_buffers: 1GB
  work_mem: 16MB
  maintenance_work_mem: 256MB
  effective_cache_size: 3GB
  
  # Backup settings
  backup_hour: 3
  backup_minute: 0
  backup_retention_days: 7
```

### 9.2 Redis Configuration

```yaml
# DigitalOcean Redis Settings
redis:
  cluster_name: catalyst-trading-cache
  version: "6"
  size: db-s-1vcpu-1gb
  region: nyc3
  
  # Persistence
  persistence: rdb
  
  # Eviction policy
  maxmemory_policy: allkeys-lru
  
  # Security
  require_auth: true
```

### 9.3 Private Networking

```python
# Environment variables for private networking
DATABASE_HOST = "private-catalyst-trading-db-do-user-xxx.db.ondigitalocean.com"
DATABASE_PORT = "25060"
DATABASE_SSLMODE = "require"

REDIS_HOST = "private-catalyst-trading-cache-do-user-xxx.db.ondigitalocean.com"
REDIS_PORT = "25061"
REDIS_SSL = "true"
```

---

## 10. Performance & Monitoring

### 10.1 Query Performance Monitoring

```python
class PerformanceMonitor:
    """Monitor database performance"""
    
    def get_slow_queries(self, threshold_ms=1000):
        """Identify slow queries"""
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        query,
                        calls,
                        mean_exec_time,
                        max_exec_time,
                        total_exec_time
                    FROM pg_stat_statements
                    WHERE mean_exec_time > %s
                    ORDER BY mean_exec_time DESC
                    LIMIT 20
                """, (threshold_ms,))
                
                return cur.fetchall()
    
    def get_table_statistics(self):
        """Get table size and performance stats"""
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                        n_live_tup as row_count,
                        n_dead_tup as dead_rows,
                        last_vacuum,
                        last_analyze
                    FROM pg_stat_user_tables
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                """)
                
                return cur.fetchall()
```

### 10.2 Real-time Metrics Dashboard

```python
def get_database_metrics():
    """Aggregate metrics for monitoring dashboard"""
    
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'connections': {
            'active': get_active_connection_count(),
            'idle': get_idle_connection_count(),
            'waiting': get_waiting_connection_count()
        },
        'performance': {
            'cache_hit_ratio': get_cache_hit_ratio(),
            'index_hit_ratio': get_index_hit_ratio(),
            'avg_query_time': get_average_query_time()
        },
        'tables': {
            'news_raw': get_table_metrics('news_raw'),
            'trade_records': get_table_metrics('trade_records'),
            'trading_signals': get_table_metrics('trading_signals')
        },
        'replication': {
            'lag_bytes': get_replication_lag(),
            'status': get_replication_status()
        }
    }
    
    return metrics
```

---

## Implementation Best Practices

### Connection Management
1. Always use connection pooling via database_utils.py
2. Never create direct connections in services
3. Use context managers for automatic cleanup
4. Implement retry logic with exponential backoff

### Data Integrity
1. Use transactions for multi-table updates
2. Implement foreign key constraints
3. Validate data before persistence
4. Log all database errors with context

### Performance Optimization
1. Use batch inserts for bulk data
2. Create appropriate indexes
3. Monitor slow queries regularly
4. Use EXPLAIN ANALYZE for query optimization

### Caching Strategy
1. Cache read-heavy data (indicators, patterns)
2. Set appropriate TTLs based on data volatility
3. Invalidate on writes
4. Monitor cache hit rates

### Backup Policy
1. Automated daily full backups
2. Hourly incremental backups during trading hours
3. Test restore procedures monthly
4. Keep 30 days of backup history

## Summary

The Database Services & Data Management v2.1.0 provides:

1. **Unified Access**: All services use database_utils.py v2.3.5
2. **Connection Pooling**: Efficient resource management
3. **Catalyst Tracking**: Full news-to-trade traceability
4. **Performance**: Redis caching and query optimization
5. **Reliability**: Health monitoring and retry logic
6. **Data Integrity**: Transaction management and constraints
7. **Scalability**: Ready for DigitalOcean managed services

This architecture ensures the Catalyst Trading System maintains data consistency while supporting the complete catalyst-driven trading workflow defined in Schema v2.1.0.