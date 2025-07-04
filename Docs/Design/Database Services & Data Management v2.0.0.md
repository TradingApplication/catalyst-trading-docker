# Catalyst Trading System - Database Services & Data Management v2.0.0

**Version**: 2.0.0  
**Date**: June 28, 2025  
**Platform**: DigitalOcean  
**Purpose**: Define database services and data persistence management

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
│  │                  Database Services Layer                  │   │
│  │                                                           │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ Connection   │  │    Cache     │  │   Data Sync   │  │   │
│  │  │  Manager     │  │   Service    │  │   Service     │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  │                                                           │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ Persistence  │  │  Migration   │  │ Backup/Restore│  │   │
│  │  │  Service     │  │   Service    │  │   Service     │  │   │
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

### 1.2 Service Responsibilities

| Service | Purpose | Port |
|---------|---------|------|
| Connection Manager | Database connection pooling & management | Internal |
| Cache Service | Redis integration for hot data | Internal |
| Persistence Service | Handle data writes and consistency | Internal |
| Migration Service | Schema updates and data migration | 5020 |
| Backup Service | Automated backups to DO Spaces | 5021 |
| Sync Service | Multi-region data synchronization | 5022 |

---

## 2. Database Service Architecture

### 2.1 Core Components

```python
# database_services.py
class DatabaseServicesManager:
    """Central manager for all database services"""
    
    def __init__(self, config):
        self.config = config
        self.connection_manager = ConnectionManager(config)
        self.cache_service = CacheService(config)
        self.persistence_service = PersistenceService(config)
        self.migration_service = MigrationService(config)
        self.backup_service = BackupService(config)
        self.sync_service = DataSyncService(config)
```

### 2.2 Configuration

```python
DATABASE_CONFIG = {
    # PostgreSQL (DigitalOcean Managed Database)
    'postgres': {
        'host': 'db-catalyst-trading.b.db.ondigitalocean.com',
        'port': 25060,
        'database': 'catalyst_trading',
        'user': 'catalyst_app',
        'password': os.getenv('DB_PASSWORD'),
        'sslmode': 'require',
        'pool_size': 20,
        'max_overflow': 10,
        'pool_timeout': 30,
        'pool_recycle': 3600
    },
    
    # Redis (DigitalOcean Managed Redis)
    'redis': {
        'host': 'redis-catalyst.b.db.ondigitalocean.com',
        'port': 25061,
        'password': os.getenv('REDIS_PASSWORD'),
        'ssl': True,
        'decode_responses': True,
        'max_connections': 50
    },
    
    # DigitalOcean Spaces (S3-compatible)
    'spaces': {
        'region': 'sgp1',
        'endpoint': 'https://sgp1.digitaloceanspaces.com',
        'bucket': 'catalyst-trading-backups',
        'access_key': os.getenv('SPACES_ACCESS_KEY'),
        'secret_key': os.getenv('SPACES_SECRET_KEY')
    }
}
```

---

## 3. Connection Management Service

### 3.1 Connection Pool Manager

```python
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
import threading

class ConnectionManager:
    """Manages PostgreSQL connection pooling"""
    
    def __init__(self, config):
        self.config = config['postgres']
        self._pool = None
        self._lock = threading.Lock()
        self.initialize_pool()
    
    def initialize_pool(self):
        """Create connection pool"""
        self._pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=5,
            maxconn=self.config['pool_size'],
            host=self.config['host'],
            port=self.config['port'],
            database=self.config['database'],
            user=self.config['user'],
            password=self.config['password'],
            sslmode=self.config['sslmode']
        )
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with context manager"""
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                self._pool.putconn(conn)
    
    def execute_with_retry(self, query, params=None, retries=3):
        """Execute query with automatic retry"""
        for attempt in range(retries):
            try:
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(query, params)
                        if cur.description:
                            return cur.fetchall()
                        return cur.rowcount
            except psycopg2.OperationalError as e:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
```

### 3.2 Connection Health Monitoring

```python
class ConnectionHealthMonitor:
    """Monitor database connection health"""
    
    def __init__(self, connection_manager):
        self.conn_manager = connection_manager
        self.health_status = {
            'connections_active': 0,
            'connections_idle': 0,
            'connections_total': 0,
            'last_error': None,
            'uptime': 0
        }
    
    def check_health(self):
        """Perform health check"""
        try:
            # Test query
            result = self.conn_manager.execute_with_retry("SELECT 1")
            
            # Update pool stats
            pool = self.conn_manager._pool
            self.health_status.update({
                'connections_active': len(pool._used),
                'connections_idle': len(pool._pool),
                'connections_total': pool.maxconn,
                'status': 'healthy',
                'last_check': datetime.utcnow().isoformat()
            })
            
            return True
            
        except Exception as e:
            self.health_status.update({
                'status': 'unhealthy',
                'last_error': str(e),
                'last_check': datetime.utcnow().isoformat()
            })
            return False
```

---

## 4. Data Persistence Service

### 4.1 Write-Through Persistence

```python
class PersistenceService:
    """Handles data persistence with write-through caching"""
    
    def __init__(self, config):
        self.conn_manager = ConnectionManager(config)
        self.cache_service = CacheService(config)
        self.write_queue = queue.Queue(maxsize=1000)
        self.start_write_workers()
    
    def persist_news_item(self, news_data):
        """Persist news with caching"""
        news_id = news_data['news_id']
        
        # Write to database
        with self.conn_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO news_raw (news_id, symbol, headline, ...)
                    VALUES (%(news_id)s, %(symbol)s, %(headline)s, ...)
                    ON CONFLICT (news_id) DO NOTHING
                """, news_data)
        
        # Cache hot data
        self.cache_service.set(
            f"news:{news_id}",
            news_data,
            expire=3600  # 1 hour
        )
        
        # Update symbol cache
        self.cache_service.add_to_set(
            f"news:symbol:{news_data['symbol']}",
            news_id
        )
    
    def bulk_persist(self, table_name, records):
        """Bulk insert with transaction"""
        with self.conn_manager.get_connection() as conn:
            with conn.cursor() as cur:
                # Build insert query dynamically
                if records:
                    columns = records[0].keys()
                    values_template = ','.join(['%s'] * len(columns))
                    
                    args_str = ','.join(
                        cur.mogrify(f"({values_template})", 
                                   tuple(r[c] for c in columns)).decode('utf-8')
                        for r in records
                    )
                    
                    query = f"""
                        INSERT INTO {table_name} ({','.join(columns)})
                        VALUES {args_str}
                        ON CONFLICT DO NOTHING
                    """
                    
                    cur.execute(query)
                    return cur.rowcount
```

### 4.2 Read Optimization

```python
class OptimizedReader:
    """Optimized read operations with caching"""
    
    def __init__(self, conn_manager, cache_service):
        self.conn_manager = conn_manager
        self.cache = cache_service
    
    def get_trading_candidates(self, limit=5):
        """Get candidates with cache"""
        cache_key = f"candidates:top:{limit}"
        
        # Try cache first
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Query database
        with self.conn_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM trading_candidates
                    WHERE selection_timestamp > NOW() - INTERVAL '1 hour'
                    ORDER BY catalyst_score DESC
                    LIMIT %s
                """, (limit,))
                
                results = cur.fetchall()
                
                # Cache results
                self.cache.set(cache_key, results, expire=300)  # 5 min
                
                return results
    
    def get_recent_news_with_outcomes(self, symbol, hours=24):
        """Complex query with multiple joins"""
        return self.conn_manager.execute_with_retry("""
            SELECT 
                n.*,
                t.pnl_percentage,
                t.exit_reason,
                sm.accuracy_rate as source_accuracy
            FROM news_raw n
            LEFT JOIN trade_records t ON t.entry_news_id = n.news_id
            LEFT JOIN source_metrics sm ON sm.source_name = n.source
            WHERE n.symbol = %s
            AND n.published_timestamp > NOW() - INTERVAL '%s hours'
            ORDER BY n.published_timestamp DESC
        """, (symbol, hours))
```

---

## 5. Cache Management Service

### 5.1 Redis Cache Service

```python
import redis
import json
import pickle

class CacheService:
    """Redis caching layer"""
    
    def __init__(self, config):
        self.redis_config = config['redis']
        self.redis_client = redis.Redis(
            host=self.redis_config['host'],
            port=self.redis_config['port'],
            password=self.redis_config['password'],
            ssl=self.redis_config['ssl'],
            decode_responses=self.redis_config['decode_responses'],
            connection_pool=redis.ConnectionPool(
                max_connections=self.redis_config['max_connections']
            )
        )
        
    def get(self, key):
        """Get value from cache"""
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
        return None
    
    def set(self, key, value, expire=None):
        """Set value in cache"""
        try:
            self.redis_client.set(
                key, 
                json.dumps(value, default=str),
                ex=expire
            )
            return True
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
            return False
    
    def invalidate_pattern(self, pattern):
        """Invalidate cache keys matching pattern"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            self.logger.error(f"Cache invalidation error: {e}")
```

### 5.2 Cache Warming

```python
class CacheWarmer:
    """Pre-populate cache with frequently accessed data"""
    
    def __init__(self, cache_service, conn_manager):
        self.cache = cache_service
        self.db = conn_manager
    
    def warm_market_open_cache(self):
        """Warm cache before market open"""
        # Cache top news stories
        recent_news = self.db.execute_with_retry("""
            SELECT * FROM news_raw
            WHERE published_timestamp > NOW() - INTERVAL '4 hours'
            AND is_pre_market = TRUE
            ORDER BY catalyst_score DESC
            LIMIT 100
        """)
        
        for news in recent_news:
            self.cache.set(f"news:{news['news_id']}", news, expire=7200)
        
        # Cache active candidates
        candidates = self.db.execute_with_retry("""
            SELECT * FROM trading_candidates
            WHERE selection_timestamp > NOW() - INTERVAL '1 hour'
            AND NOT traded
        """)
        
        self.cache.set("candidates:active", candidates, expire=1800)
```

---

## 6. Database Migration Service

### 6.1 Migration Manager

```python
class MigrationService:
    """Handle database schema migrations"""
    
    def __init__(self, config):
        self.conn_manager = ConnectionManager(config)
        self.migrations_table = 'schema_migrations'
        self.ensure_migrations_table()
    
    def ensure_migrations_table(self):
        """Create migrations tracking table"""
        with self.conn_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
    
    def run_migrations(self):
        """Execute pending migrations"""
        applied = self.get_applied_migrations()
        pending = self.get_pending_migrations(applied)
        
        for migration in pending:
            self.apply_migration(migration)
    
    def apply_migration(self, migration):
        """Apply a single migration"""
        version = migration['version']
        
        try:
            with self.conn_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Execute migration SQL
                    cur.execute(migration['up'])
                    
                    # Record migration
                    cur.execute("""
                        INSERT INTO schema_migrations (version, name)
                        VALUES (%s, %s)
                    """, (version, migration['name']))
                    
            self.logger.info(f"Applied migration {version}: {migration['name']}")
            
        except Exception as e:
            self.logger.error(f"Migration {version} failed: {e}")
            raise
```

### 6.2 Migration Definitions

```python
MIGRATIONS = [
    {
        'version': 1,
        'name': 'initial_schema',
        'up': """
            -- Create initial tables
            CREATE TABLE news_raw (...);
            CREATE TABLE trading_candidates (...);
            -- etc
        """,
        'down': """
            DROP TABLE IF EXISTS news_raw;
            DROP TABLE IF EXISTS trading_candidates;
        """
    },
    {
        'version': 2,
        'name': 'add_source_alignment',
        'up': """
            ALTER TABLE news_raw 
            ADD COLUMN source_tier INTEGER DEFAULT 5,
            ADD COLUMN confirmation_status TEXT DEFAULT 'unconfirmed',
            ADD COLUMN narrative_cluster_id TEXT;
            
            CREATE INDEX idx_news_source_tier ON news_raw(source_tier);
        """,
        'down': """
            ALTER TABLE news_raw 
            DROP COLUMN source_tier,
            DROP COLUMN confirmation_status,
            DROP COLUMN narrative_cluster_id;
        """
    }
]
```

---

## 7. Backup & Recovery Service

### 7.1 Automated Backup Service

```python
import boto3
from datetime import datetime

class BackupService:
    """Automated database backups to DigitalOcean Spaces"""
    
    def __init__(self, config):
        self.config = config
        self.spaces_config = config['spaces']
        
        # S3-compatible client for DO Spaces
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.spaces_config['endpoint'],
            aws_access_key_id=self.spaces_config['access_key'],
            aws_secret_access_key=self.spaces_config['secret_key']
        )
    
    def backup_database(self):
        """Create and upload database backup"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_file = f"catalyst_trading_backup_{timestamp}.sql"
        
        try:
            # Create backup using pg_dump
            backup_command = f"""
                PGPASSWORD={self.config['postgres']['password']} \
                pg_dump -h {self.config['postgres']['host']} \
                -p {self.config['postgres']['port']} \
                -U {self.config['postgres']['user']} \
                -d {self.config['postgres']['database']} \
                -f /tmp/{backup_file}
            """
            
            os.system(backup_command)
            
            # Compress backup
            os.system(f"gzip /tmp/{backup_file}")
            
            # Upload to Spaces
            with open(f"/tmp/{backup_file}.gz", 'rb') as f:
                self.s3_client.put_object(
                    Bucket=self.spaces_config['bucket'],
                    Key=f"backups/{backup_file}.gz",
                    Body=f,
                    ACL='private'
                )
            
            # Clean up local file
            os.remove(f"/tmp/{backup_file}.gz")
            
            self.logger.info(f"Backup completed: {backup_file}.gz")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False
    
    def restore_database(self, backup_key):
        """Restore database from backup"""
        try:
            # Download from Spaces
            local_file = f"/tmp/restore_{datetime.utcnow().timestamp()}.sql.gz"
            self.s3_client.download_file(
                self.spaces_config['bucket'],
                backup_key,
                local_file
            )
            
            # Decompress
            os.system(f"gunzip {local_file}")
            
            # Restore
            restore_command = f"""
                PGPASSWORD={self.config['postgres']['password']} \
                psql -h {self.config['postgres']['host']} \
                -p {self.config['postgres']['port']} \
                -U {self.config['postgres']['user']} \
                -d {self.config['postgres']['database']} \
                -f {local_file[:-3]}
            """
            
            os.system(restore_command)
            
            # Clean up
            os.remove(local_file[:-3])
            
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False
```

### 7.2 Backup Schedule

```python
class BackupScheduler:
    """Schedule automated backups"""
    
    def __init__(self, backup_service):
        self.backup_service = backup_service
        self.schedules = {
            'hourly': self.hourly_backup,
            'daily': self.daily_backup,
            'weekly': self.weekly_backup
        }
    
    def hourly_backup(self):
        """Incremental backup of hot tables"""
        tables = ['news_raw', 'trading_candidates', 'trading_signals', 'trade_records']
        for table in tables:
            self.backup_table(table, retention_hours=24)
    
    def daily_backup(self):
        """Full database backup"""
        self.backup_service.backup_database()
        self.cleanup_old_backups(days=7)
    
    def weekly_backup(self):
        """Weekly archive with extended retention"""
        backup_file = self.backup_service.backup_database()
        self.tag_backup(backup_file, 'weekly', retention_days=90)
```

---

## 8. Data Synchronization Service

### 8.1 Real-time Sync Service

```python
class DataSyncService:
    """Synchronize data between regions/environments"""
    
    def __init__(self, config):
        self.primary_db = ConnectionManager(config['primary'])
        self.replica_db = ConnectionManager(config['replica'])
        self.sync_queue = queue.Queue()
        self.start_sync_workers()
    
    def sync_table_changes(self, table_name, changes):
        """Sync changes to replica"""
        for change in changes:
            self.sync_queue.put({
                'table': table_name,
                'operation': change['op'],
                'data': change['data'],
                'timestamp': datetime.utcnow()
            })
    
    def process_sync_queue(self):
        """Worker to process sync queue"""
        while True:
            try:
                item = self.sync_queue.get(timeout=1)
                self.apply_sync_item(item)
                self.sync_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Sync error: {e}")
```

### 8.2 Data Consistency Checker

```python
class ConsistencyChecker:
    """Verify data consistency between databases"""
    
    def check_table_consistency(self, table_name):
        """Compare table data between primary and replica"""
        # Get checksums
        primary_checksum = self.get_table_checksum(self.primary_db, table_name)
        replica_checksum = self.get_table_checksum(self.replica_db, table_name)
        
        if primary_checksum != replica_checksum:
            # Find differences
            differences = self.find_differences(table_name)
            self.reconcile_differences(table_name, differences)
    
    def get_table_checksum(self, db, table_name):
        """Calculate table checksum"""
        result = db.execute_with_retry(f"""
            SELECT MD5(CAST(array_agg(t.*) AS text))
            FROM {table_name} t
        """)
        return result[0][0] if result else None
```

---

## 9. DigitalOcean Integration

### 9.1 Managed Database Configuration

```python
# DigitalOcean Managed PostgreSQL setup
DO_DATABASE_CONFIG = {
    'cluster_name': 'catalyst-trading-db',
    'region': 'sgp1',  # Singapore for low latency to Perth
    'size': 'db-s-2vcpu-4gb',
    'node_count': 2,  # High availability
    'version': '14',
    
    # Connection pooling
    'connection_pool': {
        'name': 'catalyst-pool',
        'mode': 'transaction',
        'size': 20,
        'db': 'catalyst_trading',
        'user': 'catalyst_app'
    },
    
    # Read replica
    'read_replica': {
        'name': 'catalyst-read-replica',
        'region': 'sgp1',
        'size': 'db-s-1vcpu-2gb'
    }
}

# Redis configuration
DO_REDIS_CONFIG = {
    'cluster_name': 'catalyst-redis',
    'region': 'sgp1',
    'size': 'db-s-1vcpu-1gb',
    'version': '6',
    'eviction_policy': 'allkeys-lru',
    'persistence': True
}
```

### 9.2 App Platform Database Service

```yaml
# .do/database-services.yaml
services:
  - name: db-connection-manager
    github:
      repo: your-repo/catalyst-trading
      branch: main
      deploy_on_push: true
    source_dir: /database_services
    dockerfile_path: Dockerfile.dbservices
    instance_count: 2
    instance_size_slug: basic-xs
    internal_ports:
      - 5030
    env:
      - key: DATABASE_URL
        scope: RUN_TIME
        type: SECRET
      - key: REDIS_URL
        scope: RUN_TIME
        type: SECRET
      - key: SERVICE_TYPE
        value: "connection_manager"
    
  - name: db-migration-service
    github:
      repo: your-repo/catalyst-trading
      branch: main
    source_dir: /database_services
    dockerfile_path: Dockerfile.migration
    instance_count: 1
    instance_size_slug: basic-xs
    http_port: 5020
    routes:
      - path: /api/migrations
    env:
      - key: DATABASE_URL
        scope: RUN_TIME
        type: SECRET

databases:
  - name: catalyst-postgres
    engine: PG
    production: true
    cluster_name: catalyst-trading-db
    
  - name: catalyst-redis
    engine: REDIS
    production: true
    cluster_name: catalyst-redis
```

---

## 10. Performance & Monitoring

### 10.1 Database Performance Metrics

```python
class DatabaseMetricsCollector:
    """Collect and report database performance metrics"""
    
    def collect_metrics(self):
        """Gather comprehensive metrics"""
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'connections': self.get_connection_metrics(),
            'queries': self.get_query_metrics(),
            'tables': self.get_table_metrics(),
            'replication': self.get_replication_lag()
        }
        
        # Send to monitoring
        self.send_to_monitoring(metrics)
        
        return metrics
    
    def get_query_metrics(self):
        """Get query performance stats"""
        return self.db.execute_with_retry("""
            SELECT 
                query,
                calls,
                mean_exec_time,
                max_exec_time,
                total_exec_time
            FROM pg_stat_statements
            WHERE calls > 100
            ORDER BY mean_exec_time DESC
            LIMIT 20
        """)
    
    def get_table_metrics(self):
        """Get table size and performance"""
        return self.db.execute_with_retry("""
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                n_tup_ins,
                n_tup_upd,
                n_tup_del,
                n_live_tup,
                n_dead_tup
            FROM pg_stat_user_tables
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """)
```

### 10.2 Monitoring Dashboard

```python
class DatabaseMonitoringDashboard:
    """Real-time database monitoring"""
    
    def get_dashboard_data(self):
        """Aggregate monitoring data for dashboard"""
        return {
            'connection_pool': {
                'active': self.conn_manager.get_active_connections(),
                'idle': self.conn_manager.get_idle_connections(),
                'waiting': self.conn_manager.get_waiting_count()
            },
            'cache_stats': {
                'hit_rate': self.cache.get_hit_rate(),
                'memory_used': self.cache.get_memory_usage(),
                'keys_count': self.cache.get_key_count()
            },
            'replication': {
                'lag_bytes': self.get_replication_lag(),
                'status': self.get_replication_status()
            },
            'backup': {
                'last_backup': self.get_last_backup_time(),
                'next_scheduled': self.get_next_backup_time(),
                'backup_size': self.get_backup_storage_used()
            }
        }
```

---

## Implementation Best Practices

### Connection Management
1. Always use connection pooling
2. Set appropriate pool sizes based on load
3. Monitor connection health
4. Implement retry logic

### Caching Strategy
1. Cache frequently accessed data
2. Use appropriate TTLs
3. Invalidate on updates
4. Monitor cache hit rates

### Backup Policy
1. Hourly incremental backups
2. Daily full backups
3. Weekly archives
4. Test restore procedures

### Performance Optimization
1. Regular VACUUM and ANALYZE
2. Monitor slow queries
3. Optimize indexes
4. Use read replicas for analytics

This comprehensive database services layer ensures reliable, performant data management for the Catalyst Trading System on DigitalOcean infrastructure.

## Summary

The Database Services layer provides:

1. **Connection Management**: Pooled connections with health monitoring
2. **Data Persistence**: Write-through caching and bulk operations
3. **Cache Management**: Redis integration for hot data
4. **Migration Service**: Version-controlled schema updates
5. **Backup & Recovery**: Automated backups to DO Spaces
6. **Data Synchronization**: Multi-region support
7. **Performance Monitoring**: Real-time metrics and alerts

This architecture ensures:
- **High Availability**: Managed databases with replicas
- **Data Integrity**: Transaction management and consistency checks
- **Performance**: Caching and query optimization
- **Security**: Private networking and encrypted backups
- **Scalability**: Ready for growth with DigitalOcean's managed services