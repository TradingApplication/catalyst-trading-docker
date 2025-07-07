# ============================================================================
# Name of Application: Catalyst Trading System
# Name of file: database_migration_manager.py
# Version: 1.0.0
# Last Updated: 2025-01-27
# Purpose: Manage database migration and setup for enhanced data collection
# 
# REVISION HISTORY:
# v1.0.0 (2025-01-27) - Initial version for enhanced schema migration
# - Database connection and migration management
# - Partition creation automation
# - Initial data seeding
# - Migration verification
# 
# Description of Service:
# This service manages the database migration process for enhanced data collection:
# 1. Safely applies schema changes
# 2. Creates partitions for time-series data
# 3. Verifies migration success
# 4. Provides rollback capabilities
# ============================================================================

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseMigrationManager:
    """Manages database migrations for enhanced data collection"""
    
    def __init__(self):
        """Initialize migration manager with database connection"""
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'catalyst_trading'),
            'user': os.getenv('DB_USER', 'catalyst_user'),
            'password': os.getenv('DB_PASSWORD')
        }
        
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            logger.info("✅ Connected to database successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise
            
    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Disconnected from database")
        
    def check_existing_schema(self) -> Dict[str, bool]:
        """Check which tables already exist"""
        tables_to_check = [
            'security_data_high_freq',
            'security_data_hourly',
            'security_data_daily',
            'security_tracking_state',
            'ml_pattern_discoveries',
            'security_correlations',
            'tracking_performance_metrics'
        ]
        
        existing_tables = {}
        
        for table in tables_to_check:
            self.cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """, (table,))
            
            exists = self.cursor.fetchone()[0]
            existing_tables[table] = exists
            
        return existing_tables
        
    def create_backup_schema(self):
        """Create backup of existing schema before migration"""
        backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_schema = f'backup_{backup_timestamp}'
        
        try:
            # Create backup schema
            self.cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {backup_schema};")
            
            # Get list of existing tables
            self.cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE';
            """)
            
            tables = self.cursor.fetchall()
            
            # Copy each table to backup schema
            for (table_name,) in tables:
                logger.info(f"Backing up table: {table_name}")
                self.cursor.execute(f"""
                    CREATE TABLE {backup_schema}.{table_name} 
                    AS SELECT * FROM public.{table_name};
                """)
                
            self.conn.commit()
            logger.info(f"✅ Backup created in schema: {backup_schema}")
            return backup_schema
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ Backup failed: {e}")
            raise
            
    def execute_migration(self, migration_sql_file: str):
        """Execute the migration SQL file"""
        try:
            # Read migration SQL
            with open(migration_sql_file, 'r') as f:
                migration_sql = f.read()
                
            logger.info("Executing migration...")
            self.cursor.execute(migration_sql)
            self.conn.commit()
            logger.info("✅ Migration executed successfully")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ Migration failed: {e}")
            raise
            
    def create_time_partitions(self, months_ahead: int = 3):
        """Create time-based partitions for high-frequency data"""
        logger.info(f"Creating partitions for next {months_ahead} months...")
        
        base_date = datetime.now().replace(day=1)
        
        for i in range(months_ahead):
            # Calculate partition dates
            start_date = base_date + timedelta(days=30 * i)
            end_date = start_date + timedelta(days=30)
            
            # Format partition name
            partition_name = f"security_data_high_freq_{start_date.strftime('%Y_%m')}"
            
            # Check if partition exists
            self.cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM pg_tables 
                    WHERE tablename = %s
                );
            """, (partition_name,))
            
            if not self.cursor.fetchone()[0]:
                # Create partition
                try:
                    self.cursor.execute(f"""
                        CREATE TABLE {partition_name} 
                        PARTITION OF security_data_high_freq
                        FOR VALUES FROM ('{start_date.strftime('%Y-%m-%d')}') 
                        TO ('{end_date.strftime('%Y-%m-%d')}');
                    """)
                    logger.info(f"✅ Created partition: {partition_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not create partition {partition_name}: {e}")
            else:
                logger.info(f"Partition already exists: {partition_name}")
                
        self.conn.commit()
        
    def create_partition_maintenance_function(self):
        """Create function to automatically maintain partitions"""
        maintenance_sql = """
        -- Function to automatically create monthly partitions
        CREATE OR REPLACE FUNCTION create_monthly_partitions()
        RETURNS void AS $$
        DECLARE
            start_date date;
            end_date date;
            partition_name text;
        BEGIN
            -- Create partitions for next 3 months
            FOR i IN 0..2 LOOP
                start_date := date_trunc('month', CURRENT_DATE + (i || ' months')::interval);
                end_date := start_date + '1 month'::interval;
                partition_name := 'security_data_high_freq_' || to_char(start_date, 'YYYY_MM');
                
                -- Check if partition exists
                IF NOT EXISTS (
                    SELECT 1 FROM pg_tables 
                    WHERE tablename = partition_name
                ) THEN
                    EXECUTE format(
                        'CREATE TABLE %I PARTITION OF security_data_high_freq FOR VALUES FROM (%L) TO (%L)',
                        partition_name, start_date, end_date
                    );
                    RAISE NOTICE 'Created partition %', partition_name;
                END IF;
            END LOOP;
        END;
        $$ LANGUAGE plpgsql;
        
        -- Create scheduled job if pg_cron is available
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
                PERFORM cron.schedule(
                    'create-partitions', 
                    '0 0 1 * *', 
                    'SELECT create_monthly_partitions()'
                );
                RAISE NOTICE 'Scheduled monthly partition creation';
            END IF;
        END $$;
        """
        
        try:
            self.cursor.execute(maintenance_sql)
            self.conn.commit()
            logger.info("✅ Created partition maintenance function")
        except Exception as e:
            logger.warning(f"⚠️ Could not create maintenance function: {e}")
            
    def verify_migration(self) -> Dict[str, any]:
        """Verify migration was successful"""
        verification_results = {}
        
        # Check all tables exist
        tables_check = self.check_existing_schema()
        all_tables_exist = all(tables_check.values())
        verification_results['all_tables_exist'] = all_tables_exist
        verification_results['tables'] = tables_check
        
        # Check indexes
        self.cursor.execute("""
            SELECT 
                tablename,
                indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
            AND (tablename LIKE 'security_%' 
                 OR tablename LIKE 'ml_%' 
                 OR tablename LIKE 'tracking_%')
            ORDER BY tablename, indexname;
        """)
        
        indexes = self.cursor.fetchall()
        verification_results['index_count'] = len(indexes)
        
        # Check views
        self.cursor.execute("""
            SELECT viewname 
            FROM pg_views 
            WHERE schemaname = 'public' 
            AND viewname IN ('v_active_securities', 'v_pattern_candidates');
        """)
        
        views = self.cursor.fetchall()
        verification_results['views_created'] = len(views) == 2
        
        # Check partitions
        self.cursor.execute("""
            SELECT 
                parent.relname AS parent_table,
                COUNT(child.relname) AS partition_count
            FROM pg_inherits
            JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
            JOIN pg_class child ON pg_inherits.inhrelid = child.oid
            WHERE parent.relname = 'security_data_high_freq'
            GROUP BY parent.relname;
        """)
        
        partition_info = self.cursor.fetchone()
        if partition_info:
            verification_results['partitions_created'] = partition_info[1]
        else:
            verification_results['partitions_created'] = 0
            
        return verification_results
        
    def initialize_tracking_state(self, initial_symbols: List[str] = None):
        """Initialize tracking state for initial symbols"""
        if not initial_symbols:
            initial_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
            
        logger.info(f"Initializing tracking state for {len(initial_symbols)} symbols...")
        
        for symbol in initial_symbols:
            try:
                self.cursor.execute("""
                    INSERT INTO security_tracking_state (
                        symbol,
                        first_seen,
                        last_updated,
                        collection_frequency,
                        data_points_collected,
                        metadata
                    ) VALUES (
                        %s,
                        CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP,
                        'high_freq',
                        0,
                        '{"source": "initial_setup"}'::jsonb
                    )
                    ON CONFLICT (symbol) DO NOTHING;
                """, (symbol,))
                
            except Exception as e:
                logger.warning(f"Could not initialize {symbol}: {e}")
                
        self.conn.commit()
        logger.info("✅ Tracking state initialized")
        
    def run_full_migration(self):
        """Run the complete migration process"""
        logger.info("=" * 60)
        logger.info("Starting Enhanced Data Collection Migration")
        logger.info("=" * 60)
        
        try:
            # Connect to database
            self.connect()
            
            # Check existing schema
            logger.info("\n📋 Checking existing schema...")
            existing = self.check_existing_schema()
            for table, exists in existing.items():
                status = "✅" if exists else "❌"
                logger.info(f"  {status} {table}")
                
            # Create backup if any tables exist
            if any(existing.values()):
                logger.info("\n💾 Creating backup...")
                backup_schema = self.create_backup_schema()
            
            # Execute migration
            logger.info("\n🚀 Executing migration...")
            self.execute_migration('enhanced_schema_migration.sql')
            
            # Create partitions
            logger.info("\n📊 Creating partitions...")
            self.create_time_partitions()
            
            # Create maintenance function
            logger.info("\n🔧 Setting up partition maintenance...")
            self.create_partition_maintenance_function()
            
            # Initialize tracking state
            logger.info("\n📈 Initializing tracking state...")
            self.initialize_tracking_state()
            
            # Verify migration
            logger.info("\n✔️ Verifying migration...")
            results = self.verify_migration()
            
            logger.info("\n📊 Migration Results:")
            logger.info(f"  - All tables created: {results['all_tables_exist']}")
            logger.info(f"  - Indexes created: {results['index_count']}")
            logger.info(f"  - Views created: {results['views_created']}")
            logger.info(f"  - Partitions created: {results['partitions_created']}")
            
            if results['all_tables_exist']:
                logger.info("\n✅ Migration completed successfully!")
            else:
                logger.warning("\n⚠️ Migration completed with issues - please review")
                
        except Exception as e:
            logger.error(f"\n❌ Migration failed: {e}")
            raise
        finally:
            self.disconnect()


def main():
    """Main entry point"""
    manager = DatabaseMigrationManager()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--verify':
            manager.connect()
            results = manager.verify_migration()
            print(f"\nVerification Results: {results}")
            manager.disconnect()
        elif sys.argv[1] == '--partitions':
            manager.connect()
            manager.create_time_partitions()
            manager.disconnect()
    else:
        # Run full migration
        manager.run_full_migration()


if __name__ == "__main__":
    main()