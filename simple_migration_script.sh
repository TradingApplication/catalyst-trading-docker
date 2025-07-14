#!/bin/bash
# ============================================================================
# Name of Application: Catalyst Trading System
# Name of file: run_migration.sh
# Version: 1.0.0
# Last Updated: 2025-01-27
# Purpose: Simple bash script to run database migration without Python dependencies
# 
# REVISION HISTORY:
# v1.0.0 (2025-01-27) - Simple migration runner
# 
# Description of Service:
# This script runs the database migration directly using psql command
# ============================================================================

# Database connection details
DB_HOST="${DB_HOST:-catalyst-trading-db-do-user-23488393-0.l.db.ondigitalocean.com}"
DB_PORT="${DB_PORT:-25060}"
DB_NAME="${DB_NAME:-catalyst_trading}"
DB_USER="${DB_USER:-doadmin}"

echo "=================================================="
echo "Catalyst Trading System - Enhanced Schema Migration"
echo "=================================================="
echo ""
echo "Database: $DB_NAME on $DB_HOST:$DB_PORT"
echo "User: $DB_USER"
echo ""

# Check if migration SQL file exists
if [ ! -f "enhanced_schema_migration.sql" ]; then
    echo "❌ Error: enhanced_schema_migration.sql not found!"
    echo "Please ensure the migration SQL file is in the current directory."
    exit 1
fi

# Prompt for password
echo -n "Enter database password: "
read -s DB_PASSWORD
echo ""

# Export password for psql
export PGPASSWORD=$DB_PASSWORD

# Create backup
echo ""
echo "📦 Creating database backup..."
BACKUP_FILE="backup_$(date +%Y%m%d_%H%M%S).sql"
pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME > $BACKUP_FILE

if [ $? -eq 0 ]; then
    echo "✅ Backup created: $BACKUP_FILE"
else
    echo "❌ Backup failed! Aborting migration."
    exit 1
fi

# Run migration
echo ""
echo "🚀 Running migration..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f enhanced_schema_migration.sql

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Migration completed successfully!"
else
    echo ""
    echo "❌ Migration failed!"
    exit 1
fi

# Verify tables were created
echo ""
echo "📊 Verifying migration..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << EOF
SELECT 
    'Tables created: ' || COUNT(*)::text as status
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN (
    'security_data_high_freq',
    'security_data_hourly',
    'security_data_daily',
    'security_tracking_state',
    'ml_pattern_discoveries',
    'security_correlations',
    'tracking_performance_metrics'
);

SELECT 
    table_name,
    pg_size_pretty(pg_total_relation_size(table_schema||'.'||table_name)) as size
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN (
    'security_data_high_freq',
    'security_data_hourly',
    'security_data_daily',
    'security_tracking_state',
    'ml_pattern_discoveries',
    'security_correlations',
    'tracking_performance_metrics'
)
ORDER BY table_name;
EOF

echo ""
echo "🎉 Migration process complete!"
echo ""
echo "Next steps:"
echo "1. Update your scanner_service.py to use the new tables"
echo "2. Start collecting data for top 100 securities"
echo "3. Monitor storage growth with: SELECT * FROM v_active_securities;"
echo ""

# Clear password
unset PGPASSWORD