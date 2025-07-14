#!/bin/bash
# Script to run market_data table migration on DigitalOcean PostgreSQL

echo "========================================"
echo "Market Data Tables Migration Script"
echo "========================================"

# Check if we have the SQL file
if [ ! -f "add_market_data_tables.sql" ]; then
    echo "Error: add_market_data_tables.sql not found!"
    echo "Please ensure the SQL file is in the current directory."
    exit 1
fi

# Check if environment variables are set
if [ -z "$DATABASE_HOST" ] || [ -z "$DATABASE_USER" ] || [ -z "$DATABASE_PASSWORD" ] || [ -z "$DATABASE_NAME" ]; then
    echo "Error: Database environment variables not set!"
    echo ""
    echo "Please source your .env file first:"
    echo "  source .env"
    echo ""
    echo "Required variables:"
    echo "  DATABASE_HOST"
    echo "  DATABASE_PORT (optional, defaults to 5432)"
    echo "  DATABASE_USER"
    echo "  DATABASE_PASSWORD"
    echo "  DATABASE_NAME"
    exit 1
fi

# Set default port if not provided
DATABASE_PORT=${DATABASE_PORT:-5432}

echo "Connecting to database..."
echo "  Host: $DATABASE_HOST"
echo "  Port: $DATABASE_PORT"
echo "  Database: $DATABASE_NAME"
echo "  User: $DATABASE_USER"
echo ""

# Run the migration using individual parameters
PGPASSWORD="$DATABASE_PASSWORD" psql \
    -h "$DATABASE_HOST" \
    -p "$DATABASE_PORT" \
    -U "$DATABASE_USER" \
    -d "$DATABASE_NAME" \
    --set=sslmode=require \
    < add_market_data_tables.sql

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Migration completed successfully!"
    echo ""
    echo "Verifying tables were created..."
    
    # Check if tables exist
    PGPASSWORD="$DATABASE_PASSWORD" psql \
        -h "$DATABASE_HOST" \
        -p "$DATABASE_PORT" \
        -U "$DATABASE_USER" \
        -d "$DATABASE_NAME" \
        --set=sslmode=require \
        -c "\dt market_data*" \
        -c "SELECT COUNT(*) as table_count FROM pg_tables WHERE tablename LIKE 'market_data%';"
else
    echo ""
    echo "❌ Migration failed!"
    echo ""
    echo "Common issues:"
    echo "1. Check your database credentials in .env"
    echo "2. Ensure the database host is accessible"
    echo "3. Verify SSL mode is enabled on your DigitalOcean database"
    echo "4. Check if you have CREATE TABLE permissions"
    exit 1
fi

echo ""
echo "Next steps:"
echo "1. Restart the scanner service to use the new tables"
echo "2. Monitor logs for any data storage issues"
echo "3. Check data is being stored: psql ... -c 'SELECT COUNT(*) FROM market_data;'"