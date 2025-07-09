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

# Method 1: Using DATABASE_URL from environment
if [ -n "$DATABASE_URL" ]; then
    echo "Using DATABASE_URL from environment..."
    psql "$DATABASE_URL" < add_market_data_tables.sql
    
    if [ $? -eq 0 ]; then
        echo "✅ Migration completed successfully!"
    else
        echo "❌ Migration failed. Trying alternative method..."
        
        # Method 2: Extract components and use flags
        echo "Attempting connection with extracted components..."
        
        # Extract components from DATABASE_URL
        # Format: postgresql://user:password@host:port/database?sslmode=require
        if [[ $DATABASE_URL =~ postgresql://([^:]+):([^@]+)@([^:]+):([^/]+)/([^?]+) ]]; then
            DB_USER="${BASH_REMATCH[1]}"
            DB_PASS="${BASH_REMATCH[2]}"
            DB_HOST="${BASH_REMATCH[3]}"
            DB_PORT="${BASH_REMATCH[4]}"
            DB_NAME="${BASH_REMATCH[5]}"
            
            PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" --set=sslmode=require < add_market_data_tables.sql
            
            if [ $? -eq 0 ]; then
                echo "✅ Migration completed successfully!"
            else
                echo "❌ Migration failed with both methods."
                exit 1
            fi
        else
            echo "❌ Could not parse DATABASE_URL"
            exit 1
        fi
    fi
else
    echo "Error: DATABASE_URL not set!"
    echo ""
    echo "Please either:"
    echo "1. Source your .env file: source .env"
    echo "2. Or export DATABASE_URL manually:"
    echo "   export DATABASE_URL='postgresql://user:password@host:port/database?sslmode=require'"
    exit 1
fi

echo ""
echo "To verify the tables were created, run:"
echo "psql \"\$DATABASE_URL\" -c '\\dt market_data*'"