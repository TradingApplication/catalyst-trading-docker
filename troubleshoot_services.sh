#!/bin/bash
# Catalyst Trading System - Service Troubleshooting Script (Correct docker-compose version)
# Purpose: Diagnose why news, scanner, and trading services are failing

echo "=== Catalyst Trading System Troubleshooting ==="
echo "Date: $(date)"
echo ""

# 1. Check status of all services
echo "1. SERVICE STATUS"
echo "================="
docker-compose ps
echo ""

# 2. Check logs for failing services
echo "2. CHECKING SERVICE LOGS FOR ERRORS"
echo "===================================="
echo ""

echo "📰 News Service Logs (last 50 lines):"
echo "-------------------------------------"
docker-compose logs news-service | tail -50
echo ""

echo "🔍 Scanner Service Logs (last 50 lines):"
echo "----------------------------------------"
docker-compose logs scanner-service | tail -50
echo ""

echo "💹 Trading Service Logs (last 50 lines):"
echo "----------------------------------------"
docker-compose logs trading-service | tail -50
echo ""

# 3. Check database connectivity using coordination service
echo "3. TESTING DATABASE CONNECTION"
echo "=============================="
docker-compose exec coordination-service python3 -c "
from database_utils import get_db_connection, health_check
import json
try:
    status = health_check()
    print('Health Check:', json.dumps(status, indent=2))
    
    # Test basic query
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT version()')
            print('PostgreSQL Version:', cur.fetchone()['version'])
            
            # Check if tables exist
            cur.execute(\"\"\"
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name
            \"\"\")
            tables = [row['table_name'] for row in cur.fetchall()]
            print('\nExisting tables:', ', '.join(tables))
except Exception as e:
    print('ERROR:', str(e))
    import traceback
    traceback.print_exc()
"
echo ""

# 4. Check if database_utils.py exists in containers
echo "4. CHECKING DATABASE_UTILS.PY IN CONTAINERS"
echo "==========================================="
echo "In coordination service (working):"
docker-compose exec coordination-service ls -la /app/database_utils.py 2>&1 || echo "File not found"
echo ""
echo "In news service (failing):"
docker-compose exec news-service ls -la /app/database_utils.py 2>&1 || echo "File not found or container not running"
echo ""

# 5. Check environment variables
echo "5. CHECKING ENV VARS"
echo "===================="
echo "From coordination service:"
docker-compose exec coordination-service printenv | grep -E "(DATABASE_URL|REDIS_URL)" | sed 's/PASSWORD=.*/PASSWORD=***HIDDEN***/'
echo ""

# 6. Get the exact error from news service startup
echo "6. NEWS SERVICE STARTUP ERROR"
echo "============================="
docker-compose logs news-service | grep -E "(ERROR|CRITICAL|Failed|Exception|ImportError|ModuleNotFoundError|Traceback)" | tail -20
echo ""

# 7. Check Redis
echo "7. TESTING REDIS CONNECTION"
echo "==========================="
docker-compose exec coordination-service python3 -c "
from database_utils import get_redis
try:
    r = get_redis()
    r.ping()
    print('Redis: Connected successfully')
except Exception as e:
    print('Redis ERROR:', str(e))
"
echo ""

# 8. Check docker-compose configuration
echo "8. DOCKER-COMPOSE SERVICE DEFINITIONS"
echo "===================================="
echo "News service definition:"
docker-compose config | grep -A20 "news-service:" | head -25
echo ""

# 9. Restart count check
echo "9. SERVICE RESTART COUNTS"
echo "========================="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.RestartCount}}" | grep -E "(NAME|news|scanner|trading)"