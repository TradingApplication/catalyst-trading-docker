#!/bin/bash
# Catalyst Trading System - Service Troubleshooting Script (Fixed)
# Purpose: Diagnose why news, scanner, and trading services are failing

echo "=== Catalyst Trading System Troubleshooting ==="
echo "Date: $(date)"
echo ""

# 1. Check logs for failing services
echo "1. CHECKING SERVICE LOGS FOR ERRORS"
echo "===================================="
echo ""

echo "📰 News Service Logs (last 30 lines):"
docker logs catalyst-trading-docker-news-service 2>&1 | tail -30
echo ""
echo "---"
echo ""

echo "🔍 Scanner Service Logs (last 30 lines):"
docker logs catalyst-trading-docker-scanner-service 2>&1 | tail -30
echo ""
echo "---"
echo ""

echo "💹 Trading Service Logs (last 30 lines):"
docker logs catalyst-trading-docker-trading-service 2>&1 | tail -30
echo ""

# 2. Check if containers are restarting
echo "2. CHECKING CONTAINER STATUS"
echo "============================"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.RestartCount}}" | grep -E "(NAME|news|scanner|trading|coordination)"
echo ""

# 3. Check database connectivity from a working container
echo "3. TESTING DATABASE CONNECTION"
echo "=============================="
docker exec catalyst-trading-docker-coordination-service python3 -c "
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

# 4. Check environment variables from working container
echo "4. CHECKING ENV VARS (from coordination service)"
echo "==============================================="
docker exec catalyst-trading-docker-coordination-service printenv | grep -E "(DATABASE_URL|REDIS_URL)" | sed 's/PASSWORD=.*/PASSWORD=***HIDDEN***/'
echo ""

# 5. Quick check of one failing service for import errors
echo "5. CHECKING FOR IMPORT ERRORS IN NEWS SERVICE"
echo "============================================="
docker logs catalyst-trading-docker-news-service 2>&1 | head -50 | grep -E "(ImportError|ModuleNotFoundError|from database_utils|import)"
echo ""

# 6. Check Redis connectivity
echo "6. TESTING REDIS CONNECTION"
echo "==========================="
docker exec catalyst-trading-docker-coordination-service python3 -c "
from database_utils import get_redis
try:
    r = get_redis()
    r.ping()
    print('Redis: Connected successfully')
    print('Redis info:', r.info('server')['redis_version'])
except Exception as e:
    print('Redis ERROR:', str(e))
    import traceback
    traceback.print_exc()
"
echo ""

# 7. Compare working vs failing service
echo "7. CHECKING PYTHON PATH AND IMPORTS"
echo "==================================="
echo "Working service (coordination) Python path:"
docker exec catalyst-trading-docker-coordination-service python3 -c "import sys; print('\n'.join(sys.path))"
echo ""
echo "Checking if database_utils.py exists in news service:"
docker exec catalyst-trading-docker-news-service ls -la /app/database_utils.py 2>&1 || echo "File check failed"
