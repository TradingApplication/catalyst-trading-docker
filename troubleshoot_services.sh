#!/bin/bash
# Catalyst Trading System - Service Troubleshooting Script
# Purpose: Diagnose why news, scanner, and trading services are failing

echo "=== Catalyst Trading System Troubleshooting ==="
echo "Date: $(date)"
echo ""

# 1. Check logs for failing services
echo "1. CHECKING SERVICE LOGS FOR ERRORS"
echo "===================================="
echo ""

echo "📰 News Service Logs (last 20 lines):"
docker logs catalyst-trading-docker-news-service-1 --tail 20 2>&1 | grep -E "(ERROR|CRITICAL|Failed|Exception|ImportError)"
echo ""

echo "🔍 Scanner Service Logs (last 20 lines):"
docker logs catalyst-trading-docker-scanner-service-1 --tail 20 2>&1 | grep -E "(ERROR|CRITICAL|Failed|Exception|ImportError)"
echo ""

echo "💹 Trading Service Logs (last 20 lines):"
docker logs catalyst-trading-docker-trading-service-1 --tail 20 2>&1 | grep -E "(ERROR|CRITICAL|Failed|Exception|ImportError)"
echo ""

# 2. Check if containers are restarting
echo "2. CHECKING CONTAINER STATUS"
echo "============================"
docker ps -a | grep -E "(news|scanner|trading)" | awk '{print $1, $2, $7, $8, $9, $10, $11}'
echo ""

# 3. Check database connectivity
echo "3. TESTING DATABASE CONNECTION"
echo "=============================="
docker exec catalyst-trading-docker-coordination-service-1 python3 -c "
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
except Exception as e:
    print('ERROR:', str(e))
"
echo ""

# 4. Check if required tables exist
echo "4. CHECKING DATABASE TABLES"
echo "==========================="
docker exec catalyst-trading-docker-postgres-1 psql -U doadmin -d catalyst_trading -c "\dt" 2>/dev/null || echo "Could not connect to database"
echo ""

# 5. Check environment variables
echo "5. CHECKING CRITICAL ENV VARS"
echo "============================="
echo "Checking news service env:"
docker exec catalyst-trading-docker-news-service-1 printenv | grep -E "(DATABASE_URL|REDIS_URL|NEWS_API_KEY)" | sed 's/=.*/=***HIDDEN***/'
echo ""

# 6. Get more detailed logs from one failing service
echo "6. DETAILED LOGS FROM NEWS SERVICE"
echo "=================================="
docker logs catalyst-trading-docker-news-service-1 --tail 50 2>&1
echo ""

# 7. Check Redis connectivity
echo "7. TESTING REDIS CONNECTION"
echo "==========================="
docker exec catalyst-trading-docker-coordination-service-1 python3 -c "
from database_utils import get_redis
try:
    r = get_redis()
    r.ping()
    print('Redis: Connected successfully')
except Exception as e:
    print('Redis ERROR:', str(e))
"
