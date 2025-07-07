#!/bin/bash
# Debug why scanner shows as degraded in dashboard

echo "=== Scanner Health Debug ==="
echo ""

# 1. Check the actual health response
echo "1. Scanner health endpoint response:"
curl -s http://localhost:5001/health | python -m json.tool
echo ""

# 2. Test database connection from scanner container
echo "2. Testing database connection from scanner container:"
docker exec catalyst-scanner python -c "
from database_utils import get_db_connection, health_check
import json

# Test health check
result = health_check()
print('Health check result:')
print(json.dumps(result, indent=2))

# Test direct connection
try:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT version()')
            version = cur.fetchone()
            print('\nPostgreSQL version:', version)
            print('Database connection: SUCCESS')
except Exception as e:
    print('Database connection: FAILED')
    print('Error:', str(e))
"
echo ""

# 3. Check Redis connection
echo "3. Testing Redis connection from scanner container:"
docker exec catalyst-scanner python -c "
from database_utils import get_redis
try:
    redis = get_redis()
    redis.ping()
    print('Redis connection: SUCCESS')
    print('Redis info:', redis.info('server')['redis_version'])
except Exception as e:
    print('Redis connection: FAILED')
    print('Error:', str(e))
"
echo ""

# 4. Check environment variables in container
echo "4. Database environment variables in scanner container:"
docker exec catalyst-scanner bash -c "env | grep -E 'DATABASE_URL|REDIS_URL' | sed 's/PASSWORD=.*/PASSWORD=****/g'"
echo ""

# 5. Check if tables exist
echo "5. Checking if database tables exist:"
docker exec catalyst-scanner python -c "
from database_utils import get_db_connection
try:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Check for trading_candidates table
            cur.execute(\"\"\"
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'trading_candidates'
                )
            \"\"\")
            exists = cur.fetchone()
            print('trading_candidates table exists:', exists['exists'])
            
            # List all tables
            cur.execute(\"\"\"
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            \"\"\")
            tables = cur.fetchall()
            print('\nAll tables in database:')
            for table in tables:
                print(f'  - {table[\"table_name\"]}')
except Exception as e:
    print('Error checking tables:', str(e))
"
echo ""

# 6. Compare with other services
echo "6. Comparing with other service health checks:"
echo "Coordination health:"
curl -s http://localhost:5000/health | python -m json.tool | grep -E "status|database|redis" | head -5
echo ""
echo "News service health:"
curl -s http://localhost:5008/health | python -m json.tool | grep -E "status|database|redis" | head -5
echo ""

# 7. Check what the dashboard sees
echo "7. What the coordination service reports about scanner:"
curl -s http://localhost:5000/services | python -m json.tool | grep -A10 "scanner"
