#!/bin/bash
# Diagnose why Coordination is unhealthy and Scanner is unreachable

echo "=== Diagnosing Coordination and Scanner Issues ==="
echo "Date: $(date)"
echo ""

# 1. Check Coordination Service
echo "1. COORDINATION SERVICE STATUS"
echo "=============================="
docker-compose ps coordination-service
echo ""

echo "Coordination Logs (last 50 lines):"
docker-compose logs --tail=50 coordination-service
echo ""

# 2. Check Scanner Service
echo "2. SCANNER SERVICE STATUS"
echo "========================="
docker-compose ps scanner-service
echo ""

echo "Scanner Logs (last 50 lines):"
docker-compose logs --tail=50 scanner-service
echo ""

# 3. Check if Scanner is actually running
echo "3. CONTAINER RUNTIME STATUS"
echo "==========================="
echo "All containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(NAME|coordination|scanner)"
echo ""

# 4. Test Coordination health endpoint
echo "4. TESTING COORDINATION HEALTH ENDPOINT"
echo "======================================"
echo "Trying coordination health check:"
curl -s http://localhost:5000/health | python3 -m json.tool || echo "Failed to reach coordination service"
echo ""

# 5. Test Scanner health endpoint
echo "5. TESTING SCANNER HEALTH ENDPOINT"
echo "=================================="
echo "Trying scanner health check:"
curl -s http://localhost:5001/health | python3 -m json.tool || echo "Failed to reach scanner service"
echo ""

# 6. Check network connectivity between services
echo "6. CHECKING DOCKER NETWORK"
echo "=========================="
echo "Testing if coordination can reach scanner:"
docker-compose exec coordination-service curl -s http://scanner-service:5001/health || echo "Coordination cannot reach scanner"
echo ""

# 7. Check for port conflicts
echo "7. CHECKING PORT AVAILABILITY"
echo "============================="
echo "Checking if ports are properly mapped:"
netstat -tulpn 2>/dev/null | grep -E "(5000|5001)" || ss -tulpn | grep -E "(5000|5001)" || echo "Cannot check ports (try with sudo)"
echo ""

# 8. Database connection test from coordination
echo "8. DATABASE CONNECTION FROM COORDINATION"
echo "========================================"
docker-compose exec coordination-service python3 -c "
from database_utils import health_check
import json
try:
    result = health_check()
    print('Database health from coordination:', json.dumps(result, indent=2))
except Exception as e:
    print('ERROR:', str(e))
" || echo "Could not test database from coordination"
echo ""

# 9. Check for missing environment variables
echo "9. ENVIRONMENT VARIABLES CHECK"
echo "=============================="
echo "Coordination environment (filtered):"
docker-compose exec coordination-service printenv | grep -E "(SERVICE_URL|PORT|DATABASE|REDIS)" | sort | head -20
echo ""

echo "Scanner environment (filtered):"
docker-compose exec scanner-service printenv 2>/dev/null | grep -E "(SERVICE_URL|PORT|DATABASE|REDIS)" | sort | head -20 || echo "Cannot check scanner env"
echo ""

# 10. Memory and resource check
echo "10. RESOURCE USAGE"
echo "=================="
docker stats --no-stream | grep -E "(NAME|coordination|scanner)"