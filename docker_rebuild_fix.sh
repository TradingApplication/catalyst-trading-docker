#!/bin/bash
# Force complete Docker rebuild to ensure new database_utils.py is used

echo "=== Forcing Complete Docker Rebuild ==="
echo "This will ensure all containers use the correct database_utils.py"
echo ""

# 1. Stop all services
echo "1. Stopping all services..."
docker-compose down
echo ""

# 2. Remove old images to force rebuild
echo "2. Removing old Docker images..."
docker-compose down --rmi local
# Alternative: docker images | grep catalyst-trading | awk '{print $3}' | xargs docker rmi -f
echo ""

# 3. Clear Docker build cache
echo "3. Clearing Docker build cache..."
docker system prune -f
echo ""

# 4. Verify database_utils.py is correct
echo "4. Verifying database_utils.py has correct imports..."
echo "Line 39 should have typing imports:"
sed -n '39p' database_utils.py
echo ""
echo "First 50 lines of database_utils.py:"
head -50 database_utils.py | nl
echo ""

# 5. Force rebuild without cache
echo "5. Rebuilding all services (no cache)..."
docker-compose build --no-cache --parallel
echo ""

# 6. Start services
echo "6. Starting services..."
docker-compose up -d
echo ""

# 7. Wait and check status
echo "7. Waiting 30 seconds for services to start..."
sleep 30
echo ""

echo "8. Checking service status..."
docker-compose ps
echo ""

# 9. Check if services are still restarting
echo "9. Checking for errors in previously failing services..."
echo ""
echo "News Service:"
docker-compose logs --tail=20 news-service | grep -E "(started|healthy|ERROR|ImportError|NameError)" || echo "Service starting..."
echo ""
echo "Scanner Service:"
docker-compose logs --tail=20 scanner-service | grep -E "(started|healthy|ERROR|ImportError|NameError)" || echo "Service starting..."
echo ""
echo "Trading Service:"
docker-compose logs --tail=20 trading-service | grep -E "(started|healthy|ERROR|ImportError|NameError)" || echo "Service starting..."
echo ""

# 10. Final health check
echo "10. Dashboard Health Status:"
curl -s http://localhost/api/health | python3 -m json.tool || echo "Dashboard not ready yet"