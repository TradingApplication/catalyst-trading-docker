#!/bin/bash
# Name of Application: Catalyst Trading System
# Name of file: force_rebuild.sh
# Version: 1.0.0
# Last Updated: 2025-01-05
# Purpose: Force rebuild and restart dashboard with new code

echo "=== Force Rebuilding Dashboard ==="
echo ""

# 1. Stop and remove the dashboard container
echo "Step 1: Stopping dashboard..."
docker-compose stop web-dashboard
docker-compose rm -f web-dashboard

# 2. Remove the old image to force rebuild
echo ""
echo "Step 2: Removing old dashboard image..."
docker rmi catalyst-trading-system_web-dashboard 2>/dev/null || true

# 3. Rebuild with no cache
echo ""
echo "Step 3: Rebuilding dashboard (this will take a minute)..."
docker-compose build --no-cache web-dashboard

# 4. Start the dashboard
echo ""
echo "Step 4: Starting dashboard..."
docker-compose up -d web-dashboard

# 5. Wait for startup
echo ""
echo "Step 5: Waiting for dashboard to initialize..."
sleep 20

# 6. Check health
echo ""
echo "Step 6: Checking dashboard health..."
curl -s http://localhost:5010/health | jq .

# 7. Check system status
echo ""
echo "Step 7: Checking system status via dashboard API..."
curl -s http://localhost:5010/api/dashboard_data | jq '.system_status, .services'

echo ""
echo "Dashboard rebuild complete!"
echo "Check http://localhost:5010"