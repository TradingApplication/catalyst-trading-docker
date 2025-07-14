#!/bin/bash
# Live patch for scanner health check issue

echo "=== Patching Scanner Health Check ==="

# Create a Python patch script
cat > /tmp/patch_scanner.py << 'EOF'
import re

# Read the scanner_service.py file
with open('/app/scanner_service.py', 'r') as f:
    content = f.read()

# Find and fix the health check
# Original pattern: db_health['database'] == 'healthy'
# Fixed pattern: db_health.get('database', {}).get('status', 'unknown') == 'healthy'

# Replace the problematic line
content = re.sub(
    r'"status": "healthy" if db_health\[\'database\'\] == \'healthy\' else "degraded"',
    '"status": "healthy" if db_health.get(\'database\', {}).get(\'status\', \'unknown\') == \'healthy\' and db_health.get(\'redis\', {}).get(\'status\', \'unknown\') == \'healthy\' else "degraded"',
    content
)

# Also fix the database and redis status lines if they exist
content = re.sub(
    r'"database": db_health\[\'database\'\]',
    '"database": db_health.get(\'database\', {}).get(\'status\', \'unknown\')',
    content
)

content = re.sub(
    r'"redis": db_health\[\'redis\'\]',
    '"redis": db_health.get(\'redis\', {}).get(\'status\', \'unknown\')',
    content
)

# Write the patched file
with open('/app/scanner_service.py', 'w') as f:
    f.write(content)

print("Scanner service patched successfully!")
EOF

# Apply the patch in the container
echo "Applying patch to scanner container..."
docker cp /tmp/patch_scanner.py catalyst-scanner:/tmp/patch_scanner.py
docker exec catalyst-scanner python /tmp/patch_scanner.py

# Restart the scanner service
echo "Restarting scanner service..."
docker-compose restart scanner-service

# Wait for service to come up
echo "Waiting for service to restart..."
sleep 10

# Test the health endpoint
echo ""
echo "Testing health endpoint..."
curl -s http://localhost:5001/health | jq . || curl -s http://localhost:5001/health

echo ""
echo "Patch complete! Check the dashboard now."
