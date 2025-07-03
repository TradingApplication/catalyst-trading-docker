#!/bin/bash
# Diagnostic script to check your Catalyst Trading System setup

echo "=== Catalyst Trading System - File Check ==="
echo "Current directory: $(pwd)"
echo ""

echo "=== Python Service Files ==="
for service in coordination news scanner pattern technical trading reporting dashboard; do
    if ls ${service}_service*.py 2>/dev/null | head -1 > /dev/null; then
        echo "✓ Found: $(ls ${service}_service*.py)"
    else
        echo "❌ Missing: ${service}_service.py"
    fi
done

echo ""
echo "=== Database Utils ==="
if [ -f "database_utils.py" ]; then
    echo "✓ Found: database_utils.py"
else
    echo "❌ Missing: database_utils.py"
fi

echo ""
echo "=== Configuration Files ==="
for file in docker-compose.yml requirements.txt .env; do
    if [ -f "$file" ]; then
        echo "✓ Found: $file"
    else
        echo "❌ Missing: $file"
    fi
done

echo ""
echo "=== Dockerfiles ==="
for dockerfile in Dockerfile.*; do
    if [ -f "$dockerfile" ]; then
        echo "✓ Found: $dockerfile"
    fi
done

echo ""
echo "=== Directories ==="
for dir in logs data models reports; do
    if [ -d "$dir" ]; then
        echo "✓ Directory exists: $dir/"
    else
        echo "❌ Directory missing: $dir/"
    fi
done

echo ""
echo "=== All Python Files ==="
ls -la *.py 2>/dev/null || echo "No .py files found"

echo ""
echo "=== Current Directory Structure ==="
ls -la

echo ""
echo "=== Docker Status ==="
docker --version || echo "Docker not installed"
docker-compose --version || echo "Docker Compose not installed"

echo ""
echo "=== Memory Status ==="
free -h