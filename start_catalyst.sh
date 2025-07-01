#!/bin/bash
# =============================================================================
# CATALYST TRADING SYSTEM v2.1.0 - STARTUP GUIDE
# Let's get your automated trading system running!
# =============================================================================

echo "🚀 STARTING CATALYST TRADING SYSTEM v2.1.0"
echo "==========================================="

# =============================================================================
# STEP 1: PRE-FLIGHT CHECKS
# =============================================================================

echo ""
echo "📋 STEP 1: Pre-flight Checks"
echo "============================"

# Check if we're in the right directory
if [[ ! -f "docker-compose.yml" ]]; then
    echo "❌ docker-compose.yml not found!"
    echo "Make sure you're in your Catalyst T
    rading System directory"
    exit 1
fi

echo "✅ Found docker-compose.yml"

# Check if .env file exists
if [[ ! -f ".env" ]]; then
    echo "❌ .env file not found!"
    echo "Please create your .env file with database and API configurations"
    exit 1
fi

echo "✅ Found .env file"

# Check if requirements.txt exists
if [[ ! -f "requirements.txt" ]]; then
    echo "❌ requirements.txt not found!"
    echo "Please create the streamlined requirements.txt file"
    exit 1
fi

echo "✅ Found requirements.txt"

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running!"
    echo "Please start Docker Desktop/daemon first"
    exit 1
fi

echo "✅ Docker is running"

echo ""
echo "🎯 All pre-flight checks passed!"

# =============================================================================
# STEP 2: CREATE MISSING DIRECTORIES
# =============================================================================

echo ""
echo "📁 STEP 2: Creating Required Directories"
echo "========================================"

mkdir -p logs data models reports
echo "✅ Created: logs/, data/, models/, reports/"

# =============================================================================
# STEP 3: CREATE DOCKERFILES
# =============================================================================

echo ""
echo "🐳 STEP 3: Creating Dockerfiles"
echo "==============================="

# Create base Dockerfile template
create_dockerfile() {
    local service_name=$1
    local service_file=$2
    
    cat > Dockerfile.${service_name} << EOF
FROM python:3.11-slim

# Install system dependencies including TA-Lib
RUN apt-get update && apt-get install -y \\
    build-essential \\
    wget \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library
RUN cd /tmp && \\
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \\
    tar -xzf ta-lib-0.4.0-src.tar.gz && \\
    cd ta-lib/ && \\
    ./configure --prefix=/usr && \\
    make && \\
    make install && \\
    cd / && \\
    rm -rf /tmp/ta-lib*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service files
COPY ${service_file}.py .

# Create directories
RUN mkdir -p /app/logs /app/data /app/models /app/reports

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_PATH=/app/logs
ENV DATA_PATH=/app/data

# Expose port
EXPOSE \${$(echo ${service_name^^})_SERVICE_PORT:-5000}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD curl -f http://localhost:\${$(echo ${service_name^^})_SERVICE_PORT:-5000}/health || exit 1

# Run service
CMD ["python", "${service_file}.py"]
EOF
    
    echo "✅ Created Dockerfile.${service_name}"
}

# Create Dockerfiles for each service
create_dockerfile "coordination" "coordination_service"
create_dockerfile "news" "news_service"
create_dockerfile "scanner" "scanner_service"
create_dockerfile "pattern" "pattern_service"
create_dockerfile "technical" "technical_service"
create_dockerfile "trading" "trading_service"
create_dockerfile "reporting" "reporting_service"
create_dockerfile "dashboard" "dashboard_service"

echo ""
echo "🎯 All Dockerfiles created!"

# =============================================================================
# STEP 4: CREATE DATABASE_UTILS.PY
# =============================================================================

echo ""
echo "📊 STEP 4: Creating Database Utilities"
echo "======================================"

cat > database_utils.py << 'EOF'
"""
Catalyst Trading System - Database Utilities
Shared database connection and helper functions
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
import logging

logger = logging.getLogger(__name__)

def get_db_connection():
    """Get PostgreSQL database connection"""
    try:
        conn = psycopg2.connect(
            os.getenv('DATABASE_URL'),
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

def get_redis_connection():
    """Get Redis connection"""
    try:
        return redis.from_url(os.getenv('REDIS_URL', 'redis://redis:6379/0'))
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        return None

def health_check():
    """Check database and Redis health"""
    status = {'database': 'unhealthy', 'redis': 'unhealthy'}
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        status['database'] = 'healthy'
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
    
    try:
        r = get_redis_connection()
        if r and r.ping():
            status['redis'] = 'healthy'
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
    
    return status
EOF

echo "✅ Created database_utils.py"

# =============================================================================
# STEP 5: BUILD AND START SERVICES
# =============================================================================

echo ""
echo "🏗️ STEP 5: Building Docker Images"
echo "================================="

echo "Building images... (this may take 5-10 minutes on first run)"
docker-compose build --no-cache

if [ $? -eq 0 ]; then
    echo "✅ All Docker images built successfully!"
else
    echo "❌ Docker build failed! Check the error messages above."
    exit 1
fi

# =============================================================================
# STEP 6: START THE SYSTEM
# =============================================================================

echo ""
echo "🚀 STEP 6: Starting All Services"
echo "================================"

echo "Starting Redis first..."
docker-compose up -d redis

echo "Waiting for Redis to be ready..."
sleep 10

echo "Starting all services..."
docker-compose up -d

if [ $? -eq 0 ]; then
    echo "✅ All services started!"
else
    echo "❌ Service startup failed!"
    exit 1
fi

# =============================================================================
# STEP 7: HEALTH CHECKS
# =============================================================================

echo ""
echo "🏥 STEP 7: Health Checks"
echo "========================"

echo "Waiting for services to initialize... (30 seconds)"
sleep 30

# Check each service
declare -A services=(
    ["coordination-service"]="5000"
    ["news-service"]="5008"
    ["scanner-service"]="5001"
    ["pattern-service"]="5002"
    ["technical-service"]="5003"
    ["trading-service"]="5005"
    ["reporting-service"]="5009"
    ["web-dashboard"]="5010"
)

healthy_count=0
total_services=${#services[@]}

echo "Checking service health..."
for service in "${!services[@]}"; do
    port=${services[$service]}
    echo -n "  ${service} (port ${port}): "
    
    if curl -sf http://localhost:${port}/health > /dev/null 2>&1; then
        echo "✅ Healthy"
        ((healthy_count++))
    else
        echo "❌ Unhealthy"
    fi
done

echo ""
echo "Health Summary: ${healthy_count}/${total_services} services healthy"

# =============================================================================
# STEP 8: SYSTEM STATUS
# =============================================================================

echo ""
echo "📊 STEP 8: System Status"
echo "========================"

echo "Container Status:"
docker-compose ps

echo ""
echo "Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -10

# =============================================================================
# STEP 9: ACCESS INFORMATION
# =============================================================================

echo ""
echo "🎯 CATALYST TRADING SYSTEM IS RUNNING!"
echo "======================================"
echo ""
echo "🌐 ACCESS URLS:"
echo "   • Main Dashboard:     http://localhost:5010"
echo "   • Coordination API:   http://localhost:5000/health"
echo "   • News Service:       http://localhost:5008/health"
echo "   • Scanner Service:    http://localhost:5001/health"
echo "   • Pattern Service:    http://localhost:5002/health"
echo "   • Technical Service:  http://localhost:5003/health"
echo "   • Trading Service:    http://localhost:5005/health"
echo "   • Reporting Service:  http://localhost:5009/health"
echo ""
echo "📊 MONITORING:"
echo "   • Redis:              http://localhost:6379"
echo "   • Prometheus:         http://localhost:9090"
echo ""
echo "🎛️ MANAGEMENT COMMANDS:"
echo "   • View all logs:      docker-compose logs -f"
echo "   • View single service: docker-compose logs -f [service-name]"
echo "   • Restart service:    docker-compose restart [service-name]"
echo "   • Stop system:        docker-compose down"
echo "   • Restart system:     docker-compose restart"
echo ""
echo "🔥 READY TO TRADE!"
echo ""

# =============================================================================
# STEP 10: OPEN DASHBOARD
# =============================================================================

echo "🌐 Opening dashboard in browser..."

# Try to open browser (works on most systems)
if command -v open &> /dev/null; then
    open http://localhost:5010
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:5010
elif command -v start &> /dev/null; then
    start http://localhost:5010
else
    echo "💡 Manually open: http://localhost:5010 in your browser"
fi

echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo "======================"
echo ""
echo "Your Catalyst Trading System is now live and ready for automated trading!"
echo "Monitor the dashboard and check service logs for any issues."
echo ""
echo "Happy Trading! 🚀📈"
