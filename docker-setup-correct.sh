#!/bin/bash
# =============================================================================
# CATALYST TRADING SYSTEM - DOCKER SETUP
# Uses the correct service file naming convention (no version suffixes)
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Catalyst Trading System - Docker Setup                 ║${NC}"
echo -e "${BLUE}║   Using standard service file names                      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# =============================================================================
# STEP 1: VERIFY ENVIRONMENT
# =============================================================================
echo -e "${YELLOW}STEP 1: Verifying Environment...${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}❌ Please run as root (use 'sudo' or login as root)${NC}"
    exit 1
fi

# Check if in correct directory
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}❌ Error: docker-compose.yml not found!${NC}"
    echo "Please cd to /opt/catalyst-trading-system first"
    exit 1
fi

# Check Docker
if ! command -v docker >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker not installed!${NC}"
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

# Start Docker if not running
if ! docker info >/dev/null 2>&1; then
    echo "Starting Docker..."
    systemctl start docker
    systemctl enable docker
fi

echo -e "${GREEN}✓ Environment verified${NC}"

# =============================================================================
# STEP 2: CLEAN UP AND PREPARE
# =============================================================================
echo -e "\n${YELLOW}STEP 2: Cleaning up...${NC}"

# Stop all containers
docker-compose down 2>/dev/null || true

# Clean up Docker system
docker system prune -f

# Create directories
mkdir -p logs data models reports
chmod 755 logs data models reports

echo -e "${GREEN}✓ Cleanup complete${NC}"

# =============================================================================
# STEP 3: CHECK SERVICE FILES
# =============================================================================
echo -e "\n${YELLOW}STEP 3: Checking service files...${NC}"

# Define expected service files (simple names, no version suffixes)
SERVICES=(
    "coordination_service.py"
    "news_service.py"
    "scanner_service.py"
    "pattern_service.py"
    "technical_service.py"
    "trading_service.py"
    "reporting_service.py"
    "dashboard_service.py"
)

# Check which files exist
MISSING_FILES=false
for service_file in "${SERVICES[@]}"; do
    if [ -f "$service_file" ]; then
        echo -e "${GREEN}✓ Found: $service_file${NC}"
    else
        echo -e "${RED}❌ Missing: $service_file${NC}"
        MISSING_FILES=true
    fi
done

# Check for database_utils.py
if [ -f "database_utils.py" ]; then
    echo -e "${GREEN}✓ Found: database_utils.py${NC}"
    HAS_DB_UTILS=true
else
    echo -e "${YELLOW}⚠ No database_utils.py found${NC}"
    HAS_DB_UTILS=false
fi

# Check for .env file
if [ -f ".env" ]; then
    echo -e "${GREEN}✓ Found: .env file${NC}"
else
    echo -e "${RED}❌ Missing: .env file${NC}"
    echo "Please create your .env file first!"
    exit 1
fi

if [ "$MISSING_FILES" = true ]; then
    echo -e "\n${RED}Some service files are missing!${NC}"
    echo "Make sure you have all service files without version suffixes."
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# =============================================================================
# STEP 4: CREATE DOCKERFILES
# =============================================================================
echo -e "\n${YELLOW}STEP 4: Creating Dockerfiles...${NC}"

# Service to port mapping
declare -A SERVICE_PORTS=(
    ["coordination"]=5000
    ["news"]=5008
    ["scanner"]=5001
    ["pattern"]=5002
    ["technical"]=5003
    ["trading"]=5005
    ["reporting"]=5009
    ["dashboard"]=5010
)

# Create Dockerfile for each service
for service in coordination news scanner pattern technical trading reporting dashboard; do
    service_file="${service}_service.py"
    port=${SERVICE_PORTS[$service]}
    
    # Skip if service file doesn't exist
    if [ ! -f "$service_file" ]; then
        echo -e "${YELLOW}Skipping Dockerfile.$service (no $service_file found)${NC}"
        continue
    fi
    
    echo -e "${BLUE}Creating Dockerfile.$service${NC}"
    
    # Determine if service needs TA-Lib
    needs_talib=false
    if [[ "$service" =~ ^(scanner|pattern|technical)$ ]]; then
        needs_talib=true
    fi
    
    # Create base Dockerfile
    cat > Dockerfile.$service << 'EOF'
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*
EOF

    # Add TA-Lib installation if needed
    if [ "$needs_talib" = true ]; then
        cat >> Dockerfile.$service << 'EOF'

# Install TA-Lib C library
RUN cd /tmp && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd / && \
    rm -rf /tmp/ta-lib*
EOF
    fi
    
    # Continue Dockerfile
    cat >> Dockerfile.$service << EOF

# Set working directory in container
WORKDIR /app

# Copy requirements first (for better Docker caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt
EOF

    # Add TA-Lib Python package if needed
    if [ "$needs_talib" = true ]; then
        cat >> Dockerfile.$service << 'EOF'

# Install TA-Lib Python wrapper after C library
RUN pip install TA-Lib==0.4.28
EOF
    fi
    
    # Copy service file
    cat >> Dockerfile.$service << EOF

# Copy service file
COPY ${service}_service.py .
EOF

    # Copy database_utils.py if it exists
    if [ "$HAS_DB_UTILS" = true ]; then
        cat >> Dockerfile.$service << 'EOF'

# Copy database utilities
COPY database_utils.py .
EOF
    fi
    
    # Final Dockerfile sections
    cat >> Dockerfile.$service << EOF

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/reports

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV SERVICE_NAME=${service}_service

# Expose port
EXPOSE $port

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD wget --no-verbose --tries=1 --spider http://localhost:$port/health || exit 1

# Run service
CMD ["python", "${service}_service.py"]
EOF

    echo -e "${GREEN}✓ Created Dockerfile.$service${NC}"
done

# =============================================================================
# STEP 5: CREATE MEMORY-OPTIMIZED DOCKER-COMPOSE
# =============================================================================
echo -e "\n${YELLOW}STEP 5: Creating docker-compose.yml...${NC}"

# Backup existing if present
[ -f docker-compose.yml ] && cp docker-compose.yml docker-compose.yml.backup

cat > docker-compose.yml << 'EOF'
version: '3.8'

# Memory-optimized configuration for 4GB droplet
services:
  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: catalyst-redis
    ports:
      - "6379:6379"
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    deploy:
      resources:
        limits:
          memory: 128M
        reservations:
          memory: 64M
    restart: unless-stopped
    networks:
      - catalyst-network

  # Coordination Service
  coordination-service:
    build:
      context: .
      dockerfile: Dockerfile.coordination
    container_name: catalyst-coordination
    ports:
      - "5000:5000"
    env_file:
      - .env
    depends_on:
      - redis
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    deploy:
      resources:
        limits:
          memory: 256M
        reservations:
          memory: 128M
    restart: unless-stopped
    networks:
      - catalyst-network

  # News Service
  news-service:
    build:
      context: .
      dockerfile: Dockerfile.news
    container_name: catalyst-news
    ports:
      - "5008:5008"
    env_file:
      - .env
    depends_on:
      - redis
      - coordination-service
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    deploy:
      resources:
        limits:
          memory: 256M
        reservations:
          memory: 128M
    restart: unless-stopped
    networks:
      - catalyst-network

  # Scanner Service
  scanner-service:
    build:
      context: .
      dockerfile: Dockerfile.scanner
    container_name: catalyst-scanner
    ports:
      - "5001:5001"
    env_file:
      - .env
    environment:
      - PYTHONOPTIMIZE=1
      - MALLOC_TRIM_THRESHOLD_=100000
    depends_on:
      - redis
      - news-service
      - coordination-service
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    deploy:
      resources:
        limits:
          memory: 384M
        reservations:
          memory: 256M
    restart: unless-stopped
    networks:
      - catalyst-network

  # Pattern Service
  pattern-service:
    build:
      context: .
      dockerfile: Dockerfile.pattern
    container_name: catalyst-pattern
    ports:
      - "5002:5002"
    env_file:
      - .env
    environment:
      - PYTHONOPTIMIZE=1
      - MALLOC_TRIM_THRESHOLD_=100000
    depends_on:
      - redis
      - coordination-service
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./models:/app/models
    deploy:
      resources:
        limits:
          memory: 384M
        reservations:
          memory: 256M
    restart: unless-stopped
    networks:
      - catalyst-network

  # Technical Service
  technical-service:
    build:
      context: .
      dockerfile: Dockerfile.technical
    container_name: catalyst-technical
    ports:
      - "5003:5003"
    env_file:
      - .env
    environment:
      - PYTHONOPTIMIZE=1
      - MALLOC_TRIM_THRESHOLD_=100000
    depends_on:
      - redis
      - coordination-service
      - pattern-service
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./models:/app/models
    deploy:
      resources:
        limits:
          memory: 384M
        reservations:
          memory: 256M
    restart: unless-stopped
    networks:
      - catalyst-network

  # Trading Service
  trading-service:
    build:
      context: .
      dockerfile: Dockerfile.trading
    container_name: catalyst-trading
    ports:
      - "5005:5005"
    env_file:
      - .env
    depends_on:
      - redis
      - coordination-service
      - technical-service
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    deploy:
      resources:
        limits:
          memory: 256M
        reservations:
          memory: 128M
    restart: unless-stopped
    networks:
      - catalyst-network

  # Reporting Service
  reporting-service:
    build:
      context: .
      dockerfile: Dockerfile.reporting
    container_name: catalyst-reporting
    ports:
      - "5009:5009"
    env_file:
      - .env
    depends_on:
      - redis
      - coordination-service
    volumes:
      - ./logs:/app/logs
      - ./reports:/app/reports
    deploy:
      resources:
        limits:
          memory: 256M
        reservations:
          memory: 128M
    restart: unless-stopped
    networks:
      - catalyst-network

  # Web Dashboard
  web-dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    container_name: catalyst-dashboard
    ports:
      - "5010:5010"
      - "80:80"
    env_file:
      - .env
    depends_on:
      - redis
      - coordination-service
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    deploy:
      resources:
        limits:
          memory: 256M
        reservations:
          memory: 128M
    restart: unless-stopped
    networks:
      - catalyst-network

networks:
  catalyst-network:
    driver: bridge

volumes:
  redis_data:
    driver: local
EOF

echo -e "${GREEN}✓ Created docker-compose.yml${NC}"

# =============================================================================
# STEP 6: BUILD SERVICES
# =============================================================================
echo -e "\n${YELLOW}STEP 6: Building Docker images...${NC}"
echo -e "${YELLOW}This will take 10-20 minutes. Be patient!${NC}\n"

# Function to build with retry
build_service() {
    local service=$1
    local max_attempts=2
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo -e "${BLUE}Building $service (attempt $attempt/$max_attempts)...${NC}"
        
        if docker-compose build --no-cache $service; then
            echo -e "${GREEN}✓ $service built successfully${NC}\n"
            return 0
        else
            echo -e "${RED}❌ Failed to build $service${NC}"
            
            if [ $attempt -lt $max_attempts ]; then
                echo "Cleaning up and retrying..."
                docker system prune -f
                sleep 5
            fi
            
            ((attempt++))
        fi
    done
    
    return 1
}

# Build services in order
services=(
    "redis"
    "coordination-service"
    "news-service"
    "scanner-service"
    "pattern-service"
    "technical-service"
    "trading-service"
    "reporting-service"
    "web-dashboard"
)

failed_services=()

for service in "${services[@]}"; do
    if [ "$service" == "redis" ]; then
        echo -e "${BLUE}Pulling Redis...${NC}"
        docker-compose pull redis
        echo -e "${GREEN}✓ Redis ready${NC}\n"
    else
        if ! build_service $service; then
            failed_services+=($service)
        fi
    fi
done

# Report on failed services
if [ ${#failed_services[@]} -gt 0 ]; then
    echo -e "\n${RED}The following services failed to build:${NC}"
    for service in "${failed_services[@]}"; do
        echo -e "${RED}  - $service${NC}"
    done
    echo -e "\n${YELLOW}Try building them individually:${NC}"
    echo "docker-compose build --no-cache <service-name>"
else
    echo -e "\n${GREEN}✓ All services built successfully!${NC}"
fi

# =============================================================================
# STEP 7: START SERVICES
# =============================================================================
echo -e "\n${YELLOW}STEP 7: Starting services...${NC}"

docker-compose up -d

echo "Waiting for services to start..."
sleep 15

# =============================================================================
# STEP 8: VERIFY SERVICES
# =============================================================================
echo -e "\n${YELLOW}STEP 8: Verifying services...${NC}"

# Check health endpoints
echo -e "\n${BLUE}Service Health Status:${NC}"
for port in 5000 5001 5002 5003 5005 5008 5009 5010; do
    if curl -s -f http://localhost:$port/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Port $port - Healthy${NC}"
    else
        echo -e "${RED}❌ Port $port - Not responding${NC}"
    fi
done

# Show running containers
echo -e "\n${BLUE}Running Containers:${NC}"
docker-compose ps

# =============================================================================
# STEP 9: CREATE HELPER SCRIPTS
# =============================================================================
echo -e "\n${YELLOW}STEP 9: Creating helper scripts...${NC}"

# Memory monitor
cat > monitor-memory.sh << 'EOF'
#!/bin/bash
# Monitor memory usage for Catalyst Trading System

while true; do
    clear
    echo "=== Catalyst Trading System Memory Monitor ==="
    echo "Date: $(date)"
    echo ""
    echo "=== System Memory ==="
    free -h
    echo ""
    echo "=== Container Memory Usage ==="
    docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}"
    echo ""
    echo "Press Ctrl+C to exit"
    sleep 3
done
EOF
chmod +x monitor-memory.sh

# Logs viewer
cat > view-logs.sh << 'EOF'
#!/bin/bash
# View logs for Catalyst Trading System services

if [ -z "$1" ]; then
    echo "Usage: ./view-logs.sh <service>"
    echo ""
    echo "Available services:"
    echo "  coordination"
    echo "  news"
    echo "  scanner"
    echo "  pattern"
    echo "  technical"
    echo "  trading"
    echo "  reporting"
    echo "  dashboard"
    echo "  redis"
    echo ""
    echo "Example: ./view-logs.sh coordination"
    exit 1
fi

# Handle redis specially (no -service suffix)
if [ "$1" == "redis" ]; then
    docker-compose logs -f redis
else
    docker-compose logs -f $1-service
fi
EOF
chmod +x view-logs.sh

# Service restart script
cat > restart-service.sh << 'EOF'
#!/bin/bash
# Restart a specific service

if [ -z "$1" ]; then
    echo "Usage: ./restart-service.sh <service>"
    echo "Services: coordination, news, scanner, pattern, technical, trading, reporting, dashboard, redis"
    exit 1
fi

if [ "$1" == "redis" ]; then
    docker-compose restart redis
    docker-compose logs --tail=50 redis
else
    docker-compose restart $1-service
    docker-compose logs --tail=50 $1-service
fi
EOF
chmod +x restart-service.sh

# Status check script
cat > check-status.sh << 'EOF'
#!/bin/bash
# Check status of all services

echo "=== Catalyst Trading System Status ==="
echo "Date: $(date)"
echo ""
echo "=== Container Status ==="
docker-compose ps
echo ""
echo "=== Service Health ==="
for port in 5000 5001 5002 5003 5005 5008 5009 5010; do
    if curl -s -f http://localhost:$port/health >/dev/null 2>&1; then
        echo "✓ Port $port - Healthy"
    else
        echo "❌ Port $port - Not responding"
    fi
done
echo ""
echo "=== Resource Usage ==="
docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.CPUPerc}}"
EOF
chmod +x check-status.sh

echo -e "${GREEN}✓ Helper scripts created${NC}"

# =============================================================================
# FINAL SUMMARY
# =============================================================================
DROPLET_IP=$(curl -s http://checkip.amazonaws.com || echo "your-droplet-ip")

echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}   🎉 DOCKER SETUP COMPLETE!${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${YELLOW}Access Information:${NC}"
echo -e "Dashboard: ${GREEN}http://$DROPLET_IP${NC}"
echo -e "API Base: ${GREEN}http://$DROPLET_IP:5000${NC}"

echo -e "\n${YELLOW}Helper Scripts Created:${NC}"
echo "- ${BLUE}./monitor-memory.sh${NC} - Monitor memory usage"
echo "- ${BLUE}./view-logs.sh <service>${NC} - View service logs"
echo "- ${BLUE}./restart-service.sh <service>${NC} - Restart a service"
echo "- ${BLUE}./check-status.sh${NC} - Check all services status"

echo -e "\n${YELLOW}Quick Commands:${NC}"
echo "Check all services: ${BLUE}docker-compose ps${NC}"
echo "View all logs: ${BLUE}docker-compose logs -f${NC}"
echo "Stop all: ${BLUE}docker-compose down${NC}"
echo "Start all: ${BLUE}docker-compose up -d${NC}"

echo -e "\n${YELLOW}Troubleshooting:${NC}"
echo "- If a service fails: ${BLUE}./restart-service.sh <service-name>${NC}"
echo "- View specific logs: ${BLUE}./view-logs.sh <service-name>${NC}"
echo "- Rebuild a service: ${BLUE}docker-compose build --no-cache <service-name>-service${NC}"
echo "- Check memory: ${BLUE}./monitor-memory.sh${NC}"

echo -e "\n${GREEN}Your Catalyst Trading System is ready! 🚀${NC}"
echo -e "${GREEN}Happy Trading!${NC}\n"