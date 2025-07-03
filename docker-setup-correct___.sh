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
# STEP 5: CREATE DOCKERFILES
# =============================================================================
echo -e "\n${YELLOW}STEP 5: Creating Dockerfiles...${NC}"

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

    echo -e "${GREEN}