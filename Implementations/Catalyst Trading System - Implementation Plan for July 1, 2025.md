# Catalyst Trading System - Implementation Plan
## Date: July 1, 2025

### Objective
Deploy the Catalyst Trading System microservices architecture on DigitalOcean with PostgreSQL database.

---

## Phase 1: DigitalOcean Droplet Creation (30 minutes)

### 1.1 Create Droplet via DigitalOcean Console

**Specifications:**
- **Name**: `catalyst-trading-prod-01`
- **Region**: Choose closest to you (e.g., NYC3, SFO3)
- **Image**: Ubuntu 22.04 LTS
- **Size**: Basic Plan - 4GB RAM, 2 vCPUs, 80GB SSD ($24/month)
- **Authentication**: SSH Key (upload your public key)
- **Additional Options**:
  - ✅ Enable backups ($4.80/month extra - recommended)
  - ✅ Enable monitoring
  - ✅ IPv6

### 1.2 Initial Server Access
```bash
# Once droplet is created, note the IP address
export DROPLET_IP=xxx.xxx.xxx.xxx

# SSH into the droplet
ssh root@$DROPLET_IP

# Verify access
hostname
uname -a
```

### 1.3 Run Initial Server Setup Script
```bash
# Create and run the setup script
cat > /tmp/initial-setup.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Catalyst Trading System - Initial Server Setup ==="

# Update system
apt-get update && apt-get upgrade -y

# Install essential packages
apt-get install -y \
    curl \
    wget \
    git \
    vim \
    htop \
    ufw \
    fail2ban \
    python3-pip \
    software-properties-common

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
rm get-docker.sh

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.20.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create application directories
mkdir -p /opt/catalyst-trading-system/{data,logs,config,scripts}
mkdir -p /opt/catalyst-trading-system/services/{coordination,scanner,pattern,technical,trading,news,reporting,dashboard}

# Configure firewall
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw allow 5432/tcp  # PostgreSQL (restrict later)
ufw --force enable

# Create catalyst user
useradd -m -s /bin/bash catalyst
usermod -aG docker catalyst

echo "Initial setup complete!"
EOF

chmod +x /tmp/initial-setup.sh
/tmp/initial-setup.sh
```

---

## Phase 2: PostgreSQL Database Setup (45 minutes)

### 2.1 Create Managed PostgreSQL Database

**Via DigitalOcean Console:**
1. Go to Databases → Create Database Cluster
2. **Engine**: PostgreSQL 15
3. **Plan**: Basic - 1GB RAM, 1 vCPU ($15/month)
4. **Region**: Same as droplet
5. **Database name**: `catalyst_trading`
6. **Connection method**: Private network (VPC)

### 2.2 Configure Database Connection
```bash
# On the droplet, create environment file
cat > /opt/catalyst-trading-system/config/.env << 'EOF'
# Database Configuration
DATABASE_HOST=your-db-host.db.ondigitalocean.com
DATABASE_PORT=25060
DATABASE_NAME=catalyst_trading
DATABASE_USER=catalyst_app
DATABASE_PASSWORD=your-secure-password
DATABASE_SSL_MODE=require

# Application Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO

# Service Ports
COORDINATION_PORT=5000
SCANNER_PORT=5001
PATTERN_PORT=5002
TECHNICAL_PORT=5003
TRADING_PORT=5005
NEWS_PORT=5008
REPORTING_PORT=5009
DASHBOARD_PORT=5010

# API Keys (add your actual keys)
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
NEWSAPI_KEY=
ALPHAVANTAGE_KEY=
EOF

# Secure the file
chmod 600 /opt/catalyst-trading-system/config/.env
chown catalyst:catalyst /opt/catalyst-trading-system/config/.env
```

### 2.3 Initialize Database Schema
```bash
# Download and run the schema
cd /opt/catalyst-trading-system

# Get the schema file from your repository
wget https://raw.githubusercontent.com/TradingApplication/catalyst-trading-system/main/database/schema.sql

# Connect and create schema (adjust connection string)
PGPASSWORD=$DATABASE_PASSWORD psql -h $DATABASE_HOST -p $DATABASE_PORT -U $DATABASE_USER -d $DATABASE_NAME -f schema.sql

# Verify tables were created
PGPASSWORD=$DATABASE_PASSWORD psql -h $DATABASE_HOST -p $DATABASE_PORT -U $DATABASE_USER -d $DATABASE_NAME -c "\dt"
```

---

## Phase 3: Deploy Services to Docker (2 hours)

### 3.1 Clone Repository and Prepare Services
```bash
# Clone the repository
cd /opt/catalyst-trading-system
git clone https://github.com/TradingApplication/catalyst-trading-system.git temp_repo

# Copy service files
cp temp_repo/coordination_service.py services/coordination/
cp temp_repo/scanner_service.py services/scanner/
cp temp_repo/pattern_service.py services/pattern/
cp temp_repo/technical_service.py services/technical/
cp temp_repo/trading_service.py services/trading/
cp temp_repo/news_service.py services/news/
cp temp_repo/reporting_service.py services/reporting/
cp temp_repo/dashboard_service.py services/dashboard/

# Copy Docker files
cp temp_repo/docker-compose.yml .
cp temp_repo/Dockerfile.* .

# Copy requirements
cp temp_repo/requirements-*.txt .

# Clean up
rm -rf temp_repo
```

### 3.2 Create Shared Database Utils
```bash
# Create database_utils.py for all services
cat > /opt/catalyst-trading-system/database_utils.py << 'EOF'
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

class DatabaseServiceMixin:
    """Shared database functionality for all services"""
    
    def __init__(self, db_url=None):
        self.db_url = db_url or os.environ.get('DATABASE_URL', self._build_db_url())
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _build_db_url(self):
        """Build database URL from environment variables"""
        host = os.environ.get('DATABASE_HOST')
        port = os.environ.get('DATABASE_PORT', '25060')
        dbname = os.environ.get('DATABASE_NAME')
        user = os.environ.get('DATABASE_USER')
        password = os.environ.get('DATABASE_PASSWORD')
        
        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"
        
    def get_db_connection(self):
        """Get a database connection"""
        return psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)
        
    def execute_query(self, query, params=None):
        """Execute a query and return results"""
        with self.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                if cursor.description:
                    return cursor.fetchall()
                conn.commit()
                return cursor.rowcount
EOF

# Copy to each service directory
for service in coordination scanner pattern technical trading news reporting dashboard; do
    cp /opt/catalyst-trading-system/database_utils.py /opt/catalyst-trading-system/services/$service/
done
```

### 3.3 Create Docker Compose Configuration
```bash
# Update docker-compose.yml with PostgreSQL connection
cat > /opt/catalyst-trading-system/docker-compose.yml << 'EOF'
version: '3.8'

services:
  coordination:
    build:
      context: .
      dockerfile: Dockerfile.coordination
    container_name: catalyst-coordination
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - SERVICE_NAME=coordination
    volumes:
      - ./services/coordination:/app
      - ./logs:/app/logs
      - ./data:/app/data
    networks:
      - catalyst-network
    restart: unless-stopped

  scanner:
    build:
      context: .
      dockerfile: Dockerfile.scanner
    container_name: catalyst-scanner
    ports:
      - "5001:5001"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - SERVICE_NAME=scanner
    volumes:
      - ./services/scanner:/app
      - ./logs:/app/logs
    networks:
      - catalyst-network
    restart: unless-stopped

  pattern:
    build:
      context: .
      dockerfile: Dockerfile.patterns
    container_name: catalyst-pattern
    ports:
      - "5002:5002"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - SERVICE_NAME=pattern
    volumes:
      - ./services/pattern:/app
      - ./logs:/app/logs
    networks:
      - catalyst-network
    restart: unless-stopped

  technical:
    build:
      context: .
      dockerfile: Dockerfile.technical
    container_name: catalyst-technical
    ports:
      - "5003:5003"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - SERVICE_NAME=technical
    volumes:
      - ./services/technical:/app
      - ./logs:/app/logs
    networks:
      - catalyst-network
    restart: unless-stopped

  trading:
    build:
      context: .
      dockerfile: Dockerfile.trading
    container_name: catalyst-trading
    ports:
      - "5005:5005"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - ALPACA_API_KEY=${ALPACA_API_KEY}
      - ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
      - SERVICE_NAME=trading
    volumes:
      - ./services/trading:/app
      - ./logs:/app/logs
    networks:
      - catalyst-network
    restart: unless-stopped

  news:
    build:
      context: .
      dockerfile: Dockerfile.news
    container_name: catalyst-news
    ports:
      - "5008:5008"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - NEWSAPI_KEY=${NEWSAPI_KEY}
      - ALPHAVANTAGE_KEY=${ALPHAVANTAGE_KEY}
      - SERVICE_NAME=news
    volumes:
      - ./services/news:/app
      - ./logs:/app/logs
    networks:
      - catalyst-network
    restart: unless-stopped

  reporting:
    build:
      context: .
      dockerfile: Dockerfile.reporting
    container_name: catalyst-reporting
    ports:
      - "5009:5009"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - SERVICE_NAME=reporting
    volumes:
      - ./services/reporting:/app
      - ./logs:/app/logs
    networks:
      - catalyst-network
    restart: unless-stopped

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    container_name: catalyst-dashboard
    ports:
      - "5010:5010"
      - "80:80"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - SERVICE_NAME=dashboard
    volumes:
      - ./services/dashboard:/app
      - ./logs:/app/logs
    networks:
      - catalyst-network
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: catalyst-nginx
    ports:
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    networks:
      - catalyst-network
    depends_on:
      - dashboard
    restart: unless-stopped

networks:
  catalyst-network:
    driver: bridge

volumes:
  postgres_data:
  logs:
  data:
EOF
```

### 3.4 Build and Deploy Services
```bash
# Load environment variables
export $(cat /opt/catalyst-trading-system/config/.env | grep -v '^#' | xargs)

# Build Docker images
cd /opt/catalyst-trading-system
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

---

## Phase 4: Verification and Testing (30 minutes)

### 4.1 Health Check All Services
```bash
# Create health check script
cat > /opt/catalyst-trading-system/scripts/health-check.sh << 'EOF'
#!/bin/bash

services=("coordination:5000" "scanner:5001" "pattern:5002" "technical:5003" "trading:5005" "news:5008" "reporting:5009" "dashboard:5010")

echo "=== Catalyst Trading System Health Check ==="
echo "Date: $(date)"
echo ""

for service in "${services[@]}"; do
    name="${service%:*}"
    port="${service#*:}"
    
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health)
    
    if [ "$response" = "200" ]; then
        echo "✅ $name (port $port): HEALTHY"
    else
        echo "❌ $name (port $port): UNHEALTHY (HTTP $response)"
    fi
done
EOF

chmod +x /opt/catalyst-trading-system/scripts/health-check.sh
/opt/catalyst-trading-system/scripts/health-check.sh
```

### 4.2 Test Database Connectivity
```bash
# Test from each container
for service in coordination scanner pattern technical trading news reporting dashboard; do
    echo "Testing database from $service..."
    docker exec catalyst-$service python -c "
from database_utils import DatabaseServiceMixin
db = DatabaseServiceMixin()
conn = db.get_db_connection()
print('✅ Database connection successful from $service')
conn.close()
"
done
```

### 4.3 Test Service Communication
```bash
# Test coordination service can reach others
docker exec catalyst-coordination python -c "
import requests
services = {
    'scanner': 'http://scanner:5001/health',
    'pattern': 'http://pattern:5002/health',
    'news': 'http://news:5008/health'
}
for name, url in services.items():
    try:
        r = requests.get(url, timeout=5)
        print(f'✅ {name}: {r.status_code}')
    except Exception as e:
        print(f'❌ {name}: {e}')
"
```

---

## Phase 5: Security and Monitoring Setup (30 minutes)

### 5.1 Configure SSL with Let's Encrypt
```bash
# Install certbot
apt-get install -y certbot python3-certbot-nginx

# Get SSL certificate (replace with your domain)
certbot certonly --standalone -d catalyst-trading.yourdomain.com
```

### 5.2 Setup Basic Monitoring
```bash
# Deploy monitoring stack
cd /opt/catalyst-trading-system
docker-compose -f docker-compose.monitoring.yml up -d

# Access Grafana at http://YOUR_IP:3000
# Default login: admin/admin
```

### 5.3 Configure Backup Script
```bash
cat > /opt/catalyst-trading-system/scripts/backup.sh << 'EOF'
#!/bin/bash
# Daily backup script
BACKUP_DIR="/opt/catalyst-trading-system/backups"
mkdir -p $BACKUP_DIR

# Backup database
PGPASSWORD=$DATABASE_PASSWORD pg_dump -h $DATABASE_HOST -p $DATABASE_PORT -U $DATABASE_USER $DATABASE_NAME | gzip > $BACKUP_DIR/db_$(date +%Y%m%d).sql.gz

# Keep only last 7 days
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete
EOF

chmod +x /opt/catalyst-trading-system/scripts/backup.sh

# Add to crontab
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/catalyst-trading-system/scripts/backup.sh") | crontab -
```

---

## Completion Checklist

- [ ] Droplet created and accessible
- [ ] PostgreSQL database created and configured
- [ ] Database schema initialized
- [ ] All service files deployed to droplet
- [ ] Docker images built successfully
- [ ] All services running and healthy
- [ ] Inter-service communication verified
- [ ] Database connectivity verified from all services
- [ ] SSL certificate configured
- [ ] Monitoring deployed
- [ ] Backup script configured

---

## Next Steps (Tomorrow)

1. **Configure DNS** - Point domain to droplet IP
2. **Enable Auto-trading** - Add Alpaca credentials
3. **Test News Collection** - Verify API keys work
4. **Run Test Trades** - Paper trading validation
5. **Setup Alerts** - Configure Prometheus alerts

---

## Troubleshooting Commands

```bash
# View all logs
docker-compose logs -f

# Restart a specific service
docker-compose restart scanner

# Check disk space
df -h

# Check memory usage
free -m

# View running processes
docker ps

# Enter a container
docker exec -it catalyst-coordination bash

# Check PostgreSQL connection
docker exec catalyst-coordination pg_isready -h $DATABASE_HOST -p $DATABASE_PORT
```

---

## Important URLs

- **Dashboard**: http://YOUR_DROPLET_IP:5010
- **Grafana**: http://YOUR_DROPLET_IP:3000
- **Health Check**: http://YOUR_DROPLET_IP:5000/health

---

## Emergency Contacts

- DigitalOcean Support: https://www.digitalocean.com/support/
- Your backup admin: [Add contact info]

---

**Estimated Total Time**: 4 hours
**Estimated Cost**: $39/month (Droplet: $24 + Database: $15)