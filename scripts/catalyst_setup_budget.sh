#!/bin/bash
# Catalyst Trading System - Budget DigitalOcean Setup Script
# Infrastructure: 4GB Droplet + Managed PostgreSQL (~$40 USD/month)
# Perfect for starting small and scaling with profits!

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Catalyst Trading System - Budget Setup${NC}"
echo -e "${GREEN}Starting Small, Thinking Big!${NC}"
echo -e "${GREEN}=========================================${NC}"

# Configuration
PROJECT_NAME="catalyst-trading-budget"
DROPLET_NAME="catalyst-budget"
DB_CLUSTER_NAME="catalyst-db"
REGION="sgp1"  # Singapore - change as needed
DROPLET_SIZE="s-2vcpu-4gb"  # 4GB RAM, 2 vCPUs (~$24/month)
DB_SIZE="db-s-1vcpu-1gb"    # Smallest managed DB (~$15/month)

# Generate secure passwords
echo -e "\n${YELLOW}Generating secure passwords...${NC}"
DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
ADMIN_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
JWT_SECRET=$(openssl rand -base64 48 | tr -d "=+/" | cut -c1-32)
FLASK_SECRET=$(openssl rand -base64 48 | tr -d "=+/" | cut -c1-32)

# Save passwords to secure file
cat > catalyst-credentials.txt << EOF
# Catalyst Trading System Credentials
# Generated: $(date)
# KEEP THIS FILE SECURE!

Database Password: ${DB_PASSWORD}
Redis Password: ${REDIS_PASSWORD}
Admin Password: ${ADMIN_PASSWORD}
JWT Secret: ${JWT_SECRET}
Flask Secret: ${FLASK_SECRET}

# Budget Configuration:
# - 4GB Droplet: ~\$24 USD/month
# - Managed PostgreSQL: ~\$15 USD/month
# - Total: ~\$39 USD/month
EOF

chmod 600 catalyst-credentials.txt
echo -e "${GREEN}✓ Credentials saved to catalyst-credentials.txt${NC}"

# Check if doctl is installed
if ! command -v doctl &> /dev/null; then
    echo -e "${RED}doctl CLI not found. Please install it first:${NC}"
    echo "brew install doctl (macOS)"
    echo "or visit: https://docs.digitalocean.com/reference/doctl/how-to/install/"
    exit 1
fi

# Check authentication
echo -e "\n${YELLOW}Checking DigitalOcean authentication...${NC}"
if ! doctl account get &> /dev/null; then
    echo -e "${RED}Not authenticated. Running: doctl auth init${NC}"
    doctl auth init
fi

# Create project
echo -e "\n${YELLOW}Creating project: ${PROJECT_NAME}${NC}"
if doctl projects list --format Name --no-header | grep -q "^${PROJECT_NAME}$"; then
    echo -e "${GREEN}✓ Project already exists${NC}"
    PROJECT_ID=$(doctl projects list --format Name,ID --no-header | grep "^${PROJECT_NAME}" | awk '{print $2}')
else
    PROJECT_OUTPUT=$(doctl projects create \
        --name "${PROJECT_NAME}" \
        --description "Budget-friendly news-driven trading system" \
        --purpose "Trading and financial services" \
        --format ID --no-header)
    PROJECT_ID=$PROJECT_OUTPUT
    echo -e "${GREEN}✓ Project created: ${PROJECT_ID}${NC}"
fi

# Get SSH key
echo -e "\n${YELLOW}Checking SSH keys...${NC}"
SSH_KEYS=$(doctl compute ssh-key list --format ID --no-header)
if [ -z "$SSH_KEYS" ]; then
    echo -e "${RED}No SSH keys found. Please add one:${NC}"
    echo "doctl compute ssh-key import my-key --public-key-file ~/.ssh/id_rsa.pub"
    exit 1
fi
SSH_KEY_ID=$(echo $SSH_KEYS | awk '{print $1}')
echo -e "${GREEN}✓ Using SSH key: ${SSH_KEY_ID}${NC}"

# Create Managed Database
echo -e "\n${YELLOW}Creating Managed PostgreSQL Database...${NC}"
echo -e "${BLUE}This provides automatic backups and better data protection!${NC}"

if doctl databases list --format Name --no-header | grep -q "^${DB_CLUSTER_NAME}$"; then
    echo -e "${YELLOW}Database cluster already exists${NC}"
    DB_ID=$(doctl databases list --format Name,ID --no-header | grep "^${DB_CLUSTER_NAME}" | awk '{print $2}')
else
    echo -e "${YELLOW}Creating database cluster (this takes ~5 minutes)...${NC}"
    DB_OUTPUT=$(doctl databases create ${DB_CLUSTER_NAME} \
        --engine pg \
        --version 14 \
        --size ${DB_SIZE} \
        --region ${REGION} \
        --num-nodes 1 \
        --wait \
        --format ID --no-header)
    DB_ID=$DB_OUTPUT
    echo -e "${GREEN}✓ Database cluster created: ${DB_ID}${NC}"
fi

# Get database connection details
echo -e "\n${YELLOW}Getting database connection details...${NC}"
DB_URI=$(doctl databases connection ${DB_ID} --format URI --no-header)
echo -e "${GREEN}✓ Database URI obtained${NC}"

# Create database and user
echo -e "\n${YELLOW}Creating database and user...${NC}"
doctl databases user create ${DB_ID} catalyst_user --format Name --no-header || true
doctl databases db create ${DB_ID} catalyst_trading --format Name --no-header || true

# Create Droplet
echo -e "\n${YELLOW}Creating 4GB Droplet: ${DROPLET_NAME}${NC}"
if doctl compute droplet list --format Name --no-header | grep -q "^${DROPLET_NAME}$"; then
    echo -e "${YELLOW}Droplet already exists${NC}"
    DROPLET_ID=$(doctl compute droplet list --format Name,ID --no-header | grep "^${DROPLET_NAME}" | awk '{print $2}')
    DROPLET_IP=$(doctl compute droplet get $DROPLET_ID --format PublicIPv4 --no-header)
else
    DROPLET_OUTPUT=$(doctl compute droplet create ${DROPLET_NAME} \
        --image docker-20-04 \
        --size ${DROPLET_SIZE} \
        --region ${REGION} \
        --ssh-keys ${SSH_KEY_ID} \
        --tag-names production,trading,catalyst,budget \
        --wait \
        --format ID,PublicIPv4 --no-header)
    
    DROPLET_ID=$(echo $DROPLET_OUTPUT | awk '{print $1}')
    DROPLET_IP=$(echo $DROPLET_OUTPUT | awk '{print $2}')
    echo -e "${GREEN}✓ Droplet created: ${DROPLET_ID} (${DROPLET_IP})${NC}"
fi

# Assign to project
echo -e "\n${YELLOW}Assigning resources to project...${NC}"
doctl projects resources assign ${PROJECT_ID} --resource=droplet:${DROPLET_ID} || true
doctl projects resources assign ${PROJECT_ID} --resource=database:${DB_ID} || true

# Create firewall rules for database
echo -e "\n${YELLOW}Configuring database firewall...${NC}"
doctl databases firewalls append ${DB_ID} --rule droplet:${DROPLET_ID} || true

# Wait for droplet
echo -e "\n${YELLOW}Waiting for droplet to be ready...${NC}"
sleep 30

# Create setup script for the droplet
cat > droplet-setup.sh << 'DROPLET_SCRIPT'
#!/bin/bash
set -e

echo "Starting Catalyst Trading System setup (Budget Edition)..."

# Update system
apt update && apt upgrade -y

# Install required packages
apt install -y curl git htop iotop nethogs redis-tools postgresql-client-14

# Install Docker
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

# Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# Create trading user
if ! id -u trading &> /dev/null; then
    useradd -m -s /bin/bash trading
    usermod -aG docker trading
fi

# Setup directories
mkdir -p /opt/catalyst/{config,logs,backups}
chown -R trading:trading /opt/catalyst

# Setup swap (IMPORTANT for 4GB droplet)
if [ ! -f /swapfile ]; then
    fallocate -l 4G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "vm.swappiness=10" >> /etc/sysctl.conf
    sysctl -p
fi

# Configure system limits for better memory usage
cat >> /etc/sysctl.conf << EOF
# Catalyst Trading System optimizations
vm.overcommit_memory=1
vm.dirty_background_ratio=5
vm.dirty_ratio=10
EOF
sysctl -p

# Configure firewall
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# Install monitoring tools
apt install -y ncdu dstat

# Create memory monitoring script
cat > /opt/catalyst/monitor-memory.sh << 'MONITOR'
#!/bin/bash
echo "=== Memory Usage by Container ==="
docker stats --no-stream --format "table {{.Container}}\t{{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}"
echo ""
echo "=== System Memory ==="
free -h
echo ""
echo "=== Top Memory Processes ==="
ps aux --sort=-%mem | head -10
MONITOR
chmod +x /opt/catalyst/monitor-memory.sh

echo "Droplet setup complete!"
DROPLET_SCRIPT

# Copy and run setup script
echo -e "\n${YELLOW}Configuring droplet...${NC}"
scp -o StrictHostKeyChecking=no droplet-setup.sh root@${DROPLET_IP}:/tmp/
ssh -o StrictHostKeyChecking=no root@${DROPLET_IP} "chmod +x /tmp/droplet-setup.sh && /tmp/droplet-setup.sh"

# Create .env file
cat > .env << EOF
# Catalyst Trading System - Budget Configuration
# Generated: $(date)

# Database (Managed PostgreSQL)
DATABASE_URL=${DB_URI}
DB_BACKUP_ENABLED=true

# Redis (Local container)
REDIS_PASSWORD=${REDIS_PASSWORD}
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379

# Security
ADMIN_PASSWORD=${ADMIN_PASSWORD}
JWT_SECRET=${JWT_SECRET}
FLASK_SECRET_KEY=${FLASK_SECRET}

# Trading APIs (Add your keys here)
NEWSAPI_KEY=your_newsapi_key_here
ALPHAVANTAGE_KEY=your_alphavantage_key_here
FINNHUB_KEY=your_finnhub_key_here
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Trading Parameters (Conservative for budget)
MAX_POSITIONS=3
POSITION_SIZE_PCT=30
PREMARKET_POSITION_PCT=15
STOP_LOSS_PCT=2
MIN_CATALYST_SCORE=40
MAX_DAILY_TRADES=10

# System Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
TIMEZONE=US/Eastern
DEBUG=false

# Memory Optimization
ENABLE_MEMORY_OPTIMIZATION=true
CACHE_TTL_MINUTES=30
MAX_HISTORICAL_DAYS=30
EOF

# Copy files to droplet
echo -e "\n${YELLOW}Deploying configuration files...${NC}"
scp .env root@${DROPLET_IP}:/opt/catalyst/
ssh root@${DROPLET_IP} "chmod 600 /opt/catalyst/.env"

# Clone repository and setup
ssh root@${DROPLET_IP} << 'REMOTE_COMMANDS'
cd /opt/catalyst
git clone https://github.com/TradingApplication/catalyst-trading-system.git
cd catalyst-trading-system

# Copy the budget docker-compose
cp Implementation/DOI/docker-compose-budget.yml docker-compose.yml

# Copy other required files
cp Implementation/DOI/nginx.conf .
cp Implementation/DOI/prometheus.yml .
cp Implementation/DOI/requirements.txt .
cp Implementation/DOI/Dockerfile.* .

# Copy service files
cp *.py .

# Build images (this will take a while)
echo "Building Docker images..."
docker-compose build --parallel

# Start services
echo "Starting services..."
docker-compose up -d

# Show status
docker-compose ps
REMOTE_COMMANDS

# Create helpful scripts
cat > manage-catalyst.sh << 'MANAGE_SCRIPT'
#!/bin/bash
# Catalyst Trading System Management Script

case "$1" in
    start)
        cd /opt/catalyst/catalyst-trading-system
        docker-compose up -d
        ;;
    stop)
        cd /opt/catalyst/catalyst-trading-system
        docker-compose down
        ;;
    restart)
        cd /opt/catalyst/catalyst-trading-system
        docker-compose restart
        ;;
    logs)
        cd /opt/catalyst/catalyst-trading-system
        docker-compose logs -f $2
        ;;
    status)
        cd /opt/catalyst/catalyst-trading-system
        docker-compose ps
        ;;
    memory)
        /opt/catalyst/monitor-memory.sh
        ;;
    backup)
        echo "Database backups are handled automatically by DigitalOcean"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|status|memory|backup}"
        exit 1
        ;;
esac
MANAGE_SCRIPT

scp manage-catalyst.sh root@${DROPLET_IP}:/usr/local/bin/
ssh root@${DROPLET_IP} "chmod +x /usr/local/bin/manage-catalyst.sh"

# Final output
echo -e "\n${GREEN}=========================================${NC}"
echo -e "${GREEN}✓ Budget Setup Complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo -e "${YELLOW}Access Information:${NC}"
echo "Droplet IP: ${DROPLET_IP}"
echo "SSH: ssh root@${DROPLET_IP}"
echo "Dashboard: http://${DROPLET_IP}"
echo ""
echo -e "${YELLOW}Monthly Cost Breakdown:${NC}"
echo "4GB Droplet: ~\$24 USD/month"
echo "Managed Database: ~\$15 USD/month"
echo "Total: ~\$39 USD/month"
echo ""
echo -e "${YELLOW}Credentials:${NC} catalyst-credentials.txt"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. SSH to droplet: ssh root@${DROPLET_IP}"
echo "2. Check services: manage-catalyst.sh status"
echo "3. View logs: manage-catalyst.sh logs [service-name]"
echo "4. Monitor memory: manage-catalyst.sh memory"
echo ""
echo -e "${YELLOW}IMPORTANT:${NC}"
echo "1. Update .env file with your API keys"
echo "2. Database backups are automatic (7 days retention)"
echo "3. Monitor memory usage regularly"
echo "4. Scale up when profits allow!"
echo ""
echo -e "${BLUE}When ready to scale:${NC}"
echo "- Resize droplet to 8GB: doctl compute droplet-action resize ${DROPLET_ID} --size s-4vcpu-8gb"
echo "- Upgrade database: doctl databases resize ${DB_ID} --size db-s-2vcpu-2gb"
echo ""
echo -e "${GREEN}Happy Trading! 🚀${NC}"

# Clean up
rm -f droplet-setup.sh .env manage-catalyst.sh