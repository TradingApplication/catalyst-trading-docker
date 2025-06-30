#!/bin/bash
# Catalyst Trading System - Server Setup Script
# Run this after SSHing into your new DigitalOcean droplet

set -e  # Exit on error

echo "🚀 Starting Catalyst Trading System server setup..."

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install required packages
echo "🔧 Installing required packages..."
apt install -y \
    curl \
    git \
    htop \
    ufw \
    nginx \
    certbot \
    python3-certbot-nginx \
    postgresql-client \
    redis-tools \
    jq

# Install Docker
echo "🐳 Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
rm get-docker.sh

# Install Docker Compose
echo "🐳 Installing Docker Compose..."
apt install -y docker-compose-plugin

# Add user to docker group (if not root)
usermod -aG docker $USER

# Setup firewall
echo "🔥 Configuring firewall..."
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# Create application directory
echo "📁 Creating application directory..."
mkdir -p /opt/catalyst-trading-system
cd /opt/catalyst-trading-system

# Clone repository
echo "📥 Cloning repository..."
git clone https://github.com/yourusername/catalyst-trading-system.git .

# Create necessary directories
echo "📁 Creating required directories..."
mkdir -p logs reports models nginx/ssl

# Create swap file (helpful for 4GB droplet)
echo "💾 Creating swap file..."
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' | tee -a /etc/fstab

# Setup log rotation
echo "📋 Setting up log rotation..."
cat > /etc/logrotate.d/catalyst-trading << EOF
/opt/catalyst-trading-system/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
}
EOF

# Create systemd service for Docker Compose
echo "⚙️ Creating systemd service..."
cat > /etc/systemd/system/catalyst-trading.service << EOF
[Unit]
Description=Catalyst Trading System
Requires=docker.service
After=docker.service

[Service]
Type=simple
WorkingDirectory=/opt/catalyst-trading-system
ExecStart=/usr/bin/docker compose up
ExecStop=/usr/bin/docker compose down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable service
systemctl daemon-reload
systemctl enable catalyst-trading

echo "✅ Server setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy your .env file to /opt/catalyst-trading-system/.env"
echo "2. Configure Nginx for your domain"
echo "3. Start the services with: systemctl start catalyst-trading"
echo "4. Monitor logs with: docker compose logs -f"