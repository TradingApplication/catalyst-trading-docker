#!/bin/bash
# Catalyst Trading System - Deployment Script
# Run from your local development machine

set -e

# Configuration
DROPLET_IP="YOUR_DROPLET_IP"
SSH_USER="root"
APP_DIR="/opt/catalyst-trading-system"

echo "🚀 Deploying Catalyst Trading System..."

# Function to run commands on server
remote_exec() {
    ssh $SSH_USER@$DROPLET_IP "$1"
}

# Copy environment file
echo "📝 Copying environment configuration..."
scp .env $SSH_USER@$DROPLET_IP:$APP_DIR/.env

# Pull latest code
echo "📥 Pulling latest code..."
remote_exec "cd $APP_DIR && git pull origin main"

# Build and deploy
echo "🏗️ Building Docker images..."
remote_exec "cd $APP_DIR && docker compose build --no-cache"

# Run database migrations
echo "🗄️ Running database migrations..."
remote_exec "cd $APP_DIR && docker compose run --rm coordination python migrate.py"

# Stop existing services
echo "🛑 Stopping existing services..."
remote_exec "systemctl stop catalyst-trading || true"

# Start services
echo "🚀 Starting services..."
remote_exec "systemctl start catalyst-trading"

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Check service health
echo "🏥 Checking service health..."
remote_exec "docker compose ps"
remote_exec "curl -s http://localhost:8001/health || echo 'Coordination service not healthy'"
remote_exec "curl -s http://localhost:5000/health || echo 'Dashboard not healthy'"

# Setup SSL certificate (first time only)
echo "🔒 Setting up SSL certificate..."
remote_exec "certbot --nginx -d catalyst-trading-system2-54e6n.ondigitalocean.app --non-interactive --agree-tos -m your-email@example.com || true"

# Restart Nginx
echo "🔄 Restarting Nginx..."
remote_exec "systemctl restart nginx"

echo "✅ Deployment complete!"
echo ""
echo "Access your application at: https://catalyst-trading-system2-54e6n.ondigitalocean.app"
echo "Monitor logs with: ssh $SSH_USER@$DROPLET_IP 'cd $APP_DIR && docker compose logs -f'"