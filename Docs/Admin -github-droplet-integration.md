# GitHub to Droplet Integration Guide

## Understanding the Difference

### App Platform (What You Had)
- Automatically pulls from GitHub
- Rebuilds on every commit
- Runs `pip install -r requirements.txt`
- Restarts the app
- All managed by DigitalOcean

### Droplet (What You Have Now)
- Full control over deployment
- You decide when/how to update
- Can implement various deployment strategies
- More powerful but requires setup

---

## Method 1: Simple Git Pull (Quick Start)

### Step 1: Clone Your Repository on Droplet
```bash
cd /opt/catalyst-trading-system

# Clone your repository
git remote -v

# Or if you want to clone into current directory
git clone https://github.com/TradingApplication/catalyst-trading-docker.git .
```

### Step 2: Set Up Git Credentials (For Private Repos)
# SSH Key
ssh-keygen -t ed25519 -C "droplet@catalyst-trading"
cat ~/.ssh/id_ed25519.pub
# Add this key to GitHub Settings → SSH Keys
```

VSCode Window
├── [SSH: catalyst-trading] (bottom left - shows you're connected)
├── Explorer (left panel)
│   └── /opt/catalyst-trading-system/  (your files)
│       ├── config/
│       ├── services/
│       └── ... (all visible and editable)
├── Editor (center)
│   └── Any file you click opens here
└── Terminal (bottom)
    └── root@catalyst-trading-prod-01:~#  (runs on droplet)


### Step 3: Create Update Script
```bash
cat > /opt/catalyst-trading-system/scripts/update-from-github.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Updating from GitHub ==="
cd /opt/catalyst-trading-system

# Pull latest code
git pull origin main

# Copy service files to their directories
for service in coordination scanner pattern technical trading news reporting dashboard; do
    if [ -f "${service}_service.py" ]; then
        cp ${service}_service.py services/$service/
        echo "Updated $service"
    fi
done

# Copy shared utilities
cp database_utils.py services/*/

# Rebuild Docker images
docker-compose build

# Restart services
docker-compose down
docker-compose up -d

echo "=== Update Complete ==="
docker-compose ps
EOF

chmod +x /opt/catalyst-trading-system/scripts/update-from-github.sh
```

### Step 4: Run Updates
```bash
# Whenever you want to update from GitHub
/opt/catalyst-trading-system/scripts/update-from-github.sh
```

---

## Method 2: GitHub Actions (Automated CI/CD)

### Step 1: Create Deploy Key on Droplet
```bash
# Generate deploy key
ssh-keygen -t ed25519 -f ~/.ssh/github_deploy -N ""

# Show public key
cat ~/.ssh/github_deploy.pub
```

### Step 2: Add to GitHub
1. Go to your repo → Settings → Deploy keys
2. Add new deploy key (paste the public key)
3. Name: "Catalyst Trading Droplet"

### Step 3: Create GitHub Action
In your repo, create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Droplet

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Droplet
      uses: appleboy/ssh-action@v0.1.5
      with:
        host: ${{ secrets.DROPLET_IP }}
        username: root
        key: ${{ secrets.DROPLET_SSH_KEY }}
        script: |
          cd /opt/catalyst-trading-system
          git pull origin main
          docker-compose build
          docker-compose down
          docker-compose up -d
```

### Step 4: Add Secrets to GitHub
1. Go to repo → Settings → Secrets → Actions
2. Add:
   - `DROPLET_IP`: Your droplet IP
   - `DROPLET_SSH_KEY`: Your private SSH key content

---

## Method 3: Webhook Deployment (Advanced)

### Step 1: Create Webhook Receiver
```python
# /opt/catalyst-trading-system/webhook_server.py
from flask import Flask, request
import subprocess
import hmac
import hashlib

app = Flask(__name__)
WEBHOOK_SECRET = 'your-webhook-secret'

@app.route('/webhook', methods=['POST'])
def github_webhook():
    # Verify webhook signature
    signature = request.headers.get('X-Hub-Signature-256')
    if not verify_signature(request.data, signature):
        return 'Unauthorized', 401
    
    # Pull and restart
    subprocess.run(['/opt/catalyst-trading-system/scripts/update-from-github.sh'])
    return 'Success', 200

def verify_signature(payload, signature):
    expected = 'sha256=' + hmac.new(
        WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
```

### Step 2: Add to GitHub
1. Repo → Settings → Webhooks
2. Payload URL: `http://YOUR_DROPLET_IP:9000/webhook`
3. Secret: your-webhook-secret

---

## Method 4: Docker Compose with GitHub

### Step 1: Modify docker-compose.yml
```yaml
version: '3.8'

services:
  coordination:
    build:
      context: https://github.com/TradingApplication/catalyst-trading-system.git
      dockerfile: Dockerfile.coordination
    # ... rest of config
```

### Step 2: Direct GitHub Building
```bash
# Docker Compose can build directly from GitHub
docker-compose build --no-cache
docker-compose up -d
```

---

## Recommended Approach for Your Setup

### 1. Initial Setup (One Time)
```bash
cd /opt/catalyst-trading-system

# Clone your repository
git clone https://github.com/TradingApplication/catalyst-trading-system.git .

# Set up directory structure
mkdir -p services/{coordination,scanner,pattern,technical,trading,news,reporting,dashboard}

# Copy service files
for service in coordination scanner pattern technical trading news reporting dashboard; do
    cp ${service}_service.py services/$service/
done
```

### 2. Create Simple Update Script
```bash
cat > /opt/catalyst-trading-system/update.sh << 'EOF'
#!/bin/bash
echo "Pulling latest from GitHub..."
git pull

echo "Updating service files..."
for service in coordination scanner pattern technical trading news reporting dashboard; do
    cp ${service}_service.py services/$service/ 2>/dev/null || true
done

echo "Rebuilding Docker images..."
docker-compose build

echo "Restarting services..."
docker-compose restart

echo "Update complete!"
docker-compose ps
EOF

chmod +x update.sh
```

### 3. Database Migrations
```bash
cat > /opt/catalyst-trading-system/migrate.sh << 'EOF'
#!/bin/bash
# Run any database migrations
source config/.env
psql "$DATABASE_URL" -f migrations/latest.sql
EOF
```

---

## Best Practices

### 1. Separate Code and Config
- Keep configuration in `/opt/catalyst-trading-system/config/`
- Don't commit `.env` files to GitHub
- Use environment variables

### 2. Version Control Strategy
```bash
# Tag releases
git tag -a v1.0.0 -m "Initial production release"
git push origin v1.0.0

# Deploy specific version
git checkout v1.0.0
```

### 3. Rollback Plan
```bash
# Before updating, tag current version
git tag -a pre-update-$(date +%Y%m%d) -m "Before update"

# If something goes wrong
git checkout pre-update-20250701
docker-compose build
docker-compose up -d
```

### 4. Development Workflow
1. Develop locally or in separate branch
2. Push to GitHub
3. Test on staging (if available)
4. Deploy to production droplet

---

## Quick Start Commands

```bash
# Initial GitHub setup on your droplet
cd /opt/catalyst-trading-system
git init
git remote add origin https://github.com/TradingApplication/catalyst-trading-system.git
git pull origin main

# Create update alias
echo "alias update-catalyst='/opt/catalyst-trading-system/update.sh'" >> ~/.bashrc
source ~/.bashrc

# Now you can just run
update-catalyst
```

---

## Environment-Specific Files

Create `.gitignore` in your repo:
```
.env
config/.env
*.log
logs/
data/
__pycache__/
*.pyc
.DS_Store
```

This keeps sensitive data out of GitHub while maintaining your code.