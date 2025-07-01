#!/bin/bash
# =============================================================================
# GIT DIVERGENT BRANCHES FIX
# Resolve branch conflicts and get back to deployment
# =============================================================================

echo "🔧 FIXING GIT DIVERGENT BRANCHES"
echo "================================="

# Check current status
echo "📊 Current Git Status:"
git status

echo ""
echo "🔍 Checking remote branches:"
git branch -a

# =============================================================================
# OPTION 1: FORCE YOUR LOCAL CHANGES (RECOMMENDED FOR DEPLOYMENT)
# =============================================================================

echo ""
echo "🎯 OPTION 1: Keep Your Local Changes (Recommended)"
echo "=================================================="
echo ""
echo "This will keep your local files and force-push to remote:"
echo ""
echo "git add ."
echo "git commit -m 'Catalyst Trading System v2.1.0 - Production Ready'"
echo "git push --force-with-lease origin main"
echo ""

# =============================================================================
# OPTION 2: MERGE REMOTE CHANGES
# =============================================================================

echo "🔄 OPTION 2: Merge Remote Changes"
echo "================================="
echo ""
echo "This will merge remote changes with your local:"
echo ""
echo "git pull --no-rebase"
echo "# OR"
echo "git pull --rebase"
echo ""

# =============================================================================
# OPTION 3: RESET TO REMOTE (LOSE LOCAL CHANGES)
# =============================================================================

echo "⚠️  OPTION 3: Reset to Remote (DANGER - Loses Local Work)"
echo "========================================================="
echo ""
echo "git fetch origin"
echo "git reset --hard origin/main"
echo ""

# =============================================================================
# QUICK FIX COMMANDS
# =============================================================================

echo "⚡ QUICK FIX (Choose One):"
echo "========================="
echo ""
echo "For deployment, run these commands:"
echo ""

read -p "Do you want to keep your local changes and force-push? (y/n): " choice

if [[ $choice =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Keeping local changes and force-pushing..."
    
    # Add all changes
    git add .
    
    # Commit changes
    git commit -m "Catalyst Trading System v2.1.0 - Production Ready Deployment
    
    - Complete PostgreSQL database schema applied
    - All 8 microservices configured (v2.1.0)
    - Production .env with real API keys
    - Optimized Docker Compose configuration
    - Streamlined requirements.txt
    - Ready for DigitalOcean deployment"
    
    # Force push (safely)
    git push --force-with-lease origin main
    
    if [ $? -eq 0 ]; then
        echo "✅ Git issues resolved! Ready to deploy."
        echo ""
        echo "🚀 Now you can start your Catalyst Trading System:"
        echo "   ./start_catalyst.sh"
    else
        echo "❌ Force push failed. Try Option 2 (merge) instead."
    fi
    
elif [[ $choice =~ ^[Nn]$ ]]; then
    echo ""
    echo "🔄 Attempting to merge remote changes..."
    
    # Try to pull and merge
    git pull --no-rebase
    
    if [ $? -eq 0 ]; then
        echo "✅ Merge successful! Ready to deploy."
        echo ""
        echo "🚀 Now you can start your Catalyst Trading System:"
        echo "   ./start_catalyst.sh"
    else
        echo "❌ Merge failed. You may have conflicts to resolve."
        echo ""
        echo "📋 Manual steps:"
        echo "1. Fix merge conflicts in the conflicted files"
        echo "2. git add ."
        echo "3. git commit -m 'Resolved merge conflicts'"
        echo "4. Continue with deployment"
    fi
else
    echo ""
    echo "💡 Manual resolution needed. Choose one of these:"
    echo ""
    echo "🎯 Keep local (recommended for deployment):"
    echo "   git add ."
    echo "   git commit -m 'Catalyst v2.1.0 ready'"
    echo "   git push --force-with-lease origin main"
    echo ""
    echo "🔄 Merge remote:"
    echo "   git pull --no-rebase"
    echo ""
    echo "⚠️  Reset to remote (lose local work):"
    echo "   git reset --hard origin/main"
fi

echo ""
echo "🎯 AFTER FIXING GIT:"
echo "==================="
echo "Run your deployment script:"
echo "   chmod +x start_catalyst.sh"
echo "   ./start_catalyst.sh"
