#!/bin/bash
# Git Diagnosis Script for Catalyst Trading System

echo "=== GIT DIAGNOSIS ==="
echo "Current Directory:"
pwd

echo -e "\n=== Git Remote Info ==="
git remote -v

echo -e "\n=== Git Status (Normal) ==="
git status

echo -e "\n=== Git Status (Including Untracked) ==="
git status -u

echo -e "\n=== Check .gitignore ==="
if [ -f .gitignore ]; then
    echo "Contents of .gitignore:"
    cat .gitignore
else
    echo "No .gitignore file found"
fi

echo -e "\n=== Check for global gitignore ==="
git config --get core.excludesfile

echo -e "\n=== List ALL files git knows about ==="
git ls-files

echo -e "\n=== List IGNORED files ==="
git status --ignored

echo -e "\n=== Check if files are being ignored ==="
echo "Checking specific files:"
git check-ignore -v docker-compose.yml
git check-ignore -v coordination_service.py
git check-ignore -v Dockerfile.coordination

echo -e "\n=== Show untracked files ==="
git ls-files --others --exclude-standard

echo -e "\n=== Count of tracked vs untracked files ==="
echo "Tracked files: $(git ls-files | wc -l)"
echo "Untracked files: $(git ls-files --others --exclude-standard | wc -l)"
echo "Total files in directory: $(find . -type f -not -path "./.git/*" | wc -l)"