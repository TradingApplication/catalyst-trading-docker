#!/bin/bash
# Start script for Catalyst Trading System on DigitalOcean

echo "Starting Catalyst Trading System..."
echo "PORT: ${PORT:-8080}"

# Use gunicorn to run the app
exec gunicorn --bind 0.0.0.0:${PORT:-8080} wsgi:app \
    --worker-tmp-dir /dev/shm \
    --workers 1 \
    --threads 2 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -