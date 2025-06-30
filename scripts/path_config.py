"""
Catalyst Trading System
Name of file: path_config.py
Version: 1.0.0
Last Updated: 2025-06-30
REVISION HISTORY:
  - v1.0.0 (2025-06-30) - Centralized path configuration for DigitalOcean
"""

import os

# Detect if running on DigitalOcean (or any read-only filesystem)
IS_DIGITALOCEAN = not os.access('/workspaces', os.W_OK) if os.path.exists('/workspaces') else True

# Base paths
BASE_PATH = '/tmp' if IS_DIGITALOCEAN else '.'
LOG_PATH = os.path.join(BASE_PATH, 'logs')
DATA_PATH = os.path.join(BASE_PATH, 'data')
PATTERN_DATA_PATH = os.path.join(BASE_PATH, 'pattern_data')
CACHE_PATH = os.path.join(BASE_PATH, 'cache')
EXPORT_PATH = os.path.join(BASE_PATH, 'exports')
REPORT_PATH = os.path.join(BASE_PATH, 'reports')

# Database
DATABASE_PATH = os.path.join(BASE_PATH, 'trading_system.db')

# Ensure directories exist
for path in [LOG_PATH, DATA_PATH, PATTERN_DATA_PATH, CACHE_PATH, EXPORT_PATH, REPORT_PATH]:
    os.makedirs(path, exist_ok=True)

print(f"📁 Path Configuration:")
print(f"  - Base: {BASE_PATH}")
print(f"  - Logs: {LOG_PATH}")
print(f"  - Database: {DATABASE_PATH}")
