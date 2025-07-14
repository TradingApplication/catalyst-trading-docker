#!/usr/bin/env python3
"""
Name of Service: TRADING SYSTEM SECURITY SCANNER - DYNAMIC CATALYST VERSION
Version: 2.0.0
Last Updated: 2025-06-27
Purpose: Dynamic security scanning with news-based filtering

REVISION HISTORY:
v2.0.0 (2025-06-27) - Complete rewrite for dynamic scanning
- Dynamic universe selection (50-100 stocks)
- News catalyst integration
- Multi-stage filtering (50 → 20 → 5)
- Pre-market focus
- Real-time narrowing throughout the day

This scanner finds the best day trading opportunities by:
1. Starting with market's most active stocks
2. Filtering by news catalysts
3. Confirming with technical setups
4. Delivering top 5 high-conviction picks
"""

import os
import json
import time
import sqlite3
import logging
import requests
import random
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Handle yfinance import
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    print("✅ yfinance imported successfully")
except ImportError as e:
    print(f"⚠️ yfinance import failed: {e}")
    YFINANCE_AVAILABLE = False

# Import database utilities if available
try:
    from database_utils_old import DatabaseServiceMixin
    USE_DB_UTILS = True
except ImportError:
    USE_DB_UTILS = False
    print("Warning: database_utils not found, using direct SQLite connections")


class DynamicSecurityScanner(DatabaseServiceMixin if USE_DB_UTILS else object):
    """
    Enhanced security scanner that dynamically finds trading opportunities
    """
    
    def __init__(self, db_path='/tmp//tmp/trading_system.db'):
        if USE_DB_UTILS:
            super().__init__(db_path)
        else:
            self.db_path = db_path
            
        self.app = Flask(__name__)
        self.setup_logging()
        self.setup_routes()
        
        # Service URLs
        self.coordination_url = "http://localhost:5000"
        self.news_service_url = "http://localhost:5008"
        
        # Scanning parameters
        self.scan_params = {
            # Universe size at each stage
            'initial_universe_size': 100,    # Start with top 100 active
            'catalyst_filter_size': 20,     # Narrow to 20 with news
            'final_selection_size': 5,       # Final 5 picks
            
            # Technical criteria
            'min_price': 1.0,               # Penny stock cutoff
            'max_price': 500.0,             # Avoid super high priced
            'min_volume': 500000,           # Liquidity threshold
            'min_relative_volume': 1.5,     # 50% above average
            'min_price_change': 2.0,        # 2% minimum move
            
            # Pre-market specific
            'premarket_min_volume': 50000,  # Lower threshold pre-market
            'premarket_weight': 2.0,        # Double weight for pre-market movers
        }
        
        # Cache for scan results
        self.scan_cache = {
            'timestamp': None,
            'universe': [],
            'catalyst_filtered': [],
            'final_picks': []
        }
        
        # Initialize database
        self._init_database_schema()
        
        self.logger.info("Dynamic Security Scanner v2.0.0 initialized")
        
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('dynamic_scanner')
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory
        os.makedirs('/tmp/logs', exist_ok=True)
        
        # File handler
        fh = logging.FileHandler('/tmp//tmp/logs/dynamic_scanner.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
    def _init_database_schema(self):
        """Initialize enhanced scanning results schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced scanning results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scanning_results_v2 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan