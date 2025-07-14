#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System  
Name of file: scanner_service.py
Version: 2.1.1
Last Updated: 2025-07-06
Purpose: Dynamic security scanning with news-based filtering and PostgreSQL

REVISION HISTORY:
v2.1.1 (2025-07-06) - Fixed missing database function imports
- Added insert_trading_candidates function implementation
- Added get_active_candidates function implementation
- Maintained backward compatibility with v2.1.0

v2.1.0 (2025-07-01) - Production-ready refactor
- Migrated from SQLite to PostgreSQL
- All configuration via environment variables
- Proper database connection pooling
- Enhanced error handling
- Added market data caching
- Improved yfinance fallback handling

v2.0.0 (2025-06-27) - Complete rewrite for dynamic scanning
- Dynamic universe selection (50-100 stocks)
- News catalyst integration
- Multi-stage filtering (50 → 20 → 5)
- Pre-market focus
- Real-time narrowing throughout the day

Description of Service:
This scanner finds the best day trading opportunities by:
1. Starting with market's most active stocks
2. Filtering by news catalysts
3. Confirming with technical setups
4. Delivering top 5 high-conviction picks
"""

import os
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from structlog import get_logger
import redis

# Import only available database utilities
from database_utils import (
    get_db_connection,
    get_redis,
    health_check
)

# Handle yfinance import
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError as e:
    YFINANCE_AVAILABLE = False
    print(f"⚠️ yfinance not available: {e}")

# Database functions that are missing from database_utils.py
def insert_trading_candidates(candidates: List[Dict], scan_id: str = None) -> int:
    """Insert trading candidates into database"""
    if not candidates:
        return 0
        
    conn = None
    inserted = 0
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            for candidate in candidates:
                try:
                    cur.execute("""
                        INSERT INTO trading_candidates (
                            symbol, scan_timestamp, score, price, volume,
                            relative_volume, price_change_pct, market_cap,
                            news_sentiment, news_count, pattern_strength,
                            technical_rating, sector, industry, metadata,
                            scan_id
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        ) ON CONFLICT (symbol, scan_timestamp) DO UPDATE
                        SET score = EXCLUDED.score,
                            price = EXCLUDED.price,
                            volume = EXCLUDED.volume,
                            relative_volume = EXCLUDED.relative_volume,
                            news_sentiment = EXCLUDED.news_sentiment,
                            news_count = EXCLUDED.news_count
                    """, (
                        candidate.get('symbol'),
                        candidate.get('scan_timestamp', datetime.utcnow()),
                        candidate.get('score', 0),
                        candidate.get('price'),
                        candidate.get('volume'),
                        candidate.get('relative_volume'),
                        candidate.get('price_change_pct'),
                        candidate.get('market_cap'),
                        candidate.get('news_sentiment'),
                        candidate.get('news_count', 0),
                        candidate.get('pattern_strength'),
                        candidate.get('technical_rating'),
                        candidate.get('sector'),
                        candidate.get('industry'),
                        json.dumps(candidate.get('metadata', {})),
                        scan_id
                    ))
                    inserted += 1
                except Exception as e:
                    print(f"Failed to insert candidate {candidate.get('symbol')}: {e}")
            
            conn.commit()
            
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Failed to insert trading candidates: {e}")
        
    finally:
        if conn:
            conn.close()
            
    return inserted

def get_active_candidates(limit: int = 20) -> List[Dict]:
    """Get active trading candidates from the last scan"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM trading_candidates
                WHERE scan_timestamp > NOW() - INTERVAL '30 minutes'
                ORDER BY score DESC
                LIMIT %s
            """, (limit,))
            
            candidates = cur.fetchall()
            return candidates if candidates else []
            
    except Exception as e:
        print(f"Failed to get active candidates: {e}")
        return []
        
    finally:
        if conn:
            conn.close()


class DynamicSecurityScanner:
    """
    Enhanced security scanner that dynamically finds trading opportunities
    """
    
    def __init__(self):
        # Initialize environment
        self.setup_environment()
        
        self.app = Flask(__name__)
        self.setup_logging()
        self.setup_routes()
        
        # Initialize Redis client
        self.redis_client = get_redis()
        
        # Service URLs from environment
        self.coordination_url = os.getenv('COORDINATION_URL', 'http://coordination-service:5000')
        self.news_service_url = os.getenv('NEWS_SERVICE_URL', 'http://news-service:5008')
        
        # Scanning parameters from environment
        self.scan_params = {
            # Universe size at each stage
            'initial_universe_size': int(os.getenv('INITIAL_UNIVERSE_SIZE', '100')),
            'catalyst_filter_size': int(os.getenv('CATALYST_FILTER_SIZE', '20')),
            'final_selection_size': int(os.getenv('FINAL_SELECTION_SIZE', '5')),
            
            # Technical criteria
            'min_price': float(os.getenv('MIN_PRICE', '1.0')),
            'max_price': float(os.getenv('MAX_PRICE', '500.0')),
            'min_volume': int(os.getenv('MIN_VOLUME', '500000')),
            'min_relative_volume': float(os.getenv('MIN_RELATIVE_VOLUME', '1.5')),
            'min_price_change': float(os.getenv('MIN_PRICE_CHANGE', '2.0')),
            
            # Pre-market specific
            'premarket_min_volume': int(os.getenv('PREMARKET_MIN_VOLUME', '50000')),
            'premarket_weight': float(os.getenv('PREMARKET_WEIGHT', '2.0')),
            
            # Cache settings
            'cache_ttl': int(os.getenv('SCANNER_CACHE_TTL', '300')),  # 5 minutes
            'concurrent_requests': int(os.getenv('SCANNER_CONCURRENT', '10'))
        }
        
        # Default stock universe (expanded)
        self.default_universe = [
            # Tech giants
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM',
            'ORCL', 'CSCO', 'ADBE', 'NFLX', 'PYPL', 'UBER', 'SNAP', 'SQ', 'SHOP', 'ROKU',
            
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA', 'BLK',
            
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'CVS', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'GILD',
            
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'OXY', 'MRO', 'HAL', 'DVN', 'APA', 'EOG',
            
            # Consumer
            'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'SBUX', 'DIS', 'COST',
            
            # ETFs for market sentiment
            'SPY', 'QQQ', 'IWM', 'DIA', 'VXX', 'GLD', 'SLV', 'USO', 'TLT', 'HYG'
        ]
        
        # Cache for scan results
        self.scan_cache = {
            'timestamp': None,
            'universe': [],
            'catalyst_filtered': [],
            'final_picks': []
        }
        
        # Register with coordination
        self._register_with_coordination()
        
        self.logger.info("Dynamic Security Scanner v2.1.1 initialized",
                        environment=os.getenv('ENVIRONMENT', 'development'),
                        universe_size=len(self.default_universe))
        
    def setup_environment(self):
        """Setup environment variables and paths"""
        # Paths
        self.log_path = os.getenv('LOG_PATH', '/app/logs')
        self.data_path = os.getenv('DATA_PATH', '/app/data')
        
        # Create directories
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        
        # Service configuration
        self.service_name = 'security_scanner'
        self.port = int(os.getenv('PORT', '5001'))
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
    def setup_logging(self):
        """Setup structured logging"""
        self.logger = get_logger()
        self.logger = self.logger.bind(service=self.service_name)
        
    def setup_routes(self):
        """Setup Flask routes"""
        @self.app.route('/health', methods=['GET'])
        def health():
            db_health = health_check()
            return jsonify({
                "status": "healthy",
                "service": "scanner",
                "version": "2.1.1",
                "mode": "dynamic",
                "database": db_health.get('postgresql', {}).get('status', 'unknown'),
                "redis": db_health.get('redis', {}).get('status', 'unknown'),
                "timestamp": datetime.now().isoformat(),
                "yfinance": YFINANCE_AVAILABLE
            })
            
        @self.app.route('/scan', methods=['POST'])
        def scan():
            """Run a security scan"""
            try:
                data = request.json or {}
                mode = data.get('mode', 'normal')
                force_refresh = data.get('force_refresh', False)
                
                # Check cache unless forced refresh
                if not force_refresh and self._is_cache_valid():
                    self.logger.info("Returning cached scan results")
                    return jsonify({
                        'status': 'success',
                        'source': 'cache',
                        'final_picks': self.scan_cache['final_picks']
                    })
                
                # Run the scan
                results = self.run_dynamic_scan(mode)
                
                return jsonify({
                    'status': 'success',
                    'source': 'fresh',
                    'scan_timestamp': datetime.now().isoformat(),
                    'mode': mode,
                    'universe_size': len(results['universe']),
                    'catalyst_filtered': len(results['catalyst_filtered']),
                    'final_picks': results['final_picks']
                })
                
            except Exception as e:
                self.logger.error("Scan failed", error=str(e))
                return jsonify({
                    'status': 'error',
                    'error': str(e)
                }), 500
                
        @self.app.route('/candidates', methods=['GET'])
        def get_candidates():
            """Get active trading candidates"""
            try:
                limit = int(request.args.get('limit', 20))
                candidates = get_active_candidates(limit)
                
                return jsonify({
                    'status': 'success',
                    'count': len(candidates),
                    'candidates': candidates
                })
                
            except Exception as e:
                self.logger.error("Failed to get candidates", error=str(e))
                return jsonify({
                    'status': 'error',
                    'error': str(e)
                }), 500
    
    def run_dynamic_scan(self, mode: str = 'normal') -> Dict:
        """
        Run the complete dynamic scanning workflow
        """
        self.logger.info("Starting dynamic scan", mode=mode)
        scan_id = f"SCAN_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{mode}"
        
        # Stage 1: Build universe (50-100 stocks)
        universe = self._build_dynamic_universe(mode)
        self.logger.info("Universe built", size=len(universe))
        
        # Stage 2: Filter by catalyst (narrow to ~20)
        catalyst_filtered = self._filter_by_catalyst(universe)
        self.logger.info("Catalyst filter applied", 
                        before=len(universe), 
                        after=len(catalyst_filtered))
        
        # Stage 3: Score and rank (select top 5)
        final_picks = self._score_and_select(catalyst_filtered, mode)
        self.logger.info("Final selection complete", picks=len(final_picks))
        
        # Cache results
        self.scan_cache = {
            'timestamp': datetime.now(),
            'universe': universe,
            'catalyst_filtered': catalyst_filtered,
            'final_picks': final_picks
        }
        
        # Save to database
        if final_picks:
            saved = insert_trading_candidates(final_picks, scan_id)
            self.logger.info("Candidates saved to database", count=saved)
        
        return {
            'universe': universe,
            'catalyst_filtered': catalyst_filtered,
            'final_picks': final_picks
        }
    
    def _build_dynamic_universe(self, mode: str) -> List[Dict]:
        """
        Build dynamic universe based on market conditions
        """
        universe = []
        
        # For testing/development, use mock data if yfinance not available
        if not YFINANCE_AVAILABLE:
            self.logger.warning("Using mock data - yfinance not available")
            for symbol in self.default_universe[:self.scan_params['initial_universe_size']]:
                universe.append(self._get_mock_symbol_data(symbol))
            return universe
        
        # Real implementation would:
        # 1. Get market movers
        # 2. Get high volume stocks
        # 3. Get gapping stocks (pre-market)
        # 4. Combine and deduplicate
        
        # For now, scan default universe
        with ThreadPoolExecutor(max_workers=self.scan_params['concurrent_requests']) as executor:
            futures = {executor.submit(self._get_symbol_data, symbol): symbol 
                      for symbol in self.default_universe}
            
            for future in as_completed(futures):
                try:
                    data = future.result()
                    if data and self._meets_basic_criteria(data, mode):
                        universe.append(data)
                        
                    # Stop when we have enough
                    if len(universe) >= self.scan_params['initial_universe_size']:
                        break
                        
                except Exception as e:
                    symbol = futures[future]
                    self.logger.debug(f"Failed to get data for {symbol}: {e}")
        
        return universe
    
    def _get_symbol_data(self, symbol: str) -> Optional[Dict]:
        """Get current data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current price and volume
            history = ticker.history(period='1d', interval='1m')
            if history.empty:
                return None
                
            current_price = history['Close'].iloc[-1]
            current_volume = history['Volume'].sum()
            
            # Calculate metrics
            price_change = current_price - info.get('previousClose', current_price)
            price_change_pct = (price_change / info.get('previousClose', 1)) * 100
            
            return {
                'symbol': symbol,
                'price': current_price,
                'volume': current_volume,
                'avg_volume': info.get('averageVolume', 0),
                'relative_volume': current_volume / max(info.get('averageVolume', 1), 1),
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'market_cap': info.get('marketCap', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.debug(f"Error getting data for {symbol}: {e}")
            return None
    
    def _meets_basic_criteria(self, data: Dict, mode: str) -> bool:
        """Check if symbol meets basic scanning criteria"""
        # Price criteria
        if data['price'] < self.scan_params['min_price'] or data['price'] > self.scan_params['max_price']:
            return False
            
        # Volume criteria
        min_volume = (self.scan_params['premarket_min_volume'] 
                     if mode == 'premarket' 
                     else self.scan_params['min_volume'])
        
        if data['volume'] < min_volume:
            return False
            
        # Relative volume
        if data['relative_volume'] < self.scan_params['min_relative_volume']:
            return False
            
        # Price movement
        if abs(data['price_change_pct']) < self.scan_params['min_price_change']:
            return False
            
        return True
    
    def _filter_by_catalyst(self, universe: List[Dict]) -> List[Dict]:
        """Filter universe by news catalysts"""
        catalyst_filtered = []
        
        try:
            # Get news data from news service
            response = requests.post(
                f"{self.news_service_url}/bulk_analysis",
                json={'symbols': [s['symbol'] for s in universe]},
                timeout=10
            )
            
            if response.status_code == 200:
                news_data = response.json()
                
                # Merge news data with universe data
                for stock in universe:
                    symbol = stock['symbol']
                    if symbol in news_data:
                        stock['news_sentiment'] = news_data[symbol].get('sentiment', 0)
                        stock['news_count'] = news_data[symbol].get('count', 0)
                        stock['has_catalyst'] = news_data[symbol].get('has_catalyst', False)
                        
                        # Include if has catalyst
                        if stock['has_catalyst']:
                            catalyst_filtered.append(stock)
                            
        except Exception as e:
            self.logger.error("Failed to get news data", error=str(e))
            # Fall back to including high movers without news check
            catalyst_filtered = sorted(universe, 
                                     key=lambda x: abs(x['price_change_pct']), 
                                     reverse=True)[:self.scan_params['catalyst_filter_size']]
        
        return catalyst_filtered
    
    def _score_and_select(self, candidates: List[Dict], mode: str) -> List[Dict]:
        """Score candidates and select top picks"""
        # Calculate composite score for each candidate
        for candidate in candidates:
            score = 0
            
            # Price movement score (0-30 points)
            move_score = min(abs(candidate['price_change_pct']) * 3, 30)
            score += move_score
            
            # Volume score (0-25 points)
            vol_score = min(candidate['relative_volume'] * 5, 25)
            score += vol_score
            
            # News catalyst score (0-25 points)
            if candidate.get('has_catalyst'):
                news_score = 15 + min(candidate.get('news_sentiment', 0) * 10, 10)
                score += news_score
            
            # Pre-market bonus (0-20 points)
            if mode == 'premarket':
                score *= self.scan_params['premarket_weight']
            
            # Add score to candidate
            candidate['score'] = round(score, 2)
            candidate['scan_timestamp'] = datetime.now()
        
        # Sort by score and select top picks
        candidates.sort(key=lambda x: x['score'], reverse=True)
        final_picks = candidates[:self.scan_params['final_selection_size']]
        
        # Add ranking
        for i, pick in enumerate(final_picks):
            pick['rank'] = i + 1
            
        return final_picks
    
    def _is_cache_valid(self) -> bool:
        """Check if cached results are still valid"""
        if not self.scan_cache['timestamp']:
            return False
            
        cache_time = self.scan_cache['timestamp']
        if not isinstance(cache_time, datetime):
            return False
            
        age = (datetime.now() - cache_time).total_seconds()
        
        return age < self.scan_params['cache_ttl']
        
    def _get_mock_symbol_data(self, symbol: str) -> Dict:
        """Generate mock data for testing"""
        import random
        
        # Generate somewhat realistic data based on symbol hash
        random.seed(hash(symbol))
        
        base_price = random.uniform(10, 200)
        volatility = random.uniform(0.01, 0.05)
        
        return {
            'symbol': symbol,
            'price': base_price,
            'volume': random.randint(500000, 10000000),
            'avg_volume': random.randint(1000000, 5000000),
            'relative_volume': random.uniform(0.5, 3.0),
            'price_change': base_price * random.uniform(-volatility, volatility),
            'price_change_pct': random.uniform(-5, 5),
            'market_cap': random.randint(1000000000, 50000000000),
            'sector': random.choice(['Technology', 'Healthcare', 'Financial', 'Energy']),
            'industry': 'Mock Industry',
            'timestamp': datetime.now().isoformat()
        }
        
    def _register_with_coordination(self):
        """Register with coordination service"""
        try:
            response = requests.post(
                f"{self.coordination_url}/register_service",
                json={
                    'service_name': 'security_scanner',
                    'service_info': {
                        'url': f"http://scanner-service:{self.port}",
                        'port': self.port,
                        'version': '2.1.1',
                        'capabilities': ['dynamic_scanning', 'catalyst_filtering', 'pre_market']
                    }
                },
                timeout=5
            )
            if response.status_code == 200:
                self.logger.info("Successfully registered with coordination service")
        except Exception as e:
            self.logger.warning(f"Could not register with coordination", error=str(e))
            
    def run(self):
        """Start the scanner service"""
        self.logger.info("Starting Dynamic Security Scanner",
                        version="2.1.1",
                        port=self.port,
                        environment=os.getenv('ENVIRONMENT', 'development'))
        
        self.app.run(host='0.0.0.0', port=self.port, debug=False)


if __name__ == "__main__":
    service = DynamicSecurityScanner()
    service.run()