#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: scanner_service.py
Version: 2.1.0
Last Updated: 2025-07-01
Purpose: Dynamic security scanning with news-based filtering and PostgreSQL

REVISION HISTORY:
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

# Import database utilities
from database_utils import (
    get_db_connection,
    insert_trading_candidates,
    get_active_candidates,
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
        
        self.logger.info("Dynamic Security Scanner v2.1.0 initialized",
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
                "status": "healthy" if db_health['database'] == 'healthy' else "degraded",
                "service": "security_scanner",
                "version": "2.1.0",
                "mode": "dynamic-catalyst",
                "database": db_health['database'],
                "redis": db_health['redis'],
                "timestamp": datetime.now().isoformat()
            })
            
        @self.app.route('/scan', methods=['GET'])
        def scan():
            """Regular market hours scan"""
            mode = request.args.get('mode', 'normal')
            force_refresh = request.args.get('force', 'false').lower() == 'true'
            
            result = self.perform_dynamic_scan(mode, force_refresh)
            return jsonify(result)
            
        @self.app.route('/scan_premarket', methods=['GET'])
        def scan_premarket():
            """Pre-market aggressive scan"""
            result = self.perform_dynamic_scan('aggressive', force_refresh=True)
            return jsonify(result)
            
        @self.app.route('/active_symbols', methods=['GET'])
        def active_symbols():
            """Get currently active symbols"""
            # Check cache first
            cached = self.redis_client.get('active_symbols')
            if cached:
                return jsonify(json.loads(cached))
                
            symbols = self._get_market_movers()
            
            # Cache for 5 minutes
            self.redis_client.setex(
                'active_symbols',
                self.scan_params['cache_ttl'],
                json.dumps({'symbols': symbols})
            )
            
            return jsonify({'symbols': symbols})
            
        @self.app.route('/scan_symbol', methods=['POST'])
        def scan_symbol():
            """Scan specific symbol"""
            data = request.json
            symbol = data.get('symbol')
            
            if not symbol:
                return jsonify({'error': 'Symbol required'}), 400
                
            result = self._scan_single_symbol(symbol)
            return jsonify(result)
            
        @self.app.route('/candidates', methods=['GET'])
        def get_candidates():
            """Get current trading candidates"""
            limit = request.args.get('limit', 10, type=int)
            candidates = get_active_candidates(limit)
            
            return jsonify({
                'count': len(candidates),
                'candidates': candidates
            })
            
    def perform_dynamic_scan(self, mode: str = 'normal', 
                           force_refresh: bool = False) -> Dict:
        """
        Perform dynamic security scan with news catalyst filtering
        """
        start_time = datetime.now()
        
        # Check cache unless forced refresh
        if not force_refresh and self._is_cache_valid():
            self.logger.info("Returning cached scan results")
            return self.scan_cache
            
        self.logger.info("Starting dynamic scan", mode=mode)
        
        try:
            # Step 1: Get initial universe (100 stocks)
            universe = self._get_initial_universe()
            self.logger.info("Initial universe selected", count=len(universe))
            
            # Step 2: Enrich with market data
            enriched_universe = self._enrich_with_market_data(universe)
            
            # Step 3: Filter by news catalysts (100 → 20)
            catalyst_filtered = self._filter_by_catalysts(enriched_universe, mode)
            self.logger.info("Catalyst filtering complete", 
                           before=len(enriched_universe),
                           after=len(catalyst_filtered))
            
            # Step 4: Apply technical filters and scoring
            technical_filtered = self._apply_technical_filters(catalyst_filtered)
            
            # Step 5: Final selection (20 → 5)
            final_picks = self._select_final_picks(technical_filtered, mode)
            self.logger.info("Final selection complete", count=len(final_picks))
            
            # Generate scan ID
            scan_id = f"SCAN_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{mode}"
            
            # Save to database
            if final_picks:
                saved_count = insert_trading_candidates(final_picks, scan_id)
                self.logger.info("Saved candidates to database", count=saved_count)
            
            # Update cache
            self.scan_cache = {
                'scan_id': scan_id,
                'timestamp': datetime.now().isoformat(),
                'mode': mode,
                'universe': universe,
                'catalyst_filtered': [c['symbol'] for c in catalyst_filtered],
                'final_picks': final_picks,
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
            
            # Cache in Redis
            self.redis_client.setex(
                f"scan_results:{mode}",
                self.scan_params['cache_ttl'],
                json.dumps(self.scan_cache)
            )
            
            return self.scan_cache
            
        except Exception as e:
            self.logger.error("Scan error", error=str(e), mode=mode)
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'mode': mode
            }
            
    def _get_initial_universe(self) -> List[str]:
        """Get initial universe of stocks to scan"""
        try:
            # Try to get market movers
            movers = self._get_market_movers()
            
            # Combine with default universe
            combined = list(set(movers + self.default_universe))
            
            # Limit to configured size
            return combined[:self.scan_params['initial_universe_size']]
            
        except Exception as e:
            self.logger.warning("Could not get market movers, using defaults",
                              error=str(e))
            return self.default_universe[:self.scan_params['initial_universe_size']]
            
    def _get_market_movers(self) -> List[str]:
        """Get today's market movers"""
        movers = []
        
        if not YFINANCE_AVAILABLE:
            return []
            
        try:
            # Get various screeners
            screeners = [
                'most_actives',
                'gainers',
                'losers',
                'trending_tickers'
            ]
            
            for screener in screeners:
                try:
                    # Check cache first
                    cache_key = f"movers:{screener}"
                    cached = self.redis_client.get(cache_key)
                    if cached:
                        movers.extend(json.loads(cached))
                        continue
                    
                    # Get from yfinance
                    tickers = yf.Tickers('')  # Empty tickers object
                    screen_data = getattr(tickers, screener, None)
                    
                    if screen_data and hasattr(screen_data, 'tickers'):
                        symbols = [t.ticker for t in screen_data.tickers[:20]]
                        movers.extend(symbols)
                        
                        # Cache the results
                        self.redis_client.setex(
                            cache_key,
                            self.scan_params['cache_ttl'],
                            json.dumps(symbols)
                        )
                        
                except Exception as e:
                    self.logger.debug(f"Could not get {screener}", error=str(e))
                    
            # Remove duplicates
            return list(set(movers))
            
        except Exception as e:
            self.logger.error("Error getting market movers", error=str(e))
            return []
            
    def _enrich_with_market_data(self, symbols: List[str]) -> List[Dict]:
        """Enrich symbols with current market data"""
        enriched = []
        
        # Use ThreadPoolExecutor for concurrent requests
        with ThreadPoolExecutor(max_workers=self.scan_params['concurrent_requests']) as executor:
            futures = {executor.submit(self._get_symbol_data, symbol): symbol 
                      for symbol in symbols}
            
            for future in as_completed(futures):
                try:
                    data = future.result()
                    if data:
                        enriched.append(data)
                except Exception as e:
                    symbol = futures[future]
                    self.logger.error(f"Error enriching {symbol}", error=str(e))
                    
        return enriched
        
    def _get_symbol_data(self, symbol: str) -> Optional[Dict]:
        """Get market data for a single symbol"""
        try:
            # Check cache first
            cache_key = f"symbol_data:{symbol}"
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            if YFINANCE_AVAILABLE:
                ticker = yf.Ticker(symbol)
                
                # Get current data
                info = ticker.info
                history = ticker.history(period="5d", interval="1d")
                
                if history.empty:
                    return None
                    
                # Calculate metrics
                current_price = history['Close'].iloc[-1]
                prev_close = history['Close'].iloc[-2] if len(history) > 1 else current_price
                volume = history['Volume'].iloc[-1]
                avg_volume = history['Volume'].mean()
                
                data = {
                    'symbol': symbol,
                    'price': current_price,
                    'volume': volume,
                    'avg_volume': avg_volume,
                    'relative_volume': volume / avg_volume if avg_volume > 0 else 0,
                    'price_change': current_price - prev_close,
                    'price_change_pct': ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0,
                    'market_cap': info.get('marketCap', 0),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Cache the data
                self.redis_client.setex(
                    cache_key,
                    self.scan_params['cache_ttl'],
                    json.dumps(data)
                )
                
                return data
            else:
                # Mock data for testing
                return self._get_mock_symbol_data(symbol)
                
        except Exception as e:
            self.logger.error(f"Error getting data for {symbol}", error=str(e))
            return None
            
    def _filter_by_catalysts(self, universe: List[Dict], mode: str) -> List[Dict]:
        """Filter stocks by news catalysts"""
        catalyst_filtered = []
        
        # Get news for all symbols
        try:
            response = requests.post(
                f"{self.news_service_url}/search_news",
                json={
                    'symbols': [s['symbol'] for s in universe],
                    'hours': 24 if mode == 'normal' else 48,
                    'market_state': 'pre-market' if mode == 'aggressive' else None
                },
                timeout=30
            )
            
            if response.status_code != 200:
                self.logger.error("News service error", status=response.status_code)
                return universe[:self.scan_params['catalyst_filter_size']]
                
            news_data = response.json()
            
            # Count news by symbol
            news_counts = {}
            catalyst_types = {}
            
            for article in news_data.get('results', []):
                symbol = article.get('symbol')
                if symbol:
                    news_counts[symbol] = news_counts.get(symbol, 0) + 1
                    
                    # Track catalyst types
                    keywords = article.get('keywords', [])
                    if symbol not in catalyst_types:
                        catalyst_types[symbol] = set()
                    catalyst_types[symbol].update(keywords)
                    
                # Also check mentioned tickers
                for ticker in article.get('mentioned_tickers', []):
                    news_counts[ticker] = news_counts.get(ticker, 0) + 1
                    
        except Exception as e:
            self.logger.error("Error getting news", error=str(e))
            news_counts = {}
            catalyst_types = {}
            
        # Score and filter stocks
        for stock in universe:
            symbol = stock['symbol']
            
            # Calculate catalyst score
            news_count = news_counts.get(symbol, 0)
            catalysts = list(catalyst_types.get(symbol, []))
            
            # Base score from news count
            catalyst_score = min(100, news_count * 20)
            
            # Boost for specific catalyst types
            high_impact_catalysts = {'earnings', 'fda', 'merger', 'analyst'}
            if any(c in high_impact_catalysts for c in catalysts):
                catalyst_score *= 1.5
                
            # Apply pre-market weight if applicable
            if mode == 'aggressive' and stock.get('pre_market_volume', 0) > 0:
                catalyst_score *= self.scan_params['premarket_weight']
                
            # Add catalyst info to stock data
            stock['catalyst_score'] = catalyst_score
            stock['news_count'] = news_count
            stock['catalysts'] = catalysts
            stock['has_catalyst'] = news_count > 0
            
            catalyst_filtered.append(stock)
            
        # Sort by catalyst score and take top N
        catalyst_filtered.sort(key=lambda x: x['catalyst_score'], reverse=True)
        
        return catalyst_filtered[:self.scan_params['catalyst_filter_size']]
        
    def _apply_technical_filters(self, stocks: List[Dict]) -> List[Dict]:
        """Apply technical criteria filters"""
        filtered = []
        
        for stock in stocks:
            # Apply filters
            if stock['price'] < self.scan_params['min_price']:
                continue
            if stock['price'] > self.scan_params['max_price']:
                continue
            if stock['volume'] < self.scan_params['min_volume']:
                continue
            if stock['relative_volume'] < self.scan_params['min_relative_volume']:
                continue
            if abs(stock['price_change_pct']) < self.scan_params['min_price_change']:
                continue
                
            # Calculate technical score
            technical_score = 0
            
            # Volume score (0-40 points)
            if stock['relative_volume'] > 3:
                technical_score += 40
            elif stock['relative_volume'] > 2:
                technical_score += 30
            else:
                technical_score += 20
                
            # Price movement score (0-30 points)
            price_move = abs(stock['price_change_pct'])
            if price_move > 5:
                technical_score += 30
            elif price_move > 3:
                technical_score += 20
            else:
                technical_score += 10
                
            # Momentum alignment (0-30 points)
            if stock['price_change_pct'] > 0 and 'bullish' in str(stock.get('catalysts', [])):
                technical_score += 30
            elif stock['price_change_pct'] < 0 and 'bearish' in str(stock.get('catalysts', [])):
                technical_score += 30
            else:
                technical_score += 15
                
            stock['technical_score'] = technical_score
            
            # Combined score
            stock['combined_score'] = (
                stock['catalyst_score'] * 0.6 +  # 60% weight on catalyst
                stock['technical_score'] * 0.4    # 40% weight on technicals
            )
            
            filtered.append(stock)
            
        return filtered
        
    def _select_final_picks(self, stocks: List[Dict], mode: str) -> List[Dict]:
        """Select final trading candidates"""
        # Sort by combined score
        stocks.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Take top N
        final_picks = stocks[:self.scan_params['final_selection_size']]
        
        # Add selection metadata
        for i, pick in enumerate(final_picks):
            pick['selection_rank'] = i + 1
            pick['selection_timestamp'] = datetime.now().isoformat()
            pick['scan_mode'] = mode
            
            # Determine primary catalyst
            if pick.get('catalysts'):
                pick['primary_catalyst'] = pick['catalysts'][0]
            else:
                pick['primary_catalyst'] = 'technical'
                
            # Add any pre-market specific data
            if mode == 'aggressive':
                pick['pre_market_scan'] = True
                # Could add pre-market volume/price here if available
                
        return final_picks
        
    def _scan_single_symbol(self, symbol: str) -> Dict:
        """Scan a single symbol on demand"""
        try:
            # Get symbol data
            data = self._get_symbol_data(symbol)
            if not data:
                return {'error': f'No data available for {symbol}'}
                
            # Get news data
            response = requests.get(
                f"{self.news_service_url}/news/{symbol}",
                params={'hours': 24},
                timeout=10
            )
            
            news_count = 0
            catalysts = []
            
            if response.status_code == 200:
                news_data = response.json()
                news_count = news_data.get('count', 0)
                
                # Extract catalysts
                catalyst_set = set()
                for article in news_data.get('articles', []):
                    catalyst_set.update(article.get('headline_keywords', []))
                catalysts = list(catalyst_set)
                
            # Calculate scores
            catalyst_score = min(100, news_count * 20)
            data['catalyst_score'] = catalyst_score
            data['news_count'] = news_count
            data['catalysts'] = catalysts
            
            # Apply technical filters
            technical_filtered = self._apply_technical_filters([data])
            
            if technical_filtered:
                return technical_filtered[0]
            else:
                return {
                    'symbol': symbol,
                    'status': 'Does not meet criteria',
                    'data': data
                }
                
        except Exception as e:
            self.logger.error(f"Error scanning {symbol}", error=str(e))
            return {'error': str(e), 'symbol': symbol}
            
    def _is_cache_valid(self) -> bool:
        """Check if scan cache is still valid"""
        if not self.scan_cache.get('timestamp'):
            return False
            
        cache_time = datetime.fromisoformat(self.scan_cache['timestamp'])
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
                        'version': '2.1.0',
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
                        version="2.1.0",
                        port=self.port,
                        environment=os.getenv('ENVIRONMENT', 'development'))
        
        self.app.run(host='0.0.0.0', port=self.port, debug=False)


if __name__ == "__main__":
    service = DynamicSecurityScanner()
    service.run()