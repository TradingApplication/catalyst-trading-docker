#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: scanner_service.py
Version: 2.1.1
Last Updated: 2025-07-10
Purpose: Dynamic security scanning with news-based filtering

REVISION HISTORY:
v2.1.1 (2025-07-10) - Fixed import error
- Removed invalid import of insert_trading_candidates from database_utils
- Added local insert_trading_candidates function definition
- Fixed get_active_candidates function definition

v2.1.0 (2025-07-01) - Production deployment version
- PostgreSQL integration
- Environment-based configuration
- Docker service discovery
- Connection pooling

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


# Database functions specific to scanner service
def insert_trading_candidates(candidates: List[Dict], scan_id: str) -> int:
    """Insert trading candidates into database"""
    count = 0
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for candidate in candidates:
                    cur.execute("""
                        INSERT INTO trading_candidates (
                            symbol, scan_id, catalyst_score, news_count,
                            price, volume, relative_volume, price_change_pct,
                            market_cap, sector, industry, 
                            selection_reason, scan_timestamp
                        ) VALUES (
                            %(symbol)s, %(scan_id)s, %(catalyst_score)s, %(news_count)s,
                            %(price)s, %(volume)s, %(relative_volume)s, %(price_change_pct)s,
                            %(market_cap)s, %(sector)s, %(industry)s,
                            %(selection_reason)s, %(scan_timestamp)s
                        )
                        ON CONFLICT (symbol, scan_id) DO UPDATE SET
                            catalyst_score = EXCLUDED.catalyst_score,
                            news_count = EXCLUDED.news_count,
                            updated_at = CURRENT_TIMESTAMP
                    """, {
                        'symbol': candidate['symbol'],
                        'scan_id': scan_id,
                        'catalyst_score': candidate.get('catalyst_score', 0),
                        'news_count': candidate.get('news_count', 0),
                        'price': candidate.get('price', 0),
                        'volume': candidate.get('volume', 0),
                        'relative_volume': candidate.get('relative_volume', 1.0),
                        'price_change_pct': candidate.get('price_change_pct', 0),
                        'market_cap': candidate.get('market_cap', 0),
                        'sector': candidate.get('sector', 'Unknown'),
                        'industry': candidate.get('industry', 'Unknown'),
                        'selection_reason': candidate.get('selection_reason', 'Catalyst-driven'),
                        'scan_timestamp': candidate.get('scan_timestamp', datetime.now())
                    })
                    count += 1
                conn.commit()
    except Exception as e:
        logger = get_logger()
        logger.error("Error inserting trading candidates", error=str(e))
        
    return count


def get_active_candidates(limit: int = 10) -> List[Dict]:
    """Get active trading candidates from the database"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM trading_candidates
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                    ORDER BY catalyst_score DESC, created_at DESC
                    LIMIT %s
                """, (limit,))
                return cur.fetchall()
    except Exception as e:
        logger = get_logger()
        logger.error("Error fetching active candidates", error=str(e))
        return []


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
            # Fixed: Use correct keys from health_check response
            return jsonify({
                "status": "healthy" if db_health['postgresql']['status'] == 'healthy' else "degraded",
                "service": "security_scanner",
                "version": "2.1.1",
                "mode": "dynamic-catalyst",
                "database": db_health['postgresql']['status'],
                "redis": db_health['redis']['status'],
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
            
            return self.scan_cache
            
        except Exception as e:
            self.logger.error("Scan failed", error=str(e))
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
    def _get_initial_universe(self) -> List[str]:
        """Get initial universe of stocks to scan"""
        try:
            # Try to get market movers first
            movers = self._get_market_movers()
            if movers:
                return movers[:self.scan_params['initial_universe_size']]
        except Exception as e:
            self.logger.warning("Could not fetch market movers", error=str(e))
            
        # Fallback to default universe
        return self.default_universe[:self.scan_params['initial_universe_size']]
        
    def _get_market_movers(self) -> List[str]:
        """Get current market movers from various sources"""
        movers = set()
        
        if not YFINANCE_AVAILABLE:
            return list(self.default_universe)
            
        try:
            # Get day gainers
            gainers = yf.Tickers('^GSPC').tickers['^GSPC'].info
            # This is a simplified example - in production you'd use proper APIs
            
            # For now, return enhanced default universe
            return list(self.default_universe)
            
        except Exception as e:
            self.logger.error("Error fetching market movers", error=str(e))
            return list(self.default_universe)
            
    def _enrich_with_market_data(self, symbols: List[str]) -> List[Dict]:
        """Enrich symbols with current market data"""
        enriched = []
        
        if not YFINANCE_AVAILABLE:
            # Return mock data for testing
            for symbol in symbols:
                enriched.append({
                    'symbol': symbol,
                    'price': 100 + np.random.uniform(-50, 50),
                    'volume': np.random.randint(1000000, 10000000),
                    'price_change_pct': np.random.uniform(-5, 5),
                    'relative_volume': np.random.uniform(0.5, 3.0),
                    'market_cap': np.random.randint(1000000000, 100000000000)
                })
            return enriched
            
        # Use ThreadPoolExecutor for concurrent fetching
        with ThreadPoolExecutor(max_workers=self.scan_params['concurrent_requests']) as executor:
            future_to_symbol = {
                executor.submit(self._fetch_symbol_data, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                result = future.result()
                if result:
                    enriched.append(result)
                    
        return enriched
        
    def _fetch_symbol_data(self, symbol: str) -> Optional[Dict]:
        """Fetch market data for a single symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='1d', interval='5m')
            
            if hist.empty:
                return None
                
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            price_change = (current_price - hist['Open'].iloc[0]) / hist['Open'].iloc[0] * 100
            
            return {
                'symbol': symbol,
                'price': float(current_price),
                'volume': int(current_volume),
                'avg_volume': int(avg_volume),
                'relative_volume': float(current_volume / avg_volume) if avg_volume > 0 else 1.0,
                'price_change_pct': float(price_change),
                'market_cap': info.get('marketCap', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
            
        except Exception as e:
            self.logger.debug(f"Could not fetch data for {symbol}: {e}")
            return None
            
    def _filter_by_catalysts(self, stocks: List[Dict], mode: str) -> List[Dict]:
        """Filter stocks by news catalysts"""
        catalyst_stocks = []
        
        # Batch request to news service
        try:
            symbols = [s['symbol'] for s in stocks]
            response = requests.post(
                f"{self.news_service_url}/batch_catalyst_check",
                json={'symbols': symbols, 'mode': mode},
                timeout=30
            )
            
            if response.status_code == 200:
                catalyst_data = response.json()
                
                # Merge catalyst data with stock data
                for stock in stocks:
                    symbol = stock['symbol']
                    if symbol in catalyst_data and catalyst_data[symbol]['has_catalyst']:
                        stock['catalyst_score'] = catalyst_data[symbol]['catalyst_score']
                        stock['news_count'] = catalyst_data[symbol]['news_count']
                        stock['catalyst_data'] = catalyst_data[symbol]
                        catalyst_stocks.append(stock)
                        
        except Exception as e:
            self.logger.error("Error checking catalysts", error=str(e))
            # Fallback: return all stocks with neutral scores
            for stock in stocks:
                stock['catalyst_score'] = 50
                stock['news_count'] = 0
                catalyst_stocks.append(stock)
                
        # Sort by catalyst score and return top N
        catalyst_stocks.sort(key=lambda x: x['catalyst_score'], reverse=True)
        return catalyst_stocks[:self.scan_params['catalyst_filter_size']]
        
    def _apply_technical_filters(self, stocks: List[Dict]) -> List[Dict]:
        """Apply technical filters to stocks"""
        filtered = []
        
        for stock in stocks:
            # Basic technical filters
            if (stock['price'] >= self.scan_params['min_price'] and
                stock['price'] <= self.scan_params['max_price'] and
                stock['volume'] >= self.scan_params['min_volume'] and
                stock['relative_volume'] >= self.scan_params['min_relative_volume'] and
                abs(stock['price_change_pct']) >= self.scan_params['min_price_change']):
                
                # Calculate technical score
                tech_score = self._calculate_technical_score(stock)
                stock['technical_score'] = tech_score
                
                # Combined score (catalyst + technical)
                stock['combined_score'] = (
                    stock['catalyst_score'] * 0.6 +  # 60% weight to catalyst
                    tech_score * 0.4  # 40% weight to technicals
                )
                
                filtered.append(stock)
                
        return filtered
        
    def _calculate_technical_score(self, stock: Dict) -> float:
        """Calculate technical score for a stock"""
        score = 50  # Base score
        
        # Volume surge bonus
        if stock['relative_volume'] > 2.0:
            score += 10
        elif stock['relative_volume'] > 1.5:
            score += 5
            
        # Price movement bonus
        if abs(stock['price_change_pct']) > 5:
            score += 15
        elif abs(stock['price_change_pct']) > 3:
            score += 10
        elif abs(stock['price_change_pct']) > 2:
            score += 5
            
        # Market cap consideration
        if stock['market_cap'] > 10000000000:  # Large cap
            score += 5
        elif stock['market_cap'] < 1000000000:  # Small cap
            score += 10  # More volatile, better for day trading
            
        return min(score, 100)  # Cap at 100
        
    def _select_final_picks(self, stocks: List[Dict], mode: str) -> List[Dict]:
        """Select final trading candidates"""
        # Sort by combined score
        stocks.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Take top N
        final_picks = stocks[:self.scan_params['final_selection_size']]
        
        # Add selection metadata
        for i, pick in enumerate(final_picks):
            pick['rank'] = i + 1
            pick['selection_reason'] = self._get_selection_reason(pick)
            pick['scan_timestamp'] = datetime.now()
            
        return final_picks
        
    def _get_selection_reason(self, stock: Dict) -> str:
        """Generate human-readable selection reason"""
        reasons = []
        
        if stock['catalyst_score'] > 70:
            reasons.append("Strong news catalyst")
        elif stock['catalyst_score'] > 50:
            reasons.append("Moderate news catalyst")
            
        if stock['relative_volume'] > 2.0:
            reasons.append("High volume surge")
            
        if abs(stock['price_change_pct']) > 3:
            reasons.append(f"{stock['price_change_pct']:.1f}% price movement")
            
        return "; ".join(reasons) if reasons else "Catalyst-driven selection"
        
    def _scan_single_symbol(self, symbol: str) -> Dict:
        """Scan a single symbol"""
        try:
            # Get market data
            data = self._fetch_symbol_data(symbol)
            if not data:
                return {'error': f'Could not fetch data for {symbol}'}
                
            # Check for catalyst
            response = requests.get(
                f"{self.news_service_url}/catalyst_check",
                params={'symbol': symbol},
                timeout=10
            )
            
            if response.status_code == 200:
                catalyst_data = response.json()
                data.update(catalyst_data)
                
            # Calculate scores
            data['technical_score'] = self._calculate_technical_score(data)
            data['combined_score'] = (
                data.get('catalyst_score', 50) * 0.6 +
                data['technical_score'] * 0.4
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error scanning {symbol}", error=str(e))
            return {'error': str(e)}
            
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self.scan_cache.get('timestamp'):
            return False
            
        cache_time = datetime.fromisoformat(self.scan_cache['timestamp'])
        age = (datetime.now() - cache_time).total_seconds()
        
        return age < self.scan_params['cache_ttl']
        
    def _register_with_coordination(self):
        """Register with coordination service"""
        try:
            response = requests.post(
                f"{self.coordination_url}/register_service",
                json={
                    'service_name': 'scanner',
                    'service_info': {
                        'url': f"http://scanner-service:{self.port}",
                        'version': '2.1.1',
                        'capabilities': ['dynamic_scan', 'catalyst_filter', 'premarket']
                    }
                },
                timeout=5
            )
            
            if response.status_code == 200:
                self.logger.info("Registered with coordination service")
            else:
                self.logger.warning("Failed to register with coordination", 
                                  status=response.status_code)
                                  
        except Exception as e:
            self.logger.warning("Could not register with coordination", error=str(e))
            
    def run(self):
        """Start the scanner service"""
        self.logger.info("Starting Dynamic Security Scanner",
                        port=self.port,
                        environment=os.getenv('ENVIRONMENT', 'development'))
        
        # Use host='0.0.0.0' for Docker
        self.app.run(
            host='0.0.0.0',
            port=self.port,
            debug=False
        )


if __name__ == '__main__':
    scanner = DynamicSecurityScanner()
    scanner.run()
