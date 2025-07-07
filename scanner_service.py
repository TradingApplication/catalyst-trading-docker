#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: scanner_service.py
Version: 2.2.0
Last Updated: 2025-07-07
Purpose: Enhanced dynamic security scanning with top 100 tracking

REVISION HISTORY:
v2.2.0 (2025-07-07) - Enhanced data collection implementation
- Track top 100 securities (not just top 5)
- Intelligent data aging system
- Pattern discovery preparation
- Storage optimization with tiered collection

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
5. NEW: Tracking top 100 for pattern discovery
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
import threading

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


class EnhancedSecurityScanner:
    """
    Enhanced security scanner that tracks top 100 for pattern discovery
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
        
        # Enhanced scanning parameters
        self.scan_params = {
            # NEW: Track more securities
            'track_top_n': int(os.getenv('TRACK_TOP_N', '100')),  # Track top 100
            'trade_top_n': int(os.getenv('TRADE_TOP_N', '5')),    # Trade top 5
            
            # Universe sizing
            'initial_universe_size': int(os.getenv('INITIAL_UNIVERSE_SIZE', '500')),
            'catalyst_filter_size': int(os.getenv('CATALYST_FILTER_SIZE', '100')),
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
            'cache_ttl': int(os.getenv('SCANNER_CACHE_TTL', '300'))  # 5 minutes
        }
        
        # NEW: Tracking state for top 100
        self.tracked_securities = {}  # symbol -> tracking_info
        self.tracking_lock = threading.Lock()
        
        # Data collection frequencies
        self.collection_frequencies = {
            'ultra_high': timedelta(minutes=1),    # Breaking news/high volatility
            'high': timedelta(minutes=15),         # Active tracking
            'medium': timedelta(hours=1),          # Normal tracking
            'low': timedelta(hours=6),             # Aging securities
            'archive': timedelta(days=1)           # Historical only
        }
        
        # Register with coordination
        self._register_with_coordination()
        
        # Start background tracking thread
        self._start_tracking_thread()
        
        self.logger.info("Enhanced Security Scanner v2.2.0 initialized",
                        environment=os.getenv('ENVIRONMENT', 'development'),
                        track_top_n=self.scan_params['track_top_n'])
        
    def setup_environment(self):
        """Setup environment variables and paths"""
        self.log_path = os.getenv('LOG_PATH', '/app/logs')
        self.data_path = os.getenv('DATA_PATH', '/app/data')
        
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        
        self.service_name = 'scanner_service'
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
            """Health check endpoint"""
            db_health = health_check()
            
            # Handle different database_utils versions
            db_status = False
            redis_status = False
            
            if 'postgresql' in db_health:
                db_status = db_health['postgresql'].get('status') == 'healthy'
                redis_status = db_health['redis'].get('status') == 'healthy'
            elif 'database' in db_health:
                db_status = db_health['database']
                redis_status = db_health['redis']
            
            return jsonify({
                "status": "healthy" if db_status else "degraded",
                "service": self.service_name,
                "version": "2.2.0",
                "database": db_status,
                "redis": redis_status,
                "tracking_count": len(self.tracked_securities),
                "timestamp": datetime.now().isoformat()
            })
            
        @self.app.route('/scan', methods=['GET'])
        def scan():
            """Regular market scan - now returns top 100 for tracking"""
            mode = request.args.get('mode', 'normal')
            results = self.scan_market(mode)
            return jsonify(results)
            
        @self.app.route('/scan_premarket', methods=['GET'])
        def scan_premarket():
            """Pre-market aggressive scan"""
            results = self.scan_market('aggressive')
            return jsonify(results)
            
        @self.app.route('/tracked_securities', methods=['GET'])
        def get_tracked():
            """Get list of all tracked securities"""
            with self.tracking_lock:
                tracked = [
                    {
                        'symbol': symbol,
                        'tracking_since': info['first_seen'].isoformat(),
                        'frequency': info['collection_frequency'],
                        'data_points': info['data_points_collected'],
                        'last_update': info['last_updated'].isoformat()
                    }
                    for symbol, info in self.tracked_securities.items()
                ]
            return jsonify({
                'count': len(tracked),
                'securities': tracked
            })
            
        @self.app.route('/tracking_stats', methods=['GET'])
        def tracking_stats():
            """Get tracking statistics"""
            stats = self._calculate_tracking_stats()
            return jsonify(stats)
            
    def scan_market(self, mode: str = 'normal') -> Dict:
        """
        Enhanced market scan that tracks top 100 securities
        """
        self.logger.info("Starting enhanced market scan",
                        mode=mode,
                        track_top_n=self.scan_params['track_top_n'])
        
        scan_start = time.time()
        
        # 1. Get news-mentioned securities
        news_securities = self._get_news_mentioned_securities()
        self.logger.info(f"Found {len(news_securities)} securities with news")
        
        # 2. Get market movers
        market_movers = self._get_market_movers()
        
        # 3. Combine and deduplicate
        all_candidates = list(set(news_securities + market_movers))
        self.logger.info(f"Total candidates to evaluate: {len(all_candidates)}")
        
        # 4. Score all candidates
        scored_candidates = self._score_candidates(all_candidates, mode)
        
        # 5. Get top 100 for tracking (NEW)
        top_100 = scored_candidates[:self.scan_params['track_top_n']]
        top_5_trading = scored_candidates[:self.scan_params['trade_top_n']]
        
        # 6. Start tracking all top 100
        for candidate in top_100:
            self._start_tracking_security(candidate)
        
        # 7. Store results
        self._store_scan_results(top_100, top_5_trading)
        
        scan_time = time.time() - scan_start
        
        return {
            'timestamp': datetime.now().isoformat(),
            'mode': mode,
            'scan_time_seconds': round(scan_time, 2),
            'total_evaluated': len(all_candidates),
            'tracking_candidates': top_100,      # NEW: All 100 for data collection
            'trading_candidates': top_5_trading,  # Top 5 for actual trading
            'tracked_securities_total': len(self.tracked_securities)
        }
        
    def _start_tracking_security(self, candidate: Dict):
        """Start or update tracking for a security"""
        symbol = candidate['symbol']
        
        with self.tracking_lock:
            if symbol in self.tracked_securities:
                # Update existing tracking
                tracking_info = self.tracked_securities[symbol]
                tracking_info['last_catalyst_score'] = candidate['score']
                tracking_info['catalyst_events'].append({
                    'timestamp': datetime.now(),
                    'score': candidate['score'],
                    'catalyst_type': candidate.get('catalyst_type'),
                    'news_count': candidate.get('news_count', 0)
                })
                
                # Potentially increase collection frequency
                if candidate['score'] > 80:
                    tracking_info['collection_frequency'] = 'high'
                    
            else:
                # New security to track
                self.tracked_securities[symbol] = {
                    'symbol': symbol,
                    'first_seen': datetime.now(),
                    'last_updated': datetime.now(),
                    'collection_frequency': 'high' if candidate['score'] > 70 else 'medium',
                    'data_points_collected': 0,
                    'catalyst_events': [{
                        'timestamp': datetime.now(),
                        'score': candidate['score'],
                        'catalyst_type': candidate.get('catalyst_type'),
                        'news_count': candidate.get('news_count', 0)
                    }],
                    'last_catalyst_score': candidate['score'],
                    'flatline_periods': 0,
                    'last_price': None,
                    'last_volume': None
                }
                
                self.logger.info(f"Started tracking {symbol}",
                               score=candidate['score'],
                               frequency=self.tracked_securities[symbol]['collection_frequency'])
                
    def _collect_security_data(self, symbol: str, tracking_info: Dict):
        """Collect comprehensive data for a tracked security"""
        try:
            # Get price and volume data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d', interval='15m')
            
            if hist.empty:
                return None
                
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            
            # Calculate metrics
            price_change = 0
            if tracking_info['last_price']:
                price_change = (current_price - tracking_info['last_price']) / tracking_info['last_price']
                
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Determine if flatlined
            is_flatlined = abs(price_change) < 0.001 and volume_ratio < 0.5
            
            # Store data based on frequency
            data = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'price': current_price,
                'volume': current_volume,
                'price_change': price_change,
                'volume_ratio': volume_ratio,
                'frequency': tracking_info['collection_frequency'],
                'rsi': self._calculate_rsi(hist['Close'].values),
                'vwap': self._calculate_vwap(hist),
                'catalyst_score': tracking_info['last_catalyst_score']
            }
            
            # Store in database
            self._store_security_data(data)
            
            # Update tracking info
            tracking_info['last_updated'] = datetime.now()
            tracking_info['data_points_collected'] += 1
            tracking_info['last_price'] = current_price
            tracking_info['last_volume'] = current_volume
            
            if is_flatlined:
                tracking_info['flatline_periods'] += 1
            else:
                tracking_info['flatline_periods'] = 0
                
            # Age the security if needed
            self._check_aging_criteria(symbol, tracking_info, is_flatlined)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error collecting data for {symbol}", error=str(e))
            return None
            
    def _check_aging_criteria(self, symbol: str, tracking_info: Dict, is_flatlined: bool):
        """Determine if security should change collection frequency"""
        current_freq = tracking_info['collection_frequency']
        hours_since_catalyst = (datetime.now() - tracking_info['catalyst_events'][-1]['timestamp']).total_seconds() / 3600
        
        # Promotion criteria (increase frequency)
        if tracking_info['last_catalyst_score'] > 80 and current_freq != 'high':
            tracking_info['collection_frequency'] = 'high'
            self.logger.info(f"Promoted {symbol} to high frequency")
            
        # Demotion criteria (decrease frequency)
        elif is_flatlined and tracking_info['flatline_periods'] > 4:
            if current_freq == 'high':
                tracking_info['collection_frequency'] = 'medium'
            elif current_freq == 'medium':
                tracking_info['collection_frequency'] = 'low'
            self.logger.info(f"Demoted {symbol} to {tracking_info['collection_frequency']} frequency")
            
        # Old catalyst decay
        elif hours_since_catalyst > 48 and current_freq == 'high':
            tracking_info['collection_frequency'] = 'medium'
            
        # Archive very old securities
        elif hours_since_catalyst > 168 and current_freq != 'archive':  # 1 week
            tracking_info['collection_frequency'] = 'archive'
            self.logger.info(f"Archived {symbol}")
            
    def _store_security_data(self, data: Dict):
        """Store security data in appropriate table based on frequency"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Determine table based on frequency
                    if data['frequency'] in ['ultra_high', 'high']:
                        table = 'security_data_high_freq'
                    elif data['frequency'] == 'medium':
                        table = 'security_data_hourly'
                    else:
                        table = 'security_data_daily'
                        
                    cur.execute(f"""
                        INSERT INTO {table} 
                        (symbol, timestamp, close, volume, rsi_14, vwap, 
                         news_count, catalyst_active)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        data['symbol'],
                        data['timestamp'],
                        data['price'],
                        data['volume'],
                        data.get('rsi'),
                        data.get('vwap'),
                        0,  # Will update with actual news count
                        data['catalyst_score'] > 60
                    ))
                    conn.commit()
                    
        except Exception as e:
            self.logger.error(f"Error storing security data", error=str(e))
            
    def _tracking_thread(self):
        """Background thread for collecting data on tracked securities"""
        self.logger.info("Starting tracking thread")
        
        while True:
            try:
                current_time = datetime.now()
                
                with self.tracking_lock:
                    for symbol, tracking_info in list(self.tracked_securities.items()):
                        # Check if it's time to collect
                        freq = tracking_info['collection_frequency']
                        interval = self.collection_frequencies.get(freq, timedelta(hours=1))
                        
                        if current_time - tracking_info['last_updated'] >= interval:
                            # Collect data in thread pool to avoid blocking
                            self._collect_security_data(symbol, tracking_info)
                            
                # Run pattern discovery every hour
                if current_time.minute == 0:
                    self._run_pattern_discovery()
                    
            except Exception as e:
                self.logger.error("Error in tracking thread", error=str(e))
                
            # Sleep for 30 seconds before next check
            time.sleep(30)
            
    def _run_pattern_discovery(self):
        """Run pattern discovery on tracked securities"""
        try:
            # Get recent data for all tracked securities
            with self.tracking_lock:
                symbols = list(self.tracked_securities.keys())
                
            if len(symbols) < 10:
                return  # Need minimum securities for patterns
                
            # Find correlation patterns
            correlations = self._find_correlation_patterns(symbols)
            
            # Find catalyst sympathy patterns
            sympathy_patterns = self._find_catalyst_sympathy(symbols)
            
            # Store discoveries
            for pattern in correlations + sympathy_patterns:
                self._store_pattern_discovery(pattern)
                
            self.logger.info(f"Pattern discovery complete",
                           correlations=len(correlations),
                           sympathy=len(sympathy_patterns))
                           
        except Exception as e:
            self.logger.error("Error in pattern discovery", error=str(e))
            
    def _find_correlation_patterns(self, symbols: List[str]) -> List[Dict]:
        """Find securities that move together"""
        patterns = []
        
        try:
            # Get price data for all symbols
            price_data = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d', interval='1h')
                if not hist.empty:
                    price_data[symbol] = hist['Close'].pct_change().dropna()
                    
            # Calculate correlations
            if len(price_data) >= 2:
                df = pd.DataFrame(price_data)
                corr_matrix = df.corr()
                
                # Find high correlations
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        symbol1 = corr_matrix.columns[i]
                        symbol2 = corr_matrix.columns[j]
                        correlation = corr_matrix.iloc[i, j]
                        
                        if abs(correlation) > 0.7:
                            pattern = {
                                'type': 'correlation',
                                'symbols': [symbol1, symbol2],
                                'correlation': correlation,
                                'timestamp': datetime.now(),
                                'confidence': abs(correlation)
                            }
                            patterns.append(pattern)
                            
        except Exception as e:
            self.logger.error("Error finding correlations", error=str(e))
            
        return patterns
        
    def _find_catalyst_sympathy(self, symbols: List[str]) -> List[Dict]:
        """Find securities that react to similar catalysts"""
        patterns = []
        
        # Group by recent catalyst events
        catalyst_groups = {}
        
        with self.tracking_lock:
            for symbol, info in self.tracked_securities.items():
                if symbol not in symbols:
                    continue
                    
                # Get recent catalyst
                if info['catalyst_events']:
                    latest = info['catalyst_events'][-1]
                    catalyst_type = latest.get('catalyst_type', 'unknown')
                    
                    if catalyst_type not in catalyst_groups:
                        catalyst_groups[catalyst_type] = []
                    catalyst_groups[catalyst_type].append(symbol)
                    
        # Find groups that moved together
        for catalyst_type, group_symbols in catalyst_groups.items():
            if len(group_symbols) >= 2:
                pattern = {
                    'type': 'catalyst_sympathy',
                    'catalyst': catalyst_type,
                    'symbols': group_symbols,
                    'timestamp': datetime.now(),
                    'confidence': 0.7
                }
                patterns.append(pattern)
                
        return patterns
        
    def _store_pattern_discovery(self, pattern: Dict):
        """Store discovered pattern in database"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO ml_pattern_discoveries
                        (pattern_type, securities_involved, pattern_confidence,
                         trigger_conditions, discovery_date)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        pattern['type'],
                        json.dumps(pattern['symbols']),
                        pattern['confidence'],
                        json.dumps(pattern),
                        pattern['timestamp']
                    ))
                    conn.commit()
                    
        except Exception as e:
            self.logger.error("Error storing pattern discovery", error=str(e))
            
    def _calculate_tracking_stats(self) -> Dict:
        """Calculate statistics about tracked securities"""
        with self.tracking_lock:
            freq_counts = {}
            for info in self.tracked_securities.values():
                freq = info['collection_frequency']
                freq_counts[freq] = freq_counts.get(freq, 0) + 1
                
            total_data_points = sum(
                info['data_points_collected'] 
                for info in self.tracked_securities.values()
            )
            
            return {
                'total_tracked': len(self.tracked_securities),
                'frequency_distribution': freq_counts,
                'total_data_points': total_data_points,
                'active_high_freq': freq_counts.get('high', 0),
                'archived': freq_counts.get('archive', 0)
            }
            
    def _start_tracking_thread(self):
        """Start the background tracking thread"""
        thread = threading.Thread(target=self._tracking_thread, daemon=True)
        thread.start()
        
    def _register_with_coordination(self):
        """Register with coordination service"""
        try:
            response = requests.post(
                f"{self.coordination_url}/register_service",
                json={
                    'service_name': 'scanner_service',
                    'service_info': {
                        'url': f"http://scanner-service:{self.port}",
                        'port': self.port,
                        'version': '2.2.0',
                        'capabilities': [
                            'market_scanning', 
                            'catalyst_scoring',
                            'top_100_tracking',  # NEW capability
                            'pattern_discovery'   # NEW capability
                        ]
                    }
                },
                timeout=5
            )
            if response.status_code == 200:
                self.logger.info("Successfully registered with coordination service")
            else:
                self.logger.warning(f"Registration returned status {response.status_code}")
        except Exception as e:
            self.logger.warning(f"Could not register with coordination", error=str(e))
            
    def run(self):
        """Start the scanner service"""
        self.logger.info("Starting Enhanced Scanner Service",
                        version="2.2.0",
                        port=self.port,
                        environment=os.getenv('ENVIRONMENT', 'development'))
        
        self.app.run(host='0.0.0.0', port=self.port, debug=False)


if __name__ == "__main__":
    service = EnhancedSecurityScanner()
    service.run()