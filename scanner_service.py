#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: scanner_service.py
Version: 3.0.0
Last Updated: 2025-01-27
Purpose: Enhanced security scanning to track top 100 securities with intelligent data collection

REVISION HISTORY:
v3.0.0 (2025-01-27) - Enhanced for top 100 tracking
- Tracks top 100 securities (not just top 5)
- Populates new database tables (security_data_high_freq, etc.)
- Implements intelligent data aging
- Maintains backward compatibility for trading top 5
- Added security tracking state management
- Implements collection frequency logic

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
This enhanced scanner finds the best day trading opportunities while building
a comprehensive data lake for pattern discovery:
1. Starting with market's most active stocks
2. Filtering by news catalysts
3. Tracking top 100 for pattern analysis
4. Returning top 5 for immediate trading
"""

# Standard library imports
import os
import sys
import json
import time
import asyncio
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Third-party imports
from flask import Flask, jsonify, request
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import redis
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from psycopg2.pool import ThreadedConnectionPool

# Technical analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("WARNING: TA-Lib not available, using fallback calculations")

# Custom imports
from database_utils import (
    get_logger, get_db_connection, execute_query, health_check,
    insert_trading_candidates, get_active_candidates
)


class EnhancedDynamicSecurityScanner:
    """
    Enhanced scanner that tracks top 100 securities for pattern detection
    while returning top 5 for active trading
    """
    
    def __init__(self):
        """Initialize the enhanced scanner service"""
        self.setup_environment()
        self.setup_logging()
        
        # Flask app
        self.app = Flask(__name__)
        
        # Database connection pool
        self.db_pool = ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            host=os.getenv('DATABASE_HOST', 'localhost'),
            port=int(os.getenv('DATABASE_PORT', '5432')),
            database=os.getenv('DATABASE_NAME', 'catalyst_trading'),
            user=os.getenv('DATABASE_USER'),
            password=os.getenv('DATABASE_PASSWORD')
        )
        
        # Redis client
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        
        # Service URLs
        self.news_service_url = os.getenv('NEWS_SERVICE_URL', 'http://news-service:5008')
        self.coordination_url = os.getenv('COORDINATION_URL', 'http://coordination-service:5000')
        
        # Enhanced scan parameters
        self.scan_params = {
            'initial_universe_size': int(os.getenv('INITIAL_UNIVERSE_SIZE', '200')),
            'top_tracking_size': int(os.getenv('TOP_TRACKING_SIZE', '100')),  # Track top 100
            'catalyst_filter_size': int(os.getenv('CATALYST_FILTER_SIZE', '50')),
            'final_selection_size': int(os.getenv('FINAL_SELECTION_SIZE', '5')),  # Trade top 5
            'min_price': float(os.getenv('MIN_PRICE', '1.0')),
            'max_price': float(os.getenv('MAX_PRICE', '500.0')),
            'min_volume': int(os.getenv('MIN_VOLUME', '500000')),
            'cache_ttl': int(os.getenv('SCANNER_CACHE_TTL', '300')),  # 5 minutes
            'concurrent_requests': int(os.getenv('SCANNER_CONCURRENT', '10'))
        }
        
        # Collection frequencies (in minutes)
        self.collection_frequencies = {
            'ultra_high': 1,    # Top 5 - every minute
            'high_freq': 15,    # Top 20 - every 15 minutes
            'medium_freq': 60,  # Top 50 - every hour
            'low_freq': 360,    # Top 100 - every 6 hours
            'archive': 1440     # Inactive - daily
        }
        
        # Tracking state cache
        self.tracking_state = {}  # symbol -> tracking info
        self._load_tracking_state()
        
        # Setup routes
        self.setup_routes()
        
        # Register with coordination
        self._register_with_coordination()
        
        self.logger.info("Enhanced Dynamic Security Scanner v3.0.0 initialized",
                        environment=os.getenv('ENVIRONMENT', 'development'),
                        tracking_size=self.scan_params['top_tracking_size'])
        
    def setup_environment(self):
        """Setup environment variables and paths"""
        self.log_path = os.getenv('LOG_PATH', '/app/logs')
        self.data_path = os.getenv('DATA_PATH', '/app/data')
        
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        
        self.service_name = 'enhanced_security_scanner'
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
                "service": "enhanced_security_scanner",
                "version": "3.0.0",
                "mode": "top-100-tracking",
                "tracking_count": len(self.tracking_state),
                "database": db_health['database'],
                "redis": db_health['redis'],
                "timestamp": datetime.now().isoformat()
            })
            
        @self.app.route('/scan', methods=['GET'])
        def scan():
            """Enhanced scan - tracks 100, returns 5"""
            mode = request.args.get('mode', 'normal')
            force_refresh = request.args.get('force', 'false').lower() == 'true'
            
            result = self.perform_enhanced_scan(mode, force_refresh)
            return jsonify(result)
            
        @self.app.route('/tracking_state', methods=['GET'])
        def tracking_state():
            """Get current tracking state for all securities"""
            return jsonify({
                'tracking_count': len(self.tracking_state),
                'securities': list(self.tracking_state.keys()),
                'frequency_breakdown': self._get_frequency_breakdown()
            })
            
        @self.app.route('/collect_data', methods=['POST'])
        def collect_data():
            """Manually trigger data collection for tracked securities"""
            self._collect_all_tracked_data()
            return jsonify({'status': 'collection_started'})
            
    def perform_enhanced_scan(self, mode: str = 'normal', 
                            force_refresh: bool = False) -> Dict:
        """
        Enhanced scan that tracks top 100 securities while returning top 5
        """
        start_time = datetime.now()
        
        self.logger.info("Starting enhanced scan", mode=mode)
        
        try:
            # Step 1: Get initial universe (200 stocks)
            universe = self._get_initial_universe()
            self.logger.info("Initial universe selected", count=len(universe))
            
            # Step 2: Enrich with market data and news scores
            enriched_universe = self._enrich_universe_data(universe)
            
            # Step 3: Score and rank all candidates
            scored_candidates = self._score_all_candidates(enriched_universe)
            
            # Step 4: Get top 100 for tracking
            top_100 = scored_candidates[:self.scan_params['top_tracking_size']]
            top_5 = scored_candidates[:self.scan_params['final_selection_size']]
            
            # Step 5: Update tracking state for all top 100
            self._update_tracking_states(top_100)
            
            # Step 6: Collect data for all tracked securities
            asyncio.create_task(self._async_collect_tracked_data())
            
            # Step 7: Age out inactive securities
            self._age_out_inactive_securities()
            
            # Step 8: Save scan results
            scan_id = f"SCAN_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{mode}"
            
            # Save top 5 as trading candidates (backward compatibility)
            if top_5:
                saved_count = insert_trading_candidates(top_5, scan_id)
                self.logger.info("Saved trading candidates", count=saved_count)
            
            # Store scan metrics
            self._store_scan_metrics(len(universe), len(top_100), len(top_5))
            
            # Return results (backward compatible - returns top 5 for trading)
            return {
                'scan_id': scan_id,
                'timestamp': datetime.now().isoformat(),
                'mode': mode,
                'securities': top_5,  # Top 5 for trading
                'metadata': {
                    'total_scanned': len(universe),
                    'total_tracked': len(top_100),
                    'total_selected': len(top_5),
                    'execution_time': (datetime.now() - start_time).total_seconds()
                }
            }
            
        except Exception as e:
            self.logger.error("Scan failed", error=str(e), traceback=traceback.format_exc())
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
            
    def _get_initial_universe(self) -> List[str]:
        """Get initial universe of stocks to scan"""
        universe = set()
        
        # 1. Get market movers
        movers = self._get_market_movers()
        universe.update(movers)
        
        # 2. Get news-mentioned securities
        news_symbols = self._get_news_mentioned_symbols()
        universe.update(news_symbols)
        
        # 3. Add high-volume stocks
        volume_leaders = self._get_volume_leaders()
        universe.update(volume_leaders)
        
        # 4. Include previous tracked securities still active
        active_tracked = [s for s, info in self.tracking_state.items() 
                         if info.get('collection_frequency') != 'archive']
        universe.update(active_tracked)
        
        # Limit to initial universe size
        universe_list = list(universe)[:self.scan_params['initial_universe_size']]
        
        return universe_list
        
    def _enrich_universe_data(self, symbols: List[str]) -> List[Dict]:
        """Enrich symbols with market data and news scores"""
        enriched = []
        
        with ThreadPoolExecutor(max_workers=self.scan_params['concurrent_requests']) as executor:
            future_to_symbol = {
                executor.submit(self._enrich_single_symbol, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data:
                        enriched.append(data)
                except Exception as e:
                    self.logger.warning(f"Failed to enrich {symbol}", error=str(e))
                    
        return enriched
        
    def _enrich_single_symbol(self, symbol: str) -> Optional[Dict]:
        """Enrich a single symbol with all data"""
        try:
            # Get market data
            market_data = self._get_symbol_market_data(symbol)
            if not market_data:
                return None
                
            # Get news data
            news_data = self._get_symbol_news_data(symbol)
            
            # Get technical indicators
            technical_data = self._calculate_technical_indicators(symbol, market_data)
            
            # Combine all data
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                **market_data,
                **news_data,
                **technical_data
            }
            
        except Exception as e:
            self.logger.warning(f"Error enriching {symbol}", error=str(e))
            return None
            
    def _score_all_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Score and rank all candidates"""
        for candidate in candidates:
            # Calculate composite score
            score = self._calculate_composite_score(candidate)
            candidate['composite_score'] = score
            candidate['score_components'] = {
                'catalyst_score': candidate.get('catalyst_score', 0),
                'volume_score': candidate.get('volume_score', 0),
                'technical_score': candidate.get('technical_score', 0),
                'momentum_score': candidate.get('momentum_score', 0)
            }
            
        # Sort by composite score
        candidates.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return candidates
        
    def _calculate_composite_score(self, data: Dict) -> float:
        """Calculate composite score for ranking"""
        # Catalyst score (0-40 points)
        catalyst_score = min(40, data.get('news_count', 0) * 10)
        
        # Volume score (0-30 points)
        rel_volume = data.get('relative_volume', 1.0)
        volume_score = min(30, rel_volume * 10)
        
        # Technical score (0-20 points)
        technical_score = 0
        if data.get('rsi'):
            if 30 < data['rsi'] < 70:
                technical_score += 10
        if data.get('above_vwap'):
            technical_score += 10
            
        # Momentum score (0-10 points)
        price_change = data.get('price_change_pct', 0)
        momentum_score = min(10, abs(price_change))
        
        # Store component scores
        data['catalyst_score'] = catalyst_score
        data['volume_score'] = volume_score
        data['technical_score'] = technical_score
        data['momentum_score'] = momentum_score
        
        return catalyst_score + volume_score + technical_score + momentum_score
        
    def _update_tracking_states(self, top_100: List[Dict]):
        """Update tracking state for top 100 securities"""
        current_time = datetime.now()
        
        # Mark all existing as candidates for aging
        for symbol in self.tracking_state:
            self.tracking_state[symbol]['last_in_top_100'] = False
            
        # Update top 100
        for i, candidate in enumerate(top_100):
            symbol = candidate['symbol']
            
            if symbol in self.tracking_state:
                # Update existing
                state = self.tracking_state[symbol]
                state['last_updated'] = current_time
                state['last_score'] = candidate['composite_score']
                state['last_in_top_100'] = True
                state['current_rank'] = i + 1
                
                # Update frequency based on rank
                if i < 5:
                    state['collection_frequency'] = 'ultra_high'
                elif i < 20:
                    state['collection_frequency'] = 'high_freq'
                elif i < 50:
                    state['collection_frequency'] = 'medium_freq'
                else:
                    state['collection_frequency'] = 'low_freq'
                    
            else:
                # New security
                self.tracking_state[symbol] = {
                    'symbol': symbol,
                    'first_seen': current_time,
                    'last_updated': current_time,
                    'collection_frequency': self._get_initial_frequency(i),
                    'data_points_collected': 0,
                    'last_score': candidate['composite_score'],
                    'current_rank': i + 1,
                    'last_in_top_100': True,
                    'catalyst_events': []
                }
                
        # Save tracking state to database
        self._save_tracking_state_to_db()
        
    def _get_initial_frequency(self, rank: int) -> str:
        """Get initial collection frequency based on rank"""
        if rank < 5:
            return 'ultra_high'
        elif rank < 20:
            return 'high_freq'
        elif rank < 50:
            return 'medium_freq'
        else:
            return 'low_freq'
            
    async def _async_collect_tracked_data(self):
        """Asynchronously collect data for all tracked securities"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._collect_all_tracked_data
            )
        except Exception as e:
            self.logger.error("Error in async data collection", error=str(e))
            
    def _collect_all_tracked_data(self):
        """Collect data for all tracked securities based on their frequencies"""
        current_time = datetime.now()
        
        for symbol, state in self.tracking_state.items():
            # Skip archived securities
            if state['collection_frequency'] == 'archive':
                continue
                
            # Check if it's time to collect
            last_updated = state.get('last_updated')
            if last_updated:
                time_since_update = (current_time - last_updated).total_seconds() / 60
                frequency_minutes = self.collection_frequencies[state['collection_frequency']]
                
                if time_since_update < frequency_minutes:
                    continue
                    
            # Collect data for this security
            self._collect_security_data(symbol, state)
            
    def _collect_security_data(self, symbol: str, state: Dict):
        """Collect comprehensive data for a security"""
        try:
            # Get all data
            data = self._enrich_single_symbol(symbol)
            if not data:
                return
                
            # Store in high-frequency table
            self._store_high_freq_data(symbol, data)
            
            # Update tracking state
            state['last_updated'] = datetime.now()
            state['data_points_collected'] += 1
            
            # Check if we should age this security
            self._check_aging_criteria(symbol, state, data)
            
        except Exception as e:
            self.logger.error(f"Error collecting data for {symbol}", error=str(e))
            
    def _store_high_freq_data(self, symbol: str, data: Dict):
        """Store data in high-frequency table"""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO security_data_high_freq (
                        symbol, timestamp, open, high, low, close, volume,
                        bid_ask_spread, order_imbalance,
                        rsi_14, macd, macd_signal, bb_upper, bb_lower, vwap,
                        news_count, catalyst_active, catalyst_score
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s
                    )
                """, (
                    symbol,
                    data['timestamp'],
                    data.get('open'),
                    data.get('high'),
                    data.get('low'),
                    data.get('close'),
                    data.get('volume'),
                    data.get('bid_ask_spread'),
                    data.get('order_imbalance'),
                    data.get('rsi'),
                    data.get('macd'),
                    data.get('macd_signal'),
                    data.get('bb_upper'),
                    data.get('bb_lower'),
                    data.get('vwap'),
                    data.get('news_count', 0),
                    data.get('catalyst_score', 0) > 20,
                    data.get('catalyst_score', 0)
                ))
                conn.commit()
        finally:
            self.db_pool.putconn(conn)
            
    def _check_aging_criteria(self, symbol: str, state: Dict, data: Dict):
        """Check if security should be aged to lower frequency"""
        # Criteria for demotion
        if state['collection_frequency'] == 'ultra_high':
            # Demote if no longer in top 5
            if state.get('current_rank', 100) > 5:
                state['collection_frequency'] = 'high_freq'
                
        elif state['collection_frequency'] == 'high_freq':
            # Demote if low activity
            if data.get('relative_volume', 1.0) < 0.8 and data.get('news_count', 0) < 2:
                state['collection_frequency'] = 'medium_freq'
                
        elif state['collection_frequency'] == 'medium_freq':
            # Demote if very low activity
            if data.get('relative_volume', 1.0) < 0.5 and data.get('news_count', 0) == 0:
                state['collection_frequency'] = 'low_freq'
                
    def _age_out_inactive_securities(self):
        """Move inactive securities to archive"""
        current_time = datetime.now()
        
        for symbol, state in self.tracking_state.items():
            # Skip if recently in top 100
            if state.get('last_in_top_100', False):
                continue
                
            # Check time since last in top 100
            last_updated = state.get('last_updated')
            if last_updated:
                hours_inactive = (current_time - last_updated).total_seconds() / 3600
                
                # Archive after 48 hours of inactivity
                if hours_inactive > 48:
                    state['collection_frequency'] = 'archive'
                    self.logger.info(f"Archived {symbol} after {hours_inactive:.1f} hours inactive")
                    
    def _save_tracking_state_to_db(self):
        """Save tracking state to database"""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cursor:
                for symbol, state in self.tracking_state.items():
                    cursor.execute("""
                        INSERT INTO security_tracking_state (
                            symbol, first_seen, last_updated,
                            collection_frequency, data_points_collected,
                            last_price, last_volume, flatline_periods,
                            last_catalyst_score, catalyst_events,
                            hours_since_catalyst, activity_score,
                            metadata
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (symbol) DO UPDATE SET
                            last_updated = EXCLUDED.last_updated,
                            collection_frequency = EXCLUDED.collection_frequency,
                            data_points_collected = EXCLUDED.data_points_collected,
                            last_catalyst_score = EXCLUDED.last_catalyst_score,
                            metadata = EXCLUDED.metadata
                    """, (
                        symbol,
                        state['first_seen'],
                        state['last_updated'],
                        state['collection_frequency'],
                        state['data_points_collected'],
                        None,  # last_price (update from market data)
                        None,  # last_volume
                        0,     # flatline_periods
                        state.get('last_score', 0),
                        json.dumps(state.get('catalyst_events', [])),
                        None,  # hours_since_catalyst
                        None,  # activity_score
                        json.dumps({'rank': state.get('current_rank')})
                    ))
                conn.commit()
        finally:
            self.db_pool.putconn(conn)
            
    def _load_tracking_state(self):
        """Load tracking state from database on startup"""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM security_tracking_state
                    WHERE collection_frequency != 'archive'
                """)
                
                for row in cursor.fetchall():
                    self.tracking_state[row['symbol']] = {
                        'symbol': row['symbol'],
                        'first_seen': row['first_seen'],
                        'last_updated': row['last_updated'],
                        'collection_frequency': row['collection_frequency'],
                        'data_points_collected': row['data_points_collected'],
                        'last_score': row['last_catalyst_score'],
                        'catalyst_events': json.loads(row['catalyst_events'] or '[]')
                    }
                    
                self.logger.info(f"Loaded tracking state for {len(self.tracking_state)} securities")
        finally:
            self.db_pool.putconn(conn)
            
    def _store_scan_metrics(self, total_scanned: int, total_tracked: int, total_selected: int):
        """Store scan metrics in performance table"""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO tracking_performance_metrics (
                        date, total_securities_tracked,
                        active_high_freq, active_medium_freq, archived_count,
                        total_data_points, storage_gb_used
                    ) VALUES (
                        CURRENT_DATE, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (date) DO UPDATE SET
                        total_securities_tracked = EXCLUDED.total_securities_tracked
                """, (
                    total_tracked,
                    sum(1 for s in self.tracking_state.values() 
                        if s['collection_frequency'] in ['ultra_high', 'high_freq']),
                    sum(1 for s in self.tracking_state.values() 
                        if s['collection_frequency'] == 'medium_freq'),
                    sum(1 for s in self.tracking_state.values() 
                        if s['collection_frequency'] == 'archive'),
                    sum(s['data_points_collected'] for s in self.tracking_state.values()),
                    0  # storage_gb_used (calculate separately)
                ))
                conn.commit()
        finally:
            self.db_pool.putconn(conn)
            
    def _get_frequency_breakdown(self) -> Dict[str, int]:
        """Get breakdown of securities by collection frequency"""
        breakdown = {
            'ultra_high': 0,
            'high_freq': 0,
            'medium_freq': 0,
            'low_freq': 0,
            'archive': 0
        }
        
        for state in self.tracking_state.values():
            freq = state.get('collection_frequency', 'low_freq')
            if freq in breakdown:
                breakdown[freq] += 1
                
        return breakdown
        
    # Helper methods (abbreviated for space - include all from original)
    def _get_market_movers(self) -> List[str]:
        """Get market movers from various sources"""
        # Implementation from original scanner_service.py
        pass
        
    def _get_news_mentioned_symbols(self) -> List[str]:
        """Get symbols mentioned in recent news"""
        try:
            response = requests.get(
                f"{self.news_service_url}/trending",
                params={'hours': 24},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return [item['symbol'] for item in data.get('trending', [])]
        except Exception as e:
            self.logger.warning("Could not get news symbols", error=str(e))
        return []
        
    def _get_volume_leaders(self) -> List[str]:
        """Get high volume stocks"""
        # Implementation similar to original
        pass
        
    def _get_symbol_market_data(self, symbol: str) -> Optional[Dict]:
        """Get market data for symbol"""
        # Implementation from original
        pass
        
    def _get_symbol_news_data(self, symbol: str) -> Dict:
        """Get news data for symbol"""
        try:
            response = requests.get(
                f"{self.news_service_url}/news/{symbol}",
                params={'hours': 24},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    'news_count': data.get('count', 0),
                    'news_sentiment': data.get('sentiment'),
                    'catalyst_keywords': data.get('catalyst_keywords', [])
                }
        except Exception as e:
            self.logger.warning(f"Could not get news for {symbol}", error=str(e))
            
        return {'news_count': 0}
        
    def _calculate_technical_indicators(self, symbol: str, market_data: Dict) -> Dict:
        """Calculate technical indicators"""
        # Implementation from original
        pass
        
    def _register_with_coordination(self):
        """Register with coordination service"""
        try:
            response = requests.post(
                f"{self.coordination_url}/register_service",
                json={
                    'service_name': 'enhanced_security_scanner',
                    'service_info': {
                        'url': f"http://scanner-service:{self.port}",
                        'port': self.port,
                        'version': '3.0.0',
                        'capabilities': [
                            'dynamic_scanning', 
                            'catalyst_filtering', 
                            'top_100_tracking',
                            'pattern_data_collection'
                        ]
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
        self.logger.info("Starting Enhanced Dynamic Security Scanner",
                        version="3.0.0",
                        port=self.port,
                        tracking_size=self.scan_params['top_tracking_size'],
                        environment=os.getenv('ENVIRONMENT', 'development'))
        
        self.app.run(host='0.0.0.0', port=self.port, debug=False)


if __name__ == "__main__":
    service = EnhancedDynamicSecurityScanner()
    service.run()