#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: scanner_service.py
Version: 3.1.0
Last Updated: 2025-07-10
Purpose: Enhanced scanner with comprehensive market data collection

REVISION HISTORY:
v3.1.0 (2025-07-10) - Enhanced market data collection
- Now stores ALL tracked securities to market_data table (not just top 20)
- Added comprehensive market data fields (RSI, SMA, VWAP, etc.)
- Improved technical indicator calculations
- Added daily aggregation updates
- Better error handling for data storage

v3.0.2 (2025-01-27) - Fixed compatibility issues
- Removed dependency on missing database_utils functions
- Direct database operations
- Maintains all v3.0.0 functionality

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
import structlog

# Technical analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("WARNING: TA-Lib not available, using fallback calculations")

# Only import what exists in database_utils
from database_utils import (
    get_db_connection, health_check,
    get_active_candidates
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
            'top_tracking_size': int(os.getenv('TOP_TRACKING_SIZE', '100')),
            'catalyst_filter_size': int(os.getenv('CATALYST_FILTER_SIZE', '50')),
            'final_selection_size': int(os.getenv('FINAL_SELECTION_SIZE', '5')),
            'min_price': float(os.getenv('MIN_PRICE', '1.0')),
            'max_price': float(os.getenv('MAX_PRICE', '500.0')),
            'min_volume': int(os.getenv('MIN_VOLUME', '500000')),
            'cache_ttl': int(os.getenv('SCANNER_CACHE_TTL', '300')),
            'concurrent_requests': int(os.getenv('SCANNER_CONCURRENT', '10'))
        }
        
        # Collection frequencies (in minutes)
        self.collection_frequencies = {
            'ultra_high': 1,
            'high_freq': 15,
            'medium_freq': 60,
            'low_freq': 360,
            'archive': 1440
        }
        
        # Tracking state cache
        self.tracking_state = {}
        self._load_tracking_state()
        
        # Setup routes
        self.setup_routes()
        
        # Register with coordination
        self._register_with_coordination()
        
        self.logger.info("Enhanced Dynamic Security Scanner v3.1.0 initialized",
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
        self.logger = structlog.get_logger()
        self.logger = self.logger.bind(service=self.service_name)
        
    def setup_routes(self):
        """Setup Flask routes"""
        @self.app.route('/health', methods=['GET'])
        def health():
            db_health = health_check()
            return jsonify({
                "status": "healthy" if db_health['postgresql']['status'] == 'healthy' else "degraded",
                "service": "enhanced_security_scanner",
                "version": "3.1.0",
                "mode": "top-100-tracking",
                "tracking_count": len(self.tracking_state),
                "database": db_health['postgresql']['status'] ,
                "redis": db_health['redis']['status'],,
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
            
    def perform_enhanced_scan(self, mode: str = 'normal', 
                            force_refresh: bool = False) -> Dict:
        """
        Enhanced scan that tracks top 100 securities while returning top 5
        """
        start_time = datetime.now()
        
        self.logger.info("Starting enhanced scan", mode=mode)
        
        try:
            # Generate scan ID
            scan_id = f"SCAN_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{mode}"
            
            # Step 1: Get market movers
            universe = self._get_market_movers()
            self.logger.info("Initial universe selected", count=len(universe))
            
            # Step 2: Get enriched data with technical indicators
            enriched_data = []
            with ThreadPoolExecutor(max_workers=self.scan_params['concurrent_requests']) as executor:
                futures = {executor.submit(self._get_enriched_symbol_data, symbol): symbol 
                          for symbol in universe[:self.scan_params['initial_universe_size']]}
                
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        data = future.result()
                        if data:
                            enriched_data.append(data)
                    except Exception as e:
                        self.logger.warning(f"Error getting data for {symbol}", error=str(e))
                    
            # Step 3: Score candidates with enhanced scoring
            for candidate in enriched_data:
                score = self._calculate_comprehensive_score(candidate)
                candidate['composite_score'] = score
                
            # Sort by score
            enriched_data.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
            
            # Step 4: Get top 100 and top 5
            top_100 = enriched_data[:self.scan_params['top_tracking_size']]
            top_5 = enriched_data[:self.scan_params['final_selection_size']]
            
            # Step 5: Update tracking state
            for i, candidate in enumerate(top_100):
                symbol = candidate['symbol']
                self.tracking_state[symbol] = {
                    'symbol': symbol,
                    'last_updated': datetime.now(),
                    'collection_frequency': self._get_frequency_by_rank(i),
                    'last_score': candidate.get('composite_score', 0),
                    'rank': i + 1
                }
                
            # Step 6: Store comprehensive market data for ALL tracked securities
            self._store_comprehensive_scan_data(top_100, scan_id)
            
            # Step 7: Update daily aggregates
            self._update_daily_aggregates()
            
            # Save top 5 as trading candidates
            if top_5:
                saved_count = insert_trading_candidates(top_5, scan_id)
                self.logger.info("Saved trading candidates", count=saved_count)
                
            # Return results (backward compatible)
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
            
    def _get_market_movers(self) -> List[str]:
        """Get market movers - simplified version"""
        # Start with default universe
        movers = list(self.default_universe)
        
        # Try to get trending from news service
        try:
            response = requests.get(
                f"{self.news_service_url}/trending",
                params={'hours': 24},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                news_symbols = [item['symbol'] for item in data.get('trending', [])]
                movers = news_symbols + movers
        except Exception as e:
            self.logger.warning("Could not get news trending", error=str(e))
            
        # Remove duplicates
        return list(dict.fromkeys(movers))
        
    def _get_enriched_symbol_data(self, symbol: str) -> Optional[Dict]:
        """Get enriched symbol data with technical indicators"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            
            # Get price history for technical indicators
            hist = ticker.history(period="1mo", interval="1d")
            if hist.empty:
                return None
                
            # Get current data
            current_price = info.get('currentPrice', info.get('regularMarketPrice', hist['Close'].iloc[-1]))
            current_volume = info.get('volume', info.get('regularMarketVolume', hist['Volume'].iloc[-1]))
            
            # Calculate technical indicators
            close_prices = hist['Close'].values
            high_prices = hist['High'].values
            low_prices = hist['Low'].values
            volumes = hist['Volume'].values
            
            # Calculate indicators
            technical_data = self._calculate_technical_indicators(
                close_prices, high_prices, low_prices, volumes
            )
            
            # Build comprehensive data
            data = {
                'symbol': symbol,
                'scan_timestamp': datetime.now(),
                'price': float(current_price),
                'open_price': float(hist['Open'].iloc[-1]),
                'high_price': float(hist['High'].iloc[-1]),
                'low_price': float(hist['Low'].iloc[-1]),
                'previous_close': float(hist['Close'].iloc[-2]) if len(hist) > 1 else float(current_price),
                'volume': int(current_volume),
                'average_volume': int(volumes[-20:].mean()) if len(volumes) >= 20 else int(current_volume),
                'market_cap': info.get('marketCap', 0),
                'news_count': 0,
                'has_news': False,
                **technical_data
            }
            
            # Calculate derived metrics
            data['price_change'] = data['price'] - data['previous_close']
            data['price_change_pct'] = (data['price_change'] / data['previous_close'] * 100) if data['previous_close'] > 0 else 0
            data['gap_pct'] = ((data['open_price'] - data['previous_close']) / data['previous_close'] * 100) if data['previous_close'] > 0 else 0
            data['relative_volume'] = (data['volume'] / data['average_volume']) if data['average_volume'] > 0 else 1
            data['dollar_volume'] = data['price'] * data['volume']
            data['day_range_pct'] = ((data['high_price'] - data['low_price']) / data['low_price'] * 100) if data['low_price'] > 0 else 0
            
            # Get news data
            try:
                response = requests.get(
                    f"{self.news_service_url}/news/{symbol}",
                    params={'hours': 24},
                    timeout=3
                )
                if response.status_code == 200:
                    news_data = response.json()
                    data['news_count'] = news_data.get('count', 0)
                    data['has_news'] = data['news_count'] > 0
                    data['primary_catalyst'] = news_data.get('primary_catalyst', '')
                    data['news_recency_hours'] = news_data.get('most_recent_hours', 24)
            except:
                pass
                
            return data
            
        except Exception as e:
            self.logger.debug(f"Error getting enriched data for {symbol}", error=str(e))
            return None
            
    def _calculate_technical_indicators(self, close_prices, high_prices, low_prices, volumes) -> Dict:
        """Calculate technical indicators using TA-Lib or fallback methods"""
        indicators = {}
        
        try:
            if TALIB_AVAILABLE and len(close_prices) >= 20:
                # RSI
                rsi = talib.RSI(close_prices, timeperiod=14)
                indicators['rsi_14'] = float(rsi[-1]) if not np.isnan(rsi[-1]) else None
                
                # Moving averages
                sma_20 = talib.SMA(close_prices, timeperiod=20)
                indicators['sma_20'] = float(sma_20[-1]) if not np.isnan(sma_20[-1]) else None
                
                if len(close_prices) >= 50:
                    sma_50 = talib.SMA(close_prices, timeperiod=50)
                    indicators['sma_50'] = float(sma_50[-1]) if not np.isnan(sma_50[-1]) else None
                else:
                    indicators['sma_50'] = None
                    
                # VWAP approximation (simplified)
                typical_price = (high_prices + low_prices + close_prices) / 3
                indicators['vwap'] = float(np.sum(typical_price[-20:] * volumes[-20:]) / np.sum(volumes[-20:]))
                
            else:
                # Fallback calculations
                if len(close_prices) >= 14:
                    # Simple RSI calculation
                    deltas = np.diff(close_prices[-15:])
                    gains = deltas[deltas > 0].sum() / 14
                    losses = -deltas[deltas < 0].sum() / 14
                    rs = gains / losses if losses > 0 else 100
                    indicators['rsi_14'] = float(100 - (100 / (1 + rs)))
                else:
                    indicators['rsi_14'] = None
                    
                # Simple moving averages
                indicators['sma_20'] = float(np.mean(close_prices[-20:])) if len(close_prices) >= 20 else None
                indicators['sma_50'] = float(np.mean(close_prices[-50:])) if len(close_prices) >= 50 else None
                
                # VWAP approximation
                if len(close_prices) >= 20:
                    typical_price = (high_prices[-20:] + low_prices[-20:] + close_prices[-20:]) / 3
                    indicators['vwap'] = float(np.sum(typical_price * volumes[-20:]) / np.sum(volumes[-20:]))
                else:
                    indicators['vwap'] = None
                    
        except Exception as e:
            self.logger.debug("Error calculating indicators", error=str(e))
            indicators = {'rsi_14': None, 'sma_20': None, 'sma_50': None, 'vwap': None}
            
        return indicators
        
    def _calculate_comprehensive_score(self, data: Dict) -> float:
        """Calculate comprehensive score with multiple factors"""
        score = 0
        
        # News catalyst score (0-40 points)
        news_score = 0
        if data.get('has_news', False):
            news_score += 20
            news_count = data.get('news_count', 0)
            news_score += min(15, news_count * 3)
            # Recency bonus
            recency = data.get('news_recency_hours', 24)
            if recency < 1:
                news_score += 5
            elif recency < 4:
                news_score += 3
        score += news_score
        
        # Volume score (0-30 points)
        volume_score = 0
        rel_volume = data.get('relative_volume', 1)
        if rel_volume > 3:
            volume_score += 30
        elif rel_volume > 2:
            volume_score += 20
        elif rel_volume > 1.5:
            volume_score += 10
        score += volume_score
        
        # Price action score (0-20 points)
        price_score = 0
        price_change_pct = data.get('price_change_pct', 0)
        if abs(price_change_pct) > 5:
            price_score += 20
        elif abs(price_change_pct) > 3:
            price_score += 15
        elif abs(price_change_pct) > 1:
            price_score += 10
        score += price_score
        
        # Technical score (0-10 points)
        tech_score = 0
        rsi = data.get('rsi_14')
        if rsi and (rsi > 70 or rsi < 30):
            tech_score += 5
        if data.get('price') and data.get('sma_20'):
            if data['price'] > data['sma_20']:
                tech_score += 5
        score += tech_score
        
        # Market cap adjustment
        market_cap = data.get('market_cap', 0)
        if market_cap > 10000000000:  # 10B+
            score *= 1.1
        elif market_cap < 100000000:  # <100M
            score *= 0.8
            
        return round(score, 2)
        
    def _get_frequency_by_rank(self, rank: int) -> str:
        """Get collection frequency by rank"""
        if rank < 5:
            return 'ultra_high'
        elif rank < 20:
            return 'high_freq'
        elif rank < 50:
            return 'medium_freq'
        else:
            return 'low_freq'
            
    def _store_comprehensive_scan_data(self, securities: List[Dict], scan_id: str):
        """Store comprehensive scan data for ALL tracked securities"""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Prepare batch insert data
                insert_data = []
                
                for i, security in enumerate(securities):
                    try:
                        # Determine catalyst score
                        catalyst_score = security.get('composite_score', 0) * (security.get('news_count', 0) / 10)
                        
                        insert_data.append((
                            scan_id,
                            security['symbol'],
                            security.get('scan_timestamp', datetime.now()),
                            security.get('price'),
                            security.get('open_price'),
                            security.get('high_price'),
                            security.get('low_price'),
                            security.get('previous_close'),
                            security.get('volume'),
                            security.get('average_volume'),
                            security.get('relative_volume'),
                            security.get('dollar_volume'),
                            security.get('price_change'),
                            security.get('price_change_pct'),
                            security.get('gap_pct'),
                            security.get('day_range_pct'),
                            security.get('rsi_14'),
                            security.get('sma_20'),
                            security.get('sma_50'),
                            security.get('vwap'),
                            security.get('has_news', False),
                            security.get('news_count', 0),
                            catalyst_score,
                            security.get('primary_catalyst'),
                            security.get('news_recency_hours'),
                            i + 1,  # scan_rank
                            i < 20,  # made_top_20
                            i < 5,   # made_top_5
                            i < 5,   # selected_for_trading
                            security.get('market_cap'),
                            security.get('sector'),
                            security.get('industry')
                        ))
                        
                    except Exception as e:
                        self.logger.error(f"Error preparing data for {security.get('symbol', 'UNKNOWN')}", 
                                        error=str(e))
                        continue
                
                # Batch insert to scan_market_data table
                if insert_data:
                    execute_batch(cursor, """
                        INSERT INTO scan_market_data (
                            scan_id, symbol, scan_timestamp,
                            price, open_price, high_price, low_price, previous_close,
                            volume, average_volume, relative_volume, dollar_volume,
                            price_change, price_change_pct, gap_pct, day_range_pct,
                            rsi_14, sma_20, sma_50, vwap,
                            has_news, news_count, catalyst_score, primary_catalyst, news_recency_hours,
                            scan_rank, made_top_20, made_top_5, selected_for_trading,
                            market_cap, sector, industry
                        ) VALUES %s
                        ON CONFLICT (scan_id, symbol) DO NOTHING
                    """, insert_data)
                    
                    self.logger.info(f"Stored scan market data for {len(insert_data)} securities")
                
                # Also store high-frequency data for top securities
                for security in securities[:20]:
                    try:
                        cursor.execute("""
                            INSERT INTO security_data_high_freq (
                                symbol, timestamp, close, volume,
                                news_count, catalyst_active, catalyst_score
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s
                            )
                        """, (
                            security['symbol'],
                            security.get('scan_timestamp', datetime.now()),
                            security.get('price'),
                            security.get('volume'),
                            security.get('news_count', 0),
                            security.get('has_news', False),
                            catalyst_score
                        ))
                    except Exception as e:
                        self.logger.debug(f"Error storing high-freq data for {security['symbol']}", 
                                        error=str(e))
                        
                conn.commit()
                
        except Exception as e:
            self.logger.error("Error storing scan data", error=str(e))
            conn.rollback()
        finally:
            self.db_pool.putconn(conn)
            
    def _update_daily_aggregates(self):
        """Update daily aggregate data"""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Call the aggregation function
                cursor.execute("SELECT update_scan_market_data_daily()")
                conn.commit()
                self.logger.info("Updated daily aggregates")
                
        except Exception as e:
            self.logger.error("Error updating daily aggregates", error=str(e))
            conn.rollback()
        finally:
            self.db_pool.putconn(conn)
            
    def _load_tracking_state(self):
        """Load tracking state from database"""
        try:
            conn = self.db_pool.getconn()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM security_tracking_state
                    WHERE collection_frequency != 'archive'
                    LIMIT 200
                """)
                
                for row in cursor.fetchall():
                    self.tracking_state[row['symbol']] = {
                        'symbol': row['symbol'],
                        'last_updated': row['last_updated'],
                        'collection_frequency': row['collection_frequency']
                    }
                    
        except Exception as e:
            # Table might not exist yet
            self.logger.info("Could not load tracking state", error=str(e))
        finally:
            if 'conn' in locals():
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
                        'version': '3.1.0',
                        'capabilities': ['top_100_tracking', 'pattern_data_collection', 'comprehensive_market_data']
                    }
                },
                timeout=5
            )
            if response.status_code == 200:
                self.logger.info("Successfully registered with coordination service")
        except Exception as e:
            self.logger.warning(f"Could not register with coordination", error=str(e))
            
    # Default universe
    default_universe = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA',
        'JNJ', 'PFE', 'UNH', 'CVS', 'ABBV', 'MRK', 'LLY',
        'XOM', 'CVX', 'COP', 'SLB', 'OXY',
        'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'DIS',
        'SPY', 'QQQ', 'IWM', 'DIA', 'VXX'
    ]
            
    def run(self):
        """Start the scanner service"""
        self.logger.info("Starting Enhanced Dynamic Security Scanner",
                        version="3.1.0",
                        port=self.port,
                        environment=os.getenv('ENVIRONMENT', 'development'))
        
        self.app.run(host='0.0.0.0', port=self.port, debug=False)


if __name__ == "__main__":
    service = EnhancedDynamicSecurityScanner()
    service.run()