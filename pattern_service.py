#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: pattern_service.py
Version: 2.1.2
Last Updated: 2025-07-07
Purpose: Technical pattern detection and analysis with news catalyst weighting

REVISION HISTORY:
v2.1.2 (2025-07-07) - Fixed health check compatibility with database_utils v2.3.1
- Updated health check to handle new postgresql/redis status format
- Maintains backward compatibility with older database_utils versions

v2.1.1 (2025-07-07) - Fixed missing _register_with_coordination method
- Added service registration method
- Fixed initialization sequence

v2.1.0 (2025-07-01) - Production-ready refactor
- Migrated from SQLite to PostgreSQL
- All configuration via environment variables
- Proper database connection pooling
- Enhanced error handling and retry logic
- Added source tier classification

v2.0.0 (2025-06-27) - Complete rewrite for catalyst-aware patterns
- News context integration
- Pre-market emphasis
- Pattern confidence weighting by catalyst
- Machine learning readiness

Description of Service:
This service analyzes technical patterns in stock charts with special
awareness of news catalysts.

KEY INNOVATION:
- Bullish patterns + positive catalyst = 50% confidence boost
- Bearish patterns + negative catalyst = 50% confidence boost  
- Misaligned patterns (bullish + bad news) = 30% confidence reduction
- Pre-market patterns with catalysts = Double weight

PATTERN TYPES DETECTED:
- Reversal: Hammer, Shooting Star, Engulfing, Doji
- Continuation: Three White Soldiers, Three Black Crows
- Momentum: Gap patterns, Volume surges
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from structlog import get_logger
import redis

# Import database utilities
from database_utils import (
    get_db_connection,
    insert_pattern_detection,
    get_redis,
    health_check
)

# Handle technical analysis library imports
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib not available, using manual calculations")

# Handle yfinance import
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance not available, using mock data")


class CatalystAwarePatternAnalysis:
    """
    Pattern analysis that understands news context
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
        
        # Pattern configuration with catalyst weights
        self.pattern_config = self._load_pattern_config()
        
        # Pre-market multiplier
        self.premarket_multiplier = float(os.getenv('PREMARKET_MULTIPLIER', '2.0'))
        
        # Cache settings
        self.cache_ttl = int(os.getenv('PATTERN_CACHE_TTL', '300'))  # 5 minutes
        
        # Register with coordination
        self._register_with_coordination()
        
        self.logger.info("Catalyst-Aware Pattern Analysis v2.1.2 initialized",
                        environment=os.getenv('ENVIRONMENT', 'development'),
                        talib_available=TALIB_AVAILABLE)
        
    def setup_environment(self):
        """Setup environment variables and paths"""
        # Paths
        self.log_path = os.getenv('LOG_PATH', '/app/logs')
        self.data_path = os.getenv('DATA_PATH', '/app/data')
        
        # Create directories
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        
        # Service configuration
        self.service_name = 'pattern_analysis'
        self.port = int(os.getenv('PORT', '5002'))
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # Pattern detection parameters
        self.min_confidence = float(os.getenv('MIN_PATTERN_CONFIDENCE', '60'))
        self.lookback_periods = int(os.getenv('PATTERN_LOOKBACK_PERIODS', '20'))
        
    def setup_logging(self):
        """Setup structured logging"""
        self.logger = get_logger()
        self.logger = self.logger.bind(service=self.service_name)
        
    def _register_with_coordination(self):
        """Register with coordination service"""
        try:
            response = requests.post(
                f"{self.coordination_url}/register_service",
                json={
                    'service_name': 'pattern_analysis',
                    'service_info': {
                        'url': f"http://pattern-service:{self.port}",
                        'port': self.port,
                        'version': '2.1.2',
                        'capabilities': ['pattern_detection', 'catalyst_weighting', 'technical_patterns']
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
            # Don't fail initialization if coordination is not available yet
            
    def _load_pattern_config(self) -> Dict:
        """Load pattern configuration with environment overrides"""
        return {
            'reversal_patterns': {
                'hammer': {
                    'base_confidence': float(os.getenv('HAMMER_BASE_CONFIDENCE', '65')),
                    'catalyst_boost': {
                        'positive': float(os.getenv('HAMMER_POSITIVE_BOOST', '1.5')),
                        'negative': float(os.getenv('HAMMER_NEGATIVE_BOOST', '0.7')),
                        'neutral': 1.0
                    },
                    'min_shadow_ratio': float(os.getenv('HAMMER_MIN_SHADOW', '2.0'))
                },
                'shooting_star': {
                    'base_confidence': float(os.getenv('SHOOTING_STAR_BASE_CONFIDENCE', '65')),
                    'catalyst_boost': {
                        'positive': float(os.getenv('SHOOTING_STAR_POSITIVE_BOOST', '0.7')),
                        'negative': float(os.getenv('SHOOTING_STAR_NEGATIVE_BOOST', '1.5')),
                        'neutral': 1.0
                    },
                    'min_shadow_ratio': float(os.getenv('SHOOTING_STAR_MIN_SHADOW', '2.0'))
                },
                'engulfing': {
                    'base_confidence': float(os.getenv('ENGULFING_BASE_CONFIDENCE', '70')),
                    'catalyst_boost': {
                        'positive': float(os.getenv('ENGULFING_POSITIVE_BOOST', '1.4')),
                        'negative': float(os.getenv('ENGULFING_NEGATIVE_BOOST', '1.4')),
                        'neutral': 1.0
                    }
                },
                'doji': {
                    'base_confidence': float(os.getenv('DOJI_BASE_CONFIDENCE', '55')),
                    'catalyst_boost': {
                        'positive': float(os.getenv('DOJI_POSITIVE_BOOST', '1.3')),
                        'negative': float(os.getenv('DOJI_NEGATIVE_BOOST', '1.3')),
                        'neutral': 1.0
                    },
                    'max_body_ratio': float(os.getenv('DOJI_MAX_BODY', '0.1'))
                }
            },
            'continuation_patterns': {
                'three_white_soldiers': {
                    'base_confidence': float(os.getenv('THREE_WHITE_BASE_CONFIDENCE', '75')),
                    'catalyst_boost': {
                        'positive': float(os.getenv('THREE_WHITE_POSITIVE_BOOST', '1.6')),
                        'negative': float(os.getenv('THREE_WHITE_NEGATIVE_BOOST', '0.5')),
                        'neutral': 1.0
                    }
                },
                'three_black_crows': {
                    'base_confidence': float(os.getenv('THREE_BLACK_BASE_CONFIDENCE', '75')),
                    'catalyst_boost': {
                        'positive': float(os.getenv('THREE_BLACK_POSITIVE_BOOST', '0.5')),
                        'negative': float(os.getenv('THREE_BLACK_NEGATIVE_BOOST', '1.6')),
                        'neutral': 1.0
                    }
                }
            },
            'momentum_patterns': {
                'gap_up': {
                    'base_confidence': float(os.getenv('GAP_UP_BASE_CONFIDENCE', '60')),
                    'catalyst_boost': {
                        'positive': float(os.getenv('GAP_UP_POSITIVE_BOOST', '1.7')),
                        'negative': float(os.getenv('GAP_UP_NEGATIVE_BOOST', '0.4')),
                        'neutral': 1.0
                    },
                    'min_gap_percent': float(os.getenv('GAP_UP_MIN_PERCENT', '1.0'))
                },
                'gap_down': {
                    'base_confidence': float(os.getenv('GAP_DOWN_BASE_CONFIDENCE', '60')),
                    'catalyst_boost': {
                        'positive': float(os.getenv('GAP_DOWN_POSITIVE_BOOST', '0.4')),
                        'negative': float(os.getenv('GAP_DOWN_NEGATIVE_BOOST', '1.7')),
                        'neutral': 1.0
                    },
                    'min_gap_percent': float(os.getenv('GAP_DOWN_MIN_PERCENT', '1.0'))
                },
                'volume_surge': {
                    'base_confidence': float(os.getenv('VOLUME_SURGE_BASE_CONFIDENCE', '50')),
                    'catalyst_boost': {
                        'positive': float(os.getenv('VOLUME_SURGE_POSITIVE_BOOST', '1.5')),
                        'negative': float(os.getenv('VOLUME_SURGE_NEGATIVE_BOOST', '1.5')),
                        'neutral': 1.0
                    },
                    'min_volume_ratio': float(os.getenv('VOLUME_SURGE_MIN_RATIO', '2.0'))
                }
            }
        }
        
    def setup_routes(self):
        """Setup Flask routes"""
        @self.app.route('/health', methods=['GET'])
        def healthcheck():
            """Health check endpoint"""
            db_health = health_check()
            
            # Handle different database_utils versions
            db_status = False
            redis_status = False
            
            if 'postgresql' in db_health:
                # v2.3.1 format
                db_status = db_health['postgresql'].get('status') == 'healthy'
                redis_status = db_health['redis'].get('status') == 'healthy'
            elif 'database' in db_health:
                # older format
                db_status = db_health['database']
                redis_status = db_health['redis']
            
            return jsonify({
                "status": "healthy" if db_status else "degraded",
                "service": self.service_name,
                "version": "2.1.2",
                "database": db_status,
                "redis": redis_status,
                "talib": TALIB_AVAILABLE,
                "timestamp": datetime.now().isoformat()
            })
            
        @self.app.route('/analyze_pattern', methods=['POST'])
        def analyze_pattern():
            """Analyze single symbol with catalyst context"""
            data = request.json
            symbol = data.get('symbol')
            timeframe = data.get('timeframe', '5min')
            context = data.get('context', {})
            
            if not symbol:
                return jsonify({'error': 'Symbol required'}), 400
                
            result = self.analyze_with_catalyst_context(symbol, timeframe, context)
            return jsonify(result)
            
        @self.app.route('/batch_analyze', methods=['POST'])
        def batch_analyze():
            """Analyze multiple symbols"""
            data = request.json
            symbols = data.get('symbols', [])
            timeframe = data.get('timeframe', '5min')
            context = data.get('context', {})
            
            results = []
            for symbol in symbols:
                try:
                    result = self.analyze_with_catalyst_context(symbol, timeframe, context)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error analyzing {symbol}", error=str(e))
                    
            return jsonify({'results': results})
            
        @self.app.route('/pattern_statistics', methods=['GET'])
        def pattern_statistics():
            """Get pattern success statistics"""
            pattern = request.args.get('pattern')
            catalyst = request.args.get('catalyst')
            hours = request.args.get('hours', 168, type=int)  # Default 1 week
            
            stats = self._get_pattern_statistics(pattern, catalyst, hours)
            return jsonify(stats)
            
        @self.app.route('/update_pattern_outcome', methods=['POST'])
        def update_pattern_outcome():
            """Update pattern with actual outcome for ML"""
            data = request.json
            pattern_id = data.get('pattern_id')
            outcome = data.get('outcome')
            
            if not pattern_id or not outcome:
                return jsonify({'error': 'pattern_id and outcome required'}), 400
                
            result = self._update_pattern_outcome(pattern_id, outcome)
            return jsonify(result)

    def analyze_with_catalyst_context(self, symbol: str, timeframe: str = '5min', 
                                    context: Dict = None) -> Dict:
        """
        Analyze patterns with catalyst awareness
        """
        self.logger.info(f"Analyzing {symbol} with catalyst context",
                        symbol=symbol,
                        timeframe=timeframe,
                        has_context=bool(context))
        
        # Check cache first
        cache_key = f"pattern:{symbol}:{timeframe}"
        cached = self.redis_client.get(cache_key)
        if cached and not context:  # Don't use cache if specific context provided
            self.logger.debug("Using cached pattern data", symbol=symbol)
            return json.loads(cached)
        
        # Get price data
        price_data = self._get_price_data(symbol, timeframe)
        if price_data is None or len(price_data) < self.lookback_periods:
            return {
                'symbol': symbol,
                'status': 'insufficient_data',
                'patterns': []
            }
            
        # Extract catalyst information
        catalyst_info = self._extract_catalyst_info(context)
        
        # Detect patterns
        detected_patterns = []
        
        # Check reversal patterns
        for pattern_name, config in self.pattern_config['reversal_patterns'].items():
            pattern = self._detect_reversal_pattern(
                price_data, pattern_name, config, catalyst_info
            )
            if pattern and pattern['final_confidence'] >= self.min_confidence:
                detected_patterns.append(pattern)
                
        # Check continuation patterns
        for pattern_name, config in self.pattern_config['continuation_patterns'].items():
            pattern = self._detect_continuation_pattern(
                price_data, pattern_name, config, catalyst_info
            )
            if pattern and pattern['final_confidence'] >= self.min_confidence:
                detected_patterns.append(pattern)
                
        # Check momentum patterns
        for pattern_name, config in self.pattern_config['momentum_patterns'].items():
            pattern = self._detect_momentum_pattern(
                price_data, pattern_name, config, catalyst_info
            )
            if pattern and pattern['final_confidence'] >= self.min_confidence:
                detected_patterns.append(pattern)
                
        # Sort by confidence
        detected_patterns.sort(key=lambda x: x['final_confidence'], reverse=True)
        
        # Build result
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'timeframe': timeframe,
            'status': 'analyzed',
            'patterns': detected_patterns[:5],  # Top 5 patterns
            'catalyst_context': catalyst_info,
            'data_points': len(price_data),
            'analysis_time_ms': 0  # TODO: Add timing
        }
        
        # Cache result
        if not context:  # Only cache generic results
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(result)
            )
            
        # Save to database
        for pattern in detected_patterns:
            self._save_pattern_detection(pattern, symbol, timeframe, catalyst_info)
            
        return result
        
    def _get_price_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get price data for analysis"""
        if not YFINANCE_AVAILABLE:
            return self._get_mock_price_data(symbol, timeframe)
            
        try:
            ticker = yf.Ticker(symbol)
            
            # Determine period based on timeframe
            period_map = {
                '1min': '7d',
                '5min': '1mo',
                '15min': '2mo',
                '30min': '3mo',
                '1h': '6mo',
                '1d': '2y'
            }
            
            period = period_map.get(timeframe, '1mo')
            data = ticker.history(period=period, interval=timeframe)
            
            if data.empty:
                self.logger.warning(f"No data retrieved for {symbol}")
                return None
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}", error=str(e))
            return None
            
    def _extract_catalyst_info(self, context: Dict) -> Dict:
        """Extract catalyst information from context"""
        if not context:
            return {
                'has_catalyst': False,
                'catalyst_type': None,
                'catalyst_sentiment': 'neutral',
                'catalyst_score': 0,
                'is_pre_market': False
            }
            
        # Determine catalyst sentiment based on type
        catalyst_type = context.get('catalyst_type')
        sentiment_map = {
            'earnings_beat': 'positive',
            'earnings_miss': 'negative',
            'fda_approval': 'positive',
            'fda_rejection': 'negative',
            'merger_announcement': 'positive',
            'lawsuit': 'negative',
            'upgrade': 'positive',
            'downgrade': 'negative',
            'guidance_raised': 'positive',
            'guidance_lowered': 'negative',
            'insider_buying': 'positive',
            'insider_selling': 'negative'
        }
        
        catalyst_sentiment = sentiment_map.get(catalyst_type, 'neutral')
        
        # Special handling for earnings
        if catalyst_type == 'earnings' and 'earnings_result' in context:
            catalyst_sentiment = 'positive' if context['earnings_result'] == 'beat' else 'negative'
            
        return {
            'has_catalyst': context.get('has_catalyst', False),
            'catalyst_type': catalyst_type,
            'catalyst_sentiment': catalyst_sentiment,
            'catalyst_score': context.get('catalyst_score', 0),
            'is_pre_market': context.get('market_state') == 'pre-market',
            'news_count': context.get('news_count', 0)
        }
        
    def _detect_reversal_pattern(self, data: pd.DataFrame, pattern_name: str, 
                                config: Dict, catalyst_info: Dict) -> Optional[Dict]:
        """Detect reversal patterns with catalyst weighting"""
        
        if len(data) < 2:
            return None
            
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        pattern_detected = False
        pattern_direction = None
        
        if pattern_name == 'hammer':
            # Bullish reversal - long lower shadow, small body at top
            body = abs(latest['Close'] - latest['Open'])
            lower_shadow = min(latest['Open'], latest['Close']) - latest['Low']
            upper_shadow = latest['High'] - max(latest['Open'], latest['Close'])
            
            if lower_shadow > body * config['min_shadow_ratio'] and upper_shadow < body:
                pattern_detected = True
                pattern_direction = 'bullish'
                
        elif pattern_name == 'shooting_star':
            # Bearish reversal - long upper shadow, small body at bottom
            body = abs(latest['Close'] - latest['Open'])
            upper_shadow = latest['High'] - max(latest['Open'], latest['Close'])
            lower_shadow = min(latest['Open'], latest['Close']) - latest['Low']
            
            if upper_shadow > body * config['min_shadow_ratio'] and lower_shadow < body:
                pattern_detected = True
                pattern_direction = 'bearish'
                
        elif pattern_name == 'engulfing':
            # Check for bullish or bearish engulfing
            if latest['Close'] > latest['Open'] and prev['Close'] < prev['Open']:
                # Potential bullish engulfing
                if latest['Open'] < prev['Close'] and latest['Close'] > prev['Open']:
                    pattern_detected = True
                    pattern_direction = 'bullish'
            elif latest['Close'] < latest['Open'] and prev['Close'] > prev['Open']:
                # Potential bearish engulfing
                if latest['Open'] > prev['Close'] and latest['Close'] < prev['Open']:
                    pattern_detected = True
                    pattern_direction = 'bearish'
                    
        elif pattern_name == 'doji':
            # Indecision pattern - very small body
            body = abs(latest['Close'] - latest['Open'])
            total_range = latest['High'] - latest['Low']
            
            if total_range > 0 and body / total_range < config['max_body_ratio']:
                pattern_detected = True
                pattern_direction = 'neutral'
                
        if not pattern_detected:
            return None
            
        # Calculate confidence with catalyst adjustment
        base_confidence = config['base_confidence']
        catalyst_multiplier = config['catalyst_boost'].get(catalyst_info['catalyst_sentiment'], 1.0)
        
        # Adjust for pattern-catalyst alignment
        if pattern_direction == 'bullish' and catalyst_info['catalyst_sentiment'] == 'positive':
            catalyst_multiplier *= 1.2  # Extra boost for alignment
        elif pattern_direction == 'bearish' and catalyst_info['catalyst_sentiment'] == 'negative':
            catalyst_multiplier *= 1.2  # Extra boost for alignment
        elif pattern_direction != 'neutral' and catalyst_info['catalyst_sentiment'] != 'neutral':
            if pattern_direction != catalyst_info['catalyst_sentiment']:
                catalyst_multiplier *= 0.7  # Penalty for misalignment
                
        # Pre-market boost
        if catalyst_info['is_pre_market']:
            catalyst_multiplier *= self.premarket_multiplier
            
        final_confidence = min(base_confidence * catalyst_multiplier, 100)
        
        return {
            'pattern_name': pattern_name,
            'pattern_type': 'reversal',
            'pattern_direction': pattern_direction,
            'base_confidence': base_confidence,
            'catalyst_multiplier': catalyst_multiplier,
            'final_confidence': round(final_confidence, 2),
            'detected_at': datetime.now().isoformat(),
            'price': float(latest['Close']),
            'volume': int(latest['Volume']),
            'catalyst_aligned': pattern_direction == catalyst_info['catalyst_sentiment']
        }
        
    def _detect_continuation_pattern(self, data: pd.DataFrame, pattern_name: str,
                                   config: Dict, catalyst_info: Dict) -> Optional[Dict]:
        """Detect continuation patterns"""
        
        if len(data) < 3:
            return None
            
        last_three = data.iloc[-3:]
        pattern_detected = False
        pattern_direction = None
        
        if pattern_name == 'three_white_soldiers':
            # Three consecutive bullish candles
            all_bullish = all(row['Close'] > row['Open'] for _, row in last_three.iterrows())
            ascending = all(last_three['Close'].iloc[i] > last_three['Close'].iloc[i-1] 
                          for i in range(1, 3))
            
            if all_bullish and ascending:
                pattern_detected = True
                pattern_direction = 'bullish'
                
        elif pattern_name == 'three_black_crows':
            # Three consecutive bearish candles
            all_bearish = all(row['Close'] < row['Open'] for _, row in last_three.iterrows())
            descending = all(last_three['Close'].iloc[i] < last_three['Close'].iloc[i-1] 
                           for i in range(1, 3))
            
            if all_bearish and descending:
                pattern_detected = True
                pattern_direction = 'bearish'
                
        if not pattern_detected:
            return None
            
        # Calculate confidence
        base_confidence = config['base_confidence']
        catalyst_multiplier = config['catalyst_boost'].get(catalyst_info['catalyst_sentiment'], 1.0)
        
        # Strong alignment bonus for continuation patterns
        if pattern_direction == catalyst_info['catalyst_sentiment']:
            catalyst_multiplier *= 1.3
            
        # Pre-market boost
        if catalyst_info['is_pre_market']:
            catalyst_multiplier *= self.premarket_multiplier
            
        final_confidence = min(base_confidence * catalyst_multiplier, 100)
        
        return {
            'pattern_name': pattern_name,
            'pattern_type': 'continuation',
            'pattern_direction': pattern_direction,
            'base_confidence': base_confidence,
            'catalyst_multiplier': catalyst_multiplier,
            'final_confidence': round(final_confidence, 2),
            'detected_at': datetime.now().isoformat(),
            'price': float(data.iloc[-1]['Close']),
            'volume': int(data.iloc[-1]['Volume']),
            'catalyst_aligned': pattern_direction == catalyst_info['catalyst_sentiment']
        }
        
    def _detect_momentum_pattern(self, data: pd.DataFrame, pattern_name: str,
                               config: Dict, catalyst_info: Dict) -> Optional[Dict]:
        """Detect momentum patterns"""
        
        if len(data) < 2:
            return None
            
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        pattern_detected = False
        pattern_direction = None
        
        if pattern_name == 'gap_up':
            # Gap up pattern
            gap_percent = ((latest['Open'] - prev['Close']) / prev['Close']) * 100
            
            if gap_percent >= config['min_gap_percent']:
                pattern_detected = True
                pattern_direction = 'bullish'
                
        elif pattern_name == 'gap_down':
            # Gap down pattern
            gap_percent = ((prev['Close'] - latest['Open']) / prev['Close']) * 100
            
            if gap_percent >= config['min_gap_percent']:
                pattern_detected = True
                pattern_direction = 'bearish'
                
        elif pattern_name == 'volume_surge':
            # Volume surge pattern
            avg_volume = data['Volume'].iloc[:-1].mean()
            volume_ratio = latest['Volume'] / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio >= config['min_volume_ratio']:
                pattern_detected = True
                # Direction based on price action
                pattern_direction = 'bullish' if latest['Close'] > latest['Open'] else 'bearish'
                
        if not pattern_detected:
            return None
            
        # Calculate confidence
        base_confidence = config['base_confidence']
        catalyst_multiplier = config['catalyst_boost'].get(catalyst_info['catalyst_sentiment'], 1.0)
        
        # Momentum patterns get huge boost with catalyst alignment
        if pattern_direction == catalyst_info['catalyst_sentiment']:
            catalyst_multiplier *= 1.5
            
        # Pre-market gaps are especially significant
        if catalyst_info['is_pre_market'] and pattern_name in ['gap_up', 'gap_down']:
            catalyst_multiplier *= 1.5
            
        final_confidence = min(base_confidence * catalyst_multiplier, 100)
        
        return {
            'pattern_name': pattern_name,
            'pattern_type': 'momentum',
            'pattern_direction': pattern_direction,
            'base_confidence': base_confidence,
            'catalyst_multiplier': catalyst_multiplier,
            'final_confidence': round(final_confidence, 2),
            'detected_at': datetime.now().isoformat(),
            'price': float(latest['Close']),
            'volume': int(latest['Volume']),
            'catalyst_aligned': pattern_direction == catalyst_info['catalyst_sentiment']
        }
        
    def _save_pattern_detection(self, pattern: Dict, symbol: str, 
                              timeframe: str, catalyst_info: Dict):
        """Save pattern detection to database"""
        try:
            pattern_data = {
                'symbol': symbol,
                'pattern_name': pattern['pattern_name'],
                'pattern_type': pattern['pattern_type'],
                'base_confidence': pattern['base_confidence'],
                'final_confidence': pattern['final_confidence'],
                'timeframe': timeframe,
                'metadata': {
                    'pattern_direction': pattern['pattern_direction'],
                    'catalyst_aligned': pattern['catalyst_aligned'],
                    'catalyst_type': catalyst_info['catalyst_type'],
                    'catalyst_sentiment': catalyst_info['catalyst_sentiment'],
                    'is_pre_market': catalyst_info['is_pre_market'],
                    'price': pattern['price'],
                    'volume': pattern['volume']
                },
                'detected_at': datetime.now()
            }
            
            insert_pattern_detection(pattern_data)
            
        except Exception as e:
            self.logger.error("Failed to save pattern detection", error=str(e))
            
    def _get_pattern_statistics(self, pattern: Optional[str], 
                              catalyst: Optional[str], hours: int) -> Dict:
        """Get historical pattern performance statistics"""
        # This would query the database for pattern performance
        # For now, return mock data
        return {
            'pattern': pattern,
            'catalyst': catalyst,
            'period_hours': hours,
            'total_detections': 0,
            'successful_trades': 0,
            'success_rate': 0,
            'average_confidence': 0,
            'best_performing_combo': None
        }
        
    def _update_pattern_outcome(self, pattern_id: str, outcome: Dict) -> Dict:
        """Update pattern with trading outcome for ML training"""
        # This would update the database with pattern outcome
        return {
            'pattern_id': pattern_id,
            'outcome_recorded': True,
            'timestamp': datetime.now().isoformat()
        }
        
    def _get_mock_price_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Generate mock price data for testing"""
        # Generate 100 periods of random walk data
        periods = 100
        base_price = 100
        
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')
        
        # Random walk
        returns = np.random.normal(0, 0.02, periods)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # OHLC with some realistic relationships
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.005, 0.005, periods)),
            'High': prices * (1 + np.random.uniform(0, 0.01, periods)),
            'Low': prices * (1 + np.random.uniform(-0.01, 0, periods)),
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, periods)
        }, index=dates)
        
        # Ensure high is highest and low is lowest
        data['High'] = data[['Open', 'High', 'Low', 'Close']].max(axis=1)
        data['Low'] = data[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        return data
        
    def run(self):
        """Start the pattern analysis service"""
        self.logger.info("Starting Pattern Analysis Service",
                        version="2.1.2",
                        port=self.port,
                        environment=os.getenv('ENVIRONMENT', 'development'))
        
        self.app.run(host='0.0.0.0', port=self.port, debug=False)


if __name__ == "__main__":
    service = CatalystAwarePatternAnalysis()
    service.run()