#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: pattern_service.py
Version: 2.1.0
Last Updated: 2025-07-01
Purpose: Detect technical patterns with news catalyst context weighting using PostgreSQL

REVISION HISTORY:
v2.1.0 (2025-07-01) - Production-ready refactor
- Migrated from SQLite to PostgreSQL
- All configuration via environment variables
- Enhanced pattern detection algorithms
- Added pattern caching for performance
- Improved ML feature collection
- Better error handling and logging

v2.0.0 (2025-06-28) - Complete rewrite for catalyst-aware analysis
- Context-weighted pattern detection
- News alignment scoring
- Pre-market pattern emphasis
- ML data collection for patterns
- Catalyst type influences pattern interpretation

Description of Service:
This service detects traditional candlestick patterns but weights their
significance based on the presence and type of news catalysts.

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
        
        self.logger.info("Catalyst-Aware Pattern Analysis v2.1.0 initialized",
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
                'bullish_engulfing': {
                    'base_confidence': float(os.getenv('BULL_ENGULF_BASE_CONFIDENCE', '70')),
                    'catalyst_boost': {
                        'positive': float(os.getenv('BULL_ENGULF_POSITIVE_BOOST', '1.5')),
                        'negative': float(os.getenv('BULL_ENGULF_NEGATIVE_BOOST', '0.6')),
                        'neutral': 1.0
                    },
                    'min_body_ratio': float(os.getenv('BULL_ENGULF_MIN_BODY', '1.5'))
                },
                'bearish_engulfing': {
                    'base_confidence': float(os.getenv('BEAR_ENGULF_BASE_CONFIDENCE', '70')),
                    'catalyst_boost': {
                        'positive': float(os.getenv('BEAR_ENGULF_POSITIVE_BOOST', '0.6')),
                        'negative': float(os.getenv('BEAR_ENGULF_NEGATIVE_BOOST', '1.5')),
                        'neutral': 1.0
                    },
                    'min_body_ratio': float(os.getenv('BEAR_ENGULF_MIN_BODY', '1.5'))
                },
                'doji': {
                    'base_confidence': float(os.getenv('DOJI_BASE_CONFIDENCE', '60')),
                    'catalyst_boost': {
                        'positive': float(os.getenv('DOJI_POSITIVE_BOOST', '1.2')),
                        'negative': float(os.getenv('DOJI_NEGATIVE_BOOST', '1.2')),
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
                    },
                    'min_consecutive': 3
                },
                'three_black_crows': {
                    'base_confidence': float(os.getenv('THREE_BLACK_BASE_CONFIDENCE', '75')),
                    'catalyst_boost': {
                        'positive': float(os.getenv('THREE_BLACK_POSITIVE_BOOST', '0.5')),
                        'negative': float(os.getenv('THREE_BLACK_NEGATIVE_BOOST', '1.6')),
                        'neutral': 1.0
                    },
                    'min_consecutive': 3
                }
            },
            'momentum_patterns': {
                'gap_up': {
                    'base_confidence': float(os.getenv('GAP_UP_BASE_CONFIDENCE', '70')),
                    'catalyst_boost': {
                        'positive': float(os.getenv('GAP_UP_POSITIVE_BOOST', '1.7')),
                        'negative': float(os.getenv('GAP_UP_NEGATIVE_BOOST', '0.4')),
                        'neutral': 1.0
                    },
                    'min_gap_percent': float(os.getenv('GAP_UP_MIN_PERCENT', '2.0'))
                },
                'gap_down': {
                    'base_confidence': float(os.getenv('GAP_DOWN_BASE_CONFIDENCE', '70')),
                    'catalyst_boost': {
                        'positive': float(os.getenv('GAP_DOWN_POSITIVE_BOOST', '0.4')),
                        'negative': float(os.getenv('GAP_DOWN_NEGATIVE_BOOST', '1.7')),
                        'neutral': 1.0
                    },
                    'min_gap_percent': float(os.getenv('GAP_DOWN_MIN_PERCENT', '2.0'))
                },
                'volume_surge': {
                    'base_confidence': float(os.getenv('VOLUME_SURGE_BASE_CONFIDENCE', '65')),
                    'catalyst_boost': {
                        'positive': float(os.getenv('VOLUME_SURGE_POSITIVE_BOOST', '1.4')),
                        'negative': float(os.getenv('VOLUME_SURGE_NEGATIVE_BOOST', '1.4')),
                        'neutral': 1.0
                    },
                    'min_volume_ratio': float(os.getenv('VOLUME_SURGE_MIN_RATIO', '2.0'))
                }
            }
        }
        
    def setup_routes(self):
        """Setup Flask routes"""
        @self.app.route('/health', methods=['GET'])
        def health():
            db_health = health_check()
            return jsonify({
                "status": "healthy" if db_health['database'] == 'healthy' else "degraded",
                "service": "pattern_analysis",
                "version": "2.1.0",
                "mode": "catalyst-aware",
                "database": db_health['database'],
                "redis": db_health['redis'],
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
                
        # Add technical indicators
        self._add_technical_indicators(price_data, detected_patterns)
        
        # Sort by confidence
        detected_patterns.sort(key=lambda x: x['final_confidence'], reverse=True)
        
        # Save patterns to database
        for pattern in detected_patterns:
            self._save_pattern(symbol, pattern, catalyst_info)
            
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'timeframe': timeframe,
            'catalyst_present': catalyst_info['has_catalyst'],
            'catalyst_type': catalyst_info.get('catalyst_type'),
            'patterns': detected_patterns[:3],  # Top 3 patterns
            'recommendation': self._generate_recommendation(detected_patterns, catalyst_info)
        }
        
        # Cache result if no specific context
        if not context:
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(result)
            )
        
        return result
        
    def _get_price_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get price data for analysis"""
        if not YFINANCE_AVAILABLE:
            return self._get_mock_price_data(symbol)
            
        try:
            ticker = yf.Ticker(symbol)
            
            # Determine period based on timeframe
            period_map = {
                '1min': '1d',
                '5min': '5d',
                '15min': '1mo',
                '30min': '1mo',
                '1h': '3mo',
                '1d': '1y'
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
        pattern

if __name__ == "__main__":
    service = CatalystAwarePatternAnalysis()
    service.run()