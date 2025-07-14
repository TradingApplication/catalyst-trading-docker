#!/usr/bin/env python3
"""
Name of Service: TRADING SYSTEM PATTERN ANALYSIS - CATALYST-AWARE VERSION
Version: 2.0.0
Last Updated: 2025-06-28
Purpose: Detect technical patterns with news catalyst context weighting

REVISION HISTORY:
v2.0.0 (2025-06-28) - Complete rewrite for catalyst-aware analysis
- Context-weighted pattern detection
- News alignment scoring
- Pre-market pattern emphasis
- ML data collection for patterns
- Catalyst type influences pattern interpretation

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
import sqlite3
import logging
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Import database utilities if available
try:
    from database_utils_old import DatabaseServiceMixin
    USE_DB_UTILS = True
except ImportError:
    USE_DB_UTILS = False
    print("Warning: database_utils not found, using direct SQLite connections")

# Handle yfinance import
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available, using mock data")


class CatalystAwarePatternAnalysis(DatabaseServiceMixin if USE_DB_UTILS else object):
    """
    Pattern analysis that understands news context
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
        
        # Pattern configuration with catalyst weights
        self.pattern_config = {
            'reversal_patterns': {
                'hammer': {
                    'base_confidence': 65,
                    'catalyst_boost': {'positive': 1.5, 'negative': 0.7, 'neutral': 1.0},
                    'min_shadow_ratio': 2.0
                },
                'shooting_star': {
                    'base_confidence': 65,
                    'catalyst_boost': {'positive': 0.7, 'negative': 1.5, 'neutral': 1.0},
                    'min_shadow_ratio': 2.0
                },
                'bullish_engulfing': {
                    'base_confidence': 70,
                    'catalyst_boost': {'positive': 1.5, 'negative': 0.6, 'neutral': 1.0},
                    'min_body_ratio': 1.5
                },
                'bearish_engulfing': {
                    'base_confidence': 70,
                    'catalyst_boost': {'positive': 0.6, 'negative': 1.5, 'neutral': 1.0},
                    'min_body_ratio': 1.5
                },
                'doji': {
                    'base_confidence': 60,
                    'catalyst_boost': {'positive': 1.2, 'negative': 1.2, 'neutral': 1.0},
                    'max_body_ratio': 0.1
                }
            },
            'continuation_patterns': {
                'three_white_soldiers': {
                    'base_confidence': 75,
                    'catalyst_boost': {'positive': 1.6, 'negative': 0.5, 'neutral': 1.0},
                    'min_consecutive': 3
                },
                'three_black_crows': {
                    'base_confidence': 75,
                    'catalyst_boost': {'positive': 0.5, 'negative': 1.6, 'neutral': 1.0},
                    'min_consecutive': 3
                }
            },
            'momentum_patterns': {
                'gap_up': {
                    'base_confidence': 70,
                    'catalyst_boost': {'positive': 1.7, 'negative': 0.4, 'neutral': 1.0},
                    'min_gap_percent': 2.0
                },
                'gap_down': {
                    'base_confidence': 70,
                    'catalyst_boost': {'positive': 0.4, 'negative': 1.7, 'neutral': 1.0},
                    'min_gap_percent': 2.0
                },
                'volume_surge': {
                    'base_confidence': 65,
                    'catalyst_boost': {'positive': 1.4, 'negative': 1.4, 'neutral': 1.0},
                    'min_volume_ratio': 2.0
                }
            }
        }
        
        # Pre-market multiplier
        self.premarket_multiplier = 2.0
        
        # Initialize database
        self._init_database_schema()
        
        # Register with coordination
        self._register_with_coordination()
        
        self.logger.info("Catalyst-Aware Pattern Analysis v2.0.0 initialized")
        
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('pattern_analysis')
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory
        os.makedirs('/tmp/logs', exist_ok=True)
        
        # File handler
        fh = logging.FileHandler('/tmp//tmp/logs/pattern_analysis.log')
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
        """Initialize pattern analysis tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced pattern analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    timeframe TEXT DEFAULT '5min',
                    
                    -- Pattern details
                    pattern_type TEXT NOT NULL,
                    pattern_name TEXT NOT NULL,
                    pattern_direction TEXT,  -- bullish, bearish, neutral
                    base_confidence DECIMAL(5,2),
                    
                    -- Catalyst context
                    has_catalyst BOOLEAN DEFAULT FALSE,
                    catalyst_type TEXT,  -- earnings, fda, merger, etc
                    catalyst_sentiment TEXT,  -- positive, negative, neutral
                    catalyst_alignment BOOLEAN,  -- Does pattern align with catalyst?
                    catalyst_score DECIMAL(5,2),
                    
                    -- Adjusted metrics
                    catalyst_adjusted_confidence DECIMAL(5,2),
                    is_pre_market BOOLEAN DEFAULT FALSE,
                    final_confidence DECIMAL(5,2),
                    
                    -- Pattern metrics
                    pattern_strength DECIMAL(5,2),
                    support_level DECIMAL(10,2),
                    resistance_level DECIMAL(10,2),
                    
                    -- Volume context
                    volume_confirmation BOOLEAN,
                    relative_volume DECIMAL(5,2),
                    
                    -- Technical context
                    trend_alignment BOOLEAN,
                    price_at_detection DECIMAL(10,2),
                    
                    -- Outcome tracking (for ML)
                    pattern_completed BOOLEAN,
                    actual_move_percent DECIMAL(5,2),
                    success BOOLEAN,
                    time_to_completion_minutes INTEGER,
                    
                    -- ML features
                    ml_features JSON,  -- Store all context for training
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Pattern statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_name TEXT NOT NULL,
                    catalyst_type TEXT,
                    
                    -- Success rates
                    total_detected INTEGER DEFAULT 0,
                    successful_patterns INTEGER DEFAULT 0,
                    failed_patterns INTEGER DEFAULT 0,
                    success_rate DECIMAL(5,2),
                    
                    -- Performance metrics
                    avg_move_percent DECIMAL(5,2),
                    max_move_percent DECIMAL(5,2),
                    avg_time_to_target_minutes INTEGER,
                    
                    -- Catalyst correlation
                    with_catalyst_success_rate DECIMAL(5,2),
                    without_catalyst_success_rate DECIMAL(5,2),
                    catalyst_improvement DECIMAL(5,2),
                    
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(pattern_name, catalyst_type)
                )
            ''')
            
            # Create indexes
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_pattern_symbol 
                ON pattern_analysis(symbol, detection_timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_pattern_catalyst 
                ON pattern_analysis(has_catalyst, catalyst_type)
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database schema initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
            
    def setup_routes(self):
        """Setup Flask routes"""
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                "status": "healthy",
                "service": "pattern_analysis",
                "version": "2.0.0",
                "mode": "catalyst-aware",
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
            
            results = []
            for symbol in symbols:
                try:
                    result = self.analyze_with_catalyst_context(symbol, timeframe)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error analyzing {symbol}: {e}")
                    
            return jsonify({'results': results})
            
        @self.app.route('/pattern_statistics', methods=['GET'])
        def pattern_statistics():
            """Get pattern success statistics"""
            pattern = request.args.get('pattern')
            catalyst = request.args.get('catalyst')
            
            stats = self._get_pattern_statistics(pattern, catalyst)
            return jsonify(stats)
            
        @self.app.route('/update_pattern_outcome', methods=['POST'])
        def update_pattern_outcome():
            """Update pattern with actual outcome for ML"""
            data = request.json
            pattern_id = data.get('pattern_id')
            outcome = data.get('outcome')
            
            result = self._update_pattern_outcome(pattern_id, outcome)
            return jsonify(result)
            
    def analyze_with_catalyst_context(self, symbol: str, timeframe: str = '5min', 
                                    context: Dict = None) -> Dict:
        """
        Analyze patterns with catalyst awareness
        """
        self.logger.info(f"Analyzing {symbol} with catalyst context: {context}")
        
        # Get price data
        price_data = self._get_price_data(symbol, timeframe)
        if price_data is None or len(price_data) < 20:
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
            if pattern:
                detected_patterns.append(pattern)
                
        # Check continuation patterns
        for pattern_name, config in self.pattern_config['continuation_patterns'].items():
            pattern = self._detect_continuation_pattern(
                price_data, pattern_name, config, catalyst_info
            )
            if pattern:
                detected_patterns.append(pattern)
                
        # Check momentum patterns
        for pattern_name, config in self.pattern_config['momentum_patterns'].items():
            pattern = self._detect_momentum_pattern(
                price_data, pattern_name, config, catalyst_info
            )
            if pattern:
                detected_patterns.append(pattern)
                
        # Sort by confidence
        detected_patterns.sort(key=lambda x: x['final_confidence'], reverse=True)
        
        # Save patterns to database
        for pattern in detected_patterns:
            self._save_pattern(symbol, pattern, catalyst_info)
            
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'catalyst_present': catalyst_info['has_catalyst'],
            'catalyst_type': catalyst_info.get('catalyst_type'),
            'patterns': detected_patterns[:3],  # Top 3 patterns
            'recommendation': self._generate_recommendation(detected_patterns, catalyst_info)
        }
        
    def _get_price_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get price data for analysis"""
        if not YFINANCE_AVAILABLE:
            return self._get_mock_price_data(symbol)
            
        try:
            ticker = yf.Ticker(symbol)
            
            # Determine period based on timeframe
            if timeframe == '1min':
                period = '1d'
            elif timeframe == '5min':
                period = '5d'
            else:
                period = '1mo'
                
            data = ticker.history(period=period, interval=timeframe)
            
            if data.empty:
                self.logger.warning(f"No data retrieved for {symbol}")
                return None
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
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
            'downgrade': 'negative'
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
        pattern_strength = 0
        
        if pattern_name == 'hammer':
            # Hammer: Small body at top, long lower shadow
            body = abs(latest['Close'] - latest['Open'])
            lower_shadow = min(latest['Open'], latest['Close']) - latest['Low']
            upper_shadow = latest['High'] - max(latest['Open'], latest['Close'])
            
            if lower_shadow > body * config['min_shadow_ratio'] and upper_shadow < body:
                pattern_detected = True
                pattern_direction = 'bullish'
                pattern_strength = min(100, (lower_shadow / body) * 20)
                
        elif pattern_name == 'shooting_star':
            # Shooting star: Small body at bottom, long upper shadow
            body = abs(latest['Close'] - latest['Open'])
            upper_shadow = latest['High'] - max(latest['Open'], latest['Close'])
            lower_shadow = min(latest['Open'], latest['Close']) - latest['Low']
            
            if upper_shadow > body * config['min_shadow_ratio'] and lower_shadow < body:
                pattern_detected = True
                pattern_direction = 'bearish'
                pattern_strength = min(100, (upper_shadow / body) * 20)
                
        elif pattern_name == 'bullish_engulfing':
            # Current green candle engulfs previous red candle
            if (prev['Close'] < prev['Open'] and  # Previous is red
                latest['Close'] > latest['Open'] and  # Current is green
                latest['Open'] <= prev['Close'] and  # Opens below prev close
                latest['Close'] >= prev['Open']):  # Closes above prev open
                
                pattern_detected = True
                pattern_direction = 'bullish'
                body_ratio = (latest['Close'] - latest['Open']) / (prev['Open'] - prev['Close'])
                pattern_strength = min(100, body_ratio * 30)
                
        elif pattern_name == 'bearish_engulfing':
            # Current red candle engulfs previous green candle
            if (prev['Close'] > prev['Open'] and  # Previous is green
                latest['Close'] < latest['Open'] and  # Current is red
                latest['Open'] >= prev['Close'] and  # Opens above prev close
                latest['Close'] <= prev['Open']):  # Closes below prev open
                
                pattern_detected = True
                pattern_direction = 'bearish'
                body_ratio = (latest['Open'] - latest['Close']) / (prev['Close'] - prev['Open'])
                pattern_strength = min(100, body_ratio * 30)
                
        elif pattern_name == 'doji':
            # Very small body relative to range
            body = abs(latest['Close'] - latest['Open'])
            range_size = latest['High'] - latest['Low']
            
            if range_size > 0 and body / range_size < config['max_body_ratio']:
                pattern_detected = True
                pattern_direction = 'neutral'
                pattern_strength = min(100, (1 - body / range_size) * 100)
                
        if not pattern_detected:
            return None
            
        # Calculate catalyst-adjusted confidence
        base_confidence = config['base_confidence']
        catalyst_boost = config['catalyst_boost'][catalyst_info['catalyst_sentiment']]
        
        # Check alignment
        alignment = self._check_pattern_catalyst_alignment(
            pattern_direction, catalyst_info['catalyst_sentiment']
        )
        
        catalyst_adjusted_confidence = base_confidence * catalyst_boost
        
        # Apply pre-market multiplier if applicable
        if catalyst_info['is_pre_market'] and catalyst_info['has_catalyst']:
            catalyst_adjusted_confidence *= self.premarket_multiplier
            
        final_confidence = min(100, catalyst_adjusted_confidence)
        
        return {
            'pattern_type': 'reversal',
            'pattern_name': pattern_name,
            'pattern_direction': pattern_direction,
            'base_confidence': base_confidence,
            'catalyst_adjusted_confidence': catalyst_adjusted_confidence,
            'final_confidence': final_confidence,
            'pattern_strength': pattern_strength,
            'catalyst_alignment': alignment,
            'support_level': latest['Low'],
            'resistance_level': latest['High'],
            'detection_price': latest['Close'],
            'detection_time': datetime.now().isoformat()
        }
        
    def _detect_continuation_pattern(self, data: pd.DataFrame, pattern_name: str,
                                   config: Dict, catalyst_info: Dict) -> Optional[Dict]:
        """Detect continuation patterns"""
        
        if len(data) < config.get('min_consecutive', 3):
            return None
            
        pattern_detected = False
        pattern_direction = None
        pattern_strength = 0
        
        if pattern_name == 'three_white_soldiers':
            # Three consecutive bullish candles
            bullish_count = 0
            for i in range(-3, 0):
                if data.iloc[i]['Close'] > data.iloc[i]['Open']:
                    bullish_count += 1
                    
            if bullish_count == 3:
                pattern_detected = True
                pattern_direction = 'bullish'
                # Calculate total move
                total_move = (data.iloc[-1]['Close'] - data.iloc[-3]['Open']) / data.iloc[-3]['Open'] * 100
                pattern_strength = min(100, total_move * 10)
                
        elif pattern_name == 'three_black_crows':
            # Three consecutive bearish candles
            bearish_count = 0
            for i in range(-3, 0):
                if data.iloc[i]['Close'] < data.iloc[i]['Open']:
                    bearish_count += 1
                    
            if bearish_count == 3:
                pattern_detected = True
                pattern_direction = 'bearish'
                # Calculate total move
                total_move = (data.iloc[-3]['Open'] - data.iloc[-1]['Close']) / data.iloc[-3]['Open'] * 100
                pattern_strength = min(100, total_move * 10)
                
        if not pattern_detected:
            return None
            
        # Apply catalyst weighting
        base_confidence = config['base_confidence']
        catalyst_boost = config['catalyst_boost'][catalyst_info['catalyst_sentiment']]
        
        catalyst_adjusted_confidence = base_confidence * catalyst_boost
        
        if catalyst_info['is_pre_market'] and catalyst_info['has_catalyst']:
            catalyst_adjusted_confidence *= self.premarket_multiplier
            
        final_confidence = min(100, catalyst_adjusted_confidence)
        
        return {
            'pattern_type': 'continuation',
            'pattern_name': pattern_name,
            'pattern_direction': pattern_direction,
            'base_confidence': base_confidence,
            'catalyst_adjusted_confidence': catalyst_adjusted_confidence,
            'final_confidence': final_confidence,
            'pattern_strength': pattern_strength,
            'catalyst_alignment': self._check_pattern_catalyst_alignment(
                pattern_direction, catalyst_info['catalyst_sentiment']
            ),
            'support_level': data.iloc[-3:]['Low'].min(),
            'resistance_level': data.iloc[-3:]['High'].max(),
            'detection_price': data.iloc[-1]['Close'],
            'detection_time': datetime.now().isoformat()
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
        pattern_strength = 0
        
        if pattern_name == 'gap_up':
            gap_percent = (latest['Open'] - prev['Close']) / prev['Close'] * 100
            if gap_percent >= config['min_gap_percent']:
                pattern_detected = True
                pattern_direction = 'bullish'
                pattern_strength = min(100, gap_percent * 10)
                
        elif pattern_name == 'gap_down':
            gap_percent = (prev['Close'] - latest['Open']) / prev['Close'] * 100
            if gap_percent >= config['min_gap_percent']:
                pattern_detected = True
                pattern_direction = 'bearish'
                pattern_strength = min(100, gap_percent * 10)
                
        elif pattern_name == 'volume_surge':
            if 'Volume' in data.columns:
                avg_volume = data['Volume'].iloc[-20:-1].mean()
                if avg_volume > 0:
                    volume_ratio = latest['Volume'] / avg_volume
                    if volume_ratio >= config['min_volume_ratio']:
                        pattern_detected = True
                        # Direction based on price action
                        pattern_direction = 'bullish' if latest['Close'] > latest['Open'] else 'bearish'
                        pattern_strength = min(100, (volume_ratio - 1) * 20)
                        
        if not pattern_detected:
            return None
            
        # Apply catalyst weighting
        base_confidence = config['base_confidence']
        catalyst_boost = config['catalyst_boost'][catalyst_info['catalyst_sentiment']]
        
        catalyst_adjusted_confidence = base_confidence * catalyst_boost
        
        # Momentum patterns get extra boost with catalysts
        if catalyst_info['has_catalyst']:
            catalyst_adjusted_confidence *= 1.2
            
        if catalyst_info['is_pre_market'] and catalyst_info['has_catalyst']:
            catalyst_adjusted_confidence *= self.premarket_multiplier
            
        final_confidence = min(100, catalyst_adjusted_confidence)
        
        return {
            'pattern_type': 'momentum',
            'pattern_name': pattern_name,
            'pattern_direction': pattern_direction,
            'base_confidence': base_confidence,
            'catalyst_adjusted_confidence': catalyst_adjusted_confidence,
            'final_confidence': final_confidence,
            'pattern_strength': pattern_strength,
            'catalyst_alignment': self._check_pattern_catalyst_alignment(
                pattern_direction, catalyst_info['catalyst_sentiment']
            ),
            'support_level': latest['Low'],
            'resistance_level': latest['High'],
            'detection_price': latest['Close'],
            'detection_time': datetime.now().isoformat()
        }
        
    def _check_pattern_catalyst_alignment(self, pattern_direction: str, 
                                        catalyst_sentiment: str) -> bool:
        """Check if pattern aligns with catalyst sentiment"""
        if pattern_direction == 'bullish' and catalyst_sentiment == 'positive':
            return True
        elif pattern_direction == 'bearish' and catalyst_sentiment == 'negative':
            return True
        elif pattern_direction == 'neutral':
            return True  # Neutral patterns align with everything
        else:
            return False
            
    def _save_pattern(self, symbol: str, pattern: Dict, catalyst_info: Dict):
        """Save detected pattern to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare ML features
            ml_features = {
                'catalyst_score': catalyst_info.get('catalyst_score', 0),
                'news_count': catalyst_info.get('news_count', 0),
                'pattern_strength': pattern['pattern_strength'],
                'volume_data': pattern.get('volume_data', {}),
                'technical_indicators': pattern.get('technical_indicators', {})
            }
            
            cursor.execute('''
                INSERT INTO pattern_analysis (
                    symbol, timeframe, pattern_type, pattern_name,
                    pattern_direction, base_confidence,
                    has_catalyst, catalyst_type, catalyst_sentiment,
                    catalyst_alignment, catalyst_score,
                    catalyst_adjusted_confidence, is_pre_market,
                    final_confidence, pattern_strength,
                    support_level, resistance_level,
                    price_at_detection, ml_features
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, '5min', pattern['pattern_type'], pattern['pattern_name'],
                pattern['pattern_direction'], pattern['base_confidence'],
                catalyst_info['has_catalyst'], catalyst_info.get('catalyst_type'),
                catalyst_info['catalyst_sentiment'], pattern['catalyst_alignment'],
                catalyst_info.get('catalyst_score', 0),
                pattern['catalyst_adjusted_confidence'], catalyst_info['is_pre_market'],
                pattern['final_confidence'], pattern['pattern_strength'],
                pattern['support_level'], pattern['resistance_level'],
                pattern['detection_price'], json.dumps(ml_features)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving pattern: {e}")
            
    def _generate_recommendation(self, patterns: List[Dict], catalyst_info: Dict) -> Dict:
        """Generate trading recommendation based on patterns and catalyst"""
        if not patterns:
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reason': 'No significant patterns detected'
            }
            
        # Get strongest pattern
        top_pattern = patterns[0]
        
        # Generate recommendation
        if top_pattern['final_confidence'] >= 80:
            action = 'STRONG_' + ('BUY' if top_pattern['pattern_direction'] == 'bullish' else 'SELL')
            reason = f"High confidence {top_pattern['pattern_name']} pattern"
        elif top_pattern['final_confidence'] >= 65:
            action = 'BUY' if top_pattern['pattern_direction'] == 'bullish' else 'SELL'
            reason = f"Moderate confidence {top_pattern['pattern_name']} pattern"
        else:
            action = 'HOLD'
            reason = "Pattern confidence below threshold"
            
        # Add catalyst context to reason
        if catalyst_info['has_catalyst'] and top_pattern['catalyst_alignment']:
            reason += f" aligned with {catalyst_info['catalyst_type']} catalyst"
        elif catalyst_info['has_catalyst'] and not top_pattern['catalyst_alignment']:
            reason += f" (caution: conflicts with {catalyst_info['catalyst_type']} catalyst)"
            
        return {
            'action': action,
            'confidence': top_pattern['final_confidence'],
            'reason': reason,
            'pattern': top_pattern['pattern_name'],
            'catalyst_aligned': top_pattern['catalyst_alignment']
        }
        
    def _get_pattern_statistics(self, pattern_name: Optional[str] = None,
                               catalyst_type: Optional[str] = None) -> Dict:
        """Get pattern performance statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM pattern_statistics WHERE 1=1"
            params = []
            
            if pattern_name:
                query += " AND pattern_name = ?"
                params.append(pattern_name)
                
            if catalyst_type:
                query += " AND catalyst_type = ?"
                params.append(catalyst_type)
                
            cursor.execute(query, params)
            
            stats = []
            for row in cursor.fetchall():
                stats.append({
                    'pattern_name': row[1],
                    'catalyst_type': row[2],
                    'total_detected': row[3],
                    'success_rate': row[6],
                    'avg_move_percent': row[7],
                    'catalyst_improvement': row[11]
                })
                
            conn.close()
            
            return {'statistics': stats}
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
            
    def _update_pattern_outcome(self, pattern_id: int, outcome: Dict) -> Dict:
        """Update pattern with actual outcome for ML training"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE pattern_analysis
                SET pattern_completed = ?,
                    actual_move_percent = ?,
                    success = ?,
                    time_to_completion_minutes = ?
                WHERE id = ?
            ''', (
                outcome.get('completed', False),
                outcome.get('move_percent', 0),
                outcome.get('success', False),
                outcome.get('time_minutes', 0),
                pattern_id
            ))
            
            conn.commit()
            conn.close()
            
            # Update statistics
            self._update_pattern_statistics()
            
            return {'status': 'success', 'pattern_id': pattern_id}
            
        except Exception as e:
            self.logger.error(f"Error updating outcome: {e}")
            return {'error': str(e)}
            
    def _update_pattern_statistics(self):
        """Update pattern performance statistics"""
        # This would run periodically to update success rates
        pass
        
    def _get_mock_price_data(self, symbol: str) -> pd.DataFrame:
        """Generate mock price data for testing"""
        # Create realistic OHLCV data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
        
        # Generate random walk
        np.random.seed(hash(symbol) % 1000)
        base_price = 100
        returns = np.random.normal(0.0002, 0.01, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLCV
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.002, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, len(dates))
        }, index=dates)
        
        # Ensure OHLC relationship
        df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        return df
        
    def _register_with_coordination(self):
        """Register with coordination service"""
        try:
            response = requests.post(
                f"{self.coordination_url}/register_service",
                json={
                    'service_name': 'pattern_analysis',
                    'service_info': {
                        'url': 'http://localhost:5002',
                        'port': 5002,
                        'version': '2.0.0',
                        'capabilities': ['catalyst_aware', 'ml_ready']
                    }
                }
            )
            if response.status_code == 200:
                self.logger.info("Successfully registered with coordination service")
        except Exception as e:
            self.logger.warning(f"Could not register with coordination: {e}")
            
    def run(self):
        """Start the Flask application"""
        self.logger.info("Starting Catalyst-Aware Pattern Analysis v2.0.0 on port 5002")
        self.logger.info("Pattern detection with news context weighting enabled")
        self.logger.info("ML data collection active for future pattern learning")
        
        self.app.run(host='0.0.0.0', port=5002, debug=False)


if __name__ == "__main__":
    service = CatalystAwarePatternAnalysis()
    service.run()