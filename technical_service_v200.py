#!/usr/bin/env python3
"""
Name of Service: TRADING SYSTEM TECHNICAL ANALYSIS - CATALYST-WEIGHTED VERSION
Version: 2.0.0
Last Updated: 2025-06-28
Purpose: Generate trading signals combining patterns, indicators, and catalysts

REVISION HISTORY:
v2.0.0 (2025-06-28) - Complete rewrite for catalyst integration
- Combines patterns + indicators + catalyst scores
- Dynamic entry/exit based on catalyst strength  
- Risk management varies with news confidence
- Pre-market aggressive positioning
- ML data collection for signal outcomes

This is the DECISION ENGINE of the system. It takes:
1. Patterns from Pattern Analysis (hammer, engulfing, etc.)
2. Technical indicators (RSI, MACD, Moving Averages)
3. Catalyst data (news type, strength, alignment)

And produces:
- BUY/SELL/HOLD signals
- Entry price recommendations
- Stop loss levels (tighter with strong catalysts)
- Profit targets (bigger with aligned catalysts)
- Position sizing (larger with high confidence)

KEY FORMULA:
Signal Confidence = (Catalyst Score × 35%) + (Pattern Score × 35%) + 
                   (Technical Score × 20%) + (Volume Score × 10%)

SIGNAL THRESHOLDS:
- > 70%: Strong signal (full position)
- 50-70%: Normal signal (half position)
- 30-50%: Weak signal (quarter position)
- < 30%: No trade
"""

import os
import json
import time
import sqlite3
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from typing import Dict, List, Optional, Tuple

# Import database utilities if available
try:
    from database_utils_old import DatabaseServiceMixin
    USE_DB_UTILS = True
except ImportError:
    USE_DB_UTILS = False
    print("Warning: database_utils not found, using direct SQLite connections")

# Handle technical analysis library
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available, using custom calculations")


class CatalystWeightedTechnicalAnalysis(DatabaseServiceMixin if USE_DB_UTILS else object):
    """
    Technical analysis that understands catalyst context
    Generates actionable trading signals
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
        self.pattern_service_url = "http://localhost:5002"
        self.coordination_url = "http://localhost:5000"
        
        # Signal generation weights
        self.signal_weights = {
            'catalyst_score': 0.35,    # 35% - News drives everything
            'pattern_score': 0.35,     # 35% - Technical patterns
            'indicator_score': 0.20,   # 20% - RSI, MACD, etc.
            'volume_score': 0.10       # 10% - Volume confirmation
        }
        
        # Risk parameters that adjust with catalysts
        self.risk_params = {
            'base_stop_loss_pct': 2.0,     # 2% default stop
            'catalyst_stop_adjustment': {
                'strong': 0.5,              # Tighter stop with strong catalyst (1.5%)
                'moderate': 1.0,            # Normal stop (2%)
                'weak': 1.5                 # Wider stop with weak catalyst (3%)
            },
            'position_size_map': {
                'strong_signal': 1.0,       # Full position
                'normal_signal': 0.5,       # Half position
                'weak_signal': 0.25         # Quarter position
            },
            'target_multipliers': {
                'target_1': 1.5,            # 1.5x risk for first target
                'target_2': 3.0             # 3x risk for second target
            }
        }
        
        # Indicator thresholds
        self.indicator_config = {
            'rsi': {
                'period': 14,
                'oversold': 30,
                'overbought': 70,
                'weight': 0.3
            },
            'macd': {
                'fast': 12,
                'slow': 26,
                'signal': 9,
                'weight': 0.3
            },
            'moving_averages': {
                'sma_20': 20,
                'sma_50': 50,
                'ema_9': 9,
                'weight': 0.2
            },
            'volume': {
                'avg_period': 20,
                'surge_threshold': 1.5,
                'weight': 0.2
            }
        }
        
        # Initialize database
        self._init_database_schema()
        
        # Register with coordination
        self._register_with_coordination()
        
        self.logger.info("Catalyst-Weighted Technical Analysis v2.0.0 initialized")
        
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('technical_analysis')
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory
        os.makedirs('/tmp/logs', exist_ok=True)
        
        # File handler
        fh = logging.FileHandler('/tmp//tmp/logs/technical_analysis.log')
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
        """Initialize trading signals table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced trading signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    signal_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    generated_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Signal details
                    signal_type TEXT NOT NULL,  -- BUY, SELL, HOLD
                    confidence DECIMAL(5,2),    -- 0-100
                    
                    -- Component scores (how we got here)
                    catalyst_score DECIMAL(5,2),
                    pattern_score DECIMAL(5,2),
                    indicator_score DECIMAL(5,2),
                    volume_score DECIMAL(5,2),
                    
                    -- Entry/Exit parameters
                    recommended_entry DECIMAL(10,2),
                    stop_loss DECIMAL(10,2),
                    target_1 DECIMAL(10,2),
                    target_2 DECIMAL(10,2),
                    
                    -- Context (why this signal)
                    catalyst_type TEXT,
                    catalyst_strength TEXT,  -- strong, moderate, weak
                    detected_patterns JSON,
                    technical_indicators JSON,
                    key_factors JSON,  -- Human-readable reasons
                    
                    -- Risk parameters
                    position_size_pct DECIMAL(5,2),
                    risk_reward_ratio DECIMAL(5,2),
                    max_loss_amount DECIMAL(10,2),
                    
                    -- Pre-market flag
                    is_pre_market BOOLEAN DEFAULT FALSE,
                    
                    -- Execution tracking
                    executed BOOLEAN DEFAULT FALSE,
                    execution_timestamp TIMESTAMP,
                    actual_entry DECIMAL(10,2),
                    
                    -- Outcome tracking (for ML)
                    signal_outcome TEXT,  -- success, stopped_out, partial_success
                    actual_pnl DECIMAL(10,2),
                    ml_features JSON,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Signal performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_type TEXT,
                    catalyst_type TEXT,
                    confidence_range TEXT,  -- 70-100, 50-70, 30-50
                    
                    -- Performance metrics
                    total_signals INTEGER DEFAULT 0,
                    successful_signals INTEGER DEFAULT 0,
                    failed_signals INTEGER DEFAULT 0,
                    success_rate DECIMAL(5,2),
                    
                    -- P&L metrics
                    total_pnl DECIMAL(10,2),
                    avg_win DECIMAL(10,2),
                    avg_loss DECIMAL(10,2),
                    profit_factor DECIMAL(5,2),
                    
                    -- Timing
                    avg_time_to_target_minutes INTEGER,
                    avg_time_to_stop_minutes INTEGER,
                    
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(signal_type, catalyst_type, confidence_range)
                )
            ''')
            
            # Create indexes
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_signals_symbol 
                ON trading_signals(symbol, generated_timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_signals_pending 
                ON trading_signals(executed, confidence DESC)
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
                "service": "technical_analysis",
                "version": "2.0.0",
                "mode": "catalyst-weighted",
                "timestamp": datetime.now().isoformat()
            })
            
        @self.app.route('/generate_signal', methods=['POST'])
        def generate_signal():
            """Generate trading signal for single symbol"""
            data = request.json
            symbol = data.get('symbol')
            patterns = data.get('patterns', [])
            catalyst_data = data.get('catalyst_data', {})
            
            if not symbol:
                return jsonify({'error': 'Symbol required'}), 400
                
            signal = self.generate_catalyst_weighted_signal(
                symbol, patterns, catalyst_data
            )
            return jsonify(signal)
            
        @self.app.route('/batch_signals', methods=['POST'])
        def batch_signals():
            """Generate signals for multiple symbols"""
            data = request.json
            candidates = data.get('candidates', [])
            
            signals = []
            for candidate in candidates:
                try:
                    signal = self.generate_catalyst_weighted_signal(
                        candidate['symbol'],
                        candidate.get('patterns', []),
                        candidate.get('catalyst_data', {})
                    )
                    signals.append(signal)
                except Exception as e:
                    self.logger.error(f"Error generating signal for {candidate['symbol']}: {e}")
                    
            return jsonify({'signals': signals})
            
        @self.app.route('/signal_performance', methods=['GET'])
        def signal_performance():
            """Get signal performance statistics"""
            signal_type = request.args.get('signal_type')
            catalyst_type = request.args.get('catalyst_type')
            
            stats = self._get_signal_performance(signal_type, catalyst_type)
            return jsonify(stats)
            
        @self.app.route('/update_signal_outcome', methods=['POST'])
        def update_signal_outcome():
            """Update signal with actual outcome"""
            data = request.json
            signal_id = data.get('signal_id')
            outcome = data.get('outcome')
            
            result = self._update_signal_outcome(signal_id, outcome)
            return jsonify(result)
            
    def generate_catalyst_weighted_signal(self, symbol: str, patterns: List[Dict], 
                                        catalyst_data: Dict) -> Dict:
        """
        Generate trading signal with catalyst weighting
        This is where the magic happens!
        """
        self.logger.info(f"Generating signal for {symbol}")
        
        # Get technical indicators
        indicators = self._calculate_technical_indicators(symbol)
        if not indicators:
            return self._create_no_trade_signal(symbol, "Failed to calculate indicators")
            
        # Score each component
        catalyst_score = self._score_catalyst(catalyst_data)
        pattern_score = self._score_patterns(patterns)
        indicator_score = self._score_indicators(indicators)
        volume_score = self._score_volume(indicators.get('volume_data', {}))
        
        # Calculate weighted confidence
        confidence = (
            catalyst_score * self.signal_weights['catalyst_score'] +
            pattern_score * self.signal_weights['pattern_score'] +
            indicator_score * self.signal_weights['indicator_score'] +
            volume_score * self.signal_weights['volume_score']
        )
        
        self.logger.info(f"Scores for {symbol}: Catalyst={catalyst_score:.1f}, "
                        f"Pattern={pattern_score:.1f}, Indicator={indicator_score:.1f}, "
                        f"Volume={volume_score:.1f}, Final={confidence:.1f}")
        
        # Determine signal type
        signal_type = self._determine_signal_type(
            confidence, patterns, indicators, catalyst_data
        )
        
        # Skip if no actionable signal
        if signal_type == 'HOLD' and confidence < 30:
            return self._create_no_trade_signal(symbol, "Confidence below threshold")
            
        # Calculate entry/exit levels
        current_price = indicators.get('current_price', 0)
        levels = self._calculate_entry_exit_levels(
            signal_type, current_price, confidence, catalyst_data
        )
        
        # Determine position sizing
        position_size = self._calculate_position_size(confidence)
        
        # Create key factors explanation
        key_factors = self._generate_key_factors(
            catalyst_data, patterns, indicators, confidence
        )
        
        # Generate signal ID
        signal_id = f"SIG_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare ML features
        ml_features = {
            'catalyst_score': catalyst_score,
            'pattern_score': pattern_score,
            'indicator_score': indicator_score,
            'volume_score': volume_score,
            'patterns': patterns,
            'indicators': indicators,
            'catalyst_data': catalyst_data
        }
        
        signal = {
            'signal_id': signal_id,
            'symbol': symbol,
            'signal_type': signal_type,
            'confidence': round(confidence, 2),
            'catalyst_score': round(catalyst_score, 2),
            'pattern_score': round(pattern_score, 2),
            'indicator_score': round(indicator_score, 2),
            'volume_score': round(volume_score, 2),
            'recommended_entry': levels['entry'],
            'stop_loss': levels['stop_loss'],
            'target_1': levels['target_1'],
            'target_2': levels['target_2'],
            'position_size_pct': position_size,
            'risk_reward_ratio': levels['risk_reward'],
            'catalyst_type': catalyst_data.get('type'),
            'catalyst_strength': self._classify_catalyst_strength(catalyst_score),
            'detected_patterns': patterns,
            'key_factors': key_factors,
            'is_pre_market': catalyst_data.get('is_pre_market', False),
            'ml_features': ml_features
        }
        
        # Save signal to database
        self._save_signal(signal)
        
        return signal
        
    def _calculate_technical_indicators(self, symbol: str) -> Optional[Dict]:
        """Calculate all technical indicators"""
        try:
            # Get price data (in production, this would fetch real data)
            price_data = self._get_price_data(symbol)
            if price_data is None or len(price_data) < 50:
                return None
                
            close_prices = price_data['Close'].values
            high_prices = price_data['High'].values
            low_prices = price_data['Low'].values
            volume = price_data['Volume'].values
            
            indicators = {
                'current_price': close_prices[-1],
                'price_change_pct': ((close_prices[-1] - close_prices[-2]) / close_prices[-2]) * 100
            }
            
            # RSI (Relative Strength Index)
            if TALIB_AVAILABLE:
                indicators['rsi'] = talib.RSI(close_prices, timeperiod=14)[-1]
            else:
                indicators['rsi'] = self._calculate_rsi(close_prices, 14)
                
            # MACD (Moving Average Convergence Divergence)
            if TALIB_AVAILABLE:
                macd, signal, hist = talib.MACD(close_prices, 
                                               fastperiod=12, 
                                               slowperiod=26, 
                                               signalperiod=9)
                indicators['macd'] = macd[-1]
                indicators['macd_signal'] = signal[-1]
                indicators['macd_histogram'] = hist[-1]
            else:
                indicators.update(self._calculate_macd(close_prices))
                
            # Moving Averages
            indicators['sma_20'] = self._calculate_sma(close_prices, 20)
            indicators['sma_50'] = self._calculate_sma(close_prices, 50)
            indicators['ema_9'] = self._calculate_ema(close_prices, 9)
            
            # Support and Resistance
            indicators['support'] = low_prices[-20:].min()
            indicators['resistance'] = high_prices[-20:].max()
            
            # Volume analysis
            indicators['volume_data'] = {
                'current_volume': volume[-1],
                'avg_volume': volume[-20:].mean(),
                'volume_ratio': volume[-1] / volume[-20:].mean() if volume[-20:].mean() > 0 else 1
            }
            
            # Trend detection
            indicators['trend'] = self._detect_trend(close_prices)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}: {e}")
            return None
            
    def _score_catalyst(self, catalyst_data: Dict) -> float:
        """
        Score catalyst strength (0-100)
        This is critical - strong catalysts drive trading decisions
        """
        if not catalyst_data or not catalyst_data.get('score'):
            return 0
            
        base_score = catalyst_data.get('score', 0)
        
        # Boost for specific catalyst types
        catalyst_type = catalyst_data.get('type', '').lower()
        type_multipliers = {
            'earnings_beat': 1.3,
            'fda_approval': 1.5,
            'merger': 1.4,
            'upgrade': 1.2,
            'earnings_miss': 1.3,  # Strong negative catalyst
            'downgrade': 1.2,
            'lawsuit': 1.1
        }
        
        multiplier = type_multipliers.get(catalyst_type, 1.0)
        
        # Pre-market boost
        if catalyst_data.get('is_pre_market'):
            multiplier *= 1.2
            
        # Multiple news sources boost
        news_count = catalyst_data.get('news_count', 1)
        if news_count > 3:
            multiplier *= 1.1
            
        final_score = min(100, base_score * multiplier)
        
        return final_score
        
    def _score_patterns(self, patterns: List[Dict]) -> float:
        """Score detected patterns (0-100)"""
        if not patterns:
            return 0
            
        # Use the highest confidence pattern
        max_confidence = max(p.get('final_confidence', 0) for p in patterns)
        
        # Bonus for multiple confirming patterns
        if len(patterns) > 1:
            # Check if patterns align
            directions = [p.get('pattern_direction') for p in patterns]
            if len(set(directions)) == 1:  # All same direction
                max_confidence = min(100, max_confidence * 1.1)
                
        return max_confidence
        
    def _score_indicators(self, indicators: Dict) -> float:
        """
        Score technical indicators (0-100)
        Combines RSI, MACD, and moving average signals
        """
        if not indicators:
            return 0
            
        score = 50  # Start neutral
        
        # RSI scoring
        rsi = indicators.get('rsi', 50)
        if rsi < self.indicator_config['rsi']['oversold']:
            score += 20  # Oversold = bullish
        elif rsi > self.indicator_config['rsi']['overbought']:
            score -= 20  # Overbought = bearish
            
        # MACD scoring
        macd_hist = indicators.get('macd_histogram', 0)
        if macd_hist > 0:
            score += 15  # Positive histogram = bullish
        else:
            score -= 15  # Negative histogram = bearish
            
        # Moving average scoring
        current_price = indicators.get('current_price', 0)
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)
        
        if current_price > sma_20 > sma_50:
            score += 15  # Price above both MAs = bullish
        elif current_price < sma_20 < sma_50:
            score -= 15  # Price below both MAs = bearish
            
        # Trend alignment
        trend = indicators.get('trend', 'neutral')
        if trend == 'bullish':
            score += 10
        elif trend == 'bearish':
            score -= 10
            
        return max(0, min(100, score))
        
    def _score_volume(self, volume_data: Dict) -> float:
        """Score volume characteristics (0-100)"""
        if not volume_data:
            return 50  # Neutral if no volume data
            
        volume_ratio = volume_data.get('volume_ratio', 1.0)
        
        # High volume confirms moves
        if volume_ratio >= 2.0:
            return 90
        elif volume_ratio >= 1.5:
            return 70
        elif volume_ratio >= 1.0:
            return 50
        else:
            return 30  # Low volume = weak signal
            
    def _determine_signal_type(self, confidence: float, patterns: List[Dict],
                              indicators: Dict, catalyst_data: Dict) -> str:
        """
        Determine BUY/SELL/HOLD based on all factors
        """
        # Start with confidence-based decision
        if confidence < 30:
            return 'HOLD'
            
        # Look at pattern directions
        if patterns:
            bullish_patterns = sum(1 for p in patterns if p.get('pattern_direction') == 'bullish')
            bearish_patterns = sum(1 for p in patterns if p.get('pattern_direction') == 'bearish')
            
            # Strong pattern bias
            if bullish_patterns > bearish_patterns:
                signal_direction = 'BUY'
            elif bearish_patterns > bullish_patterns:
                signal_direction = 'SELL'
            else:
                signal_direction = 'HOLD'
        else:
            signal_direction = 'HOLD'
            
        # Confirm with indicators
        rsi = indicators.get('rsi', 50)
        macd_hist = indicators.get('macd_histogram', 0)
        
        # RSI extremes can override
        if rsi < 30 and signal_direction != 'SELL':
            signal_direction = 'BUY'
        elif rsi > 70 and signal_direction != 'BUY':
            signal_direction = 'SELL'
            
        # Catalyst sentiment can strengthen or weaken signal
        catalyst_sentiment = catalyst_data.get('sentiment', 'neutral')
        if catalyst_sentiment == 'positive' and signal_direction == 'SELL':
            signal_direction = 'HOLD'  # Don't fight positive catalyst
        elif catalyst_sentiment == 'negative' and signal_direction == 'BUY':
            signal_direction = 'HOLD'  # Don't fight negative catalyst
            
        return signal_direction
        
    def _calculate_entry_exit_levels(self, signal_type: str, current_price: float,
                                   confidence: float, catalyst_data: Dict) -> Dict:
        """
        Calculate entry, stop loss, and targets
        Tighter stops with strong catalysts!
        """
        levels = {}
        
        # Entry price (can adjust for better fills)
        if signal_type == 'BUY':
            # Enter slightly below current for better fill
            levels['entry'] = current_price * 0.999
        elif signal_type == 'SELL':
            # Enter slightly above current for better fill
            levels['entry'] = current_price * 1.001
        else:
            levels['entry'] = current_price
            
        # Stop loss - tighter with strong catalysts
        catalyst_strength = self._classify_catalyst_strength(
            catalyst_data.get('score', 0)
        )
        stop_adjustment = self.risk_params['catalyst_stop_adjustment'][catalyst_strength]
        stop_loss_pct = self.risk_params['base_stop_loss_pct'] * stop_adjustment
        
        if signal_type == 'BUY':
            levels['stop_loss'] = levels['entry'] * (1 - stop_loss_pct / 100)
            risk_amount = levels['entry'] - levels['stop_loss']
            levels['target_1'] = levels['entry'] + (risk_amount * self.risk_params['target_multipliers']['target_1'])
            levels['target_2'] = levels['entry'] + (risk_amount * self.risk_params['target_multipliers']['target_2'])
        else:  # SELL
            levels['stop_loss'] = levels['entry'] * (1 + stop_loss_pct / 100)
            risk_amount = levels['stop_loss'] - levels['entry']
            levels['target_1'] = levels['entry'] - (risk_amount * self.risk_params['target_multipliers']['target_1'])
            levels['target_2'] = levels['entry'] - (risk_amount * self.risk_params['target_multipliers']['target_2'])
            
        # Calculate risk/reward
        if signal_type != 'HOLD':
            reward = abs(levels['target_1'] - levels['entry'])
            risk = abs(levels['entry'] - levels['stop_loss'])
            levels['risk_reward'] = round(reward / risk, 2) if risk > 0 else 0
        else:
            levels['risk_reward'] = 0
            
        # Round all prices
        for key in levels:
            if key != 'risk_reward':
                levels[key] = round(levels[key], 2)
                
        return levels
        
    def _classify_catalyst_strength(self, catalyst_score: float) -> str:
        """Classify catalyst as strong/moderate/weak"""
        if catalyst_score >= 70:
            return 'strong'
        elif catalyst_score >= 40:
            return 'moderate'
        else:
            return 'weak'
            
    def _calculate_position_size(self, confidence: float) -> float:
        """
        Calculate position size based on confidence
        Higher confidence = larger position
        """
        if confidence >= 70:
            return self.risk_params['position_size_map']['strong_signal'] * 100
        elif confidence >= 50:
            return self.risk_params['position_size_map']['normal_signal'] * 100
        else:
            return self.risk_params['position_size_map']['weak_signal'] * 100
            
    def _generate_key_factors(self, catalyst_data: Dict, patterns: List[Dict],
                            indicators: Dict, confidence: float) -> List[str]:
        """
        Generate human-readable reasons for the signal
        This helps traders understand WHY we're trading
        """
        factors = []
        
        # Catalyst factors
        if catalyst_data.get('type'):
            factors.append(f"{catalyst_data['type'].replace('_', ' ').title()} catalyst")
            
        # Pattern factors
        if patterns:
            top_pattern = patterns[0]
            factors.append(f"{top_pattern['pattern_name'].replace('_', ' ').title()} pattern detected")
            
        # Indicator factors
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            factors.append("RSI oversold")
        elif rsi > 70:
            factors.append("RSI overbought")
            
        # Volume factors
        volume_ratio = indicators.get('volume_data', {}).get('volume_ratio', 1)
        if volume_ratio > 2:
            factors.append("High volume confirmation")
            
        # Confidence factor
        if confidence >= 70:
            factors.append("High confidence setup")
            
        # Pre-market factor
        if catalyst_data.get('is_pre_market'):
            factors.append("Pre-market catalyst")
            
        return factors
        
    def _create_no_trade_signal(self, symbol: str, reason: str) -> Dict:
        """Create a HOLD signal with explanation"""
        return {
            'symbol': symbol,
            'signal_type': 'HOLD',
            'confidence': 0,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
    def _save_signal(self, signal: Dict):
        """Save signal to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trading_signals (
                    signal_id, symbol, signal_type, confidence,
                    catalyst_score, pattern_score, indicator_score, volume_score,
                    recommended_entry, stop_loss, target_1, target_2,
                    catalyst_type, catalyst_strength, detected_patterns,
                    key_factors, position_size_pct, risk_reward_ratio,
                    is_pre_market, ml_features
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['signal_id'], signal['symbol'], signal['signal_type'],
                signal['confidence'], signal['catalyst_score'], signal['pattern_score'],
                signal['indicator_score'], signal['volume_score'],
                signal['recommended_entry'], signal['stop_loss'],
                signal['target_1'], signal['target_2'],
                signal.get('catalyst_type'), signal.get('catalyst_strength'),
                json.dumps(signal.get('detected_patterns', [])),
                json.dumps(signal.get('key_factors', [])),
                signal.get('position_size_pct'), signal.get('risk_reward_ratio'),
                signal.get('is_pre_market', False),
                json.dumps(signal.get('ml_features', {}))
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Saved signal {signal['signal_id']} for {signal['symbol']}")
            
        except Exception as e:
            self.logger.error(f"Error saving signal: {e}")
            
    def _get_signal_performance(self, signal_type: Optional[str] = None,
                               catalyst_type: Optional[str] = None) -> Dict:
        """Get signal performance statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM signal_performance WHERE 1=1"
            params = []
            
            if signal_type:
                query += " AND signal_type = ?"
                params.append(signal_type)
                
            if catalyst_type:
                query += " AND catalyst_type = ?"
                params.append(catalyst_type)
                
            cursor.execute(query, params)
            
            stats = []
            for row in cursor.fetchall():
                stats.append({
                    'signal_type': row[1],
                    'catalyst_type': row[2],
                    'confidence_range': row[3],
                    'total_signals': row[4],
                    'success_rate': row[7],
                    'avg_win': row[9],
                    'avg_loss': row[10],
                    'profit_factor': row[11]
                })
                
            conn.close()
            
            return {'performance': stats}
            
        except Exception as e:
            self.logger.error(f"Error getting performance: {e}")
            return {'error': str(e)}
            
    def _update_signal_outcome(self, signal_id: str, outcome: Dict) -> Dict:
        """Update signal with actual trading outcome"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE trading_signals
                SET signal_outcome = ?,
                    actual_pnl = ?,
                    executed = TRUE
                WHERE signal_id = ?
            ''', (
                outcome.get('result'),  # success, stopped_out, partial_success
                outcome.get('pnl', 0),
                signal_id
            ))
            
            conn.commit()
            conn.close()
            
            # Update performance statistics
            self._update_performance_stats()
            
            return {'status': 'success', 'signal_id': signal_id}
            
        except Exception as e:
            self.logger.error(f"Error updating outcome: {e}")
            return {'error': str(e)}
            
    def _update_performance_stats(self):
        """Update signal performance statistics"""
        # This would run periodically to calculate success rates
        pass
        
    # Technical indicator calculations (fallbacks when TA-Lib not available)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI manually"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_macd(self, prices: np.ndarray) -> Dict:
        """Calculate MACD manually"""
        exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return {
            'macd': macd.iloc[-1],
            'macd_signal': signal.iloc[-1],
            'macd_histogram': histogram.iloc[-1]
        }
        
    def _calculate_sma(self, prices: np.ndarray, period: int) -> float:
        """Simple Moving Average"""
        if len(prices) < period:
            return prices[-1]
        return np.mean(prices[-period:])
        
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Exponential Moving Average"""
        return pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1]
        
    def _detect_trend(self, prices: np.ndarray) -> str:
        """Detect price trend"""
        if len(prices) < 20:
            return 'neutral'
            
        sma_20 = self._calculate_sma(prices, 20)
        sma_50 = self._calculate_sma(prices, 50) if len(prices) >= 50 else sma_20
        
        current = prices[-1]
        
        if current > sma_20 > sma_50:
            return 'bullish'
        elif current < sma_20 < sma_50:
            return 'bearish'
        else:
            return 'neutral'
            
    def _get_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get price data for analysis"""
        # In production, this would fetch from yfinance or data provider
        # For now, return mock data
        return self._get_mock_price_data(symbol)
        
    def _get_mock_price_data(self, symbol: str) -> pd.DataFrame:
        """Generate mock price data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
        
        # Generate realistic price movement
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
                    'service_name': 'technical_analysis',
                    'service_info': {
                        'url': 'http://localhost:5003',
                        'port': 5003,
                        'version': '2.0.0',
                        'capabilities': ['catalyst_weighted', 'ml_ready']
                    }
                }
            )
            if response.status_code == 200:
                self.logger.info("Successfully registered with coordination service")
        except Exception as e:
            self.logger.warning(f"Could not register with coordination: {e}")
            
    def run(self):
        """Start the Flask application"""
        self.logger.info("Starting Catalyst-Weighted Technical Analysis v2.0.0 on port 5003")
        self.logger.info("Signal generation with catalyst integration active")
        self.logger.info("Signal formula: 35% Catalyst + 35% Pattern + 20% Indicators + 10% Volume")
        
        self.app.run(host='0.0.0.0', port=5003, debug=False)


if __name__ == "__main__":
    service = CatalystWeightedTechnicalAnalysis()
    service.run()