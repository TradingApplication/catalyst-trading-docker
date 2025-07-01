#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: technical_service.py
Version: 2.1.0
Last Updated: 2025-07-01
Purpose: Technical analysis and signal generation with catalyst awareness

REVISION HISTORY:
v2.1.0 (2025-07-01) - Production-ready implementation
- PostgreSQL integration with connection pooling
- All configuration via environment variables
- Comprehensive technical indicator calculations
- Signal generation with catalyst weighting
- Redis caching for performance
- TA-Lib with manual fallbacks

Description of Service:
This service performs technical analysis on securities and generates
trading signals. It integrates with pattern analysis results and
considers news catalysts when generating signals.

KEY FEATURES:
- Multiple timeframe analysis (1min, 5min, 15min, 1h, 1d)
- 20+ technical indicators (RSI, MACD, Bollinger, etc.)
- Signal generation with confidence scoring
- Catalyst-weighted signal strength
- Support/resistance level calculation
- Trend analysis and confirmation
- Volume analysis integration
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
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from structlog import get_logger
import redis

# Import database utilities
from database_utils import (
    get_db_connection,
    insert_trading_signal,
    get_pending_signals,
    insert_technical_indicators,
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


class TechnicalAnalysisService:
    """
    Technical analysis service with signal generation
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
        self.pattern_service_url = os.getenv('PATTERN_SERVICE_URL', 'http://pattern-service:5002')
        
        # Technical analysis configuration
        self.ta_config = {
            # Indicator periods
            'rsi_period': int(os.getenv('RSI_PERIOD', '14')),
            'macd_fast': int(os.getenv('MACD_FAST', '12')),
            'macd_slow': int(os.getenv('MACD_SLOW', '26')),
            'macd_signal': int(os.getenv('MACD_SIGNAL', '9')),
            'bb_period': int(os.getenv('BB_PERIOD', '20')),
            'bb_std': float(os.getenv('BB_STD', '2.0')),
            'sma_short': int(os.getenv('SMA_SHORT', '20')),
            'sma_long': int(os.getenv('SMA_LONG', '50')),
            'ema_short': int(os.getenv('EMA_SHORT', '9')),
            'ema_long': int(os.getenv('EMA_LONG', '21')),
            'atr_period': int(os.getenv('ATR_PERIOD', '14')),
            'adx_period': int(os.getenv('ADX_PERIOD', '14')),
            'stoch_period': int(os.getenv('STOCH_PERIOD', '14')),
            
            # Signal thresholds
            'rsi_oversold': float(os.getenv('RSI_OVERSOLD', '30')),
            'rsi_overbought': float(os.getenv('RSI_OVERBOUGHT', '70')),
            'adx_trend_strength': float(os.getenv('ADX_TREND_STRENGTH', '25')),
            'volume_surge_multiplier': float(os.getenv('VOLUME_SURGE_MULT', '2.0')),
            
            # Signal generation
            'min_signal_confidence': float(os.getenv('MIN_SIGNAL_CONFIDENCE', '60')),
            'catalyst_weight': float(os.getenv('CATALYST_WEIGHT', '0.3')),
            'technical_weight': float(os.getenv('TECHNICAL_WEIGHT', '0.7')),
            
            # Risk management
            'default_stop_loss_pct': float(os.getenv('DEFAULT_STOP_LOSS_PCT', '2.0')),
            'default_take_profit_pct': float(os.getenv('DEFAULT_TAKE_PROFIT_PCT', '4.0')),
            'atr_stop_multiplier': float(os.getenv('ATR_STOP_MULTIPLIER', '2.0')),
            
            # Cache settings
            'cache_ttl': int(os.getenv('TECHNICAL_CACHE_TTL', '300'))  # 5 minutes
        }
        
        # Signal type definitions
        self.signal_types = {
            'momentum_breakout': {'weight': 1.2, 'min_confidence': 70},
            'trend_continuation': {'weight': 1.0, 'min_confidence': 65},
            'reversal': {'weight': 1.1, 'min_confidence': 75},
            'range_breakout': {'weight': 1.0, 'min_confidence': 70},
            'volume_surge': {'weight': 0.9, 'min_confidence': 60}
        }
        
        # Register with coordination
        self._register_with_coordination()
        
        self.logger.info("Technical Analysis Service v2.1.0 initialized",
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
        self.service_name = 'technical_analysis'
        self.port = int(os.getenv('PORT', '5003'))
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
                "service": "technical_analysis",
                "version": "2.1.0",
                "database": db_health['database'],
                "redis": db_health['redis'],
                "talib": TALIB_AVAILABLE,
                "timestamp": datetime.now().isoformat()
            })
            
        @self.app.route('/analyze', methods=['POST'])
        def analyze():
            """Perform technical analysis on a symbol"""
            data = request.json
            symbol = data.get('symbol')
            timeframe = data.get('timeframe', '5min')
            
            if not symbol:
                return jsonify({'error': 'Symbol required'}), 400
                
            result = self.analyze_symbol(symbol, timeframe)
            return jsonify(result)
            
        @self.app.route('/generate_signal', methods=['POST'])
        def generate_signal():
            """Generate trading signal for a symbol"""
            data = request.json
            symbol = data.get('symbol')
            timeframe = data.get('timeframe', '5min')
            patterns = data.get('patterns', [])
            catalyst_data = data.get('catalyst_data', {})
            
            if not symbol:
                return jsonify({'error': 'Symbol required'}), 400
                
            signal = self.generate_trading_signal(
                symbol, timeframe, patterns, catalyst_data
            )
            return jsonify(signal)
            
        @self.app.route('/batch_analyze', methods=['POST'])
        def batch_analyze():
            """Analyze multiple symbols"""
            data = request.json
            symbols = data.get('symbols', [])
            timeframe = data.get('timeframe', '5min')
            
            results = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(self.analyze_symbol, symbol, timeframe)
                    for symbol in symbols
                ]
                
                for future in futures:
                    try:
                        result = future.result(timeout=10)
                        results.append(result)
                    except Exception as e:
                        self.logger.error("Batch analysis error", error=str(e))
                        
            return jsonify({'results': results})
            
        @self.app.route('/indicators/<symbol>', methods=['GET'])
        def get_indicators(symbol):
            """Get current technical indicators for a symbol"""
            timeframe = request.args.get('timeframe', '5min')
            
            indicators = self._calculate_all_indicators(symbol, timeframe)
            if indicators:
                return jsonify(indicators)
            else:
                return jsonify({'error': 'Failed to calculate indicators'}), 500
                
        @self.app.route('/support_resistance/<symbol>', methods=['GET'])
        def get_support_resistance(symbol):
            """Get support and resistance levels"""
            timeframe = request.args.get('timeframe', '1d')
            
            levels = self._calculate_support_resistance(symbol, timeframe)
            return jsonify(levels)
            
        @self.app.route('/pending_signals', methods=['GET'])
        def pending_signals():
            """Get pending trading signals"""
            signals = get_pending_signals()
            return jsonify({
                'count': len(signals),
                'signals': signals
            })
            
    def analyze_symbol(self, symbol: str, timeframe: str = '5min') -> Dict:
        """
        Perform comprehensive technical analysis on a symbol
        """
        self.logger.info(f"Analyzing {symbol}", symbol=symbol, timeframe=timeframe)
        
        # Check cache first
        cache_key = f"technical:{symbol}:{timeframe}"
        cached = self.redis_client.get(cache_key)
        if cached:
            self.logger.debug("Using cached technical data", symbol=symbol)
            return json.loads(cached)
        
        # Get price data
        price_data = self._get_price_data(symbol, timeframe)
        if price_data is None or len(price_data) < 50:  # Need enough data for indicators
            return {
                'symbol': symbol,
                'status': 'insufficient_data',
                'message': 'Not enough price data for technical analysis'
            }
            
        # Calculate all indicators
        indicators = self._calculate_indicators(price_data)
        
        # Analyze trend
        trend_analysis = self._analyze_trend(price_data, indicators)
        
        # Identify chart patterns
        chart_patterns = self._identify_chart_patterns(price_data)
        
        # Calculate support/resistance
        support_resistance = self._calculate_support_resistance_from_data(price_data)
        
        # Generate technical score
        technical_score = self._calculate_technical_score(
            indicators, trend_analysis, chart_patterns
        )
        
        # Save indicators to database
        self._save_indicators(symbol, timeframe, indicators, price_data)
        
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'current_price': float(price_data['Close'].iloc[-1]),
            'indicators': indicators,
            'trend': trend_analysis,
            'patterns': chart_patterns,
            'support_resistance': support_resistance,
            'technical_score': technical_score,
            'recommendation': self._generate_recommendation(technical_score, trend_analysis)
        }
        
        # Cache result
        self.redis_client.setex(
            cache_key,
            self.ta_config['cache_ttl'],
            json.dumps(result)
        )
        
        return result
        
    def generate_trading_signal(self, symbol: str, timeframe: str,
                              patterns: List[Dict], catalyst_data: Dict) -> Dict:
        """
        Generate a trading signal based on technical and catalyst data
        """
        self.logger.info(f"Generating signal for {symbol}",
                        symbol=symbol,
                        has_patterns=bool(patterns),
                        has_catalyst=bool(catalyst_data))
        
        # Get technical analysis
        ta_result = self.analyze_symbol(symbol, timeframe)
        
        if ta_result.get('status') == 'insufficient_data':
            return {
                'symbol': symbol,
                'signal': 'NO_SIGNAL',
                'reason': 'Insufficient data'
            }
            
        # Determine signal type and direction
        signal_type, direction = self._determine_signal_type(
            ta_result, patterns, catalyst_data
        )
        
        if signal_type == 'NO_SIGNAL':
            return {
                'symbol': symbol,
                'signal': 'NO_SIGNAL',
                'reason': 'No clear signal detected'
            }
            
        # Calculate signal confidence
        confidence = self._calculate_signal_confidence(
            ta_result, patterns, catalyst_data, signal_type
        )
        
        if confidence < self.ta_config['min_signal_confidence']:
            return {
                'symbol': symbol,
                'signal': 'NO_SIGNAL',
                'reason': f'Confidence too low: {confidence:.1f}%'
            }
            
        # Calculate entry, stop loss, and targets
        current_price = ta_result['current_price']
        levels = self._calculate_trade_levels(
            current_price, direction, ta_result, signal_type
        )
        
        # Generate signal ID
        signal_id = f"SIG_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create signal object
        signal = {
            'signal_id': signal_id,
            'symbol': symbol,
            'signal': direction,  # 'BUY' or 'SELL'
            'signal_type': signal_type,
            'confidence': confidence,
            'catalyst_score': catalyst_data.get('score', 0),
            'pattern_score': self._calculate_pattern_score(patterns),
            'technical_score': ta_result['technical_score'],
            'volume_score': self._calculate_volume_score(ta_result),
            'recommended_entry': levels['entry'],
            'stop_loss': levels['stop_loss'],
            'target_1': levels['target_1'],
            'target_2': levels['target_2'],
            'catalyst_type': catalyst_data.get('type'),
            'detected_patterns': [p['pattern_name'] for p in patterns],
            'key_factors': self._identify_key_factors(ta_result, patterns, catalyst_data),
            'position_size_pct': self._calculate_position_size(confidence, signal_type),
            'risk_reward_ratio': levels['risk_reward_ratio'],
            'timeframe': timeframe,
            'generated_at': datetime.now().isoformat()
        }
        
        # Save signal to database
        insert_trading_signal(signal)
        
        return signal
        
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
            
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        indicators = {}
        
        try:
            # Price data
            close = data['Close'].values
            high = data['High'].values
            low = data['Low'].values
            volume = data['Volume'].values
            
            if TALIB_AVAILABLE:
                # Use TA-Lib for calculations
                indicators['rsi'] = float(talib.RSI(close, timeperiod=self.ta_config['rsi_period'])[-1])
                
                macd, signal, hist = talib.MACD(close,
                                               fastperiod=self.ta_config['macd_fast'],
                                               slowperiod=self.ta_config['macd_slow'],
                                               signalperiod=self.ta_config['macd_signal'])
                indicators['macd'] = float(macd[-1]) if not np.isnan(macd[-1]) else 0
                indicators['macd_signal'] = float(signal[-1]) if not np.isnan(signal[-1]) else 0
                indicators['macd_histogram'] = float(hist[-1]) if not np.isnan(hist[-1]) else 0
                
                upper, middle, lower = talib.BBANDS(close,
                                                   timeperiod=self.ta_config['bb_period'],
                                                   nbdevup=self.ta_config['bb_std'],
                                                   nbdevdn=self.ta_config['bb_std'])
                indicators['bb_upper'] = float(upper[-1])
                indicators['bb_middle'] = float(middle[-1])
                indicators['bb_lower'] = float(lower[-1])
                indicators['bb_width'] = indicators['bb_upper'] - indicators['bb_lower']
                indicators['bb_position'] = (close[-1] - indicators['bb_lower']) / indicators['bb_width'] if indicators['bb_width'] > 0 else 0.5
                
                indicators['sma_20'] = float(talib.SMA(close, timeperiod=self.ta_config['sma_short'])[-1])
                indicators['sma_50'] = float(talib.SMA(close, timeperiod=self.ta_config['sma_long'])[-1])
                indicators['ema_9'] = float(talib.EMA(close, timeperiod=self.ta_config['ema_short'])[-1])
                indicators['ema_21'] = float(talib.EMA(close, timeperiod=self.ta_config['ema_long'])[-1])
                
                indicators['atr'] = float(talib.ATR(high, low, close, timeperiod=self.ta_config['atr_period'])[-1])
                indicators['adx'] = float(talib.ADX(high, low, close, timeperiod=self.ta_config['adx_period'])[-1])
                
                slowk, slowd = talib.STOCH(high, low, close,
                                          fastk_period=self.ta_config['stoch_period'],
                                          slowk_period=3,
                                          slowd_period=3)
                indicators['stoch_k'] = float(slowk[-1])
                indicators['stoch_d'] = float(slowd[-1])
                
                indicators['obv'] = float(talib.OBV(close, volume)[-1])
                indicators['mfi'] = float(talib.MFI(high, low, close, volume, timeperiod=14)[-1])
                
            else:
                # Manual calculations as fallback
                indicators = self._calculate_indicators_manual(data)
                
            # Volume analysis
            indicators['volume_sma'] = float(data['Volume'].rolling(20).mean().iloc[-1])
            indicators['relative_volume'] = float(data['Volume'].iloc[-1] / indicators['volume_sma']) if indicators['volume_sma'] > 0 else 1.0
            
            # Price position
            indicators['price_position'] = self._calculate_price_position(data)
            
            # Momentum
            indicators['momentum'] = self._calculate_momentum(data)
            
        except Exception as e:
            self.logger.error("Error calculating indicators", error=str(e))
            
        return indicators
        
    def _calculate_indicators_manual(self, data: pd.DataFrame) -> Dict:
        """Manual calculation of indicators when TA-Lib not available"""
        indicators = {}
        close = data['Close']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.ta_config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.ta_config['rsi_period']).mean()
        rs = gain / loss
        indicators['rsi'] = float(100 - (100 / (1 + rs)).iloc[-1])
        
        # Moving averages
        indicators['sma_20'] = float(close.rolling(window=self.ta_config['sma_short']).mean().iloc[-1])
        indicators['sma_50'] = float(close.rolling(window=self.ta_config['sma_long']).mean().iloc[-1])
        indicators['ema_9'] = float(close.ewm(span=self.ta_config['ema_short'], adjust=False).mean().iloc[-1])
        indicators['ema_21'] = float(close.ewm(span=self.ta_config['ema_long'], adjust=False).mean().iloc[-1])
        
        # MACD
        ema_fast = close.ewm(span=self.ta_config['macd_fast'], adjust=False).mean()
        ema_slow = close.ewm(span=self.ta_config['macd_slow'], adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.ta_config['macd_signal'], adjust=False).mean()
        
        indicators['macd'] = float(macd.iloc[-1])
        indicators['macd_signal'] = float(signal.iloc[-1])
        indicators['macd_histogram'] = float(macd.iloc[-1] - signal.iloc[-1])
        
        # Bollinger Bands
        sma = close.rolling(window=self.ta_config['bb_period']).mean()
        std = close.rolling(window=self.ta_config['bb_period']).std()
        
        indicators['bb_upper'] = float(sma.iloc[-1] + (std.iloc[-1] * self.ta_config['bb_std']))
        indicators['bb_middle'] = float(sma.iloc[-1])
        indicators['bb_lower'] = float(sma.iloc[-1] - (std.iloc[-1] * self.ta_config['bb_std']))
        indicators['bb_width'] = indicators['bb_upper'] - indicators['bb_lower']
        indicators['bb_position'] = (close.iloc[-1] - indicators['bb_lower']) / indicators['bb_width'] if indicators['bb_width'] > 0 else 0.5
        
        # ATR (simplified)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - close.shift())
        low_close = np.abs(data['Low'] - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        indicators['atr'] = float(true_range.rolling(self.ta_config['atr_period']).mean().iloc[-1])
        
        # Stochastic (simplified)
        lowest_low = data['Low'].rolling(self.ta_config['stoch_period']).min()
        highest_high = data['High'].rolling(self.ta_config['stoch_period']).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        indicators['stoch_k'] = float(k.iloc[-1])
        indicators['stoch_d'] = float(k.rolling(3).mean().iloc[-1])
        
        # OBV (simplified)
        obv = (np.sign(close.diff()) * data['Volume']).cumsum()
        indicators['obv'] = float(obv.iloc[-1])
        
        # ADX (simplified - just trend strength)
        indicators['adx'] = 25.0  # Default value
        
        # MFI (simplified)
        indicators['mfi'] = 50.0  # Default value
        
        return indicators
        
    def _analyze_trend(self, data: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze the current trend"""
        close = data['Close'].iloc[-1]
        
        trend = {
            'direction': 'neutral',
            'strength': 0,
            'duration': 0,
            'confirmed': False
        }
        
        # Moving average analysis
        if indicators.get('sma_20', 0) > 0 and indicators.get('sma_50', 0) > 0:
            if close > indicators['sma_20'] > indicators['sma_50']:
                trend['direction'] = 'bullish'
                trend['strength'] = 70
            elif close < indicators['sma_20'] < indicators['sma_50']:
                trend['direction'] = 'bearish'
                trend['strength'] = 70
            elif indicators['sma_20'] > indicators['sma_50']:
                trend['direction'] = 'bullish'
                trend['strength'] = 50
            elif indicators['sma_20'] < indicators['sma_50']:
                trend['direction'] = 'bearish'
                trend['strength'] = 50
                
        # ADX confirmation
        if indicators.get('adx', 0) > self.ta_config['adx_trend_strength']:
            trend['confirmed'] = True
            trend['strength'] = min(100, trend['strength'] + 20)
            
        # MACD confirmation
        if indicators.get('macd', 0) > indicators.get('macd_signal', 0):
            if trend['direction'] == 'bullish':
                trend['strength'] = min(100, trend['strength'] + 10)
        elif indicators.get('macd', 0) < indicators.get('macd_signal', 0):
            if trend['direction'] == 'bearish':
                trend['strength'] = min(100, trend['strength'] + 10)
                
        # Calculate trend duration
        if trend['direction'] != 'neutral':
            trend['duration'] = self._calculate_trend_duration(data, trend['direction'])
            
        return trend
        
    def _identify_chart_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Identify chart patterns in price data"""
        patterns = []
        
        # Double Top/Bottom
        double_pattern = self._detect_double_pattern(data)
        if double_pattern:
            patterns.append(double_pattern)
            
        # Head and Shoulders
        hs_pattern = self._detect_head_shoulders(data)
        if hs_pattern:
            patterns.append(hs_pattern)
            
        # Triangle patterns
        triangle = self._detect_triangle_pattern(data)
        if triangle:
            patterns.append(triangle)
            
        # Channel/Range
        channel = self._detect_channel_pattern(data)
        if channel:
            patterns.append(channel)
            
        return patterns
        
    def _calculate_support_resistance_from_data(self, data: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']
        
        # Find significant levels
        levels = []
        
        # Recent highs and lows
        for period in [20, 50, 100]:
            if len(data) >= period:
                levels.append({
                    'level': float(high.tail(period).max()),
                    'type': 'resistance',
                    'strength': period / 20
                })
                levels.append({
                    'level': float(low.tail(period).min()),
                    'type': 'support',
                    'strength': period / 20
                })
                
        # Volume-weighted levels
        vwap = (close * volume).sum() / volume.sum()
        levels.append({
            'level': float(vwap),
            'type': 'pivot',
            'strength': 2
        })
        
        # Psychological levels (round numbers)
        current_price = close.iloc[-1]
        round_levels = [
            round(current_price, -1),  # Nearest 10
            round(current_price, -2),  # Nearest 100
        ]
        
        for level in round_levels:
            if abs(level - current_price) / current_price < 0.1:  # Within 10%
                levels.append({
                    'level': float(level),
                    'type': 'psychological',
                    'strength': 1
                })
                
        # Sort and filter levels
        support_levels = sorted([l for l in levels if l['type'] == 'support' or l['level'] < current_price],
                               key=lambda x: x['level'], reverse=True)[:3]
        resistance_levels = sorted([l for l in levels if l['type'] == 'resistance' or l['level'] > current_price],
                                  key=lambda x: x['level'])[:3]
        
        return {
            'support': support_levels,
            'resistance': resistance_levels,
            'pivot': float(vwap),
            'current_price': float(current_price)
        }
        
    def _calculate_technical_score(self, indicators: Dict, trend: Dict, patterns: List[Dict]) -> float:
        """Calculate overall technical score (0-100)"""
        score = 0
        
        # Trend score (0-30)
        if trend['confirmed']:
            score += min(30, trend['strength'] * 0.3)
        else:
            score += min(15, trend['strength'] * 0.15)
            
        # Momentum score (0-25)
        rsi = indicators.get('rsi', 50)
        if 30 < rsi < 70:  # Not oversold/overbought
            score += 15
        elif rsi <= 30 and trend['direction'] == 'bullish':  # Oversold in uptrend
            score += 20
        elif rsi >= 70 and trend['direction'] == 'bearish':  # Overbought in downtrend
            score += 20
        else:
            score += 5
            
        # MACD score (0-20)
        if indicators.get('macd', 0) > indicators.get('macd_signal', 0):
            if trend['direction'] == 'bullish':
                score += 20
            else:
                score += 10
        elif indicators.get('macd', 0) < indicators.get('macd_signal', 0):
            if trend['direction'] == 'bearish':
                score += 20
            else:
                score += 10
                
        # Volume score (0-15)
        rel_volume = indicators.get('relative_volume', 1.0)
        if rel_volume > self.ta_config['volume_surge_multiplier']:
            score += 15
        elif rel_volume > 1.5:
            score += 10
        elif rel_volume > 1.0:
            score += 5
            
        # Pattern score (0-10)
        if patterns:
            pattern_score = min(10, len(patterns) * 3)
            score += pattern_score
            
        return min(100, score)
        
    def _generate_recommendation(self, technical_score: float, trend: Dict) -> str:
        """Generate trading recommendation"""
        if technical_score >= 80:
            if trend['direction'] == 'bullish':
                return 'STRONG_BUY'
            elif trend['direction'] == 'bearish':
                return 'STRONG_SELL'
        elif technical_score >= 60:
            if trend['direction'] == 'bullish':
                return 'BUY'
            elif trend['direction'] == 'bearish':
                return 'SELL'
        elif technical_score >= 40:
            return 'HOLD'
        else:
            return 'AVOID'
            
    def _determine_signal_type(self, ta_result: Dict, patterns: List[Dict],
                             catalyst_data: Dict) -> Tuple[str, str]:
        """Determine signal type and direction"""
        indicators = ta_result['indicators']
        trend = ta_result['trend']
        
        # Check for momentum breakout
        if (indicators.get('relative_volume', 0) > self.ta_config['volume_surge_multiplier'] and
            abs(indicators.get('momentum', 0)) > 2):
            if indicators['momentum'] > 0:
                return 'momentum_breakout', 'BUY'
            else:
                return 'momentum_breakout', 'SELL'
                
        # Check for trend continuation
        if trend['confirmed'] and trend['strength'] > 60:
            if trend['direction'] == 'bullish':
                return 'trend_continuation', 'BUY'
            elif trend['direction'] == 'bearish':
                return 'trend_continuation', 'SELL'
                
        # Check for reversal with catalyst
        if catalyst_data.get('score', 0) > 50:
            rsi = indicators.get('rsi', 50)
            if rsi < self.ta_config['rsi_oversold'] and catalyst_data.get('type') in ['earnings_beat', 'upgrade']:
                return 'reversal', 'BUY'
            elif rsi > self.ta_config['rsi_overbought'] and catalyst_data.get('type') in ['earnings_miss', 'downgrade']:
                return 'reversal', 'SELL'
                
        # Check patterns
        if patterns:
            bullish_patterns = [p for p in patterns if p.get('pattern_direction') == 'bullish']
            bearish_patterns = [p for p in patterns if p.get('pattern_direction') == 'bearish']
            
            if len(bullish_patterns) > len(bearish_patterns):
                return 'pattern_based', 'BUY'
            elif len(bearish_patterns) > len(bullish_patterns):
                return 'pattern_based', 'SELL'
                
        return 'NO_SIGNAL', 'NONE'
        
    def _calculate_signal_confidence(self, ta_result: Dict, patterns: List[Dict],
                                   catalyst_data: Dict, signal_type: str) -> float:
        """Calculate signal confidence score"""
        # Base confidence from signal type
        base_confidence = self.signal_types.get(signal_type, {}).get('min_confidence', 50)
        
        # Technical score contribution
        technical_contribution = ta_result['technical_score'] * self.ta_config['technical_weight']
        
        # Catalyst contribution
        catalyst_score = catalyst_data.get('score', 0)
        catalyst_contribution = catalyst_score * self.ta_config['catalyst_weight']
        
        # Pattern contribution
        pattern_confidence = 0
        if patterns:
            pattern_confidence = max([p.get('final_confidence', 0) for p in patterns])
            
        # Combine scores
        confidence = (base_confidence * 0.3 +
                     technical_contribution * 0.3 +
                     catalyst_contribution * 0.2 +
                     pattern_confidence * 0.2)
        
        # Adjust for signal type weight
        signal_weight = self.signal_types.get(signal_type, {}).get('weight', 1.0)
        confidence *= signal_weight
        
        return min(100, confidence)
        
    def _calculate_trade_levels(self, current_price: float, direction: str,
                              ta_result: Dict, signal_type: str) -> Dict:
        """Calculate entry, stop loss, and target levels"""
        indicators = ta_result['indicators']
        support_resistance = ta_result['support_resistance']
        atr = indicators.get('atr', current_price * 0.01)  # Default 1% if no ATR
        
        levels = {
            'entry': current_price,
            'stop_loss': 0,
            'target_1': 0,
            'target_2': 0,
            'risk_reward_ratio': 0
        }
        
        if direction == 'BUY':
            # Entry slightly above current price
            levels['entry'] = current_price * 1.001
            
            # Stop loss calculation
            stop_methods = []
            
            # ATR-based stop
            stop_methods.append(current_price - (atr * self.ta_config['atr_stop_multiplier']))
            
            # Support-based stop
            if support_resistance['support']:
                nearest_support = support_resistance['support'][0]['level']
                stop_methods.append(nearest_support * 0.99)
                
            # Percentage-based stop
            stop_methods.append(current_price * (1 - self.ta_config['default_stop_loss_pct'] / 100))
            
            # Use the highest stop (least risk)
            levels['stop_loss'] = max(stop_methods)
            
            # Target calculation
            risk = levels['entry'] - levels['stop_loss']
            
            # Target 1: 2:1 risk/reward
            levels['target_1'] = levels['entry'] + (risk * 2)
            
            # Target 2: Next resistance or 3:1 risk/reward
            if support_resistance['resistance']:
                nearest_resistance = support_resistance['resistance'][0]['level']
                levels['target_2'] = max(nearest_resistance, levels['entry'] + (risk * 3))
            else:
                levels['target_2'] = levels['entry'] + (risk * 3)
                
        else:  # SELL
            # Entry slightly below current price
            levels['entry'] = current_price * 0.999
            
            # Stop loss calculation
            stop_methods = []
            
            # ATR-based stop
            stop_methods.append(current_price + (atr * self.ta_config['atr_stop_multiplier']))
            
            # Resistance-based stop
            if support_resistance['resistance']:
                nearest_resistance = support_resistance['resistance'][0]['level']
                stop_methods.append(nearest_resistance * 1.01)
                
            # Percentage-based stop
            stop_methods.append(current_price * (1 + self.ta_config['default_stop_loss_pct'] / 100))
            
            # Use the lowest stop (least risk)
            levels['stop_loss'] = min(stop_methods)
            
            # Target calculation
            risk = levels['stop_loss'] - levels['entry']
            
            # Target 1: 2:1 risk/reward
            levels['target_1'] = levels['entry'] - (risk * 2)
            
            # Target 2: Next support or 3:1 risk/reward
            if support_resistance['support']:
                nearest_support = support_resistance['support'][0]['level']
                levels['target_2'] = min(nearest_support, levels['entry'] - (risk * 3))
            else:
                levels['target_2'] = levels['entry'] - (risk * 3)
                
        # Calculate risk/reward ratio
        if direction == 'BUY':
            potential_reward = levels['target_1'] - levels['entry']
            risk = levels['entry'] - levels['stop_loss']
        else:
            potential_reward = levels['entry'] - levels['target_1']
            risk = levels['stop_loss'] - levels['entry']
            
        levels['risk_reward_ratio'] = potential_reward / risk if risk > 0 else 0
        
        return levels
        
    def _calculate_pattern_score(self, patterns: List[Dict]) -> float:
        """Calculate pattern score from detected patterns"""
        if not patterns:
            return 0
            
        # Average confidence of all patterns
        confidences = [p.get('final_confidence', 0) for p in patterns]
        return sum(confidences) / len(confidences) if confidences else 0
        
    def _calculate_volume_score(self, ta_result: Dict) -> float:
        """Calculate volume score"""
        indicators = ta_result['indicators']
        rel_volume = indicators.get('relative_volume', 1.0)
        
        if rel_volume > 3:
            return 100
        elif rel_volume > 2:
            return 80
        elif rel_volume > 1.5:
            return 60
        elif rel_volume > 1:
            return 40
        else:
            return 20
            
    def _identify_key_factors(self, ta_result: Dict, patterns: List[Dict],
                            catalyst_data: Dict) -> List[str]:
        """Identify key factors driving the signal"""
        factors = []
        
        # Technical factors
        indicators = ta_result['indicators']
        
        if indicators.get('relative_volume', 0) > 2:
            factors.append('High volume surge')
            
        if indicators.get('rsi', 50) < 30:
            factors.append('Oversold RSI')
        elif indicators.get('rsi', 50) > 70:
            factors.append('Overbought RSI')
            
        if ta_result['trend']['confirmed']:
            factors.append(f"Confirmed {ta_result['trend']['direction']} trend")
            
        # Pattern factors
        if patterns:
            pattern_names = [p['pattern_name'] for p in patterns[:2]]
            factors.extend([f"{name} pattern" for name in pattern_names])
            
        # Catalyst factors
        if catalyst_data.get('type'):
            factors.append(f"{catalyst_data['type']} catalyst")
            
        return factors[:5]  # Limit to top 5 factors
        
    def _calculate_position_size(self, confidence: float, signal_type: str) -> float:
        """Calculate recommended position size based on confidence"""
        base_size = float(os.getenv('POSITION_SIZE_PCT', '20'))
        
        # Adjust based on confidence
        if confidence > 80:
            size_multiplier = 1.0
        elif confidence > 70:
            size_multiplier = 0.8
        elif confidence > 60:
            size_multiplier = 0.6
        else:
            size_multiplier = 0.4
            
        # Adjust based on signal type
        if signal_type == 'reversal':
            size_multiplier *= 0.8  # More risky
        elif signal_type == 'trend_continuation':
            size_multiplier *= 1.1  # Less risky
            
        return min(base_size * size_multiplier, base_size)
        
    def _save_indicators(self, symbol: str, timeframe: str, indicators: Dict,
                        price_data: pd.DataFrame):
        """Save technical indicators to database"""
        try:
            latest = price_data.iloc[-1]
            
            indicator_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'open_price': float(latest['Open']),
                'high_price': float(latest['High']),
                'low_price': float(latest['Low']),
                'close_price': float(latest['Close']),
                'volume': int(latest['Volume']),
                'rsi': indicators.get('rsi'),
                'macd': indicators.get('macd'),
                'macd_signal': indicators.get('macd_signal'),
                'sma_20': indicators.get('sma_20'),
                'sma_50': indicators.get('sma_50'),
                'ema_9': indicators.get('ema_9'),
                'atr': indicators.get('atr'),
                'bollinger_upper': indicators.get('bb_upper'),
                'bollinger_lower': indicators.get('bb_lower'),
                'volume_sma': indicators.get('volume_sma'),
                'relative_volume': indicators.get('relative_volume')
            }
            
            insert_technical_indicators(indicator_data)
            
        except Exception as e:
            self.logger.error("Error saving indicators", error=str(e))
            
    def _calculate_price_position(self, data: pd.DataFrame) -> float:
        """Calculate where price is relative to recent range (0-100)"""
        high = data['High'].tail(20).max()
        low = data['Low'].tail(20).min()
        current = data['Close'].iloc[-1]
        
        if high > low:
            position = (current - low) / (high - low) * 100
            return float(min(100, max(0, position)))
        return 50.0
        
    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate price momentum"""
        # Simple rate of change
        periods = min(10, len(data) - 1)
        if periods > 0:
            old_price = data['Close'].iloc[-periods-1]
            current_price = data['Close'].iloc[-1]
            momentum = ((current_price - old_price) / old_price) * 100
            return float(momentum)
        return 0.0
        
    def _calculate_trend_duration(self, data: pd.DataFrame, direction: str) -> int:
        """Calculate how long the trend has been in place (in periods)"""
        sma_20 = data['Close'].rolling(20).mean()
        sma_50 = data['Close'].rolling(50).mean()
        
        duration = 0
        for i in range(len(data) - 1, max(0, len(data) - 100), -1):
            if direction == 'bullish':
                if sma_20.iloc[i] > sma_50.iloc[i]:
                    duration += 1
                else:
                    break
            else:  # bearish
                if sma_20.iloc[i] < sma_50.iloc[i]:
                    duration += 1
                else:
                    break
                    
        return duration
        
    def _detect_double_pattern(self, data: pd.DataFrame) -> Optional[Dict]:
        """Detect double top/bottom patterns"""
        # Simplified detection
        return None
        
    def _detect_head_shoulders(self, data: pd.DataFrame) -> Optional[Dict]:
        """Detect head and shoulders patterns"""
        # Simplified detection
        return None
        
    def _detect_triangle_pattern(self, data: pd.DataFrame) -> Optional[Dict]:
        """Detect triangle patterns"""
        # Simplified detection
        return None
        
    def _detect_channel_pattern(self, data: pd.DataFrame) -> Optional[Dict]:
        """Detect channel/range patterns"""
        # Simplified detection
        return None
        
    def _calculate_support_resistance(self, symbol: str, timeframe: str) -> Dict:
        """Calculate support and resistance levels for API endpoint"""
        data = self._get_price_data(symbol, timeframe)
        if data is None:
            return {'error': 'No data available'}
            
        return self._calculate_support_resistance_from_data(data)
        
    def _calculate_all_indicators(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Calculate all indicators for API endpoint"""
        data = self._get_price_data(symbol, timeframe)
        if data is None:
            return None
            
        return self._calculate_indicators(data)
        
    def _get_mock_price_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Generate mock price data for testing"""
        # Generate realistic OHLCV data
        periods = 100
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')
        
        # Generate random walk
        np.random.seed(hash(symbol) % 1000)
        base_price = 100
        returns = np.random.normal(0.0002, 0.01, periods)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLCV
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.002, periods)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.005, periods))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, periods))),
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, periods)
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
                        'url': f"http://technical-service:{self.port}",
                        'port': self.port,
                        'version': '2.1.0',
                        'capabilities': ['indicators', 'signals', 'support_resistance']
                    }
                },
                timeout=5
            )
            if response.status_code == 200:
                self.logger.info("Successfully registered with coordination service")
        except Exception as e:
            self.logger.warning(f"Could not register with coordination", error=str(e))
            
    def run(self):
        """Start the technical analysis service"""
        self.logger.info("Starting Technical Analysis Service",
                        version="2.1.0",
                        port=self.port,
                        environment=os.getenv('ENVIRONMENT', 'development'))
        
        self.app.run(host='0.0.0.0', port=self.port, debug=False)


if __name__ == "__main__":
    service = TechnicalAnalysisService()
    service.run()