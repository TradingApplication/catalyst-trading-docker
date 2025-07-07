#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: technical_service.py
Version: 2.1.2
Last Updated: 2025-07-07
Purpose: Technical analysis and signal generation with catalyst awareness

REVISION HISTORY:
v2.1.2 (2025-07-07) - Fixed multiple compatibility issues
- Fixed health check for database_utils v2.3.1 format
- Removed non-existent insert_trading_signal import
- Added inline _save_trading_signal method
- Added missing _register_with_coordination method

v2.1.1 (2025-07-07) - Fixed database utilities import
- Removed non-existent insert_trading_signal import
- Added inline database insertion for trading signals
- Fixed get_redis function name
- Added _register_with_coordination method

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
            'macd_threshold': float(os.getenv('MACD_THRESHOLD', '0')),
            'adx_trend_strength': float(os.getenv('ADX_TREND_STRENGTH', '25')),
            
            # Risk parameters
            'stop_loss_atr_multiplier': float(os.getenv('STOP_LOSS_ATR', '2.0')),
            'take_profit_atr_multiplier': float(os.getenv('TAKE_PROFIT_ATR', '3.0'))
        }
        
        # Cache settings
        self.cache_ttl = int(os.getenv('TECHNICAL_CACHE_TTL', '300'))  # 5 minutes
        
        # Register with coordination
        self._register_with_coordination()
        
        self.logger.info("Technical Analysis Service v2.1.2 initialized",
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
        
        # Signal generation parameters
        self.min_confidence = float(os.getenv('MIN_SIGNAL_CONFIDENCE', '60'))
        self.lookback_periods = int(os.getenv('TECHNICAL_LOOKBACK_PERIODS', '100'))
        
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
                    'service_name': 'technical_analysis',
                    'service_info': {
                        'url': f"http://technical-service:{self.port}",
                        'port': self.port,
                        'version': '2.1.2',
                        'capabilities': ['technical_indicators', 'signal_generation', 'risk_levels']
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
            
        @self.app.route('/analyze', methods=['POST'])
        def analyze():
            """Analyze single symbol"""
            data = request.json
            symbol = data.get('symbol')
            timeframe = data.get('timeframe', '5min')
            catalyst_data = data.get('catalyst_data', {})
            
            if not symbol:
                return jsonify({'error': 'Symbol required'}), 400
                
            result = self.analyze_with_catalyst_context(symbol, timeframe, catalyst_data)
            return jsonify(result)
            
        @self.app.route('/generate_signal', methods=['POST'])
        def generate_signal():
            """Generate trading signal with all context"""
            data = request.json
            required = ['symbol', 'patterns', 'catalyst_data']
            
            if not all(k in data for k in required):
                return jsonify({'error': f'Required fields: {required}'}), 400
                
            signal = self.generate_catalyst_weighted_signal(
                data['symbol'],
                data['patterns'],
                data['catalyst_data'],
                data.get('timeframe', '5min')
            )
            
            return jsonify(signal)
            
        @self.app.route('/batch_analyze', methods=['POST'])
        def batch_analyze():
            """Analyze multiple symbols"""
            data = request.json
            symbols = data.get('symbols', [])
            timeframe = data.get('timeframe', '5min')
            catalyst_data = data.get('catalyst_data', {})
            
            results = []
            for symbol in symbols:
                try:
                    result = self.analyze_with_catalyst_context(
                        symbol, timeframe, catalyst_data.get(symbol, {})
                    )
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error analyzing {symbol}", error=str(e))
                    
            return jsonify({'results': results})
            
        @self.app.route('/indicators/<symbol>', methods=['GET'])
        def get_indicators(symbol):
            """Get current indicators for symbol"""
            timeframe = request.args.get('timeframe', '5min')
            
            # Check cache first
            cache_key = f"indicators:{symbol}:{timeframe}"
            cached = self.redis_client.get(cache_key)
            if cached:
                return jsonify(json.loads(cached))
                
            # Calculate fresh
            price_data = self._get_price_data(symbol, timeframe)
            if price_data is None:
                return jsonify({'error': 'No data available'}), 404
                
            indicators = self._calculate_indicators(price_data)
            
            # Cache result
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(indicators)
            )
            
            return jsonify(indicators)
            
        @self.app.route('/support_resistance/<symbol>', methods=['GET'])
        def get_support_resistance(symbol):
            """Get support and resistance levels"""
            timeframe = request.args.get('timeframe', '1d')
            
            levels = self._calculate_support_resistance(symbol, timeframe)
            return jsonify(levels)

    def analyze_with_catalyst_context(self, symbol: str, timeframe: str = '5min',
                                    catalyst_data: Dict = None) -> Dict:
        """
        Perform technical analysis with catalyst awareness
        """
        self.logger.info(f"Analyzing {symbol} with catalyst context",
                        symbol=symbol,
                        timeframe=timeframe,
                        has_catalyst=bool(catalyst_data))
        
        # Get price data
        price_data = self._get_price_data(symbol, timeframe)
        if price_data is None or len(price_data) < self.lookback_periods:
            return {
                'symbol': symbol,
                'status': 'insufficient_data',
                'indicators': {}
            }
            
        # Calculate all indicators
        indicators = self._calculate_indicators(price_data)
        
        # Calculate trend
        trend = self._determine_trend(indicators, price_data)
        
        # Calculate momentum
        momentum = self._calculate_momentum(indicators)
        
        # Calculate volatility
        volatility = self._calculate_volatility(price_data)
        
        # Weight by catalyst if present
        if catalyst_data:
            catalyst_weight = self._calculate_catalyst_weight(catalyst_data)
            momentum['catalyst_adjusted'] = momentum['score'] * catalyst_weight
        else:
            momentum['catalyst_adjusted'] = momentum['score']
            
        # Compile analysis
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'timeframe': timeframe,
            'indicators': indicators,
            'trend': trend,
            'momentum': momentum,
            'volatility': volatility,
            'support_resistance': self._find_key_levels(price_data),
            'signal_strength': self._calculate_signal_strength(
                indicators, trend, momentum, catalyst_data
            )
        }
        
        return analysis
        
    def generate_catalyst_weighted_signal(self, symbol: str, patterns: List[Dict],
                                        catalyst_data: Dict, timeframe: str = '5min') -> Dict:
        """
        Generate trading signal combining patterns, indicators, and catalyst
        """
        # Get technical analysis
        ta_result = self.analyze_with_catalyst_context(symbol, timeframe, catalyst_data)
        
        if ta_result.get('status') == 'insufficient_data':
            return {
                'symbol': symbol,
                'signal': 'NO_SIGNAL',
                'reason': 'Insufficient data'
            }
            
        # Calculate combined confidence
        pattern_confidence = max([p.get('confidence', 0) for p in patterns]) if patterns else 0
        technical_confidence = ta_result['signal_strength']
        catalyst_confidence = catalyst_data.get('confidence', 50)
        
        # Weight the confidences
        weights = {
            'pattern': 0.3,
            'technical': 0.4,
            'catalyst': 0.3
        }
        
        combined_confidence = (
            pattern_confidence * weights['pattern'] +
            technical_confidence * weights['technical'] +
            catalyst_confidence * weights['catalyst']
        )
        
        # Determine signal type
        signal_type = self._determine_signal_type(
            ta_result, patterns, catalyst_data, combined_confidence
        )
        
        # Calculate entry/exit levels
        current_price = float(ta_result['indicators'].get('close', 0))
        atr = float(ta_result['volatility'].get('atr', 0))
        
        levels = self._calculate_entry_exit_levels(
            signal_type, current_price, atr, ta_result['support_resistance']
        )
        
        # Build signal
        signal = {
            'signal_id': f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'signal_type': signal_type,
            'confidence': round(combined_confidence, 2),
            'components': {
                'pattern_confidence': pattern_confidence,
                'technical_confidence': technical_confidence,
                'catalyst_confidence': catalyst_confidence
            },
            'current_price': current_price,
            'recommended_entry': levels['entry'],
            'stop_loss': levels['stop_loss'],
            'target_1': levels['target_1'],
            'target_2': levels['target_2'],
            'catalyst_type': catalyst_data.get('type'),
            'detected_patterns': [p['pattern_name'] for p in patterns],
            'key_factors': self._identify_key_factors(ta_result, patterns, catalyst_data),
            'position_size_pct': self._calculate_position_size(combined_confidence, signal_type),
            'risk_reward_ratio': levels['risk_reward_ratio'],
            'timeframe': timeframe,
            'generated_at': datetime.now().isoformat()
        }
        
        # Save signal to database inline
        self._save_trading_signal(signal)
        
        return signal
        
    def _save_trading_signal(self, signal: Dict):
        """Save trading signal to database"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO trading_signals 
                        (signal_id, symbol, generated_timestamp, signal_type, confidence,
                         catalyst_score, pattern_score, technical_score, volume_score,
                         recommended_entry, stop_loss, target_1, target_2,
                         catalyst_type, detected_patterns, key_factors,
                         position_size_pct, risk_reward_ratio)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (signal_id) DO NOTHING
                    """, (
                        signal['signal_id'],
                        signal['symbol'],
                        signal['timestamp'],
                        signal['signal_type'],
                        signal['confidence'],
                        signal['components'].get('catalyst_confidence', 0),
                        signal['components'].get('pattern_confidence', 0),
                        signal['components'].get('technical_confidence', 0),
                        0,  # volume_score - could be calculated if needed
                        signal['recommended_entry'],
                        signal['stop_loss'],
                        signal['target_1'],
                        signal['target_2'],
                        signal.get('catalyst_type'),
                        json.dumps(signal['detected_patterns']),
                        json.dumps(signal['key_factors']),
                        signal['position_size_pct'],
                        signal['risk_reward_ratio']
                    ))
                    conn.commit()
                    
        except Exception as e:
            self.logger.error(f"Failed to save signal to database", error=str(e))
            # Don't fail the signal generation if DB save fails
            
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
                
                # Bollinger Bands
                upper, middle, lower = talib.BBANDS(close,
                                                   timeperiod=self.ta_config['bb_period'],
                                                   nbdevup=self.ta_config['bb_std'],
                                                   nbdevdn=self.ta_config['bb_std'])
                indicators['bb_upper'] = float(upper[-1])
                indicators['bb_middle'] = float(middle[-1])
                indicators['bb_lower'] = float(lower[-1])
                
                # Moving averages
                indicators['sma_short'] = float(talib.SMA(close, timeperiod=self.ta_config['sma_short'])[-1])
                indicators['sma_long'] = float(talib.SMA(close, timeperiod=self.ta_config['sma_long'])[-1])
                indicators['ema_short'] = float(talib.EMA(close, timeperiod=self.ta_config['ema_short'])[-1])
                indicators['ema_long'] = float(talib.EMA(close, timeperiod=self.ta_config['ema_long'])[-1])
                
                # ATR for volatility
                indicators['atr'] = float(talib.ATR(high, low, close, timeperiod=self.ta_config['atr_period'])[-1])
                
                # ADX for trend strength
                indicators['adx'] = float(talib.ADX(high, low, close, timeperiod=self.ta_config['adx_period'])[-1])
                
                # Stochastic
                slowk, slowd = talib.STOCH(high, low, close,
                                          fastk_period=self.ta_config['stoch_period'],
                                          slowk_period=3,
                                          slowd_period=3)
                indicators['stoch_k'] = float(slowk[-1])
                indicators['stoch_d'] = float(slowd[-1])
                
            else:
                # Manual calculations as fallback
                indicators = self._calculate_indicators_manual(close, high, low, volume)
                
            # Current price info
            indicators['close'] = float(close[-1])
            indicators['volume'] = int(volume[-1])
            indicators['price_change'] = float((close[-1] - close[-2]) / close[-2] * 100)
            
            # Volume analysis
            avg_volume = np.mean(volume[:-1])
            indicators['volume_ratio'] = float(volume[-1] / avg_volume) if avg_volume > 0 else 1
            
        except Exception as e:
            self.logger.error("Error calculating indicators", error=str(e))
            
        return indicators
        
    def _calculate_indicators_manual(self, close: np.ndarray, high: np.ndarray,
                                   low: np.ndarray, volume: np.ndarray) -> Dict:
        """Manual indicator calculations when TA-Lib not available"""
        indicators = {}
        
        # RSI
        def calculate_rsi(prices, period=14):
            deltas = np.diff(prices)
            seed = deltas[:period+1]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            rs = up / down if down != 0 else 0
            rsi = np.zeros_like(prices)
            rsi[:period] = 100. - 100. / (1. + rs)
            
            for i in range(period, len(prices)):
                delta = deltas[i-1]
                if delta > 0:
                    upval = delta
                    downval = 0.
                else:
                    upval = 0.
                    downval = -delta
                    
                up = (up * (period - 1) + upval) / period
                down = (down * (period - 1) + downval) / period
                rs = up / down if down != 0 else 0
                rsi[i] = 100. - 100. / (1. + rs)
                
            return rsi
            
        indicators['rsi'] = float(calculate_rsi(close, self.ta_config['rsi_period'])[-1])
        
        # Simple moving averages
        indicators['sma_short'] = float(np.mean(close[-self.ta_config['sma_short']:]))
        indicators['sma_long'] = float(np.mean(close[-self.ta_config['sma_long']:]))
        
        # EMA approximation
        def calculate_ema(prices, period):
            ema = np.zeros_like(prices)
            ema[0] = prices[0]
            alpha = 2 / (period + 1)
            
            for i in range(1, len(prices)):
                ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
                
            return ema
            
        indicators['ema_short'] = float(calculate_ema(close, self.ta_config['ema_short'])[-1])
        indicators['ema_long'] = float(calculate_ema(close, self.ta_config['ema_long'])[-1])
        
        # MACD
        ema_fast = calculate_ema(close, self.ta_config['macd_fast'])
        ema_slow = calculate_ema(close, self.ta_config['macd_slow'])
        macd_line = ema_fast - ema_slow
        signal_line = calculate_ema(macd_line, self.ta_config['macd_signal'])
        
        indicators['macd'] = float(macd_line[-1])
        indicators['macd_signal'] = float(signal_line[-1])
        indicators['macd_histogram'] = float(macd_line[-1] - signal_line[-1])
        
        # Bollinger Bands
        bb_sma = np.mean(close[-self.ta_config['bb_period']:])
        bb_std = np.std(close[-self.ta_config['bb_period']:])
        
        indicators['bb_upper'] = float(bb_sma + self.ta_config['bb_std'] * bb_std)
        indicators['bb_middle'] = float(bb_sma)
        indicators['bb_lower'] = float(bb_sma - self.ta_config['bb_std'] * bb_std)
        
        # ATR approximation
        tr = np.maximum(high - low, 
                       np.maximum(abs(high - np.roll(close, 1)), 
                                 abs(low - np.roll(close, 1))))
        indicators['atr'] = float(np.mean(tr[-self.ta_config['atr_period']:]))
        
        # ADX (simplified)
        indicators['adx'] = 25.0  # Default neutral value
        
        # Stochastic
        lowest_low = np.min(low[-self.ta_config['stoch_period']:])
        highest_high = np.max(high[-self.ta_config['stoch_period']:])
        
        if highest_high - lowest_low > 0:
            k_percent = 100 * (close[-1] - lowest_low) / (highest_high - lowest_low)
        else:
            k_percent = 50
            
        indicators['stoch_k'] = float(k_percent)
        indicators['stoch_d'] = float(k_percent)  # Simplified
        
        return indicators
        
    def _determine_trend(self, indicators: Dict, price_data: pd.DataFrame) -> Dict:
        """Determine current trend based on indicators"""
        trend_signals = []
        
        # Moving average trend
        if indicators.get('sma_short', 0) > indicators.get('sma_long', 0):
            trend_signals.append(1)  # Bullish
        else:
            trend_signals.append(-1)  # Bearish
            
        # EMA trend
        if indicators.get('ema_short', 0) > indicators.get('ema_long', 0):
            trend_signals.append(1)
        else:
            trend_signals.append(-1)
            
        # Price vs moving averages
        current_price = indicators.get('close', 0)
        if current_price > indicators.get('sma_short', 0):
            trend_signals.append(1)
        else:
            trend_signals.append(-1)
            
        # MACD trend
        if indicators.get('macd', 0) > indicators.get('macd_signal', 0):
            trend_signals.append(1)
        else:
            trend_signals.append(-1)
            
        # Calculate trend score
        trend_score = sum(trend_signals) / len(trend_signals)
        
        # Determine trend direction
        if trend_score > 0.5:
            direction = 'bullish'
        elif trend_score < -0.5:
            direction = 'bearish'
        else:
            direction = 'neutral'
            
        # Trend strength from ADX
        adx = indicators.get('adx', 0)
        if adx > 40:
            strength = 'strong'
        elif adx > 25:
            strength = 'moderate'
        else:
            strength = 'weak'
            
        return {
            'direction': direction,
            'strength': strength,
            'score': trend_score,
            'adx': adx
        }
        
    def _calculate_momentum(self, indicators: Dict) -> Dict:
        """Calculate momentum indicators"""
        momentum_score = 0
        signals = []
        
        # RSI momentum
        rsi = indicators.get('rsi', 50)
        if rsi < self.ta_config['rsi_oversold']:
            signals.append(('rsi', 'oversold', 1))
            momentum_score += 1
        elif rsi > self.ta_config['rsi_overbought']:
            signals.append(('rsi', 'overbought', -1))
            momentum_score -= 1
        else:
            signals.append(('rsi', 'neutral', 0))
            
        # MACD momentum
        macd_hist = indicators.get('macd_histogram', 0)
        if macd_hist > 0 and indicators.get('macd', 0) > 0:
            signals.append(('macd', 'bullish', 1))
            momentum_score += 1
        elif macd_hist < 0 and indicators.get('macd', 0) < 0:
            signals.append(('macd', 'bearish', -1))
            momentum_score -= 1
        else:
            signals.append(('macd', 'neutral', 0))
            
        # Stochastic momentum
        stoch_k = indicators.get('stoch_k', 50)
        if stoch_k < 20:
            signals.append(('stochastic', 'oversold', 1))
            momentum_score += 1
        elif stoch_k > 80:
            signals.append(('stochastic', 'overbought', -1))
            momentum_score -= 1
        else:
            signals.append(('stochastic', 'neutral', 0))
            
        # Volume momentum
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 2:
            signals.append(('volume', 'high', 1))
            momentum_score += 0.5
        elif volume_ratio < 0.5:
            signals.append(('volume', 'low', -0.5))
            momentum_score -= 0.5
        else:
            signals.append(('volume', 'normal', 0))
            
        # Normalize score
        momentum_score = momentum_score / 4  # 4 indicators
        
        return {
            'score': momentum_score,
            'signals': signals,
            'direction': 'bullish' if momentum_score > 0.25 else 'bearish' if momentum_score < -0.25 else 'neutral'
        }
        
    def _calculate_volatility(self, price_data: pd.DataFrame) -> Dict:
        """Calculate volatility metrics"""
        close_prices = price_data['Close'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        
        # Historical volatility
        hist_vol = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Average True Range
        atr = self._calculate_atr(price_data)
        
        # Bollinger Band width
        bb_width = 0
        if 'bb_upper' in price_data.columns and 'bb_lower' in price_data.columns:
            bb_upper = price_data['bb_upper'].iloc[-1]
            bb_lower = price_data['bb_lower'].iloc[-1]
            bb_middle = (bb_upper + bb_lower) / 2
            bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
            
        return {
            'historical': hist_vol,
            'atr': atr,
            'bb_width': bb_width,
            'level': 'high' if hist_vol > 0.4 else 'medium' if hist_vol > 0.2 else 'low'
        }
        
    def _calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = price_data['High'].values
        low = price_data['Low'].values
        close = price_data['Close'].values
        
        tr = np.maximum(high - low, 
                       np.maximum(abs(high - np.roll(close, 1)), 
                                 abs(low - np.roll(close, 1))))
        
        # Remove first element (NaN from roll)
        tr = tr[1:]
        
        if len(tr) >= period:
            return float(np.mean(tr[-period:]))
        else:
            return float(np.mean(tr))
            
    def _find_key_levels(self, price_data: pd.DataFrame) -> Dict:
        """Find support and resistance levels"""
        high = price_data['High'].values
        low = price_data['Low'].values
        close = price_data['Close'].values
        
        # Recent highs and lows
        recent_high = np.max(high[-20:])
        recent_low = np.min(low[-20:])
        
        # Pivot points
        pivot = (recent_high + recent_low + close[-1]) / 3
        r1 = 2 * pivot - recent_low
        s1 = 2 * pivot - recent_high
        r2 = pivot + (recent_high - recent_low)
        s2 = pivot - (recent_high - recent_low)
        
        return {
            'resistance_2': float(r2),
            'resistance_1': float(r1),
            'pivot': float(pivot),
            'support_1': float(s1),
            'support_2': float(s2),
            'recent_high': float(recent_high),
            'recent_low': float(recent_low)
        }
        
    def _calculate_signal_strength(self, indicators: Dict, trend: Dict, 
                                 momentum: Dict, catalyst_data: Dict = None) -> float:
        """Calculate overall signal strength"""
        strength = 50  # Base strength
        
        # Trend contribution
        if trend['direction'] == 'bullish':
            strength += 10 if trend['strength'] == 'strong' else 5
        elif trend['direction'] == 'bearish':
            strength -= 10 if trend['strength'] == 'strong' else 5
            
        # Momentum contribution
        strength += momentum['score'] * 20
        
        # Indicator alignment
        price = indicators.get('close', 0)
        sma = indicators.get('sma_short', 0)
        
        if price > sma and trend['direction'] == 'bullish':
            strength += 5
        elif price < sma and trend['direction'] == 'bearish':
            strength += 5
            
        # Catalyst boost
        if catalyst_data:
            catalyst_score = catalyst_data.get('score', 0)
            strength += catalyst_score * 10
            
        # Ensure within bounds
        strength = max(0, min(100, strength))
        
        return strength
        
    def _calculate_catalyst_weight(self, catalyst_data: Dict) -> float:
        """Calculate weight multiplier based on catalyst strength"""
        base_weight = 1.0
        
        # Catalyst type weights
        type_weights = {
            'earnings': 1.5,
            'fda': 1.8,
            'merger': 1.6,
            'guidance': 1.4,
            'analyst': 1.2,
            'insider': 1.3
        }
        
        catalyst_type = catalyst_data.get('type')
        if catalyst_type in type_weights:
            base_weight *= type_weights[catalyst_type]
            
        # News volume weight
        news_count = catalyst_data.get('news_count', 0)
        if news_count > 10:
            base_weight *= 1.2
        elif news_count > 5:
            base_weight *= 1.1
            
        # Sentiment weight
        sentiment = catalyst_data.get('sentiment', 'neutral')
        if sentiment == 'very_positive':
            base_weight *= 1.3
        elif sentiment == 'positive':
            base_weight *= 1.1
        elif sentiment == 'negative':
            base_weight *= 0.9
        elif sentiment == 'very_negative':
            base_weight *= 0.7
            
        return base_weight
        
    def _determine_signal_type(self, ta_result: Dict, patterns: List[Dict],
                             catalyst_data: Dict, confidence: float) -> str:
        """Determine the type of signal to generate"""
        trend = ta_result.get('trend', {})
        momentum = ta_result.get('momentum', {})
        
        # Check for strong directional signals
        if trend['direction'] == 'bullish' and momentum['direction'] == 'bullish':
            if confidence > 70:
                return 'BUY'
            else:
                return 'BUY_WEAK'
        elif trend['direction'] == 'bearish' and momentum['direction'] == 'bearish':
            if confidence > 70:
                return 'SELL'
            else:
                return 'SELL_WEAK'
        else:
            return 'HOLD'
            
    def _calculate_entry_exit_levels(self, signal_type: str, current_price: float,
                                   atr: float, support_resistance: Dict) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        levels = {
            'entry': current_price,
            'stop_loss': 0,
            'target_1': 0,
            'target_2': 0,
            'risk_reward_ratio': 0
        }
        
        if signal_type in ['BUY', 'BUY_WEAK']:
            # Buy signal levels
            levels['stop_loss'] = current_price - (atr * self.ta_config['stop_loss_atr_multiplier'])
            levels['target_1'] = current_price + (atr * self.ta_config['take_profit_atr_multiplier'])
            levels['target_2'] = current_price + (atr * self.ta_config['take_profit_atr_multiplier'] * 1.5)
            
            # Adjust for support/resistance
            if support_resistance['support_1'] > levels['stop_loss']:
                levels['stop_loss'] = support_resistance['support_1'] * 0.99
                
            if support_resistance['resistance_1'] < levels['target_1']:
                levels['target_1'] = support_resistance['resistance_1'] * 0.99
                
        elif signal_type in ['SELL', 'SELL_WEAK']:
            # Sell signal levels
            levels['stop_loss'] = current_price + (atr * self.ta_config['stop_loss_atr_multiplier'])
            levels['target_1'] = current_price - (atr * self.ta_config['take_profit_atr_multiplier'])
            levels['target_2'] = current_price - (atr * self.ta_config['take_profit_atr_multiplier'] * 1.5)
            
            # Adjust for support/resistance
            if support_resistance['resistance_1'] < levels['stop_loss']:
                levels['stop_loss'] = support_resistance['resistance_1'] * 1.01
                
            if support_resistance['support_1'] > levels['target_1']:
                levels['target_1'] = support_resistance['support_1'] * 1.01
                
        # Calculate risk/reward
        risk = abs(current_price - levels['stop_loss'])
        reward = abs(levels['target_1'] - current_price)
        
        if risk > 0:
            levels['risk_reward_ratio'] = round(reward / risk, 2)
        else:
            levels['risk_reward_ratio'] = 0
            
        return levels
        
    def _identify_key_factors(self, ta_result: Dict, patterns: List[Dict],
                            catalyst_data: Dict) -> List[str]:
        """Identify key factors driving the signal"""
        factors = []
        
        # Trend factors
        trend = ta_result.get('trend', {})
        if trend['direction'] != 'neutral' and trend['strength'] in ['moderate', 'strong']:
            factors.append(f"{trend['strength']} {trend['direction']} trend")
            
        # Momentum factors
        momentum = ta_result.get('momentum', {})
        for signal_name, status, _ in momentum.get('signals', []):
            if status in ['oversold', 'overbought']:
                factors.append(f"{signal_name} {status}")
                
        # Pattern factors
        for pattern in patterns[:2]:  # Top 2 patterns
            if pattern.get('final_confidence', 0) > 70:
                factors.append(f"{pattern['pattern_name']} pattern")
                
        # Catalyst factors
        if catalyst_data:
            catalyst_type = catalyst_data.get('type')
            if catalyst_type:
                factors.append(f"{catalyst_type} catalyst")
                
        # Volume factor
        indicators = ta_result.get('indicators', {})
        if indicators.get('volume_ratio', 1) > 2:
            factors.append("high volume")
            
        return factors[:5]  # Limit to 5 key factors
        
    def _calculate_position_size(self, confidence: float, signal_type: str) -> float:
        """Calculate recommended position size based on confidence"""
        if signal_type in ['BUY', 'SELL']:
            if confidence > 80:
                return 15.0  # 15% of portfolio
            elif confidence > 70:
                return 10.0  # 10% of portfolio
            else:
                return 5.0   # 5% of portfolio
        else:
            return 0.0  # No position for weak signals or holds
            
    def _calculate_support_resistance(self, symbol: str, timeframe: str) -> Dict:
        """Calculate detailed support and resistance levels"""
        price_data = self._get_price_data(symbol, timeframe)
        
        if price_data is None or len(price_data) < 20:
            return {}
            
        return self._find_key_levels(price_data)
        
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
        """Start the technical analysis service"""
        self.logger.info("Starting Technical Analysis Service",
                        version="2.1.2",
                        port=self.port,
                        environment=os.getenv('ENVIRONMENT', 'development'))
        
        self.app.run(host='0.0.0.0', port=self.port, debug=False)


if __name__ == "__main__":
    service = TechnicalAnalysisService()
    service.run()