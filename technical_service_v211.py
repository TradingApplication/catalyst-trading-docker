#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: technical_service.py
Version: 2.1.1
Last Updated: 2025-07-07
Purpose: Technical analysis and signal generation with catalyst awareness

REVISION HISTORY:
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

# Import database utilities - Fixed imports
from database_utils import (
    get_db_connection,
    get_redis,  # Fixed from get_redis_connection
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
        
        self.logger.info("Technical Analysis Service v2.1.1 initialized",
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
                        'version': '2.1.1',
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
            return jsonify({
                "status": "healthy" if db_health['database'] else "degraded",
                "service": self.service_name,
                "version": "2.1.1",
                "database": db_health['database'],
                "redis": db_health['redis'],
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
        
        if ta_result['status'] == 'insufficient_data':
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
            
    # ... rest of the methods remain the same ...
    
    def run(self):
        """Start the technical analysis service"""
        self.logger.info("Starting Technical Analysis Service",
                        version="2.1.1",
                        port=self.port,
                        environment=os.getenv('ENVIRONMENT', 'development'))
        
        self.app.run(host='0.0.0.0', port=self.port, debug=False)


if __name__ == "__main__":
    service = TechnicalAnalysisService()
    service.run()