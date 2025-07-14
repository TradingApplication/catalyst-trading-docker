#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: pattern_service.py
Version: 2.1.1
Last Updated: 2025-07-07
Purpose: Technical pattern detection and analysis with news catalyst weighting

REVISION HISTORY:
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
        
        self.logger.info("Catalyst-Aware Pattern Analysis v2.1.1 initialized",
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
                        'version': '2.1.1',
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
            return jsonify({
                "status": "healthy" if db_health['database'] else "degraded",
                "service": self.service_name,
                "version": "2.1.1",
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

    # ... rest of the methods remain the same ...
    
    def run(self):
        """Start the pattern analysis service"""
        self.logger.info("Starting Pattern Analysis Service",
                        version="2.1.1",
                        port=self.port,
                        environment=os.getenv('ENVIRONMENT', 'development'))
        
        self.app.run(host='0.0.0.0', port=self.port, debug=False)


if __name__ == "__main__":
    service = CatalystAwarePatternAnalysis()
    service.run()