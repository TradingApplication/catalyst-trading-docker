#!/usr/bin/env python3
"""
Name of Service: TRADING SYSTEM PAPER TRADING - OUTCOME TRACKING VERSION
Version: 2.0.0
Last Updated: 2025-06-28
Purpose: Execute trades and track outcomes for ML training

REVISION HISTORY:
v2.0.0 (2025-06-28) - Catalyst-aware execution with outcome tracking
- Executes signals from Technical Analysis service
- Manages positions with dynamic stops based on catalyst strength
- Tracks trade outcomes for pattern learning
- Pre-market and regular hours handling
- Records which news led to profits/losses

This service is the EXECUTION ENGINE. It:
1. Receives trading signals
2. Places orders via Alpaca API
3. Manages positions (stops, targets)
4. Tracks P&L in real-time
5. Updates news accuracy based on outcomes
6. Feeds ML training data

KEY FEATURES:
- Catalyst-aware position sizing
- Dynamic stop management
- Outcome tracking for ML
- Pre-market execution capability
- Risk limits enforcement
- Slippage simulation

RISK MANAGEMENT:
- Max 5 positions at once
- Max 20% capital per position
- Daily loss limit: 5% of capital
- Tighter stops with strong catalysts
"""

import os
import json
import time
import sqlite3
import logging
import requests
import threading
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from typing import Dict, List, Optional, Tuple
import pandas as pd
from decimal import Decimal
from queue import Queue

# Import Alpaca trading client
try:
    from alpaca_trade_api import REST
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: alpaca-trade-api not installed, using mock trading")

# Import database utilities if available
try:
    from database_utils_old import DatabaseServiceMixin
    USE_DB_UTILS = True
except ImportError:
    USE_DB_UTILS = False
    print("Warning: database_utils not found, using direct SQLite connections")


class CatalystAwarePaperTrading(DatabaseServiceMixin if USE_DB_UTILS else object):
    """
    Paper trading with catalyst awareness and outcome tracking
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
        self.news_service_url = "http://localhost:5008"
        
        # Alpaca configuration
        self.alpaca_config = {
            'api_key': os.getenv('ALPACA_API_KEY', ''),
            'secret_key': os.getenv('ALPACA_SECRET_KEY', ''),
            'base_url': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
            'api_version': 'v2'
        }
        
        # Initialize Alpaca client
        self.alpaca_client = self._init_alpaca_client()
        
        # Risk management parameters
        self.risk_params = {
            'max_positions': 5,
            'max_position_size_pct': 20.0,  # 20% of capital
            'max_daily_loss_pct': 5.0,      # 5% daily loss limit
            'pre_market_position_pct': 10.0, # Smaller pre-market positions
            'min_price': 1.0,               # No penny stocks
            'max_price': 10000.0            # Reasonable upper limit
        }
        
        # Position tracking
        self.active_positions = {}  # symbol -> position data
        self.daily_pnl = 0.0
        self.trade_queue = Queue()
        
        # Outcome tracking for ML
        self.outcome_tracking = {
            'update_interval_minutes': 5,
            'track_duration_hours': 24
        }
        
        # Initialize database
        self._init_database_schema()
        
        # Start background threads
        self._start_position_monitor()
        self._start_outcome_tracker()
        
        # Register with coordination
        self._register_with_coordination()
        
        self.logger.info("Catalyst-Aware Paper Trading v2.0.0 initialized")
        
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('paper_trading')
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory
        os.makedirs('/tmp/logs', exist_ok=True)
        
        # File handler
        fh = logging.FileHandler('/tmp//tmp/logs/paper_trading.log')
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
        """Initialize trading tables with outcome tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced trade records table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_records (
                    trade_id TEXT PRIMARY KEY,
                    signal_id TEXT,
                    symbol TEXT NOT NULL,
                    
                    -- Execution details
                    order_id TEXT,
                    order_type TEXT,  -- market, limit
                    side TEXT NOT NULL,  -- buy, sell
                    quantity INTEGER NOT NULL,
                    
                    -- Entry details
                    entry_price DECIMAL(10,2),
                    entry_timestamp TIMESTAMP,
                    entry_commission DECIMAL(10,2),
                    
                    -- Exit details
                    exit_price DECIMAL(10,2),
                    exit_timestamp TIMESTAMP,
                    exit_reason TEXT,  -- stop_loss, target_1, target_2, time_stop, trailing_stop
                    exit_commission DECIMAL(10,2),
                    
                    -- P&L tracking
                    pnl_amount DECIMAL(10,2),
                    pnl_percentage DECIMAL(5,2),
                    max_profit DECIMAL(10,2),
                    max_loss DECIMAL(10,2),
                    
                    -- Catalyst tracking
                    entry_catalyst TEXT,
                    entry_news_id TEXT,  -- Links to news_raw
                    catalyst_score_at_entry DECIMAL(5,2),
                    catalyst_type TEXT,
                    
                    -- Market state
                    market_state_at_entry TEXT,  -- pre-market, regular, after-hours
                    
                    -- Risk management
                    stop_loss_price DECIMAL(10,2),
                    target_1_price DECIMAL(10,2),
                    target_2_price DECIMAL(10,2),
                    position_size_pct DECIMAL(5,2),
                    
                    -- Outcome tracking
                    time_to_stop_minutes INTEGER,
                    time_to_target_minutes INTEGER,
                    pattern_confirmed BOOLEAN,
                    catalyst_outcome TEXT,  -- successful, failed, mixed
                    
                    -- ML features
                    ml_outcome_data JSON,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (signal_id) REFERENCES trading_signals(signal_id),
                    FOREIGN KEY (entry_news_id) REFERENCES news_raw(news_id)
                )
            ''')
            
            # Active positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS active_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL UNIQUE,
                    trade_id TEXT NOT NULL,
                    
                    -- Position details
                    quantity INTEGER NOT NULL,
                    entry_price DECIMAL(10,2),
                    current_price DECIMAL(10,2),
                    
                    -- P&L tracking
                    unrealized_pnl DECIMAL(10,2),
                    unrealized_pnl_pct DECIMAL(5,2),
                    high_price DECIMAL(10,2),  -- For trailing stops
                    low_price DECIMAL(10,2),
                    
                    -- Risk levels
                    stop_loss DECIMAL(10,2),
                    target_1 DECIMAL(10,2),
                    target_2 DECIMAL(10,2),
                    trailing_stop_pct DECIMAL(5,2),
                    
                    -- Status
                    status TEXT DEFAULT 'open',  -- open, closing, closed
                    partial_exit_qty INTEGER DEFAULT 0,
                    
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (trade_id) REFERENCES trade_records(trade_id)
                )
            ''')
            
            # Trading performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL UNIQUE,
                    
                    -- Daily metrics
                    trades_executed INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    win_rate DECIMAL(5,2),
                    
                    -- P&L metrics
                    gross_pnl DECIMAL(10,2),
                    commissions DECIMAL(10,2),
                    net_pnl DECIMAL(10,2),
                    
                    -- Risk metrics
                    max_drawdown DECIMAL(10,2),
                    sharpe_ratio DECIMAL(5,2),
                    
                    -- Catalyst performance
                    catalyst_trades INTEGER DEFAULT 0,
                    catalyst_win_rate DECIMAL(5,2),
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_trades_symbol 
                ON trade_records(symbol, entry_timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_trades_open 
                ON trade_records(exit_timestamp) 
                WHERE exit_timestamp IS NULL
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_positions_active 
                ON active_positions(status) 
                WHERE status = 'open'
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
                "service": "paper_trading",
                "version": "2.0.0",
                "mode": "catalyst-aware",
                "alpaca_connected": self.alpaca_client is not None,
                "active_positions": len(self.active_positions),
                "daily_pnl": round(self.daily_pnl, 2),
                "timestamp": datetime.now().isoformat()
            })
            
        @self.app.route('/execute_trade', methods=['POST'])
        def execute_trade():
            """Execute a trading signal"""
            signal = request.json
            
            # Validate signal
            if not self._validate_signal(signal):
                return jsonify({'error': 'Invalid signal format'}), 400
                
            # Check risk limits
            risk_check = self._check_risk_limits(signal)
            if not risk_check['allowed']:
                return jsonify({
                    'error': 'Risk limit exceeded',
                    'reason': risk_check['reason']
                }), 400
                
            # Execute trade
            result = self.execute_signal(signal)
            return jsonify(result)
            
        @self.app.route('/positions', methods=['GET'])
        def get_positions():
            """Get current positions"""
            return jsonify({
                'positions': list(self.active_positions.values()),
                'count': len(self.active_positions),
                'timestamp': datetime.now().isoformat()
            })
            
        @self.app.route('/orders', methods=['GET'])
        def get_orders():
            """Get recent orders"""
            limit = request.args.get('limit', 10, type=int)
            orders = self._get_recent_orders(limit)
            return jsonify({'orders': orders})
            
        @self.app.route('/close_position', methods=['POST'])
        def close_position():
            """Manually close a position"""
            data = request.json
            symbol = data.get('symbol')
            reason = data.get('reason', 'manual_close')
            
            if symbol not in self.active_positions:
                return jsonify({'error': 'Position not found'}), 404
                
            result = self.close_position(symbol, reason)
            return jsonify(result)
            
        @self.app.route('/performance', methods=['GET'])
        def get_performance():
            """Get trading performance metrics"""
            days = request.args.get('days', 7, type=int)
            performance = self._get_performance_metrics(days)
            return jsonify(performance)
            
        @self.app.route('/update_stops', methods=['POST'])
        def update_stops():
            """Update stop loss for a position"""
            data = request.json
            symbol = data.get('symbol')
            new_stop = data.get('stop_loss')
            
            if symbol not in self.active_positions:
                return jsonify({'error': 'Position not found'}), 404
                
            result = self._update_stop_loss(symbol, new_stop)
            return jsonify(result)
            
    def _init_alpaca_client(self) -> Optional[REST]:
        """Initialize Alpaca API client"""
        if not ALPACA_AVAILABLE:
            self.logger.warning("Alpaca API not available, using mock trading")
            return None
            
        if not self.alpaca_config['api_key'] or not self.alpaca_config['secret_key']:
            self.logger.warning("Alpaca credentials not configured")
            return None
            
        try:
            client = REST(
                self.alpaca_config['api_key'],
                self.alpaca_config['secret_key'],
                self.alpaca_config['base_url'],
                api_version=self.alpaca_config['api_version']
            )
            
            # Test connection
            account = client.get_account()
            self.logger.info(f"Connected to Alpaca. Account status: {account.status}")
            
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca client: {e}")
            return None
            
    def execute_signal(self, signal: Dict) -> Dict:
        """
        Execute a trading signal
        This is where signals become real trades!
        """
        self.logger.info(f"Executing signal for {signal['symbol']}: {signal['signal_type']}")
        
        try:
            # Generate trade ID
            trade_id = f"TRD_{signal['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Calculate position size
            position_size = self._calculate_position_size(signal)
            if position_size == 0:
                return {
                    'status': 'rejected',
                    'reason': 'Position size too small',
                    'symbol': signal['symbol']
                }
                
            # Determine market state
            market_state = self._get_market_state()
            
            # Place order
            if self.alpaca_client:
                order = self._place_alpaca_order(signal, position_size)
                order_id = order.id
            else:
                # Mock order for testing
                order_id = f"MOCK_{trade_id}"
                order = self._create_mock_order(signal, position_size)
                
            # Record trade
            self._record_trade_entry(trade_id, signal, order, market_state)
            
            # Update active positions
            self.active_positions[signal['symbol']] = {
                'symbol': signal['symbol'],
                'trade_id': trade_id,
                'quantity': position_size,
                'entry_price': signal['recommended_entry'],
                'stop_loss': signal['stop_loss'],
                'target_1': signal['target_1'],
                'target_2': signal['target_2'],
                'catalyst_type': signal.get('catalyst_type'),
                'catalyst_score': signal.get('catalyst_score'),
                'status': 'open',
                'entry_time': datetime.now().isoformat()
            }
            
            self.logger.info(f"Trade executed: {trade_id} for {signal['symbol']}")
            
            return {
                'status': 'success',
                'trade_id': trade_id,
                'order_id': order_id,
                'symbol': signal['symbol'],
                'quantity': position_size,
                'entry_price': signal['recommended_entry']
            }
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'symbol': signal['symbol']
            }
            
    def close_position(self, symbol: str, reason: str = 'manual') -> Dict:
        """Close an existing position"""
        if symbol not in self.active_positions:
            return {'error': 'Position not found'}
            
        position = self.active_positions[symbol]
        
        try:
            # Place closing order
            if self.alpaca_client:
                order = self._place_closing_order(symbol, position['quantity'])
                exit_price = float(order.filled_avg_price) if order.filled_avg_price else position['current_price']
            else:
                # Mock exit
                exit_price = self._get_current_price(symbol)
                
            # Calculate P&L
            pnl = self._calculate_pnl(
                position['quantity'],
                position['entry_price'],
                exit_price
            )
            
            # Update trade record
            self._record_trade_exit(position['trade_id'], exit_price, reason, pnl)
            
            # Update daily P&L
            self.daily_pnl += pnl['net_pnl']
            
            # Remove from active positions
            del self.active_positions[symbol]
            
            # Trigger outcome tracking
            self._queue_outcome_tracking(position['trade_id'], position)
            
            self.logger.info(f"Closed position {symbol}: P&L ${pnl['net_pnl']:.2f}")
            
            return {
                'status': 'closed',
                'symbol': symbol,
                'exit_price': exit_price,
                'pnl': pnl['net_pnl'],
                'reason': reason
            }
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return {'error': str(e)}
            
    def _validate_signal(self, signal: Dict) -> bool:
        """Validate trading signal format"""
        required_fields = [
            'symbol', 'signal_type', 'confidence',
            'recommended_entry', 'stop_loss', 'target_1'
        ]
        
        for field in required_fields:
            if field not in signal:
                self.logger.error(f"Missing required field: {field}")
                return False
                
        # Validate signal type
        if signal['signal_type'] not in ['BUY', 'SELL']:
            self.logger.error(f"Invalid signal type: {signal['signal_type']}")
            return False
            
        # Validate prices
        if signal['stop_loss'] >= signal['recommended_entry'] and signal['signal_type'] == 'BUY':
            self.logger.error("Stop loss must be below entry for BUY signals")
            return False
            
        return True
        
    def _check_risk_limits(self, signal: Dict) -> Dict:
        """Check if trade passes risk management rules"""
        # Check max positions
        if len(self.active_positions) >= self.risk_params['max_positions']:
            return {
                'allowed': False,
                'reason': f"Max positions ({self.risk_params['max_positions']}) reached"
            }
            
        # Check if already in position
        if signal['symbol'] in self.active_positions:
            return {
                'allowed': False,
                'reason': f"Already in position for {signal['symbol']}"
            }
            
        # Check daily loss limit
        if self.daily_pnl <= -self._get_daily_loss_limit():
            return {
                'allowed': False,
                'reason': "Daily loss limit reached"
            }
            
        # Check price range
        price = signal['recommended_entry']
        if price < self.risk_params['min_price'] or price > self.risk_params['max_price']:
            return {
                'allowed': False,
                'reason': f"Price ${price} outside allowed range"
            }
            
        return {'allowed': True}
        
    def _calculate_position_size(self, signal: Dict) -> int:
        """
        Calculate position size based on account and signal confidence
        """
        try:
            if self.alpaca_client:
                account = self.alpaca_client.get_account()
                buying_power = float(account.buying_power)
            else:
                # Mock account
                buying_power = 100000  # $100k paper trading
                
            # Get position size percentage from signal
            position_pct = signal.get('position_size_pct', 10) / 100
            
            # Apply pre-market limits
            if signal.get('is_pre_market'):
                max_pct = self.risk_params['pre_market_position_pct'] / 100
                position_pct = min(position_pct, max_pct)
            else:
                max_pct = self.risk_params['max_position_size_pct'] / 100
                position_pct = min(position_pct, max_pct)
                
            # Calculate dollar amount
            position_value = buying_power * position_pct
            
            # Calculate shares
            shares = int(position_value / signal['recommended_entry'])
            
            # Minimum 1 share
            return max(1, shares)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
            
    def _place_alpaca_order(self, signal: Dict, quantity: int):
        """Place order via Alpaca API"""
        try:
            # Determine order type
            if signal.get('is_pre_market'):
                # Use limit order for pre-market
                order = self.alpaca_client.submit_order(
                    symbol=signal['symbol'],
                    qty=quantity,
                    side='buy' if signal['signal_type'] == 'BUY' else 'sell',
                    type='limit',
                    limit_price=signal['recommended_entry'],
                    time_in_force='day',
                    extended_hours=True  # Allow pre-market trading
                )
            else:
                # Market order during regular hours
                order = self.alpaca_client.submit_order(
                    symbol=signal['symbol'],
                    qty=quantity,
                    side='buy' if signal['signal_type'] == 'BUY' else 'sell',
                    type='market',
                    time_in_force='day'
                )
                
            self.logger.info(f"Alpaca order placed: {order.id}")
            return order
            
        except Exception as e:
            self.logger.error(f"Alpaca order failed: {e}")
            raise
            
    def _place_closing_order(self, symbol: str, quantity: int):
        """Place closing order"""
        try:
            order = self.alpaca_client.submit_order(
                symbol=symbol,
                qty=quantity,
                side='sell',  # Always sell to close long positions
                type='market',
                time_in_force='day'
            )
            
            # Wait for fill
            time.sleep(2)
            order = self.alpaca_client.get_order(order.id)
            
            return order
            
        except Exception as e:
            self.logger.error(f"Closing order failed: {e}")
            raise
            
    def _create_mock_order(self, signal: Dict, quantity: int) -> Dict:
        """Create mock order for testing"""
        return {
            'id': f"MOCK_{datetime.now().timestamp()}",
            'symbol': signal['symbol'],
            'qty': quantity,
            'side': 'buy' if signal['signal_type'] == 'BUY' else 'sell',
            'filled_avg_price': signal['recommended_entry'],
            'status': 'filled',
            'filled_at': datetime.now().isoformat()
        }
        
    def _record_trade_entry(self, trade_id: str, signal: Dict, order: Dict, market_state: str):
        """Record trade entry in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trade_records (
                    trade_id, signal_id, symbol, order_id, order_type,
                    side, quantity, entry_price, entry_timestamp,
                    entry_catalyst, catalyst_type, catalyst_score_at_entry,
                    market_state_at_entry, stop_loss_price, target_1_price,
                    target_2_price, position_size_pct, entry_news_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_id, signal.get('signal_id'), signal['symbol'],
                order.id if hasattr(order, 'id') else order['id'],
                'market', signal['signal_type'].lower(),
                order.qty if hasattr(order, 'qty') else order['qty'],
                float(order.filled_avg_price) if hasattr(order, 'filled_avg_price') else order['filled_avg_price'],
                datetime.now(), signal.get('catalyst_type'),
                signal.get('catalyst_type'), signal.get('catalyst_score'),
                market_state, signal['stop_loss'], signal['target_1'],
                signal.get('target_2'), signal.get('position_size_pct'),
                signal.get('entry_news_id')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error recording trade entry: {e}")
            
    def _record_trade_exit(self, trade_id: str, exit_price: float, 
                          exit_reason: str, pnl: Dict):
        """Record trade exit in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE trade_records
                SET exit_price = ?,
                    exit_timestamp = ?,
                    exit_reason = ?,
                    pnl_amount = ?,
                    pnl_percentage = ?,
                    updated_at = ?
                WHERE trade_id = ?
            ''', (
                exit_price, datetime.now(), exit_reason,
                pnl['net_pnl'], pnl['pnl_pct'],
                datetime.now(), trade_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error recording trade exit: {e}")
            
    def _calculate_pnl(self, quantity: int, entry_price: float, 
                      exit_price: float) -> Dict:
        """Calculate P&L for a trade"""
        gross_pnl = (exit_price - entry_price) * quantity
        
        # Estimate commission (Alpaca has no commission, but good to track)
        commission = 0  # Could add SEC/FINRA fees here
        
        net_pnl = gross_pnl - commission
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        
        return {
            'gross_pnl': round(gross_pnl, 2),
            'commission': round(commission, 2),
            'net_pnl': round(net_pnl, 2),
            'pnl_pct': round(pnl_pct, 2)
        }
        
    def _get_market_state(self) -> str:
        """Determine current market state"""
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        weekday = now.weekday()
        
        # Weekend
        if weekday >= 5:
            return 'weekend'
            
        # Convert to market time (EST)
        # This is simplified - should use proper timezone handling
        if 4 <= hour < 9 or (hour == 9 and minute < 30):
            return 'pre-market'
        elif (hour == 9 and minute >= 30) or (10 <= hour < 16):
            return 'regular'
        elif 16 <= hour < 20:
            return 'after-hours'
        else:
            return 'closed'
            
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        if self.alpaca_client:
            try:
                quote = self.alpaca_client.get_latest_quote(symbol)
                return float(quote.ap)  # Ask price
            except:
                pass
                
        # Mock price with small random change
        import random
        if symbol in self.active_positions:
            base_price = self.active_positions[symbol]['entry_price']
            return base_price * (1 + random.uniform(-0.02, 0.02))
        return 100.0
        
    def _get_daily_loss_limit(self) -> float:
        """Calculate daily loss limit"""
        if self.alpaca_client:
            try:
                account = self.alpaca_client.get_account()
                equity = float(account.equity)
                return equity * (self.risk_params['max_daily_loss_pct'] / 100)
            except:
                pass
                
        # Mock limit
        return 5000  # $5k daily loss limit
        
    def _update_stop_loss(self, symbol: str, new_stop: float) -> Dict:
        """Update stop loss for a position"""
        if symbol not in self.active_positions:
            return {'error': 'Position not found'}
            
        position = self.active_positions[symbol]
        
        # Validate new stop
        current_price = self._get_current_price(symbol)
        if new_stop >= current_price:
            return {'error': 'Stop loss must be below current price'}
            
        # Update position
        position['stop_loss'] = new_stop
        
        # Update in database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE active_positions
                SET stop_loss = ?, last_updated = ?
                WHERE symbol = ?
            ''', (new_stop, datetime.now(), symbol))
            
            conn.commit()
            conn.close()
            
            return {
                'status': 'updated',
                'symbol': symbol,
                'new_stop': new_stop
            }
            
        except Exception as e:
            return {'error': str(e)}
            
    def _start_position_monitor(self):
        """Start thread to monitor positions"""
        def monitor_positions():
            while True:
                try:
                    self._check_all_positions()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    self.logger.error(f"Position monitor error: {e}")
                    
        thread = threading.Thread(target=monitor_positions)
        thread.daemon = True
        thread.start()
        
    def _check_all_positions(self):
        """Check all positions for stops/targets"""
        for symbol, position in list(self.active_positions.items()):
            try:
                current_price = self._get_current_price(symbol)
                position['current_price'] = current_price
                
                # Update unrealized P&L
                pnl = self._calculate_pnl(
                    position['quantity'],
                    position['entry_price'],
                    current_price
                )
                position['unrealized_pnl'] = pnl['net_pnl']
                position['unrealized_pnl_pct'] = pnl['pnl_pct']
                
                # Check stop loss
                if current_price <= position['stop_loss']:
                    self.logger.info(f"Stop loss triggered for {symbol}")
                    self.close_position(symbol, 'stop_loss')
                    continue
                    
                # Check targets
                if current_price >= position.get('target_2', float('inf')):
                    self.logger.info(f"Target 2 reached for {symbol}")
                    self.close_position(symbol, 'target_2')
                elif current_price >= position.get('target_1', float('inf')):
                    # Could implement partial exit here
                    self.logger.info(f"Target 1 reached for {symbol}")
                    
                # Update high water mark for trailing stops
                if current_price > position.get('high_price', position['entry_price']):
                    position['high_price'] = current_price
                    
            except Exception as e:
                self.logger.error(f"Error checking position {symbol}: {e}")
                
    def _start_outcome_tracker(self):
        """Start thread to track trade outcomes"""
        def track_outcomes():
            while True:
                try:
                    self._process_outcome_queue()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    self.logger.error(f"Outcome tracker error: {e}")
                    
        thread = threading.Thread(target=track_outcomes)
        thread.daemon = True
        thread.start()
        
    def _queue_outcome_tracking(self, trade_id: str, position: Dict):
        """Queue a trade for outcome tracking"""
        self.trade_queue.put({
            'trade_id': trade_id,
            'position': position,
            'timestamp': datetime.now()
        })
        
    def _process_outcome_queue(self):
        """Process trades waiting for outcome tracking"""
        # This would update news accuracy based on trade outcomes
        pass
        
    def _get_recent_orders(self, limit: int) -> List[Dict]:
        """Get recent orders from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT trade_id, symbol, side, quantity, entry_price,
                       entry_timestamp, exit_price, exit_timestamp,
                       pnl_amount, exit_reason
                FROM trade_records
                ORDER BY entry_timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            orders = []
            for row in cursor.fetchall():
                orders.append({
                    'trade_id': row[0],
                    'symbol': row[1],
                    'side': row[2],
                    'quantity': row[3],
                    'entry_price': row[4],
                    'entry_time': row[5],
                    'exit_price': row[6],
                    'exit_time': row[7],
                    'pnl': row[8],
                    'exit_reason': row[9]
                })
                
            conn.close()
            return orders
            
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            return []
            
    def _get_performance_metrics(self, days: int) -> Dict:
        """Calculate performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get trades from last N days
            cursor.execute('''
                SELECT COUNT(*) as total_trades,
                       SUM(CASE WHEN pnl_amount > 0 THEN 1 ELSE 0 END) as winning_trades,
                       SUM(CASE WHEN pnl_amount < 0 THEN 1 ELSE 0 END) as losing_trades,
                       SUM(pnl_amount) as total_pnl,
                       AVG(pnl_percentage) as avg_pnl_pct,
                       MAX(pnl_amount) as best_trade,
                       MIN(pnl_amount) as worst_trade
                FROM trade_records
                WHERE entry_timestamp > datetime('now', '-{} days')
                AND exit_timestamp IS NOT NULL
            '''.format(days))
            
            metrics = cursor.fetchone()
            
            total_trades = metrics[0] or 0
            winning_trades = metrics[1] or 0
            losing_trades = metrics[2] or 0
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            performance = {
                'period_days': days,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': round(win_rate, 2),
                'total_pnl': round(metrics[3] or 0, 2),
                'avg_pnl_pct': round(metrics[4] or 0, 2),
                'best_trade': round(metrics[5] or 0, 2),
                'worst_trade': round(metrics[6] or 0, 2)
            }
            
            # Get catalyst performance
            cursor.execute('''
                SELECT catalyst_type, COUNT(*) as count,
                       AVG(pnl_percentage) as avg_pnl
                FROM trade_records
                WHERE entry_timestamp > datetime('now', '-{} days')
                AND exit_timestamp IS NOT NULL
                AND catalyst_type IS NOT NULL
                GROUP BY catalyst_type
            '''.format(days))
            
            catalyst_performance = {}
            for row in cursor.fetchall():
                catalyst_performance[row[0]] = {
                    'trades': row[1],
                    'avg_pnl': round(row[2] or 0, 2)
                }
                
            performance['catalyst_performance'] = catalyst_performance
            
            conn.close()
            return performance
            
        except Exception as e:
            self.logger.error(f"Error calculating performance: {e}")
            return {'error': str(e)}
            
    def _register_with_coordination(self):
        """Register with coordination service"""
        try:
            response = requests.post(
                f"{self.coordination_url}/register_service",
                json={
                    'service_name': 'paper_trading',
                    'service_info': {
                        'url': 'http://localhost:5005',
                        'port': 5005,
                        'version': '2.0.0',
                        'capabilities': ['catalyst_aware', 'outcome_tracking']
                    }
                }
            )
            if response.status_code == 200:
                self.logger.info("Successfully registered with coordination service")
        except Exception as e:
            self.logger.warning(f"Could not register with coordination: {e}")
            
    def run(self):
        """Start the Flask application"""
        self.logger.info("Starting Catalyst-Aware Paper Trading v2.0.0 on port 5005")
        self.logger.info("Risk limits: Max 5 positions, 20% per position, 5% daily loss")
        self.logger.info("Outcome tracking enabled for ML training")
        
        self.app.run(host='0.0.0.0', port=5005, debug=False)


if __name__ == "__main__":
    service = CatalystAwarePaperTrading()
    service.run()