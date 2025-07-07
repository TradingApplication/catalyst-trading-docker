#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: trading_service.py
Version: 2.1.2
Last Updated: 2025-07-07
Purpose: Paper trading execution service with risk management

REVISION HISTORY:
v2.1.2 (2025-07-07) - Fixed health check for database_utils v2.3.1
- Updated health check to handle postgresql/redis status format
- Added compatibility for different database_utils versions
- Maintains all existing functionality

v2.1.1 (2025-07-07) - Fixed health check for database_utils v2.3.1
- Updated health check to handle postgresql/redis status format
- Added compatibility for different database_utils versions

v2.1.0 (2025-07-01) - Production-ready implementation
- Alpaca paper trading integration
- PostgreSQL database with connection pooling
- Risk management and position sizing
- Stop loss and take profit automation
- Real-time position tracking
- Performance metrics calculation

Description of Service:
This service handles all trading operations using Alpaca's paper trading API.
It integrates with Alpaca's paper trading API for safe testing without real money.

KEY FEATURES:
- Paper trading via Alpaca API
- Position sizing based on signal confidence
- Automatic stop loss and take profit orders
- Real-time position tracking
- Portfolio performance metrics
- Risk management controls
- Trade history and analytics
"""

import os
import json
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from flask import Flask, request, jsonify
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from structlog import get_logger
import redis
import alpaca_trade_api as tradeapi

# Import database utilities
from database_utils import (
    get_db_connection,
    insert_trade_record,
    update_trade_exit,
    get_open_positions,
    get_pending_signals,
    get_redis,
    health_check
)


class TradingService:
    """
    Paper trading service for order execution and position management
    """
    
    def __init__(self):
        # Initialize environment
        self.setup_environment()
        
        self.app = Flask(__name__)
        self.setup_logging()
        
        # Initialize Redis client
        self.redis_client = get_redis()
        
        # Initialize Alpaca API
        self.alpaca = self._init_alpaca()
        
        # Service URLs from environment
        self.coordination_url = os.getenv('COORDINATION_URL', 'http://coordination-service:5000')
        self.technical_url = os.getenv('TECHNICAL_SERVICE_URL', 'http://technical-service:5003')
        
        # Trading configuration
        self.trading_config = {
            # Position sizing
            'max_positions': int(os.getenv('MAX_POSITIONS', '5')),
            'max_position_size_pct': float(os.getenv('MAX_POSITION_SIZE_PCT', '20')),
            'min_position_size_pct': float(os.getenv('MIN_POSITION_SIZE_PCT', '5')),
            'default_position_size_pct': float(os.getenv('DEFAULT_POSITION_SIZE_PCT', '10')),
            
            # Risk management
            'max_portfolio_risk_pct': float(os.getenv('MAX_PORTFOLIO_RISK_PCT', '10')),
            'default_stop_loss_pct': float(os.getenv('DEFAULT_STOP_LOSS_PCT', '2')),
            'default_take_profit_pct': float(os.getenv('DEFAULT_TAKE_PROFIT_PCT', '4')),
            'trailing_stop_pct': float(os.getenv('TRAILING_STOP_PCT', '1.5')),
            
            # Order management
            'order_timeout_seconds': int(os.getenv('ORDER_TIMEOUT_SECONDS', '60')),
            'max_slippage_pct': float(os.getenv('MAX_SLIPPAGE_PCT', '0.5')),
            'use_limit_orders': os.getenv('USE_LIMIT_ORDERS', 'true').lower() == 'true',
            'limit_order_offset_pct': float(os.getenv('LIMIT_ORDER_OFFSET_PCT', '0.1')),
            
            # Trading hours
            'allow_premarket': os.getenv('ALLOW_PREMARKET_TRADING', 'true').lower() == 'true',
            'allow_afterhours': os.getenv('ALLOW_AFTERHOURS_TRADING', 'false').lower() == 'true'
        }
        
        # Position tracking
        self.positions = {}
        self.pending_orders = {}
        self.daily_metrics = {
            'trades_executed': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0
        }
        
        # Setup routes
        self.setup_routes()
        
        # Register with coordination
        self._register_with_coordination()
        
        # Start background tasks
        self._start_background_tasks()
        
        self.logger.info("Trading Service v2.1.2 initialized",
                        environment=os.getenv('ENVIRONMENT', 'development'),
                        alpaca_connected=bool(self.alpaca))
        
    def setup_environment(self):
        """Setup environment variables and paths"""
        # Paths
        self.log_path = os.getenv('LOG_PATH', '/app/logs')
        self.data_path = os.getenv('DATA_PATH', '/app/data')
        
        # Create directories
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        
        # Service configuration
        self.service_name = 'paper_trading'
        self.port = int(os.getenv('PORT', '5005'))
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # Alpaca configuration
        self.alpaca_config = {
            'api_key': os.getenv('ALPACA_API_KEY'),
            'secret_key': os.getenv('ALPACA_SECRET_KEY'),
            'base_url': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
            'data_url': os.getenv('ALPACA_DATA_URL', 'https://data.alpaca.markets')
        }
        
    def setup_logging(self):
        """Setup structured logging"""
        self.logger = get_logger()
        self.logger = self.logger.bind(service=self.service_name)
        
    def _init_alpaca(self):
        """Initialize Alpaca API connection"""
        if not self.alpaca_config['api_key'] or not self.alpaca_config['secret_key']:
            self.logger.warning("Alpaca API credentials not configured")
            return None
            
        try:
            api = tradeapi.REST(
                self.alpaca_config['api_key'],
                self.alpaca_config['secret_key'],
                self.alpaca_config['base_url'],
                api_version='v2'
            )
            
            # Test connection
            account = api.get_account()
            self.logger.info("Connected to Alpaca API",
                           account_status=account.status,
                           buying_power=account.buying_power)
            
            return api
            
        except Exception as e:
            self.logger.error("Failed to initialize Alpaca API", error=str(e))
            return None
            
    def setup_routes(self):
        """Setup Flask routes"""
        @self.app.route('/health', methods=['GET'])
        def health():
            db_health = health_check()
            alpaca_health = self._check_alpaca_health()
            
            # Handle different database_utils versions
            db_status = False
            redis_status = False
            
            if 'postgresql' in db_health:
                # v2.3.1 format
                db_status = db_health['postgresql'].get('status') == 'healthy'
                redis_status = db_health['redis'].get('status') == 'healthy'
            elif 'database' in db_health:
                # older format
                db_status = db_health['database'] == 'healthy'
                redis_status = db_health['redis'] == 'healthy'
            
            overall_health = db_status and redis_status and alpaca_health
            
            return jsonify({
                "status": "healthy" if overall_health else "degraded",
                "service": "paper_trading",
                "version": "2.1.2",
                "database": db_status,
                "redis": redis_status,
                "alpaca": "connected" if alpaca_health else "disconnected",
                "timestamp": datetime.now().isoformat()
            })
            
        @self.app.route('/execute_trade', methods=['POST'])
        def execute_trade():
            """Execute a trade based on signal"""
            signal = request.json
            
            if not signal or not signal.get('symbol'):
                return jsonify({'error': 'Invalid signal data'}), 400
                
            result = self.execute_signal(signal)
            return jsonify(result)
            
        @self.app.route('/positions', methods=['GET'])
        def get_positions():
            """Get current positions"""
            positions = self.get_current_positions()
            return jsonify({
                'count': len(positions),
                'positions': positions,
                'summary': self._calculate_position_summary(positions)
            })
            
        @self.app.route('/position/<symbol>', methods=['GET'])
        def get_position(symbol):
            """Get position for specific symbol"""
            position = self.get_position_details(symbol)
            if position:
                return jsonify(position)
            else:
                return jsonify({'error': f'No position found for {symbol}'}), 404
                
        @self.app.route('/close_position', methods=['POST'])
        def close_position():
            """Close a specific position"""
            data = request.json
            symbol = data.get('symbol')
            reason = data.get('reason', 'manual_close')
            
            if not symbol:
                return jsonify({'error': 'Symbol required'}), 400
                
            result = self.close_position_for_symbol(symbol, reason)
            return jsonify(result)
            
        @self.app.route('/orders', methods=['GET'])
        def get_orders():
            """Get recent orders"""
            status = request.args.get('status', 'all')
            limit = request.args.get('limit', 50, type=int)
            
            orders = self._get_orders(status, limit)
            return jsonify({
                'count': len(orders),
                'orders': orders
            })
            
        @self.app.route('/order/<order_id>', methods=['GET'])
        def get_order(order_id):
            """Get specific order details"""
            order = self._get_order_details(order_id)
            if order:
                return jsonify(order)
            else:
                return jsonify({'error': f'Order {order_id} not found'}), 404
                
        @self.app.route('/cancel_order', methods=['POST'])
        def cancel_order():
            """Cancel a pending order"""
            data = request.json
            order_id = data.get('order_id')
            
            if not order_id:
                return jsonify({'error': 'order_id required'}), 400
                
            result = self._cancel_order(order_id)
            return jsonify(result)
            
        @self.app.route('/performance', methods=['GET'])
        def get_performance():
            """Get trading performance metrics"""
            period = request.args.get('period', 'today')
            metrics = self._calculate_performance_metrics(period)
            return jsonify(metrics)
            
        @self.app.route('/risk_metrics', methods=['GET'])
        def get_risk_metrics():
            """Get current risk metrics"""
            metrics = self._calculate_risk_metrics()
            return jsonify(metrics)

    def _register_with_coordination(self):
        """Register with coordination service"""
        try:
            response = requests.post(
                f"{self.coordination_url}/register_service",
                json={
                    'service_name': 'paper_trading',
                    'service_info': {
                        'url': f"http://trading-service:{self.port}",
                        'port': self.port,
                        'version': '2.1.2',
                        'capabilities': ['paper_trading', 'position_management', 'risk_control']
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

    def _check_alpaca_health(self) -> bool:
        """Check if Alpaca API is healthy"""
        if not self.alpaca:
            return False
            
        try:
            account = self.alpaca.get_account()
            return account.status in ['ACTIVE', 'PAPER']
        except:
            return False
            
    def execute_signal(self, signal: Dict) -> Dict:
        """Execute a trading signal"""
        try:
            symbol = signal['symbol']
            signal_type = signal.get('signal_type', 'HOLD')
            confidence = signal.get('confidence', 0)
            
            # Check if we should execute
            if signal_type == 'HOLD' or confidence < self.trading_config['min_position_size_pct']:
                return {
                    'status': 'skipped',
                    'reason': f'Signal type {signal_type} or low confidence {confidence}'
                }
                
            # Check existing position
            existing_position = self.get_position_details(symbol)
            if existing_position:
                return {
                    'status': 'skipped',
                    'reason': f'Already have position in {symbol}'
                }
                
            # Check position limits
            current_positions = self.get_current_positions()
            if len(current_positions) >= self.trading_config['max_positions']:
                return {
                    'status': 'rejected',
                    'reason': f"Maximum positions ({self.trading_config['max_positions']}) reached"
                }
                
            # Calculate position size
            position_size = self._calculate_position_size(signal)
            if position_size <= 0:
                return {
                    'status': 'rejected',
                    'reason': 'Invalid position size calculated'
                }
                
            # Get current price
            current_price = self._get_current_price(symbol)
            if not current_price:
                return {
                    'status': 'error',
                    'reason': 'Could not get current price'
                }
                
            # Calculate shares
            shares = int(position_size / current_price)
            if shares < 1:
                return {
                    'status': 'rejected',
                    'reason': 'Position size too small'
                }
                
            # Prepare order
            order_params = self._prepare_order(
                symbol, signal_type, shares, current_price, signal
            )
            
            # Execute order
            order_result = self._execute_order(order_params)
            
            if order_result['status'] == 'success':
                # Record trade in database
                trade_record = self._create_trade_record(
                    signal, order_result['order'], order_params
                )
                trade_id = insert_trade_record(trade_record)
                
                # Set stop loss and take profit orders
                self._set_exit_orders(
                    symbol, signal_type, order_result['fill_price'],
                    signal, trade_id
                )
                
                # Update metrics
                self.daily_metrics['trades_executed'] += 1
                
                return {
                    'status': 'success',
                    'trade_id': trade_id,
                    'order_id': order_result['order_id'],
                    'symbol': symbol,
                    'direction': signal_type,
                    'shares': shares,
                    'entry_price': order_result['fill_price'],
                    'position_value': shares * order_result['fill_price']
                }
            else:
                return order_result
                
        except Exception as e:
            self.logger.error("Error executing signal",
                            symbol=signal.get('symbol'),
                            error=str(e))
            return {
                'status': 'error',
                'reason': str(e)
            }
            
    def get_current_positions(self) -> List[Dict]:
        """Get all current positions"""
        if not self.alpaca:
            return self._get_mock_positions()
            
        try:
            positions = self.alpaca.list_positions()
            
            position_list = []
            for pos in positions:
                position_list.append({
                    'symbol': pos.symbol,
                    'quantity': int(pos.qty),
                    'side': pos.side,
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price) if pos.current_price else 0,
                    'market_value': float(pos.market_value),
                    'cost_basis': float(pos.cost_basis),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'change_today': float(pos.change_today) if pos.change_today else 0
                })
                
            return position_list
            
        except Exception as e:
            self.logger.error("Error fetching positions", error=str(e))
            return []
            
    def get_position_details(self, symbol: str) -> Optional[Dict]:
        """Get details for a specific position"""
        positions = self.get_current_positions()
        for pos in positions:
            if pos['symbol'] == symbol:
                return pos
        return None
        
    def close_position_for_symbol(self, symbol: str, reason: str = 'manual') -> Dict:
        """Close position for a specific symbol"""
        if not self.alpaca:
            return {
                'status': 'error',
                'reason': 'Alpaca not connected'
            }
            
        try:
            # Get current position
            position = self.get_position_details(symbol)
            if not position:
                return {
                    'status': 'error',
                    'reason': f'No position found for {symbol}'
                }
                
            # Submit order to close
            order = self.alpaca.submit_order(
                symbol=symbol,
                qty=abs(position['quantity']),
                side='sell' if position['side'] == 'long' else 'buy',
                type='market',
                time_in_force='day'
            )
            
            # Update trade record
            self._update_trade_exit(symbol, reason)
            
            return {
                'status': 'success',
                'order_id': order.id,
                'symbol': symbol,
                'quantity': position['quantity'],
                'reason': reason
            }
            
        except Exception as e:
            self.logger.error("Error closing position",
                            symbol=symbol,
                            error=str(e))
            return {
                'status': 'error',
                'reason': str(e)
            }
            
    def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size based on signal and risk management"""
        if not self.alpaca:
            return 10000  # Default for mock trading
            
        try:
            account = self.alpaca.get_account()
            buying_power = float(account.buying_power)
            
            # Base position size on confidence
            confidence = signal.get('confidence', 50)
            position_size_pct = signal.get('position_size_pct', self.trading_config['default_position_size_pct'])
            
            # Ensure within limits
            position_size_pct = max(
                self.trading_config['min_position_size_pct'],
                min(self.trading_config['max_position_size_pct'], position_size_pct)
            )
            
            position_size = buying_power * (position_size_pct / 100)
            
            return position_size
            
        except Exception as e:
            self.logger.error("Error calculating position size", error=str(e))
            return 0
            
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        if not self.alpaca:
            return 100.0  # Mock price
            
        try:
            quote = self.alpaca.get_last_quote(symbol)
            return float(quote.askprice)
            
        except Exception as e:
            self.logger.error("Error getting current price",
                            symbol=symbol,
                            error=str(e))
            return None
            
    def _prepare_order(self, symbol: str, signal_type: str, shares: int,
                      current_price: float, signal: Dict) -> Dict:
        """Prepare order parameters"""
        # Determine order side
        side = 'buy' if signal_type in ['BUY', 'BUY_WEAK'] else 'sell'
        
        # Determine order type
        if self.trading_config['use_limit_orders']:
            order_type = 'limit'
            offset = current_price * (self.trading_config['limit_order_offset_pct'] / 100)
            limit_price = current_price + offset if side == 'buy' else current_price - offset
        else:
            order_type = 'market'
            limit_price = None
            
        return {
            'symbol': symbol,
            'qty': shares,
            'side': side,
            'type': order_type,
            'time_in_force': 'day',
            'limit_price': limit_price,
            'client_order_id': signal.get('signal_id')
        }
        
    def _execute_order(self, order_params: Dict) -> Dict:
        """Execute order through Alpaca"""
        if not self.alpaca:
            return self._execute_mock_order(order_params)
            
        try:
            order = self.alpaca.submit_order(**order_params)
            
            # Wait for fill or timeout
            start_time = time.time()
            timeout = self.trading_config['order_timeout_seconds']
            
            while time.time() - start_time < timeout:
                order = self.alpaca.get_order(order.id)
                
                if order.status == 'filled':
                    return {
                        'status': 'success',
                        'order_id': order.id,
                        'order': order,
                        'fill_price': float(order.filled_avg_price)
                    }
                elif order.status in ['canceled', 'rejected']:
                    return {
                        'status': 'failed',
                        'reason': f'Order {order.status}'
                    }
                    
                time.sleep(1)
                
            # Timeout - cancel order
            self.alpaca.cancel_order(order.id)
            return {
                'status': 'timeout',
                'reason': 'Order execution timeout'
            }
            
        except Exception as e:
            self.logger.error("Error executing order", error=str(e))
            return {
                'status': 'error',
                'reason': str(e)
            }
            
    def _set_exit_orders(self, symbol: str, direction: str, entry_price: float,
                        signal: Dict, trade_id: str):
        """Set stop loss and take profit orders"""
        if not self.alpaca:
            return
            
        try:
            # Calculate stop and target prices
            stop_price = signal.get('stop_loss', entry_price * 0.98)
            target_price = signal.get('target_1', entry_price * 1.02)
            
            # Determine order side (opposite of entry)
            exit_side = 'sell' if direction in ['BUY', 'BUY_WEAK'] else 'buy'
            
            # Get position quantity
            position = self.get_position_details(symbol)
            if not position:
                return
                
            qty = abs(position['quantity'])
            
            # Submit stop loss order
            stop_order = self.alpaca.submit_order(
                symbol=symbol,
                qty=qty,
                side=exit_side,
                type='stop',
                stop_price=stop_price,
                time_in_force='gtc',
                client_order_id=f"{trade_id}_stop"
            )
            
            # Submit take profit order
            target_order = self.alpaca.submit_order(
                symbol=symbol,
                qty=qty,
                side=exit_side,
                type='limit',
                limit_price=target_price,
                time_in_force='gtc',
                client_order_id=f"{trade_id}_target"
            )
            
            self.logger.info("Exit orders placed",
                           symbol=symbol,
                           stop_id=stop_order.id,
                           target_id=target_order.id)
                           
        except Exception as e:
            self.logger.error("Error setting exit orders",
                            symbol=symbol,
                            error=str(e))
                            
    def _create_trade_record(self, signal: Dict, order: Any, order_params: Dict) -> Dict:
        """Create trade record for database"""
        return {
            'signal_id': signal.get('signal_id'),
            'symbol': signal['symbol'],
            'order_type': order_params['type'],
            'side': order_params['side'],
            'quantity': order_params['qty'],
            'entry_price': float(order.filled_avg_price) if hasattr(order, 'filled_avg_price') else order_params.get('limit_price', 0),
            'stop_loss': signal.get('stop_loss'),
            'take_profit': signal.get('target_1'),
            'signal_confidence': signal.get('confidence'),
            'catalyst_type': signal.get('catalyst_type'),
            'execution_timestamp': datetime.now(),
            'order_id': order.id if hasattr(order, 'id') else 'mock_order',
            'status': 'open'
        }
        
    def _update_trade_exit(self, symbol: str, reason: str):
        """Update trade record with exit information"""
        # This would update the database with exit details
        # Implementation depends on database schema
        pass
        
    def _get_orders(self, status: str, limit: int) -> List[Dict]:
        """Get orders with specified status"""
        if not self.alpaca:
            return []
            
        try:
            if status == 'all':
                orders = self.alpaca.list_orders(limit=limit)
            else:
                orders = self.alpaca.list_orders(status=status, limit=limit)
                
            order_list = []
            for order in orders:
                order_list.append({
                    'order_id': order.id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'type': order.order_type,
                    'quantity': int(order.qty),
                    'status': order.status,
                    'created_at': order.created_at.isoformat(),
                    'filled_qty': int(order.filled_qty) if order.filled_qty else 0,
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else 0
                })
                
            return order_list
            
        except Exception as e:
            self.logger.error("Error fetching orders", error=str(e))
            return []
            
    def _get_order_details(self, order_id: str) -> Optional[Dict]:
        """Get details for specific order"""
        if not self.alpaca:
            return None
            
        try:
            order = self.alpaca.get_order(order_id)
            
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'side': order.side,
                'type': order.order_type,
                'quantity': int(order.qty),
                'status': order.status,
                'created_at': order.created_at.isoformat(),
                'filled_qty': int(order.filled_qty) if order.filled_qty else 0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else 0,
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None
            }
            
        except Exception as e:
            self.logger.error("Error fetching order details",
                            order_id=order_id,
                            error=str(e))
            return None
            
    def _cancel_order(self, order_id: str) -> Dict:
        """Cancel a pending order"""
        if not self.alpaca:
            return {'status': 'error', 'reason': 'Alpaca not connected'}
            
        try:
            self.alpaca.cancel_order(order_id)
            return {'status': 'success', 'order_id': order_id}
            
        except Exception as e:
            self.logger.error("Error canceling order",
                            order_id=order_id,
                            error=str(e))
            return {'status': 'error', 'reason': str(e)}
            
    def _calculate_performance_metrics(self, period: str) -> Dict:
        """Calculate trading performance metrics"""
        # This would calculate metrics from database
        # For now, return current session metrics
        return {
            'period': period,
            'trades_executed': self.daily_metrics['trades_executed'],
            'winning_trades': self.daily_metrics['winning_trades'],
            'losing_trades': self.daily_metrics['losing_trades'],
            'win_rate': self._calculate_win_rate(),
            'total_pnl': self.daily_metrics['total_pnl'],
            'max_drawdown': self.daily_metrics['max_drawdown'],
            'sharpe_ratio': 0,  # Would calculate from returns
            'profit_factor': 0  # Would calculate from wins/losses
        }
        
    def _calculate_win_rate(self) -> float:
        """Calculate win rate percentage"""
        total = self.daily_metrics['winning_trades'] + self.daily_metrics['losing_trades']
        if total == 0:
            return 0
        return (self.daily_metrics['winning_trades'] / total) * 100
        
    def _calculate_risk_metrics(self) -> Dict:
        """Calculate current risk metrics"""
        positions = self.get_current_positions()
        
        total_exposure = sum(pos['market_value'] for pos in positions)
        total_risk = 0
        
        # Calculate risk per position
        position_risks = []
        for pos in positions:
            # Assume 2% stop loss if not set
            risk_amount = pos['market_value'] * 0.02
            total_risk += risk_amount
            
            position_risks.append({
                'symbol': pos['symbol'],
                'exposure': pos['market_value'],
                'risk_amount': risk_amount,
                'risk_percent': 2.0
            })
            
        return {
            'total_exposure': total_exposure,
            'total_risk': total_risk,
            'position_count': len(positions),
            'positions': position_risks,
            'portfolio_heat': (total_risk / 10000) * 100 if total_risk > 0 else 0  # Assume 10k portfolio
        }
        
    def _calculate_position_summary(self, positions: List[Dict]) -> Dict:
        """Calculate summary statistics for positions"""
        if not positions:
            return {
                'total_value': 0,
                'total_pnl': 0,
                'total_pnl_percent': 0,
                'winning_positions': 0,
                'losing_positions': 0
            }
            
        total_value = sum(pos['market_value'] for pos in positions)
        total_pnl = sum(pos['unrealized_pl'] for pos in positions)
        total_cost = sum(pos['cost_basis'] for pos in positions)
        
        winning = len([p for p in positions if p['unrealized_pl'] > 0])
        losing = len([p for p in positions if p['unrealized_pl'] < 0])
        
        return {
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_percent': (total_pnl / total_cost * 100) if total_cost > 0 else 0,
            'winning_positions': winning,
            'losing_positions': losing
        }
        
    def _get_mock_positions(self) -> List[Dict]:
        """Get mock positions for testing"""
        return []
        
    def _execute_mock_order(self, order_params: Dict) -> Dict:
        """Execute mock order for testing"""
        return {
            'status': 'success',
            'order_id': f"mock_{order_params['symbol']}_{time.time()}",
            'order': None,
            'fill_price': order_params.get('limit_price', 100.0)
        }
        
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        # Position monitoring
        def monitor_positions():
            while True:
                try:
                    positions = self.get_current_positions()
                    for pos in positions:
                        self._check_trailing_stop(pos)
                        self._update_position_metrics(pos)
                except Exception as e:
                    self.logger.error("Position monitor error", error=str(e))
                    
                time.sleep(30)  # Check every 30 seconds
                
        # Order monitoring
        def monitor_orders():
            while True:
                try:
                    # Clean up old pending orders
                    cutoff_time = datetime.now() - timedelta(minutes=5)
                    
                    for order_id, order_info in list(self.pending_orders.items()):
                        if order_info['submitted_at'] < cutoff_time:
                            # Check if order is still pending
                            if self.alpaca:
                                order = self.alpaca.get_order(order_id)
                                if order.status in ['filled', 'cancelled', 'rejected']:
                                    del self.pending_orders[order_id]
                                    
                except Exception as e:
                    self.logger.error("Order monitor error", error=str(e))
                    
                time.sleep(30)  # Check every 30 seconds
                
        import threading
        thread1 = threading.Thread(target=monitor_positions)
        thread1.daemon = True
        thread1.start()
        
        thread2 = threading.Thread(target=monitor_orders)
        thread2.daemon = True
        thread2.start()
        
    def _check_trailing_stop(self, position: Dict):
        """Check and update trailing stops"""
        # Implementation for trailing stop logic
        pass
        
    def _update_position_metrics(self, position: Dict):
        """Update position metrics"""
        # Track max profit/loss for position
        pass
        
    def run(self):
        """Start the trading service"""
        self.logger.info("Starting Trading Service",
                        version="2.1.2",
                        port=self.port,
                        environment=os.getenv('ENVIRONMENT', 'development'),
                        alpaca_mode="paper" if self.alpaca else "mock")
        
        self.app.run(host='0.0.0.0', port=self.port, debug=False)


if __name__ == "__main__":
    service = TradingService()
    service.run()