#!/usr/bin/env python3
"""
Name of System: Catalyst Trading System
Name of file: dashboard_service.py
Version: 1.0.0
Last Updated: 2025-06-29
REVISION HISTORY:
  - v1.0.0 (2025-06-29) - Initial release with standardized authentication

Purpose: Web-based dashboard for monitoring and controlling the trading system

This service provides:
1. Real-time system status monitoring
2. Trading performance visualization
3. Service health monitoring
4. Trade execution interface
5. News and pattern analysis display
6. Historical performance reports
"""

import os
import json
import sqlite3
import logging
import requests
from datetime import datetime, timedelta, date
from flask import Flask, render_template_string, jsonify, request, session
from flask_cors import CORS
from typing import Dict, List, Optional, Any
import threading
import time

# Import database utilities if available
try:
    from database_utils_old import DatabaseServiceMixin
    USE_DB_UTILS = True
except ImportError:
    USE_DB_UTILS = False
    print("Warning: database_utils not found, using direct SQLite connections")


class TradingDashboardService(DatabaseServiceMixin if USE_DB_UTILS else object):
    """
    Web dashboard service for the Catalyst Trading System
    Provides real-time monitoring and control interface
    """
    
    def __init__(self, db_path='/workspaces/trading-system//tmp/trading_system.db'):
        if USE_DB_UTILS:
            super().__init__(db_path)
        else:
            self.db_path = db_path
            
        self.app = Flask(__name__)
        self.app.secret_key = os.urandom(24)
        CORS(self.app)
        
        self.setup_logging()
        self.setup_routes()
        
        # Service endpoints
        self.services = {
            'coordination': {'url': 'http://localhost:5000', 'port': 5000},
            'scanner': {'url': 'http://localhost:5001', 'port': 5001},
            'pattern': {'url': 'http://localhost:5002', 'port': 5002},
            'technical': {'url': 'http://localhost:5003', 'port': 5003},
            'trading': {'url': 'http://localhost:5005', 'port': 5005},
            'news': {'url': 'http://localhost:5008', 'port': 5008},
            'reporting': {'url': 'http://localhost:5009', 'port': 5009}
        }
        
        # Cache for service status
        self.service_status_cache = {}
        self.last_status_check = None
        
        # Start background status monitoring
        self.start_status_monitor()
        
        # Register with coordination
        self._register_with_coordination()
        
        self.logger.info("Trading Dashboard Service v1.0.0 initialized")
        
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('dashboard_service')
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory
        os.makedirs('/tmp/logs', exist_ok=True)
        
        # File handler
        fh = logging.FileHandler('/tmp//tmp/logs/dashboard.log')
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
        
    def setup_routes(self):
        """Setup Flask routes"""
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template_string(self.get_dashboard_template())
            
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                "status": "healthy",
                "service": "dashboard",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat()
            })
            
        @self.app.route('/api/system_status', methods=['GET'])
        def system_status():
            """Get overall system status"""
            return jsonify(self.get_system_status())
            
        @self.app.route('/api/service_health', methods=['GET'])
        def service_health():
            """Get health status of all services"""
            return jsonify(self.get_service_health())
            
        @self.app.route('/api/recent_trades', methods=['GET'])
        def recent_trades():
            """Get recent trades"""
            limit = request.args.get('limit', 20, type=int)
            return jsonify(self.get_recent_trades(limit))
            
        @self.app.route('/api/active_positions', methods=['GET'])
        def active_positions():
            """Get currently active positions"""
            return jsonify(self.get_active_positions())
            
        @self.app.route('/api/performance_summary', methods=['GET'])
        def performance_summary():
            """Get performance summary"""
            days = request.args.get('days', 7, type=int)
            return jsonify(self.get_performance_summary(days))
            
        @self.app.route('/api/recent_news', methods=['GET'])
        def recent_news():
            """Get recent news items"""
            limit = request.args.get('limit', 10, type=int)
            return jsonify(self.get_recent_news(limit))
            
        @self.app.route('/api/pattern_analysis', methods=['GET'])
        def pattern_analysis():
            """Get recent pattern detections"""
            limit = request.args.get('limit', 10, type=int)
            return jsonify(self.get_recent_patterns(limit))
            
        @self.app.route('/api/start_trading_cycle', methods=['POST'])
        def start_trading_cycle():
            """Start a new trading cycle"""
            try:
                mode = request.json.get('mode', 'normal')
                response = requests.post(
                    f"{self.services['coordination']['url']}/start_trading_cycle",
                    json={'mode': mode},
                    timeout=5
                )
                return jsonify(response.json())
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/current_cycle', methods=['GET'])
        def current_cycle():
            """Get current trading cycle status"""
            try:
                response = requests.get(
                    f"{self.services['coordination']['url']}/current_cycle",
                    timeout=5
                )
                return jsonify(response.json())
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
    def get_system_status(self) -> Dict:
        """Get overall system status and metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get today's metrics
            today = date.today().isoformat()
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as trades_today,
                    SUM(pnl_amount) as pnl_today,
                    AVG(pnl_percentage) as avg_return
                FROM trade_records
                WHERE DATE(entry_timestamp) = ?
            ''', (today,))
            
            today_stats = cursor.fetchone()
            
            # Get active positions
            cursor.execute('''
                SELECT COUNT(*) as active_positions
                FROM trade_records
                WHERE exit_timestamp IS NULL
            ''')
            
            active_count = cursor.fetchone()[0]
            
            # Get service health summary
            healthy_services = sum(1 for s in self.service_status_cache.values() 
                                 if s.get('status') == 'healthy')
            total_services = len(self.services)
            
            # Get current cycle info
            try:
                response = requests.get(
                    f"{self.services['coordination']['url']}/current_cycle",
                    timeout=2
                )
                if response.status_code == 200:
                    cycle_info = response.json()
                else:
                    cycle_info = {'status': 'unknown'}
            except:
                cycle_info = {'status': 'error'}
                
            conn.close()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system_health': 'healthy' if healthy_services == total_services else 'degraded',
                'services_online': f"{healthy_services}/{total_services}",
                'today_stats': {
                    'trades': today_stats[0] or 0,
                    'pnl': round(today_stats[1] or 0, 2),
                    'avg_return': round(today_stats[2] or 0, 2) if today_stats[2] else 0
                },
                'active_positions': active_count,
                'current_cycle': cycle_info
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
            
    def get_service_health(self) -> Dict:
        """Get detailed health status of all services"""
        # Use cached status if recent
        if (self.last_status_check and 
            datetime.now() - self.last_status_check < timedelta(seconds=10)):
            return self.service_status_cache
            
        # Otherwise, check all services
        return self.check_all_services()
        
    def check_all_services(self) -> Dict:
        """Check health of all services"""
        status = {}
        
        for service_name, service_info in self.services.items():
            try:
                response = requests.get(
                    f"{service_info['url']}/health",
                    timeout=2
                )
                
                if response.status_code == 200:
                    health_data = response.json()
                    status[service_name] = {
                        'status': 'healthy',
                        'version': health_data.get('version', 'unknown'),
                        'response_time_ms': response.elapsed.total_seconds() * 1000
                    }
                else:
                    status[service_name] = {
                        'status': 'unhealthy',
                        'error': f'HTTP {response.status_code}'
                    }
                    
            except Exception as e:
                status[service_name] = {
                    'status': 'offline',
                    'error': str(e)
                }
                
        self.service_status_cache = status
        self.last_status_check = datetime.now()
        
        return status
        
    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """Get recent trade records"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    trade_id,
                    symbol,
                    entry_timestamp,
                    exit_timestamp,
                    entry_price,
                    exit_price,
                    position_size,
                    pnl_amount,
                    pnl_percentage,
                    catalyst_type,
                    exit_reason
                FROM trade_records
                ORDER BY entry_timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            trades = []
            for row in cursor.fetchall():
                trades.append({
                    'trade_id': row[0],
                    'symbol': row[1],
                    'entry_time': row[2],
                    'exit_time': row[3],
                    'entry_price': row[4],
                    'exit_price': row[5],
                    'position_size': row[6],
                    'pnl_amount': round(row[7] or 0, 2),
                    'pnl_percentage': round(row[8] or 0, 2),
                    'catalyst_type': row[9],
                    'exit_reason': row[10],
                    'status': 'closed' if row[3] else 'open'
                })
                
            conn.close()
            return trades
            
        except Exception as e:
            self.logger.error(f"Error getting recent trades: {e}")
            return []
            
    def get_active_positions(self) -> List[Dict]:
        """Get currently active positions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    trade_id,
                    symbol,
                    entry_timestamp,
                    entry_price,
                    position_size,
                    stop_loss_price,
                    take_profit_price,
                    catalyst_type,
                    position_side
                FROM trade_records
                WHERE exit_timestamp IS NULL
                ORDER BY entry_timestamp DESC
            ''')
            
            positions = []
            for row in cursor.fetchall():
                # Calculate unrealized P&L (would need current price in real implementation)
                positions.append({
                    'trade_id': row[0],
                    'symbol': row[1],
                    'entry_time': row[2],
                    'entry_price': row[3],
                    'position_size': row[4],
                    'stop_loss': row[5],
                    'take_profit': row[6],
                    'catalyst_type': row[7],
                    'side': row[8]
                })
                
            conn.close()
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting active positions: {e}")
            return []
            
    def get_performance_summary(self, days: int = 7) -> Dict:
        """Get performance summary for specified days"""
        try:
            # Get from reporting service if available
            response = requests.get(
                f"{self.services['reporting']['url']}/daily_summary",
                params={'date': date.today().isoformat()},
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                # Fallback to local calculation
                return self._calculate_local_performance(days)
                
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return self._calculate_local_performance(days)
            
    def _calculate_local_performance(self, days: int) -> Dict:
        """Calculate performance locally if reporting service unavailable"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl_amount > 0 THEN 1 ELSE 0 END) as winners,
                    SUM(pnl_amount) as total_pnl,
                    AVG(pnl_percentage) as avg_return
                FROM trade_records
                WHERE entry_timestamp > datetime('now', '-{} days')
                AND exit_timestamp IS NOT NULL
            '''.format(days))
            
            stats = cursor.fetchone()
            
            conn.close()
            
            total_trades = stats[0] or 0
            winners = stats[1] or 0
            
            return {
                'period_days': days,
                'total_trades': total_trades,
                'winning_trades': winners,
                'win_rate': round((winners / total_trades * 100) if total_trades > 0 else 0, 2),
                'total_pnl': round(stats[2] or 0, 2),
                'avg_return_pct': round(stats[3] or 0, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating local performance: {e}")
            return {}
            
    def get_recent_news(self, limit: int = 10) -> List[Dict]:
        """Get recent news items"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    news_id,
                    symbol,
                    headline,
                    source,
                    published_timestamp,
                    catalyst_score,
                    catalyst_type
                FROM news_raw
                ORDER BY published_timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            news = []
            for row in cursor.fetchall():
                news.append({
                    'news_id': row[0],
                    'symbol': row[1],
                    'headline': row[2],
                    'source': row[3],
                    'published': row[4],
                    'catalyst_score': row[5],
                    'catalyst_type': row[6]
                })
                
            conn.close()
            return news
            
        except Exception as e:
            self.logger.error(f"Error getting recent news: {e}")
            return []
            
    def get_recent_patterns(self, limit: int = 10) -> List[Dict]:
        """Get recent pattern detections"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    symbol,
                    pattern_name,
                    pattern_type,
                    detection_timestamp,
                    final_confidence,
                    has_catalyst
                FROM pattern_analysis
                ORDER BY detection_timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            patterns = []
            for row in cursor.fetchall():
                patterns.append({
                    'symbol': row[0],
                    'pattern': row[1],
                    'type': row[2],
                    'detected': row[3],
                    'confidence': row[4],
                    'has_catalyst': bool(row[5])
                })
                
            conn.close()
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error getting recent patterns: {e}")
            return []
            
    def start_status_monitor(self):
        """Start background service status monitoring"""
        def monitor():
            while True:
                try:
                    self.check_all_services()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    self.logger.error(f"Status monitor error: {e}")
                    time.sleep(60)
                    
        thread = threading.Thread(target=monitor)
        thread.daemon = True
        thread.start()
        
    def _register_with_coordination(self):
        """Register with coordination service"""
        try:
            response = requests.post(
                f"{self.services['coordination']['url']}/register_service",
                json={
                    'service_name': 'web_dashboard',
                    'service_info': {
                        'url': 'http://localhost:5010',
                        'port': 5010,
                        'version': '1.0.0',
                        'capabilities': ['monitoring', 'control', 'visualization']
                    }
                },
                timeout=5
            )
            if response.status_code == 200:
                self.logger.info("Successfully registered with coordination service")
        except Exception as e:
            self.logger.warning(f"Could not register with coordination: {e}")
            
    def get_dashboard_template(self) -> str:
        """Return the HTML template for the dashboard"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Catalyst Trading System Dashboard</title>
    <style>
        :root {
            --primary: #2196F3;
            --success: #4CAF50;
            --danger: #f44336;
            --warning: #ff9800;
            --dark: #212529;
            --light: #f8f9fa;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        
        .header {
            background-color: var(--dark);
            color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,.1);
        }
        
        .header h1 {
            font-size: 1.5rem;
            font-weight: 500;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,.1);
            transition: transform 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,.15);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #eee;
        }
        
        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--dark);
        }
        
        .status-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        .status-healthy {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        
        .status-degraded {
            background-color: #fff3e0;
            color: #e65100;
        }
        
        .status-offline {
            background-color: #ffebee;
            color: #c62828;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin: 0.5rem 0;
        }
        
        .metric-label {
            font-size: 0.875rem;
            color: #666;
        }
        
        .metric-value {
            font-size: 1.25rem;
            font-weight: 600;
        }
        
        .metric-value.positive {
            color: var(--success);
        }
        
        .metric-value.negative {
            color: var(--danger);
        }
        
        .btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 4px;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .btn-primary {
            background-color: var(--primary);
            color: white;
        }
        
        .btn-primary:hover {
            background-color: #1976D2;
        }
        
        .btn-success {
            background-color: var(--success);
            color: white;
        }
        
        .btn-success:hover {
            background-color: #388E3C;
        }
        
        .table-container {
            overflow-x: auto;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }
        
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        
        th {
            font-weight: 600;
            color: #666;
            background-color: #fafafa;
        }
        
        tr:hover {
            background-color: #f5f5f5;
        }
        
        .service-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .service-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 1rem;
            text-align: center;
            transition: all 0.2s;
        }
        
        .service-card.healthy {
            border-color: var(--success);
            background-color: #e8f5e9;
        }
        
        .service-card.offline {
            border-color: var(--danger);
            background-color: #ffebee;
        }
        
        .loading {
            text-align: center;
            color: #666;
            padding: 2rem;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Catalyst Trading System Dashboard</h1>
    </div>
    
    <div class="container">
        <!-- System Status Overview -->
        <div class="dashboard-grid">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">System Status</h2>
                    <span id="system-health-badge" class="status-badge">Loading...</span>
                </div>
                <div id="system-metrics">
                    <div class="loading">Loading system status...</div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Today's Performance</h2>
                </div>
                <div id="performance-metrics">
                    <div class="loading">Loading performance data...</div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Trading Control</h2>
                </div>
                <div id="trading-controls">
                    <button class="btn btn-primary" onclick="startTradingCycle('normal')">
                        Start Normal Cycle
                    </button>
                    <button class="btn btn-success" onclick="startTradingCycle('aggressive')">
                        Start Aggressive Cycle
                    </button>
                    <div id="cycle-status" style="margin-top: 1rem;">
                        <div class="loading">Loading cycle status...</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Service Health -->
        <div class="card">
            <div class="card-header">
                <h2 class="card-title">Service Health</h2>
            </div>
            <div id="service-health" class="service-grid">
                <div class="loading">Loading service status...</div>
            </div>
        </div>
        
        <!-- Recent Trades -->
        <div class="card" style="margin-top: 1.5rem;">
            <div class="card-header">
                <h2 class="card-title">Recent Trades</h2>
            </div>
            <div class="table-container">
                <table id="trades-table">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Entry Time</th>
                            <th>Exit Time</th>
                            <th>P&L</th>
                            <th>P&L %</th>
                            <th>Catalyst</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td colspan="7" class="loading">Loading trades...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Active Positions -->
        <div class="card" style="margin-top: 1.5rem;">
            <div class="card-header">
                <h2 class="card-title">Active Positions</h2>
            </div>
            <div class="table-container">
                <table id="positions-table">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Entry Price</th>
                            <th>Size</th>
                            <th>Stop Loss</th>
                            <th>Take Profit</th>
                            <th>Entry Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td colspan="7" class="loading">Loading positions...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        // Update functions
        async function updateSystemStatus() {
            try {
                const response = await fetch('/api/system_status');
                const data = await response.json();
                
                // Update health badge
                const badge = document.getElementById('system-health-badge');
                badge.textContent = data.system_health || 'Unknown';
                badge.className = 'status-badge status-' + (data.system_health || 'offline');
                
                // Update metrics
                const metricsDiv = document.getElementById('system-metrics');
                metricsDiv.innerHTML = `
                    <div class="metric">
                        <span class="metric-label">Services Online</span>
                        <span class="metric-value">${data.services_online || '0/0'}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Active Positions</span>
                        <span class="metric-value">${data.active_positions || 0}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Today's Trades</span>
                        <span class="metric-value">${data.today_stats?.trades || 0}</span>
                    </div>
                `;
                
                // Update performance metrics
                const perfDiv = document.getElementById('performance-metrics');
                const pnl = data.today_stats?.pnl || 0;
                const avgReturn = data.today_stats?.avg_return || 0;
                
                perfDiv.innerHTML = `
                    <div class="metric">
                        <span class="metric-label">Today's P&L</span>
                        <span class="metric-value ${pnl >= 0 ? 'positive' : 'negative'}">
                            $${pnl.toFixed(2)}
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Average Return</span>
                        <span class="metric-value ${avgReturn >= 0 ? 'positive' : 'negative'}">
                            ${avgReturn.toFixed(2)}%
                        </span>
                    </div>
                `;
                
                // Update cycle status
                updateCycleStatus(data.current_cycle);
                
            } catch (error) {
                console.error('Error updating system status:', error);
            }
        }
        
        async function updateServiceHealth() {
            try {
                const response = await fetch('/api/service_health');
                const data = await response.json();
                
                const healthDiv = document.getElementById('service-health');
                healthDiv.innerHTML = Object.entries(data).map(([service, status]) => `
                    <div class="service-card ${status.status}">
                        <div style="font-weight: 600;">${service}</div>
                        <div style="font-size: 0.875rem; color: #666;">
                            ${status.status}
                        </div>
                        ${status.version ? `<div style="font-size: 0.75rem;">v${status.version}</div>` : ''}
                    </div>
                `).join('');
                
            } catch (error) {
                console.error('Error updating service health:', error);
            }
        }
        
        async function updateRecentTrades() {
            try {
                const response = await fetch('/api/recent_trades');
                const trades = await response.json();
                
                const tbody = document.querySelector('#trades-table tbody');
                
                if (trades.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align: center;">No trades found</td></tr>';
                    return;
                }
                
                tbody.innerHTML = trades.map(trade => `
                    <tr>
                        <td>${trade.symbol}</td>
                        <td>${new Date(trade.entry_time).toLocaleString()}</td>
                        <td>${trade.exit_time ? new Date(trade.exit_time).toLocaleString() : '-'}</td>
                        <td class="${trade.pnl_amount >= 0 ? 'positive' : 'negative'}">
                            $${trade.pnl_amount.toFixed(2)}
                        </td>
                        <td class="${trade.pnl_percentage >= 0 ? 'positive' : 'negative'}">
                            ${trade.pnl_percentage.toFixed(2)}%
                        </td>
                        <td>${trade.catalyst_type || '-'}</td>
                        <td>
                            <span class="status-badge status-${trade.status === 'open' ? 'degraded' : 'healthy'}">
                                ${trade.status}
                            </span>
                        </td>
                    </tr>
                `).join('');
                
            } catch (error) {
                console.error('Error updating trades:', error);
            }
        }
        
        async function updateActivePositions() {
            try {
                const response = await fetch('/api/active_positions');
                const positions = await response.json();
                
                const tbody = document.querySelector('#positions-table tbody');
                
                if (positions.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align: center;">No active positions</td></tr>';
                    return;
                }
                
                tbody.innerHTML = positions.map(pos => `
                    <tr>
                        <td>${pos.symbol}</td>
                        <td>${pos.side || 'LONG'}</td>
                        <td>$${pos.entry_price.toFixed(2)}</td>
                        <td>${pos.position_size}</td>
                        <td>${pos.stop_loss ? '$' + pos.stop_loss.toFixed(2) : '-'}</td>
                        <td>${pos.take_profit ? '$' + pos.take_profit.toFixed(2) : '-'}</td>
                        <td>${new Date(pos.entry_time).toLocaleString()}</td>
                    </tr>
                `).join('');
                
            } catch (error) {
                console.error('Error updating positions:', error);
            }
        }
        
        function updateCycleStatus(cycle) {
            const statusDiv = document.getElementById('cycle-status');
            
            if (!cycle || cycle.status === 'no active cycle') {
                statusDiv.innerHTML = '<p style="color: #666;">No active trading cycle</p>';
                return;
            }
            
            statusDiv.innerHTML = `
                <div class="metric">
                    <span class="metric-label">Cycle ID</span>
                    <span class="metric-value">${cycle.cycle_id || 'N/A'}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Status</span>
                    <span class="metric-value">${cycle.status || 'Unknown'}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Mode</span>
                    <span class="metric-value">${cycle.mode || 'N/A'}</span>
                </div>
            `;
        }
        
        async function startTradingCycle(mode) {
            try {
                const response = await fetch('/api/start_trading_cycle', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ mode: mode })
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    alert(`Trading cycle started: ${result.cycle_id}`);
                    updateSystemStatus();
                } else {
                    alert(`Error: ${result.error || 'Failed to start cycle'}`);
                }
                
            } catch (error) {
                console.error('Error starting cycle:', error);
                alert('Failed to start trading cycle');
            }
        }
        
        // Initial load and periodic updates
        document.addEventListener('DOMContentLoaded', function() {
            updateSystemStatus();
            updateServiceHealth();
            updateRecentTrades();
            updateActivePositions();
            
            // Update every 10 seconds
            setInterval(updateSystemStatus, 10000);
            setInterval(updateServiceHealth, 30000);
            setInterval(updateRecentTrades, 15000);
            setInterval(updateActivePositions, 15000);
        });
    </script>
</body>
</html>
        '''
        
    def run(self):
        """Start the Flask application"""
        self.logger.info("Starting Trading Dashboard Service v1.0.0 on port 5010")
        self.logger.info("Access the dashboard at http://localhost:5010")
        
        self.app.run(host='0.0.0.0', port=5010, debug=False)


if __name__ == "__main__":
    service = TradingDashboardService()
    service.run()
