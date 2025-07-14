#!/usr/bin/env python3
"""
Name of Service: TRADING SYSTEM COORDINATION SERVICE - NEWS-DRIVEN VERSION
Version: 2.0.0
Last Updated: 2025-06-27
Purpose: Orchestrate news-driven trading workflow

REVISION HISTORY:
v2.0.0 (2025-06-27) - Complete rewrite for news-driven architecture
- News collection as primary driver
- Pre-market aggressive mode
- Source alignment awareness
- Outcome tracking integration
- Preserved core service management functionality

This service coordinates the new workflow:
1. News Collection → 2. Security Selection → 3. Pattern Analysis → 
4. Signal Generation → 5. Trade Execution → 6. Outcome Tracking
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
from typing import Dict, List, Optional, Any
import schedule

# Import database utilities if available
try:
    from database_utils_old import DatabaseServiceMixin
    USE_DB_UTILS = True
except ImportError:
    USE_DB_UTILS = False


class NewsDriverCoordinationService(DatabaseServiceMixin if USE_DB_UTILS else object):
    """
    Coordination service for news-driven trading system
    Orchestrates workflow from news to trades
    """
    
    def __init__(self, db_path='/tmp//tmp/trading_system.db'):
        if USE_DB_UTILS:
            super().__init__(db_path)
        else:
            self.db_path = db_path
            
        self.app = Flask(__name__)
        self.setup_logging()
        self.setup_routes()
        
        # Service registry
        self.services = {
            'news_collection': {
                'url': 'http://localhost:5008',
                'port': 5008,
                'required': True,
                'health_check': '/health'
            },
            'security_scanner': {
                'url': 'http://localhost:5001',
                'port': 5001,
                'required': True,
                'health_check': '/health'
            },
            'pattern_analysis': {
                'url': 'http://localhost:5002',
                'port': 5002,
                'required': True,
                'health_check': '/health'
            },
            'technical_analysis': {
                'url': 'http://localhost:5003',
                'port': 5003,
                'required': True,
                'health_check': '/health'
            },
            'paper_trading': {
                'url': 'http://localhost:5005',
                'port': 5005,
                'required': True,
                'health_check': '/health'
            },
            'reporting': {
                'url': 'http://localhost:5009',
                'port': 5009,
                'required': False,
                'health_check': '/health'
            },
            'web_dashboard': {
                'url': 'http://localhost:5010',
                'port': 5010,
                'required': False,
                'health_check': '/health'
            }
        }
        
        # Workflow configuration
        self.workflow_config = {
            'pre_market': {
                'enabled': True,
                'start_time': '04:00',  # 4 AM EST
                'end_time': '09:30',    # 9:30 AM EST
                'interval_minutes': 5,   # Aggressive
                'mode': 'aggressive'
            },
            'market_hours': {
                'enabled': True,
                'start_time': '09:30',
                'end_time': '16:00',
                'interval_minutes': 30,
                'mode': 'normal'
            },
            'after_hours': {
                'enabled': True,
                'start_time': '16:00',
                'end_time': '20:00',
                'interval_minutes': 60,
                'mode': 'light'
            }
        }
        
        # Trading cycle state
        self.current_cycle = None
        self.cycle_history = []
        
        # Service health status
        self.service_health = {}
        
        # Initialize database
        self._init_database()
        
        # Start background threads
        self.start_health_monitor()
        self.start_scheduler()
        
        self.logger.info("News-Driven Coordination Service v2.0.0 initialized")
        
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('coordination_service')
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory
        os.makedirs('/tmp/logs', exist_ok=True)
        
        # File handler
        fh = logging.FileHandler('/tmp//tmp/logs/coordination.log')
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
        
    def _init_database(self):
        """Initialize coordination database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Trading cycles table - tracks each workflow execution
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_cycles (
                    cycle_id TEXT PRIMARY KEY,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    status TEXT DEFAULT 'running',
                    mode TEXT,  -- aggressive, normal, light
                    
                    -- Metrics from each stage
                    news_collected INTEGER DEFAULT 0,
                    securities_scanned INTEGER DEFAULT 0,
                    candidates_selected INTEGER DEFAULT 0,
                    patterns_analyzed INTEGER DEFAULT 0,
                    signals_generated INTEGER DEFAULT 0,
                    trades_executed INTEGER DEFAULT 0,
                    
                    -- Performance
                    cycle_pnl DECIMAL(10,2),
                    success_rate DECIMAL(5,2),
                    
                    -- Metadata
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Service health table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS service_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    last_check TIMESTAMP NOT NULL,
                    response_time_ms INTEGER,
                    error_message TEXT,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Workflow execution log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workflow_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id TEXT,
                    step_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    duration_seconds DECIMAL(10,3),
                    result JSON,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (cycle_id) REFERENCES trading_cycles(cycle_id)
                )
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
                "service": "coordination",
                "version": "2.0.0",
                "mode": "news-driven",
                "timestamp": datetime.now().isoformat()
            })
            
        @self.app.route('/register_service', methods=['POST'])
        def register_service():
            """Register a service with coordination"""
            data = request.json
            service_name = data.get('service_name')
            service_info = data.get('service_info', {})
            
            if service_name in self.services:
                self.services[service_name].update(service_info)
                self.logger.info(f"Service {service_name} registered/updated")
                
            return jsonify({
                'status': 'registered',
                'service': service_name
            })
            
        @self.app.route('/start_trading_cycle', methods=['POST'])
        def start_trading_cycle():
            """Start a trading cycle"""
            data = request.json or {}
            mode = data.get('mode', 'normal')
            
            if self.current_cycle and self.current_cycle['status'] == 'running':
                return jsonify({
                    'error': 'Cycle already running',
                    'cycle_id': self.current_cycle['cycle_id']
                }), 400
                
            cycle = self.start_new_cycle(mode)
            return jsonify(cycle)
            
        @self.app.route('/current_cycle', methods=['GET'])
        def current_cycle():
            """Get current cycle status"""
            if not self.current_cycle:
                return jsonify({'status': 'no active cycle'}), 404
            return jsonify(self.current_cycle)
            
        @self.app.route('/service_health', methods=['GET'])
        def service_health():
            """Get health status of all services"""
            return jsonify({
                'services': self.service_health,
                'last_check': datetime.now().isoformat()
            })
            
        @self.app.route('/workflow_config', methods=['GET', 'POST'])
        def workflow_config():
            """Get or update workflow configuration"""
            if request.method == 'GET':
                return jsonify(self.workflow_config)
            else:
                self.workflow_config.update(request.json)
                return jsonify({
                    'status': 'updated',
                    'config': self.workflow_config
                })
                
    def start_new_cycle(self, mode: str = 'normal') -> Dict:
        """Start a new trading cycle"""
        cycle_id = f"CYCLE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_cycle = {
            'cycle_id': cycle_id,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'mode': mode,
            'progress': {}
        }
        
        # Save to database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trading_cycles (cycle_id, start_time, status, mode)
                VALUES (?, ?, ?, ?)
            ''', (cycle_id, self.current_cycle['start_time'], 'running', mode))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving cycle: {e}")
            
        # Start workflow in background thread
        thread = threading.Thread(target=self.execute_trading_workflow, args=(cycle_id, mode))
        thread.daemon = True
        thread.start()
        
        return self.current_cycle
        
    def execute_trading_workflow(self, cycle_id: str, mode: str):
        """
        Execute the complete news-driven trading workflow
        """
        self.logger.info(f"Starting trading workflow: {cycle_id} in {mode} mode")
        
        try:
            # Step 1: Collect News
            self.log_workflow_step(cycle_id, 'news_collection', 'started')
            news_result = self.collect_news(mode)
            self.current_cycle['news_collected'] = news_result.get('articles_collected', 0)
            self.log_workflow_step(cycle_id, 'news_collection', 'completed', news_result)
            
            # Step 2: Scan Securities (News-Driven)
            self.log_workflow_step(cycle_id, 'security_scanning', 'started')
            scan_result = self.scan_securities(mode)
            self.current_cycle['candidates_selected'] = len(scan_result.get('final_picks', []))
            self.log_workflow_step(cycle_id, 'security_scanning', 'completed', scan_result)
            
            # Step 3: Analyze Patterns on Selected Securities
            if scan_result.get('final_picks'):
                self.log_workflow_step(cycle_id, 'pattern_analysis', 'started')
                patterns = self.analyze_patterns(scan_result['final_picks'])
                self.current_cycle['patterns_analyzed'] = len(patterns)
                self.log_workflow_step(cycle_id, 'pattern_analysis', 'completed', patterns)
                
                # Step 4: Generate Trading Signals
                self.log_workflow_step(cycle_id, 'signal_generation', 'started')
                signals = self.generate_signals(scan_result['final_picks'], patterns)
                self.current_cycle['signals_generated'] = len(signals)
                self.log_workflow_step(cycle_id, 'signal_generation', 'completed', signals)
                
                # Step 5: Execute Trades
                if signals:
                    self.log_workflow_step(cycle_id, 'trade_execution', 'started')
                    trades = self.execute_trades(signals)
                    self.current_cycle['trades_executed'] = len(trades)
                    self.log_workflow_step(cycle_id, 'trade_execution', 'completed', trades)
            
            # Step 6: Update Cycle Complete
            self.complete_cycle(cycle_id, 'completed')
            
        except Exception as e:
            self.logger.error(f"Workflow error: {e}")
            self.complete_cycle(cycle_id, 'failed', str(e))
            
    def collect_news(self, mode: str) -> Dict:
        """Trigger news collection"""
        try:
            response = requests.post(
                f"{self.services['news_collection']['url']}/collect_news",
                json={'sources': 'all', 'mode': mode},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'News collection failed: {response.status_code}'}
                
        except Exception as e:
            self.logger.error(f"News collection error: {e}")
            return {'error': str(e)}
            
    def scan_securities(self, mode: str) -> Dict:
        """Scan for trading candidates based on news"""
        try:
            endpoint = '/scan_premarket' if mode == 'aggressive' else '/scan'
            response = requests.get(
                f"{self.services['security_scanner']['url']}{endpoint}",
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'Scanner failed: {response.status_code}'}
                
        except Exception as e:
            self.logger.error(f"Scanner error: {e}")
            return {'error': str(e)}
            
    def analyze_patterns(self, candidates: List[Dict]) -> List[Dict]:
        """Analyze patterns for selected candidates"""
        patterns = []
        
        for candidate in candidates:
            try:
                response = requests.post(
                    f"{self.services['pattern_analysis']['url']}/analyze_pattern",
                    json={
                        'symbol': candidate['symbol'],
                        'timeframe': '5min',
                        'context': {
                            'has_catalyst': True,
                            'catalyst_type': candidate.get('catalysts', [])[0] if candidate.get('catalysts') else None,
                            'market_state': 'pre-market' if candidate.get('has_pre_market_news') else 'regular'
                        }
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    pattern_data = response.json()
                    pattern_data['symbol'] = candidate['symbol']
                    patterns.append(pattern_data)
                    
            except Exception as e:
                self.logger.error(f"Pattern analysis error for {candidate['symbol']}: {e}")
                
        return patterns
        
    def generate_signals(self, candidates: List[Dict], patterns: List[Dict]) -> List[Dict]:
        """Generate trading signals"""
        signals = []
        
        # Match patterns to candidates
        pattern_map = {p['symbol']: p for p in patterns}
        
        for candidate in candidates:
            try:
                pattern_data = pattern_map.get(candidate['symbol'], {})
                
                response = requests.post(
                    f"{self.services['technical_analysis']['url']}/generate_signal",
                    json={
                        'symbol': candidate['symbol'],
                        'patterns': pattern_data.get('patterns', []),
                        'catalyst_data': {
                            'score': candidate.get('catalyst_score', 0),
                            'type': candidate.get('catalysts', [])[0] if candidate.get('catalysts') else None
                        }
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    signal = response.json()
                    if signal.get('signal') in ['BUY', 'SELL']:
                        signals.append(signal)
                        
            except Exception as e:
                self.logger.error(f"Signal generation error for {candidate['symbol']}: {e}")
                
        return signals
        
    def execute_trades(self, signals: List[Dict]) -> List[Dict]:
        """Execute trades via paper trading service"""
        trades = []
        
        for signal in signals:
            try:
                response = requests.post(
                    f"{self.services['paper_trading']['url']}/execute_trade",
                    json=signal,
                    timeout=10
                )
                
                if response.status_code == 200:
                    trade = response.json()
                    trades.append(trade)
                    self.logger.info(f"Executed trade for {signal['symbol']}: {trade}")
                    
            except Exception as e:
                self.logger.error(f"Trade execution error for {signal['symbol']}: {e}")
                
        return trades
        
    def complete_cycle(self, cycle_id: str, status: str, error: Optional[str] = None):
        """Complete a trading cycle"""
        try:
            self.current_cycle['status'] = status
            self.current_cycle['end_time'] = datetime.now().isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE trading_cycles
                SET status = ?, end_time = ?, news_collected = ?, 
                    candidates_selected = ?, patterns_analyzed = ?,
                    signals_generated = ?, trades_executed = ?
                WHERE cycle_id = ?
            ''', (
                status,
                self.current_cycle['end_time'],
                self.current_cycle.get('news_collected', 0),
                self.current_cycle.get('candidates_selected', 0),
                self.current_cycle.get('patterns_analyzed', 0),
                self.current_cycle.get('signals_generated', 0),
                self.current_cycle.get('trades_executed', 0),
                cycle_id
            ))
            
            conn.commit()
            conn.close()
            
            # Archive current cycle
            self.cycle_history.append(self.current_cycle.copy())
            if len(self.cycle_history) > 100:
                self.cycle_history.pop(0)
                
        except Exception as e:
            self.logger.error(f"Error completing cycle: {e}")
            
    def log_workflow_step(self, cycle_id: str, step: str, status: str, result: Any = None):
        """Log workflow step execution"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if status == 'started':
                cursor.execute('''
                    INSERT INTO workflow_log (cycle_id, step_name, status, start_time)
                    VALUES (?, ?, ?, ?)
                ''', (cycle_id, step, status, datetime.now()))
            else:
                cursor.execute('''
                    UPDATE workflow_log
                    SET status = ?, end_time = ?, result = ?
                    WHERE cycle_id = ? AND step_name = ? AND status = 'started'
                ''', (status, datetime.now(), json.dumps(result) if result else None, 
                      cycle_id, step))
                
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error logging workflow step: {e}")
            
    def check_service_health(self):
        """Check health of all services"""
        for service_name, service_info in self.services.items():
            try:
                response = requests.get(
                    f"{service_info['url']}{service_info['health_check']}",
                    timeout=5
                )
                
                if response.status_code == 200:
                    self.service_health[service_name] = {
                        'status': 'healthy',
                        'last_check': datetime.now().isoformat(),
                        'response_time_ms': response.elapsed.total_seconds() * 1000
                    }
                else:
                    self.service_health[service_name] = {
                        'status': 'unhealthy',
                        'last_check': datetime.now().isoformat(),
                        'error': f'HTTP {response.status_code}'
                    }
                    
            except Exception as e:
                self.service_health[service_name] = {
                    'status': 'unreachable',
                    'last_check': datetime.now().isoformat(),
                    'error': str(e)
                }
                
    def start_health_monitor(self):
        """Start background health monitoring"""
        def monitor():
            while True:
                self.check_service_health()
                time.sleep(30)  # Check every 30 seconds
                
        thread = threading.Thread(target=monitor)
        thread.daemon = True
        thread.start()
        
    def start_scheduler(self):
        """Start scheduled workflow execution"""
        def run_scheduled_jobs():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        # Schedule pre-market aggressive scanning
        if self.workflow_config['pre_market']['enabled']:
            schedule.every().day.at("04:00").do(
                lambda: self.start_new_cycle('aggressive')
            )
            
        # Schedule regular market hours scanning
        if self.workflow_config['market_hours']['enabled']:
            schedule.every(30).minutes.do(
                lambda: self.start_new_cycle('normal')
            ).tag('market_hours')
            
        thread = threading.Thread(target=run_scheduled_jobs)
        thread.daemon = True
        thread.start()
        
    def get_market_state(self) -> str:
        """Determine current market state"""
        now = datetime.now()
        hour = now.hour
        
        if 4 <= hour < 9.5:
            return 'pre_market'
        elif 9.5 <= hour < 16:
            return 'market_hours'
        elif 16 <= hour < 20:
            return 'after_hours'
        else:
            return 'closed'
            
    def run(self):
        """Start the coordination service"""
        self.logger.info("Starting News-Driven Coordination Service v2.0.0 on port 5000")
        self.logger.info("Workflow: News → Scan → Pattern → Signal → Trade → Track")
        self.logger.info("Pre-market aggressive mode enabled")
        
        self.app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == "__main__":
    service = NewsDriverCoordinationService()
    service.run()
