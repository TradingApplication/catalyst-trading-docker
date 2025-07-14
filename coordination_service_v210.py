#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: coordination_service.py
Version: 2.1.0
Last Updated: 2025-07-01
Purpose: Orchestrate news-driven trading workflow with PostgreSQL and environment variables

REVISION HISTORY:
v2.1.0 (2025-07-01) - Major refactor for production deployment
- Migrated from SQLite to PostgreSQL
- All configuration via environment variables
- Proper service discovery using Docker service names
- Fixed import paths and database utilities
- Added connection pooling and error handling

v2.0.0 (2025-06-27) - Complete rewrite for news-driven architecture
- News collection as primary driver
- Pre-market aggressive mode
- Source alignment awareness
- Outcome tracking integration

Description of Service:
This service coordinates the news-driven workflow:
1. News Collection → 2. Security Selection → 3. Pattern Analysis → 
4. Signal Generation → 5. Trade Execution → 6. Outcome Tracking
"""

import os
import json
import time
import logging
import requests
import threading
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from typing import Dict, List, Optional, Any
import schedule
from structlog import get_logger

# Import proper database utilities
from database_utils import (
    get_db_connection,
    create_trading_cycle,
    update_trading_cycle,
    log_workflow_step,
    update_service_health,
    get_configuration,
    health_check
)


class CoordinationService:
    """
    Coordination service for news-driven trading system
    Orchestrates workflow from news to trades
    """
    
    def __init__(self):
        # Initialize environment variables
        self.setup_environment()
        
        self.app = Flask(__name__)
        self.setup_logging()
        self.setup_routes()
        
        # Service registry with environment-based URLs
        self.services = {
            'news_collection': {
                'url': os.getenv('NEWS_SERVICE_URL', 'http://news-service:5008'),
                'port': 5008,
                'required': True,
                'health_check': '/health'
            },
            'security_scanner': {
                'url': os.getenv('SCANNER_SERVICE_URL', 'http://scanner-service:5001'),
                'port': 5001,
                'required': True,
                'health_check': '/health'
            },
            'pattern_analysis': {
                'url': os.getenv('PATTERN_SERVICE_URL', 'http://pattern-service:5002'),
                'port': 5002,
                'required': True,
                'health_check': '/health'
            },
            'technical_analysis': {
                'url': os.getenv('TECHNICAL_SERVICE_URL', 'http://technical-service:5003'),
                'port': 5003,
                'required': True,
                'health_check': '/health'
            },
            'paper_trading': {
                'url': os.getenv('TRADING_SERVICE_URL', 'http://trading-service:5005'),
                'port': 5005,
                'required': True,
                'health_check': '/health'
            },
            'reporting': {
                'url': os.getenv('REPORTING_SERVICE_URL', 'http://reporting-service:5009'),
                'port': 5009,
                'required': False,
                'health_check': '/health'
            },
            'web_dashboard': {
                'url': os.getenv('DASHBOARD_SERVICE_URL', 'http://web-dashboard:5010'),
                'port': 5010,
                'required': False,
                'health_check': '/health'
            }
        }
        
        # Workflow configuration
        self.workflow_config = {
            'pre_market': {
                'enabled': os.getenv('PREMARKET_ENABLED', 'true').lower() == 'true',
                'start_time': os.getenv('PREMARKET_START', '04:00'),
                'end_time': os.getenv('PREMARKET_END', '09:30'),
                'interval_minutes': int(os.getenv('PREMARKET_INTERVAL', '5')),
                'mode': 'aggressive'
            },
            'market_hours': {
                'enabled': os.getenv('MARKET_HOURS_ENABLED', 'true').lower() == 'true',
                'start_time': os.getenv('MARKET_START', '09:30'),
                'end_time': os.getenv('MARKET_END', '16:00'),
                'interval_minutes': int(os.getenv('MARKET_INTERVAL', '30')),
                'mode': 'normal'
            },
            'after_hours': {
                'enabled': os.getenv('AFTER_HOURS_ENABLED', 'true').lower() == 'true',
                'start_time': os.getenv('AFTER_HOURS_START', '16:00'),
                'end_time': os.getenv('AFTER_HOURS_END', '20:00'),
                'interval_minutes': int(os.getenv('AFTER_HOURS_INTERVAL', '60')),
                'mode': 'light'
            }
        }
        
        # Trading cycle state
        self.current_cycle = None
        self.cycle_history = []
        
        # Service health status
        self.service_health = {}
        
        # Start background threads
        self.start_health_monitor()
        self.start_scheduler()
        
        self.logger.info("Coordination Service v2.1.0 initialized", 
                        environment=os.getenv('ENVIRONMENT', 'development'))
        
    def setup_environment(self):
        """Setup environment variables and paths"""
        # Paths
        self.log_path = os.getenv('LOG_PATH', '/app/logs')
        self.data_path = os.getenv('DATA_PATH', '/app/data')
        
        # Create directories
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        
        # Service configuration
        self.service_name = 'coordination'
        self.port = int(os.getenv('PORT', '5000'))
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
                "service": "coordination",
                "version": "2.1.0",
                "mode": "news-driven",
                "database": db_health['database'],
                "redis": db_health['redis'],
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
                self.logger.info("Service registered/updated", 
                               service=service_name,
                               info=service_info)
                
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
        try:
            # Create cycle in database
            cycle_id = create_trading_cycle(mode)
            
            self.current_cycle = {
                'cycle_id': cycle_id,
                'start_time': datetime.now().isoformat(),
                'status': 'running',
                'mode': mode,
                'progress': {}
            }
            
            # Start workflow in background thread
            thread = threading.Thread(
                target=self.execute_trading_workflow, 
                args=(cycle_id, mode)
            )
            thread.daemon = True
            thread.start()
            
            return self.current_cycle
            
        except Exception as e:
            self.logger.error("Failed to start cycle", error=str(e))
            raise
        
    def execute_trading_workflow(self, cycle_id: str, mode: str):
        """
        Execute the complete news-driven trading workflow
        """
        self.logger.info("Starting trading workflow", 
                        cycle_id=cycle_id, 
                        mode=mode)
        
        try:
            # Step 1: Collect News
            workflow_id = log_workflow_step(cycle_id, 'news_collection', 'started')
            news_result = self.collect_news(mode)
            self.current_cycle['news_collected'] = news_result.get('articles_collected', 0)
            log_workflow_step(cycle_id, 'news_collection', 'completed', news_result)
            
            # Step 2: Scan Securities (News-Driven)
            workflow_id = log_workflow_step(cycle_id, 'security_scanning', 'started')
            scan_result = self.scan_securities(mode)
            self.current_cycle['candidates_selected'] = len(scan_result.get('final_picks', []))
            log_workflow_step(cycle_id, 'security_scanning', 'completed', scan_result)
            
            # Step 3: Analyze Patterns on Selected Securities
            if scan_result.get('final_picks'):
                workflow_id = log_workflow_step(cycle_id, 'pattern_analysis', 'started')
                patterns = self.analyze_patterns(scan_result['final_picks'])
                self.current_cycle['patterns_analyzed'] = len(patterns)
                log_workflow_step(cycle_id, 'pattern_analysis', 'completed', patterns)
                
                # Step 4: Generate Trading Signals
                workflow_id = log_workflow_step(cycle_id, 'signal_generation', 'started')
                signals = self.generate_signals(scan_result['final_picks'], patterns)
                self.current_cycle['signals_generated'] = len(signals)
                log_workflow_step(cycle_id, 'signal_generation', 'completed', signals)
                
                # Step 5: Execute Trades
                if signals:
                    workflow_id = log_workflow_step(cycle_id, 'trade_execution', 'started')
                    trades = self.execute_trades(signals)
                    self.current_cycle['trades_executed'] = len(trades)
                    log_workflow_step(cycle_id, 'trade_execution', 'completed', trades)
            
            # Step 6: Update Cycle Complete
            self.complete_cycle(cycle_id, 'completed')
            
        except Exception as e:
            self.logger.error("Workflow error", 
                            cycle_id=cycle_id,
                            error=str(e))
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
            self.logger.error("News collection error", error=str(e))
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
            self.logger.error("Scanner error", error=str(e))
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
                self.logger.error("Pattern analysis error",
                                symbol=candidate['symbol'],
                                error=str(e))
                
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
                self.logger.error("Signal generation error",
                                symbol=candidate['symbol'],
                                error=str(e))
                
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
                    self.logger.info("Trade executed",
                                   symbol=signal['symbol'],
                                   trade_id=trade.get('trade_id'))
                    
            except Exception as e:
                self.logger.error("Trade execution error",
                                symbol=signal['symbol'],
                                error=str(e))
                
        return trades
        
    def complete_cycle(self, cycle_id: str, status: str, error: Optional[str] = None):
        """Complete a trading cycle"""
        try:
            self.current_cycle['status'] = status
            self.current_cycle['end_time'] = datetime.now().isoformat()
            
            # Update database
            updates = {
                'status': status,
                'end_time': datetime.now(),
                'news_collected': self.current_cycle.get('news_collected', 0),
                'securities_scanned': self.current_cycle.get('candidates_selected', 0),
                'patterns_analyzed': self.current_cycle.get('patterns_analyzed', 0),
                'signals_generated': self.current_cycle.get('signals_generated', 0),
                'trades_executed': self.current_cycle.get('trades_executed', 0)
            }
            
            if error:
                updates['error_message'] = error
                
            update_trading_cycle(cycle_id, updates)
            
            # Archive current cycle
            self.cycle_history.append(self.current_cycle.copy())
            if len(self.cycle_history) > 100:
                self.cycle_history.pop(0)
                
        except Exception as e:
            self.logger.error("Error completing cycle", 
                            cycle_id=cycle_id,
                            error=str(e))
            
    def check_service_health(self):
        """Check health of all services"""
        for service_name, service_info in self.services.items():
            try:
                start_time = time.time()
                response = requests.get(
                    f"{service_info['url']}{service_info['health_check']}",
                    timeout=5
                )
                response_time = int((time.time() - start_time) * 1000)
                
                if response.status_code == 200:
                    self.service_health[service_name] = {
                        'status': 'healthy',
                        'last_check': datetime.now().isoformat(),
                        'response_time_ms': response_time
                    }
                    update_service_health(service_name, 'healthy', response_time)
                else:
                    error_msg = f'HTTP {response.status_code}'
                    self.service_health[service_name] = {
                        'status': 'unhealthy',
                        'last_check': datetime.now().isoformat(),
                        'error': error_msg
                    }
                    update_service_health(service_name, 'unhealthy', 
                                        response_time, error_msg)
                    
            except Exception as e:
                error_msg = str(e)
                self.service_health[service_name] = {
                    'status': 'unreachable',
                    'last_check': datetime.now().isoformat(),
                    'error': error_msg
                }
                update_service_health(service_name, 'unreachable', 
                                    None, error_msg)
                
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
            schedule.every().day.at(
                self.workflow_config['pre_market']['start_time']
            ).do(lambda: self.start_new_cycle('aggressive'))
            
        # Schedule regular market hours scanning
        if self.workflow_config['market_hours']['enabled']:
            interval = self.workflow_config['market_hours']['interval_minutes']
            schedule.every(interval).minutes.do(
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
        self.logger.info("Starting Coordination Service",
                        version="2.1.0",
                        port=self.port,
                        environment=os.getenv('ENVIRONMENT', 'development'))
        
        self.app.run(host='0.0.0.0', port=self.port, debug=False)


if __name__ == "__main__":
    service = CoordinationService()
    service.run()