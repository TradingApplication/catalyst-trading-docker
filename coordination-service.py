#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: coordination_service.py
Version: 2.1.3
Last Updated: 2025-07-11
Purpose: Orchestrate news-driven trading workflow with PostgreSQL and environment variables

REVISION HISTORY:
v2.1.3 (2025-07-11) - Fixed HTTP method for news collection
- Added method='POST' to news collection execute_step call
- Fixed 405 Method Not Allowed error

v2.1.2 (2025-07-10) - Fixed scanner service name typo
- Fixed scanner service name in services dictionary
- Removed extra quotes from scanner service name

v2.1.1 (2025-07-10) - Fixed health check endpoint
- Fixed KeyError: 'database' by using correct key 'postgresql'
- Maintained compatibility with database_utils health_check response

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
import sys
import json
import time
import logging
import threading
import schedule
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from typing import Dict, List, Optional, Any
from structlog import get_logger
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import database utilities
try:
    from database_utils import (
        get_db_connection,
        create_trading_cycle,
        update_trading_cycle,
        update_service_health,
        get_service_health,
        health_check,
        get_redis,
        log_workflow_step
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Attempting basic import...")
    from database_utils import get_db_connection, health_check, get_redis
    
    # Define missing functions if not available
    def create_trading_cycle(mode='normal'):
        return f"CYCLE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{mode}"
    
    def update_trading_cycle(cycle_id, updates):
        pass
    
    def update_service_health(service_name, status, response_time, error=None):
        pass
    
    def get_service_health(service_name=None, hours=24):
        return []
    
    def log_workflow_step(cycle_id, step_name, status, result=None, records_processed=None, records_output=None, error_message=None):
        pass


class NewsDriverCoordinator:
    """
    Coordinates the news-driven trading workflow
    """
    
    def __init__(self):
        # Initialize environment
        self.setup_environment()
        
        self.app = Flask(__name__)
        self.setup_logging()
        self.setup_routes()
        
        # Initialize Redis client
        self.redis_client = get_redis()
        
        # Service registry with enhanced URLs
        self.services = {
            'news': {
                'name': 'news_service',
                'url': os.getenv('NEWS_SERVICE_URL', 'http://news-service:5008'),
                'port': 5008,
                'required': True,
                'health_check': '/health'
            },
            'scanner': {
                'name': 'scanner_service',
                'url': os.getenv('SCANNER_SERVICE_URL', 'http://scanner-service:5001'),
                'port': 5001,
                'required': True,
                'health_check': '/health'
            },
            'pattern': {
                'name': 'pattern_service',
                'url': os.getenv('PATTERN_SERVICE_URL', 'http://pattern-service:5002'),
                'port': 5002,
                'required': True,
                'health_check': '/health'
            },
            'technical': {
                'name': 'technical_service',
                'url': os.getenv('TECHNICAL_SERVICE_URL', 'http://technical-service:5003'),
                'port': 5003,
                'required': True,
                'health_check': '/health'
            },
            'trading': {
                'name': 'trading_service',
                'url': os.getenv('TRADING_SERVICE_URL', 'http://trading-service:5005'),
                'port': 5005,
                'required': True,
                'health_check': '/health'
            },
            'reporting': {
                'name': 'reporting_service',
                'url': os.getenv('REPORTING_SERVICE_URL', 'http://reporting-service:5009'),
                'port': 5009,
                'required': False,
                'health_check': '/health'
            },
            'dashboard': {
                'name': 'dashboard_service',
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
        
        self.logger.info("Coordination Service v2.1.3 initialized", 
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
            # Fixed: Use 'postgresql' key instead of 'database'
            return jsonify({
                "status": "healthy" if db_health['postgresql']['status'] == 'healthy' else "degraded",
                "service": "coordination",
                "version": "2.1.3",
                "mode": "news-driven",
                "database": db_health['postgresql']['status'],
                "redis": db_health['redis']['status'],
                "timestamp": datetime.now().isoformat()
            })
            
        @self.app.route('/register_service', methods=['POST'])
        def register_service():
            """Register a service with coordination"""
            data = request.json
            service_name = data.get('service_name')
            service_info = data.get('service_info', {})
            
            # Find the service by name and update its info
            for key, service in self.services.items():
                if service['name'] == service_name:
                    service.update(service_info)
                    self.logger.info("Service registered/updated", 
                                   service=service_name,
                                   info=service_info)
                    break
                
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
            """Get health of all services"""
            self.check_service_health()
            return jsonify({
                'services': self.service_health,
                'timestamp': datetime.now().isoformat()
            })
            
        @self.app.route('/workflow_status', methods=['GET'])
        def workflow_status():
            """Get current workflow configuration and status"""
            return jsonify({
                'config': self.workflow_config,
                'current_cycle': self.current_cycle,
                'cycle_history': self.cycle_history[-10:],  # Last 10 cycles
                'timestamp': datetime.now().isoformat()
            })
            
    def start_new_cycle(self, mode: str) -> Dict:
        """Start a new trading cycle"""
        try:
            # Create cycle using database function
            cycle_id = create_trading_cycle(mode)
            
            self.current_cycle = {
                'cycle_id': cycle_id,
                'mode': mode,
                'status': 'running',
                'start_time': datetime.now(),
                'steps_completed': [],
                'errors': []
            }
            
            # Log workflow start
            log_workflow_step(cycle_id, 'workflow_start', 'started', 
                            result={'mode': mode})
            
            # Start workflow in background
            thread = threading.Thread(
                target=self.run_trading_workflow,
                args=(cycle_id, mode)
            )
            thread.daemon = True
            thread.start()
            
            return self.current_cycle
            
        except Exception as e:
            self.logger.error("Error starting cycle", error=str(e))
            return {'error': str(e)}
            
    def run_trading_workflow(self, cycle_id: str, mode: str):
        """Execute the complete trading workflow"""
        try:
            self.logger.info("Starting trading workflow", 
                           cycle_id=cycle_id, mode=mode)
            
            # Step 1: Collect News - FIXED: Added method='POST'
            log_workflow_step(cycle_id, 'news_collection', 'started')
            news_result = self.execute_step('news', '/collect_news', {
                'mode': mode,
                'cycle_id': cycle_id
            }, method='POST')  # FIXED: Added method='POST' here
            
            if news_result.get('status') == 'success' or news_result.get('articles_collected'):
                articles_collected = news_result.get('articles_collected', 0)
                self.current_cycle['news_collected'] = articles_collected
                self.current_cycle['steps_completed'].append('news_collection')
                log_workflow_step(cycle_id, 'news_collection', 'completed',
                                result=news_result, records_processed=articles_collected)
            else:
                log_workflow_step(cycle_id, 'news_collection', 'failed',
                                error_message=news_result.get('error', 'Unknown error'))
                
            # Step 2: Security Scanning
            log_workflow_step(cycle_id, 'security_scanning', 'started')
            scan_result = self.execute_step('scanner', '/scan', {
                'mode': mode,
                'force': 'true'
            })
            
            if scan_result.get('securities') or scan_result.get('final_picks'):
                candidates = scan_result.get('securities', scan_result.get('final_picks', []))
                self.current_cycle['candidates_selected'] = len(candidates)
                self.current_cycle['steps_completed'].append('security_scanning')
                
                log_workflow_step(cycle_id, 'security_scanning', 'completed',
                                result=scan_result,
                                records_processed=scan_result.get('metadata', {}).get('total_scanned', 0),
                                records_output=len(candidates))
                
                # Step 3: Pattern Analysis for each candidate
                patterns_analyzed = 0
                for candidate in candidates[:5]:  # Top 5 only
                    pattern_result = self.execute_step('pattern', '/analyze_pattern', {
                        'symbol': candidate.get('symbol'),
                        'catalyst_data': candidate.get('catalyst_data', {})
                    }, method='POST')
                    
                    if pattern_result.get('status') == 'success' or pattern_result.get('patterns'):
                        patterns_analyzed += 1
                
                if patterns_analyzed > 0:
                    self.current_cycle['patterns_analyzed'] = patterns_analyzed
                    self.current_cycle['steps_completed'].append('pattern_analysis')
                    log_workflow_step(cycle_id, 'pattern_analysis', 'completed',
                                    records_processed=len(candidates[:5]),
                                    records_output=patterns_analyzed)
                
                # Step 4: Technical Analysis & Signal Generation
                signals_generated = 0
                for candidate in candidates[:5]:
                    signal_result = self.execute_step('technical', '/analyze', {
                        'symbol': candidate.get('symbol'),
                        'catalyst_score': candidate.get('catalyst_score', candidate.get('composite_score', 0))
                    }, method='POST')
                    
                    if signal_result.get('signal'):
                        signals_generated += 1
                
                if signals_generated > 0:
                    self.current_cycle['signals_generated'] = signals_generated
                    self.current_cycle['steps_completed'].append('signal_generation')
                    log_workflow_step(cycle_id, 'signal_generation', 'completed',
                                    records_processed=len(candidates[:5]),
                                    records_output=signals_generated)
                
                # Step 5: Trade Execution (if enabled)
                if os.getenv('TRADING_ENABLED', 'false').lower() == 'true' and signals_generated > 0:
                    log_workflow_step(cycle_id, 'trade_execution', 'started')
                    trade_result = self.execute_step('trading', '/execute_signals', {
                        'cycle_id': cycle_id
                    }, method='POST')
                    
                    if trade_result.get('trades_executed'):
                        self.current_cycle['trades_executed'] = trade_result.get('trades_executed', 0)
                        self.current_cycle['steps_completed'].append('trade_execution')
                        log_workflow_step(cycle_id, 'trade_execution', 'completed',
                                        result=trade_result,
                                        records_output=trade_result.get('trades_executed', 0))
            else:
                log_workflow_step(cycle_id, 'security_scanning', 'failed',
                                error_message=scan_result.get('error', 'No candidates found'))
                        
            # Mark cycle as complete
            self.complete_cycle(cycle_id, 'completed')
            
        except Exception as e:
            self.logger.error("Error in trading workflow", 
                            cycle_id=cycle_id, error=str(e))
            self.current_cycle['errors'].append(str(e))
            log_workflow_step(cycle_id, 'workflow_error', 'failed',
                            error_message=str(e))
            self.complete_cycle(cycle_id, 'failed', str(e))
            
    def execute_step(self, service_key: str, endpoint: str, 
                    data: Dict, method: str = 'GET') -> Dict:
        """Execute a workflow step"""
        try:
            service = self.services.get(service_key)
            if not service:
                raise ValueError(f"Unknown service: {service_key}")
                
            url = f"{service['url']}{endpoint}"
            
            if method == 'GET':
                response = requests.get(url, params=data, timeout=30)
            else:
                response = requests.post(url, json=data, timeout=30)
                
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Service returned {response.status_code}"
                self.logger.error("Step execution failed", 
                                service=service_key,
                                endpoint=endpoint,
                                error=error_msg)
                return {'status': 'error', 'error': error_msg}
                
        except Exception as e:
            self.logger.error("Step execution error",
                            service=service_key,
                            endpoint=endpoint,
                            error=str(e))
            return {'status': 'error', 'error': str(e)}
            
    def complete_cycle(self, cycle_id: str, status: str, error: str = None):
        """Complete a trading cycle"""
        try:
            self.current_cycle['status'] = status
            self.current_cycle['end_time'] = datetime.now()
            self.current_cycle['duration'] = (
                self.current_cycle['end_time'] - self.current_cycle['start_time']
            ).total_seconds()
            
            # Update database
            updates = {
                'status': status,
                'end_time': datetime.now(),
                'news_collected': self.current_cycle.get('news_collected', 0),
                'securities_scanned': self.current_cycle.get('candidates_selected', 0),
                'patterns_analyzed': self.current_cycle.get('patterns_analyzed', 0),
                'signals_generated': self.current_cycle.get('signals_generated', 0),
                'trades_executed': self.current_cycle.get('trades_executed', 0),
                'cycle_pnl': 0,  # Would be calculated from actual trades
                'metadata': {
                    'steps_completed': self.current_cycle.get('steps_completed', []),
                    'errors': self.current_cycle.get('errors', [])
                }
            }
            
            if error:
                updates['error_message'] = error
                
            update_trading_cycle(cycle_id, updates)
            
            # Log workflow completion
            log_workflow_step(cycle_id, 'workflow_complete', status,
                            result={'duration_seconds': self.current_cycle['duration']})
            
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
                    update_service_health(service_info['name'], 'healthy', response_time)
                else:
                    error_msg = f'HTTP {response.status_code}'
                    self.service_health[service_name] = {
                        'status': 'unhealthy',
                        'last_check': datetime.now().isoformat(),
                        'error': error_msg
                    }
                    update_service_health(service_info['name'], 'unhealthy', 
                                        response_time, error_msg)
                    
            except Exception as e:
                error_msg = str(e)
                self.service_health[service_name] = {
                    'status': 'unreachable',
                    'last_check': datetime.now().isoformat(),
                    'error': error_msg
                }
                update_service_health(service_info['name'], 'unreachable', 0, error_msg)
                
    def start_health_monitor(self):
        """Start background health monitoring"""
        def monitor():
            while True:
                try:
                    self.check_service_health()
                except Exception as e:
                    self.logger.error("Health monitor error", error=str(e))
                time.sleep(30)  # Check every 30 seconds
                
        thread = threading.Thread(target=monitor)
        thread.daemon = True
        thread.start()
        
    def start_scheduler(self):
        """Start the workflow scheduler"""
        def schedule_loop():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        # Schedule pre-market scans
        if self.workflow_config['pre_market']['enabled']:
            schedule.every().day.at(
                self.workflow_config['pre_market']['start_time']
            ).do(lambda: self.start_new_cycle('aggressive'))
            
        # Schedule market hours scans
        if self.workflow_config['market_hours']['enabled']:
            schedule.every().day.at(
                self.workflow_config['market_hours']['start_time']
            ).do(lambda: self.start_new_cycle('normal'))
            
        thread = threading.Thread(target=schedule_loop)
        thread.daemon = True
        thread.start()
        
    def run(self):
        """Start the coordination service"""
        self.logger.info("Starting Coordination Service",
                        port=self.port,
                        environment=os.getenv('ENVIRONMENT', 'development'))
        
        # Use host='0.0.0.0' for Docker
        self.app.run(
            host='0.0.0.0',
            port=self.port,
            debug=False
        )


if __name__ == '__main__':
    coordinator = NewsDriverCoordinator()
    coordinator.run()
