#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: reporting_service.py
Version: 2.1.2
Last Updated: 2025-07-08
Purpose: Analytics and reporting service for trading performance and system health

REVISION HISTORY:
v2.1.2 (2025-07-14) - Fixed endpoint calls
- Table reference change FROM trades → FROM trade_records

v2.1.1 (2025-07-08) - Fixed endpoint calls
- Changed /service_status to /service_health in two places
- Fixed _generate_system_health_report method
- Fixed _get_service_performance_metrics method

v2.1.0 (2025-07-01) - Initial implementation for production deployment
- PostgreSQL database integration
- Comprehensive performance analytics
- Real-time system health monitoring
- Trade analytics and pattern effectiveness reporting
- Daily/weekly/monthly summary reports
- Risk management metrics
- Service orchestration integration
- Environment variable configuration
- Structured logging with rotation

Description of Service:
The Reporting Service provides comprehensive analytics and reporting capabilities:
1. Trading performance analysis (P&L, win rates, Sharpe ratio)
2. Pattern effectiveness tracking
3. System health monitoring and service status
4. Risk management metrics and alerts
5. Daily, weekly, and monthly summary reports
6. Portfolio analysis and position tracking
7. Market analysis effectiveness
8. Service uptime and performance metrics
"""

import os
import sys
import logging
import json
import traceback
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
import structlog
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection utilities
def get_db_connection():
    """Get PostgreSQL database connection"""
    try:
        conn = psycopg2.connect(
            os.getenv('DATABASE_URL', 'postgresql://catalyst_user:password@db:5432/catalyst_trading'),
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        logger.error("Database connection failed", error=str(e))
        raise

def get_redis_connection():
    """Get Redis connection for caching"""
    try:
        return redis.from_url(os.getenv('REDIS_URL', 'redis://redis:6379/0'))
    except Exception as e:
        logger.error("Redis connection failed", error=str(e))
        return None

# Configure structured logging
log_path = os.getenv('LOG_PATH', '/app/logs')
Path(log_path).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_path, 'reporting.log')),
        logging.StreamHandler()
    ]
)

logger = structlog.get_logger(__name__)

@dataclass
class PerformanceMetrics:
    """Trading performance metrics data structure"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_volume: int = 0
    total_commissions: float = 0.0

@dataclass
class RiskMetrics:
    """Risk management metrics"""
    current_positions: int = 0
    total_exposure: float = 0.0
    max_position_risk: float = 0.0
    portfolio_beta: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    correlation_matrix: Dict = None

class ReportingService:
    """Main reporting service class"""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Service configuration
        self.service_name = "reporting_service"
        self.port = int(os.getenv('REPORTING_SERVICE_PORT', 5009))
        self.coordination_url = os.getenv('COORDINATION_URL', 'http://coordination-service:5000')
        
        # Database connections
        self.db_pool = None
        self.redis_client = get_redis_connection()
        
        # Cache settings
        self.cache_ttl = int(os.getenv('CACHE_TTL_SECONDS', 300))  # 5 minutes
        
        # Initialize service
        self._setup_routes()
        self._register_with_coordinator()
        
        logger.info("Reporting Service initialized", port=self.port)

    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            try:
                # Test database connection
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                
                # Test Redis connection
                redis_status = "healthy" if self.redis_client and self.redis_client.ping() else "unavailable"
                
                return jsonify({
                    'status': 'healthy',
                    'service': self.service_name,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'database': 'healthy',
                    'redis': redis_status,
                    'version': '2.1.1'
                }), 200
                
            except Exception as e:
                logger.error("Health check failed", error=str(e))
                return jsonify({
                    'status': 'unhealthy',
                    'service': self.service_name,
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }), 500

        @self.app.route('/daily_summary', methods=['GET'])
        def daily_summary():
            """Generate daily trading summary"""
            try:
                date_str = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
                target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                
                summary = self._generate_daily_summary(target_date)
                return jsonify(summary), 200
                
            except Exception as e:
                logger.error("Daily summary generation failed", error=str(e))
                return jsonify({'error': str(e)}), 500

        @self.app.route('/trading_performance', methods=['GET'])
        def trading_performance():
            """Get comprehensive trading performance metrics"""
            try:
                period = request.args.get('period', '30')  # days
                include_positions = request.args.get('positions', 'false').lower() == 'true'
                
                performance = self._calculate_trading_performance(int(period), include_positions)
                return jsonify(performance), 200
                
            except Exception as e:
                logger.error("Trading performance calculation failed", error=str(e))
                return jsonify({'error': str(e)}), 500

        @self.app.route('/pattern_effectiveness', methods=['GET'])
        def pattern_effectiveness():
            """Analyze pattern detection effectiveness"""
            try:
                period = request.args.get('period', '30')
                pattern_type = request.args.get('pattern_type', 'all')
                
                effectiveness = self._analyze_pattern_effectiveness(int(period), pattern_type)
                return jsonify(effectiveness), 200
                
            except Exception as e:
                logger.error("Pattern effectiveness analysis failed", error=str(e))
                return jsonify({'error': str(e)}), 500

        @self.app.route('/system_health', methods=['GET'])
        def system_health():
            """Get comprehensive system health report"""
            try:
                health_report = self._generate_system_health_report()
                return jsonify(health_report), 200
                
            except Exception as e:
                logger.error("System health report generation failed", error=str(e))
                return jsonify({'error': str(e)}), 500

        @self.app.route('/risk_metrics', methods=['GET'])
        def risk_metrics():
            """Get current risk management metrics"""
            try:
                metrics = self._calculate_risk_metrics()
                return jsonify(metrics), 200
                
            except Exception as e:
                logger.error("Risk metrics calculation failed", error=str(e))
                return jsonify({'error': str(e)}), 500

        @self.app.route('/portfolio_analysis', methods=['GET'])
        def portfolio_analysis():
            """Comprehensive portfolio analysis"""
            try:
                analysis = self._analyze_portfolio()
                return jsonify(analysis), 200
                
            except Exception as e:
                logger.error("Portfolio analysis failed", error=str(e))
                return jsonify({'error': str(e)}), 500

        @self.app.route('/weekly_report', methods=['GET'])
        def weekly_report():
            """Generate weekly trading report"""
            try:
                week_start = request.args.get('week_start')
                if week_start:
                    start_date = datetime.strptime(week_start, '%Y-%m-%d').date()
                else:
                    # Default to current week
                    today = datetime.now().date()
                    start_date = today - timedelta(days=today.weekday())
                
                report = self._generate_weekly_report(start_date)
                return jsonify(report), 200
                
            except Exception as e:
                logger.error("Weekly report generation failed", error=str(e))
                return jsonify({'error': str(e)}), 500

        @self.app.route('/service_performance', methods=['GET'])
        def service_performance():
            """Get individual service performance metrics"""
            try:
                performance = self._get_service_performance_metrics()
                return jsonify(performance), 200
                
            except Exception as e:
                logger.error("Service performance metrics failed", error=str(e))
                return jsonify({'error': str(e)}), 500

    def _register_with_coordinator(self):
        """Register this service with the coordination service"""
        try:
            registration_data = {
                'service_name': self.service_name,
                'port': self.port,
                'status': 'healthy',
                'capabilities': [
                    'daily_summary',
                    'trading_performance',
                    'pattern_effectiveness',
                    'system_health',
                    'risk_metrics',
                    'portfolio_analysis'
                ]
            }
            
            response = requests.post(
                f"{self.coordination_url}/register_service",
                json=registration_data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Successfully registered with coordination service")
            else:
                logger.warning("Failed to register with coordination service", 
                             status_code=response.status_code)
                
        except Exception as e:
            logger.error("Service registration failed", error=str(e))

    def _generate_daily_summary(self, target_date) -> Dict:
        """Generate comprehensive daily trading summary"""
        cache_key = f"daily_summary:{target_date}"
        
        # Check cache first
        if self.redis_client:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get trading activity for the day
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                        SUM(pnl) as total_pnl,
                        AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                        AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
                        MAX(pnl) as largest_win,
                        MIN(pnl) as largest_loss,
                        SUM(quantity) as total_volume,
                        SUM(commission) as total_commissions
                    FROM trade_records 
                    WHERE DATE(executed_at) = %s
                """, (target_date,))
                
                trade_stats = cur.fetchone()
                
                # Get scanning activity
                cur.execute("""
                    SELECT 
                        COUNT(DISTINCT symbol) as symbols_scanned,
                        COUNT(*) as scan_results,
                        AVG(confidence_score) as avg_confidence
                    FROM trading_candidates 
                    WHERE DATE(created_at) = %s
                """, (target_date,))
                
                scan_stats = cur.fetchone()
                
                # Get pattern analysis results
                cur.execute("""
                    SELECT 
                        pattern_type,
                        COUNT(*) as count,
                        AVG(confidence) as avg_confidence
                    FROM pattern_analysis 
                    WHERE DATE(created_at) = %s
                    GROUP BY pattern_type
                """, (target_date,))
                
                pattern_stats = cur.fetchall()
        
        # Calculate derived metrics
        total_trades = trade_stats['total_trades'] or 0
        winning_trades = trade_stats['winning_trades'] or 0
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = 0
        if trade_stats['avg_loss'] and trade_stats['avg_loss'] < 0:
            gross_profit = abs(trade_stats['avg_win'] or 0) * winning_trades
            gross_loss = abs(trade_stats['avg_loss'] or 0) * (trade_stats['losing_trades'] or 0)
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        summary = {
            'date': target_date.isoformat(),
            'trading_summary': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': trade_stats['losing_trades'] or 0,
                'win_rate': round(win_rate, 2),
                'total_pnl': float(trade_stats['total_pnl'] or 0),
                'average_win': float(trade_stats['avg_win'] or 0),
                'average_loss': float(trade_stats['avg_loss'] or 0),
                'largest_win': float(trade_stats['largest_win'] or 0),
                'largest_loss': float(trade_stats['largest_loss'] or 0),
                'profit_factor': round(profit_factor, 2),
                'total_volume': trade_stats['total_volume'] or 0,
                'total_commissions': float(trade_stats['total_commissions'] or 0)
            },
            'scanning_summary': {
                'symbols_scanned': scan_stats['symbols_scanned'] or 0,
                'scan_results': scan_stats['scan_results'] or 0,
                'average_confidence': float(scan_stats['avg_confidence'] or 0)
            },
            'pattern_summary': [
                {
                    'pattern_type': pattern['pattern_type'],
                    'count': pattern['count'],
                    'average_confidence': float(pattern['avg_confidence'])
                }
                for pattern in pattern_stats
            ],
            'generated_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Cache the result
        if self.redis_client:
            self.redis_client.setex(
                cache_key, 
                self.cache_ttl, 
                json.dumps(summary, default=str)
            )
        
        return summary

    def _calculate_trading_performance(self, period_days: int, include_positions: bool) -> Dict:
        """Calculate comprehensive trading performance metrics"""
        cache_key = f"trading_performance:{period_days}:{include_positions}"
        
        if self.redis_client:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        start_date = datetime.now() - timedelta(days=period_days)
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get all trade_records in period
                cur.execute("""
                    SELECT * FROM trade_records 
                    WHERE executed_at >= %s 
                    ORDER BY executed_at
                """, (start_date,))
                
                trade_records = cur.fetchall()
                
                # Get current positions if requested
                positions = []
                if include_positions:
                    cur.execute("""
                        SELECT * FROM positions 
                        WHERE is_open = true
                    """)
                    positions = cur.fetchall()
        
        # Calculate performance metrics
        if not trade_records:
            return {
                'period_days': period_days,
                'total_trades': 0,
                'performance': PerformanceMetrics().__dict__,
                'daily_returns': [],
                'positions': positions if include_positions else None,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
        
        # Calculate metrics
        total_trades = len(trade_records)
        winning_trades = len([t for t in trade_records if (t['pnl'] or 0) > 0])
        losing_trades = len([t for t in trade_records if (t['pnl'] or 0) < 0])
        
        pnls = [float(t['pnl'] or 0) for t in trade_records]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        total_pnl = sum(pnls)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        average_win = sum(wins) / len(wins) if wins else 0
        average_loss = sum(losses) / len(losses) if losses else 0
        
        # Calculate Sharpe ratio (simplified)
        daily_returns = self._calculate_daily_returns(trade_records)
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        
        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown(daily_returns)
        
        # Calculate profit factor
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        performance = PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate, 2),
            total_pnl=round(total_pnl, 2),
            average_win=round(average_win, 2),
            average_loss=round(average_loss, 2),
            largest_win=max(pnls) if pnls else 0,
            largest_loss=min(pnls) if pnls else 0,
            profit_factor=round(profit_factor, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            max_drawdown=round(max_drawdown, 2),
            total_volume=sum([t['quantity'] or 0 for t in trade_records]),
            total_commissions=sum([float(t['commission'] or 0) for t in trade_records])
        )
        
        result = {
            'period_days': period_days,
            'performance': performance.__dict__,
            'daily_returns': daily_returns,
            'positions': positions if include_positions else None,
            'generated_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Cache result
        if self.redis_client:
            self.redis_client.setex(
                cache_key, 
                self.cache_ttl, 
                json.dumps(result, default=str)
            )
        
        return result

    def _analyze_pattern_effectiveness(self, period_days: int, pattern_type: str) -> Dict:
        """Analyze effectiveness of pattern detection"""
        start_date = datetime.now() - timedelta(days=period_days)
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Base query for pattern analysis
                base_query = """
                    SELECT 
                        pa.pattern_type,
                        pa.symbol,
                        pa.confidence,
                        pa.created_at,
                        t.pnl,
                        t.executed_at
                    FROM pattern_analysis pa
                    LEFT JOIN trading_candidates tc ON pa.symbol = tc.symbol 
                        AND DATE(pa.created_at) = DATE(tc.created_at)
                    LEFT JOIN trade_records t ON tc.symbol = t.symbol 
                        AND t.executed_at >= pa.created_at
                        AND t.executed_at <= pa.created_at + INTERVAL '24 hours'
                    WHERE pa.created_at >= %s
                """
                
                params = [start_date]
                if pattern_type != 'all':
                    base_query += " AND pa.pattern_type = %s"
                    params.append(pattern_type)
                
                cur.execute(base_query, params)
                pattern_data = cur.fetchall()
        
        # Analyze effectiveness
        pattern_stats = {}
        
        for row in pattern_data:
            ptype = row['pattern_type']
            if ptype not in pattern_stats:
                pattern_stats[ptype] = {
                    'total_detections': 0,
                    'trades_executed': 0,
                    'profitable_trades': 0,
                    'total_pnl': 0,
                    'avg_confidence': 0,
                    'confidence_sum': 0
                }
            
            stats = pattern_stats[ptype]
            stats['total_detections'] += 1
            stats['confidence_sum'] += float(row['confidence'] or 0)
            
            if row['pnl'] is not None:
                stats['trades_executed'] += 1
                pnl = float(row['pnl'])
                stats['total_pnl'] += pnl
                if pnl > 0:
                    stats['profitable_trades'] += 1
        
        # Calculate derived metrics
        for ptype, stats in pattern_stats.items():
            if stats['total_detections'] > 0:
                stats['avg_confidence'] = stats['confidence_sum'] / stats['total_detections']
                stats['execution_rate'] = (stats['trades_executed'] / stats['total_detections']) * 100
                
            if stats['trades_executed'] > 0:
                stats['success_rate'] = (stats['profitable_trades'] / stats['trades_executed']) * 100
                stats['avg_pnl_per_trade'] = stats['total_pnl'] / stats['trades_executed']
            else:
                stats['success_rate'] = 0
                stats['avg_pnl_per_trade'] = 0
            
            # Remove intermediate calculations
            del stats['confidence_sum']
        
        return {
            'period_days': period_days,
            'pattern_type_filter': pattern_type,
            'pattern_effectiveness': pattern_stats,
            'generated_at': datetime.now(timezone.utc).isoformat()
        }

    def _generate_system_health_report(self) -> Dict:
        """Generate comprehensive system health report"""
        # Get service health from coordination service
        try:
            # FIXED: Changed from /service_status to /service_health
            response = requests.get(f"{self.coordination_url}/service_health", timeout=10)
            service_health = response.json() if response.status_code == 200 else {}
        except:
            service_health = {}
        
        # Database health
        db_health = self._check_database_health()
        
        # Redis health
        redis_health = {
            'status': 'healthy' if self.redis_client and self.redis_client.ping() else 'unhealthy',
            'memory_usage': None
        }
        
        if self.redis_client:
            try:
                info = self.redis_client.info()
                redis_health['memory_usage'] = info.get('used_memory_human', 'unknown')
            except:
                pass
        
        # Recent errors from logs
        recent_errors = self._get_recent_errors()
        
        return {
            'system_status': 'healthy' if db_health['status'] == 'healthy' else 'degraded',
            'services': service_health,
            'database': db_health,
            'redis': redis_health,
            'recent_errors': recent_errors,
            'uptime_stats': self._get_uptime_stats(),
            'generated_at': datetime.now(timezone.utc).isoformat()
        }

    def _calculate_risk_metrics(self) -> Dict:
        """Calculate current risk management metrics"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Current positions
                cur.execute("""
                    SELECT 
                        COUNT(*) as position_count,
                        SUM(ABS(quantity * current_price)) as total_exposure,
                        MAX(ABS(quantity * current_price)) as max_position_value
                    FROM positions 
                    WHERE is_open = true
                """)
                
                position_data = cur.fetchone()
                
                # Recent P&L for VaR calculation
                cur.execute("""
                    SELECT pnl 
                    FROM trade_records 
                    WHERE executed_at >= NOW() - INTERVAL '30 days'
                    ORDER BY executed_at
                """)
                
                recent_pnls = [float(row['pnl'] or 0) for row in cur.fetchall()]
        
        # Calculate VaR (95% confidence)
        var_95 = 0
        if recent_pnls:
            var_95 = np.percentile(recent_pnls, 5)  # 5th percentile for VaR
        
        risk_metrics = RiskMetrics(
            current_positions=position_data['position_count'] or 0,
            total_exposure=float(position_data['total_exposure'] or 0),
            max_position_risk=float(position_data['max_position_value'] or 0),
            var_95=float(var_95)
        )
        
        return {
            'risk_metrics': risk_metrics.__dict__,
            'risk_limits': {
                'max_positions': int(os.getenv('MAX_POSITIONS', 5)),
                'max_position_size_pct': float(os.getenv('POSITION_SIZE_PCT', 20)),
                'stop_loss_pct': float(os.getenv('STOP_LOSS_PCT', 2))
            },
            'alerts': self._check_risk_alerts(risk_metrics),
            'generated_at': datetime.now(timezone.utc).isoformat()
        }

    def _analyze_portfolio(self) -> Dict:
        """Comprehensive portfolio analysis"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Portfolio composition
                cur.execute("""
                    SELECT 
                        symbol,
                        quantity,
                        entry_price,
                        current_price,
                        (current_price - entry_price) * quantity as unrealized_pnl,
                        entry_date
                    FROM positions 
                    WHERE is_open = true
                """)
                
                positions = cur.fetchall()
                
                # Sector/industry distribution (simplified)
                cur.execute("""
                    SELECT 
                        symbol,
                        COUNT(*) as trade_count,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl
                    FROM trade_records 
                    WHERE executed_at >= NOW() - INTERVAL '30 days'
                    GROUP BY symbol
                    ORDER BY total_pnl DESC
                """)
                
                symbol_performance = cur.fetchall()
        
        # Calculate portfolio metrics
        total_value = sum([float(p['quantity']) * float(p['current_price'] or 0) for p in positions])
        total_unrealized_pnl = sum([float(p['unrealized_pnl'] or 0) for p in positions])
        
        return {
            'portfolio_value': round(total_value, 2),
            'unrealized_pnl': round(total_unrealized_pnl, 2),
            'position_count': len(positions),
            'positions': [
                {
                    'symbol': p['symbol'],
                    'quantity': p['quantity'],
                    'entry_price': float(p['entry_price'] or 0),
                    'current_price': float(p['current_price'] or 0),
                    'unrealized_pnl': float(p['unrealized_pnl'] or 0),
                    'days_held': (datetime.now().date() - p['entry_date']).days if p['entry_date'] else 0
                }
                for p in positions
            ],
            'top_performers': [
                {
                    'symbol': s['symbol'],
                    'trade_count': s['trade_count'],
                    'total_pnl': float(s['total_pnl'] or 0),
                    'avg_pnl': float(s['avg_pnl'] or 0)
                }
                for s in symbol_performance[:10]
            ],
            'generated_at': datetime.now(timezone.utc).isoformat()
        }

    def _generate_weekly_report(self, start_date) -> Dict:
        """Generate comprehensive weekly report"""
        end_date = start_date + timedelta(days=6)
        
        # Generate daily summaries for the week
        daily_summaries = []
        current_date = start_date
        
        while current_date <= end_date:
            daily_summary = self._generate_daily_summary(current_date)
            daily_summaries.append(daily_summary)
            current_date += timedelta(days=1)
        
        # Calculate weekly aggregates
        weekly_trades = sum([ds['trading_summary']['total_trades'] for ds in daily_summaries])
        weekly_pnl = sum([ds['trading_summary']['total_pnl'] for ds in daily_summaries])
        weekly_volume = sum([ds['trading_summary']['total_volume'] for ds in daily_summaries])
        
        # Calculate weekly win rate
        total_winning = sum([ds['trading_summary']['winning_trades'] for ds in daily_summaries])
        weekly_win_rate = (total_winning / weekly_trades * 100) if weekly_trades > 0 else 0
        
        return {
            'week_start': start_date.isoformat(),
            'week_end': end_date.isoformat(),
            'weekly_summary': {
                'total_trades': weekly_trades,
                'total_pnl': round(weekly_pnl, 2),
                'win_rate': round(weekly_win_rate, 2),
                'total_volume': weekly_volume,
                'trading_days': len([ds for ds in daily_summaries if ds['trading_summary']['total_trades'] > 0])
            },
            'daily_breakdown': daily_summaries,
            'weekly_performance': self._calculate_trading_performance(7, False),
            'generated_at': datetime.now(timezone.utc).isoformat()
        }

    def _get_service_performance_metrics(self) -> Dict:
        """Get performance metrics for each service"""
        try:
            # FIXED: Changed from /service_status to /service_health
            response = requests.get(f"{self.coordination_url}/service_health", timeout=10)
            service_status = response.json() if response.status_code == 200 else {}
        except:
            service_status = {}
        
        # Add database query performance
        db_performance = self._get_database_performance()
        
        return {
            'services': service_status,
            'database_performance': db_performance,
            'cache_performance': self._get_cache_performance(),
            'generated_at': datetime.now(timezone.utc).isoformat()
        }

    # Helper methods
    def _calculate_daily_returns(self, trade_records: List) -> List[float]:
        """Calculate daily returns from trade_records"""
        if not trade_records:
            return []
        
        # Group trade_records by date and sum P&L
        daily_pnl = {}
        for trade in trade_records:
            date = trade['executed_at'].date() if hasattr(trade['executed_at'], 'date') else trade['executed_at']
            if isinstance(date, str):
                date = datetime.strptime(date.split()[0], '%Y-%m-%d').date()
            
            if date not in daily_pnl:
                daily_pnl[date] = 0
            daily_pnl[date] += float(trade['pnl'] or 0)
        
        return list(daily_pnl.values())

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio (simplified, assuming risk-free rate = 0)"""
        if not returns or len(returns) < 2:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        return (mean_return / std_return) if std_return > 0 else 0

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        
        return float(np.min(drawdown))

    def _check_database_health(self) -> Dict:
        """Check database health and performance"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Check basic connectivity
                    cur.execute("SELECT 1")
                    
                    # Check table sizes
                    cur.execute("""
                        SELECT 
                            schemaname,
                            tablename,
                            attname,
                            n_distinct,
                            correlation
                        FROM pg_stats 
                        WHERE schemaname = 'public'
                        LIMIT 10
                    """)
                    
                    table_stats = cur.fetchall()
            
            return {
                'status': 'healthy',
                'connection': 'successful',
                'table_stats': len(table_stats)
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

    def _get_recent_errors(self) -> List[Dict]:
        """Get recent errors from logs"""
        # This would typically read from a centralized logging system
        # For now, return a placeholder
        return []

    def _get_uptime_stats(self) -> Dict:
        """Get system uptime statistics"""
        # This would track service start times
        return {
            'system_start_time': datetime.now(timezone.utc).isoformat(),
            'uptime_hours': 0
        }

    def _check_risk_alerts(self, risk_metrics: RiskMetrics) -> List[Dict]:
        """Check for risk management alerts"""
        alerts = []
        
        max_positions = int(os.getenv('MAX_POSITIONS', 5))
        if risk_metrics.current_positions >= max_positions:
            alerts.append({
                'type': 'max_positions',
                'severity': 'warning',
                'message': f'At maximum position limit: {risk_metrics.current_positions}/{max_positions}'
            })
        
        return alerts

    def _get_database_performance(self) -> Dict:
        """Get database performance metrics"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            COUNT(*) as active_connections
                        FROM pg_stat_activity 
                        WHERE state = 'active'
                    """)
                    
                    result = cur.fetchone()
            
            return {
                'active_connections': result['active_connections'] if result else 0,
                'status': 'healthy'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def _get_cache_performance(self) -> Dict:
        """Get cache performance metrics"""
        if not self.redis_client:
            return {'status': 'unavailable'}
        
        try:
            info = self.redis_client.info()
            return {
                'status': 'healthy',
                'hit_rate': 0,  # Would need to track this
                'memory_usage': info.get('used_memory_human', 'unknown'),
                'connected_clients': info.get('connected_clients', 0)
            }
        except:
            return {'status': 'error'}

    def run(self):
        """Run the Flask application"""
        logger.info("Starting Reporting Service", port=self.port)
        self.app.run(host='0.0.0.0', port=self.port, debug=False)

if __name__ == '__main__':
    service = ReportingService()
    service.run()