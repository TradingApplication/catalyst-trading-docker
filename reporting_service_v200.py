#!/usr/bin/env python3
"""
Name of Service: TRADING SYSTEM REPORTING SERVICE - ACCURACY TRACKING VERSION
Version: 2.0.0
Last Updated: 2025-06-28
Purpose: Analyze trading performance and validate system accuracy

REVISION HISTORY:
v2.0.0 (2025-06-28) - Complete rewrite for accuracy validation
- Trading performance analytics
- Source reliability tracking
- Pattern success measurement
- Catalyst effectiveness analysis
- ML data quality reporting
- Daily/weekly/monthly summaries

This service is the TRUTH DETECTOR of our system. It answers:
1. Is our trading profitable?
2. Which news sources are accurate?
3. Which patterns actually work?
4. Do catalysts improve performance?
5. Where can we improve?

KEY METRICS TRACKED:
- Win Rate: % of profitable trades
- Profit Factor: Gross profit / Gross loss
- Sharpe Ratio: Risk-adjusted returns
- Source Accuracy: Which news sources lead to profits
- Pattern Success: Which patterns work with which catalysts
- Time Analysis: How long trades take to hit targets/stops

REPORTS GENERATED:
- Daily Trading Summary
- Source Reliability Report
- Pattern Performance Analysis
- Catalyst Effectiveness Report
- ML Training Data Quality
- Risk Management Review
"""

import os
import json
import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from flask import Flask, request, jsonify, send_file
from typing import Dict, List, Optional, Tuple, Any
import io
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')

# Import database utilities if available
try:
    from database_utils_old import DatabaseServiceMixin
    USE_DB_UTILS = True
except ImportError:
    USE_DB_UTILS = False
    print("Warning: database_utils not found, using direct SQLite connections")


class TradingAccuracyReporting(DatabaseServiceMixin if USE_DB_UTILS else object):
    """
    Comprehensive reporting service for trading system validation
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
        
        # Report configuration
        self.report_config = {
            'min_trades_for_statistics': 10,
            'confidence_buckets': [0, 30, 50, 70, 100],
            'time_buckets_minutes': [5, 15, 30, 60, 240],
            'pnl_buckets': [-100, -50, -20, -5, 0, 5, 20, 50, 100]
        }
        
        # Performance benchmarks
        self.benchmarks = {
            'target_win_rate': 60.0,
            'target_profit_factor': 1.5,
            'target_sharpe_ratio': 2.0,
            'max_acceptable_drawdown': 10.0,
            'min_trades_per_day': 3
        }
        
        # Initialize database for reporting
        self._init_reporting_tables()
        
        # Register with coordination
        self._register_with_coordination()
        
        self.logger.info("Trading Accuracy Reporting v2.0.0 initialized")
        
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('reporting_service')
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory
        os.makedirs('/tmp/logs', exist_ok=True)
        
        # File handler
        fh = logging.FileHandler('/tmp//tmp/logs/reporting.log')
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
        
    def _init_reporting_tables(self):
        """Initialize reporting-specific tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Daily performance summary
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_performance_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trading_date DATE NOT NULL UNIQUE,
                    
                    -- Trade statistics
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    win_rate DECIMAL(5,2),
                    
                    -- P&L metrics
                    gross_profit DECIMAL(10,2),
                    gross_loss DECIMAL(10,2),
                    net_pnl DECIMAL(10,2),
                    profit_factor DECIMAL(5,2),
                    
                    -- Risk metrics
                    max_drawdown DECIMAL(10,2),
                    sharpe_ratio DECIMAL(5,2),
                    avg_risk_reward DECIMAL(5,2),
                    
                    -- Catalyst performance
                    catalyst_trades INTEGER DEFAULT 0,
                    catalyst_win_rate DECIMAL(5,2),
                    best_catalyst_type TEXT,
                    
                    -- Pattern performance
                    best_pattern TEXT,
                    worst_pattern TEXT,
                    
                    -- Time analysis
                    avg_trade_duration_minutes INTEGER,
                    fastest_winner_minutes INTEGER,
                    longest_loser_minutes INTEGER,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Source accuracy tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS source_accuracy_report (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_date DATE NOT NULL,
                    source_name TEXT NOT NULL,
                    source_tier INTEGER,
                    
                    -- Accuracy metrics
                    total_news_items INTEGER DEFAULT 0,
                    news_leading_to_trades INTEGER DEFAULT 0,
                    profitable_trade_news INTEGER DEFAULT 0,
                    accuracy_rate DECIMAL(5,2),
                    
                    -- Performance impact
                    total_pnl_from_source DECIMAL(10,2),
                    avg_confidence_score DECIMAL(5,2),
                    avg_trade_pnl DECIMAL(10,2),
                    
                    -- Timing analysis
                    avg_early_minutes INTEGER,  -- How early they report
                    exclusive_scoops INTEGER DEFAULT 0,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(report_date, source_name)
                )
            ''')
            
            # Pattern performance analysis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_performance_report (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_date DATE NOT NULL,
                    pattern_name TEXT NOT NULL,
                    
                    -- Performance metrics
                    total_occurrences INTEGER DEFAULT 0,
                    successful_trades INTEGER DEFAULT 0,
                    failed_trades INTEGER DEFAULT 0,
                    success_rate DECIMAL(5,2),
                    
                    -- Context analysis
                    with_catalyst_success_rate DECIMAL(5,2),
                    without_catalyst_success_rate DECIMAL(5,2),
                    best_catalyst_combination TEXT,
                    
                    -- P&L impact
                    total_pnl DECIMAL(10,2),
                    avg_pnl_per_trade DECIMAL(10,2),
                    best_trade_pnl DECIMAL(10,2),
                    worst_trade_pnl DECIMAL(10,2),
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(report_date, pattern_name)
                )
            ''')
            
            # ML training data quality
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_data_quality_report (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_date DATE NOT NULL UNIQUE,
                    
                    -- Data completeness
                    total_trades INTEGER DEFAULT 0,
                    trades_with_outcomes INTEGER DEFAULT 0,
                    trades_with_patterns INTEGER DEFAULT 0,
                    trades_with_catalysts INTEGER DEFAULT 0,
                    data_completeness_score DECIMAL(5,2),
                    
                    -- Feature quality
                    unique_patterns INTEGER DEFAULT 0,
                    unique_catalysts INTEGER DEFAULT 0,
                    feature_diversity_score DECIMAL(5,2),
                    
                    -- Outcome distribution
                    win_loss_balance DECIMAL(5,2),  -- How balanced is the dataset
                    outcome_variety_score DECIMAL(5,2),
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_daily_performance_date 
                ON daily_performance_summary(trading_date DESC)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_source_accuracy_date 
                ON source_accuracy_report(report_date DESC, accuracy_rate DESC)
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Reporting tables initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing reporting tables: {e}")
            raise
            
    def setup_routes(self):
        """Setup Flask routes"""
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                "status": "healthy",
                "service": "reporting",
                "version": "2.0.0",
                "mode": "accuracy-tracking",
                "timestamp": datetime.now().isoformat()
            })
            
        @self.app.route('/daily_summary', methods=['GET'])
        def daily_summary():
            """Get daily trading summary"""
            date_str = request.args.get('date', date.today().isoformat())
            summary = self.generate_daily_summary(date_str)
            return jsonify(summary)
            
        @self.app.route('/source_accuracy', methods=['GET'])
        def source_accuracy():
            """Get news source accuracy report"""
            days = request.args.get('days', 30, type=int)
            report = self.analyze_source_accuracy(days)
            return jsonify(report)
            
        @self.app.route('/pattern_performance', methods=['GET'])
        def pattern_performance():
            """Get pattern performance analysis"""
            days = request.args.get('days', 30, type=int)
            pattern = request.args.get('pattern')  # Optional filter
            report = self.analyze_pattern_performance(days, pattern)
            return jsonify(report)
            
        @self.app.route('/catalyst_effectiveness', methods=['GET'])
        def catalyst_effectiveness():
            """Analyze catalyst impact on trading"""
            days = request.args.get('days', 30, type=int)
            report = self.analyze_catalyst_effectiveness(days)
            return jsonify(report)
            
        @self.app.route('/ml_data_quality', methods=['GET'])
        def ml_data_quality():
            """Check ML training data quality"""
            report = self.assess_ml_data_quality()
            return jsonify(report)
            
        @self.app.route('/performance_chart', methods=['GET'])
        def performance_chart():
            """Generate performance visualization"""
            chart_type = request.args.get('type', 'pnl_curve')
            days = request.args.get('days', 30, type=int)
            
            chart_buffer = self.generate_performance_chart(chart_type, days)
            return send_file(chart_buffer, mimetype='image/png')
            
        @self.app.route('/risk_analysis', methods=['GET'])
        def risk_analysis():
            """Analyze risk metrics"""
            days = request.args.get('days', 30, type=int)
            report = self.analyze_risk_metrics(days)
            return jsonify(report)
            
        @self.app.route('/generate_reports', methods=['POST'])
        def generate_reports():
            """Generate all reports for a date"""
            date_str = request.json.get('date', date.today().isoformat())
            self.generate_all_reports(date_str)
            return jsonify({'status': 'reports generated', 'date': date_str})
            
    def generate_daily_summary(self, date_str: str) -> Dict:
        """
        Generate comprehensive daily trading summary
        This is the main report showing overall performance
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all trades for the day
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl_amount > 0 THEN 1 ELSE 0 END) as winners,
                    SUM(CASE WHEN pnl_amount < 0 THEN 1 ELSE 0 END) as losers,
                    SUM(CASE WHEN pnl_amount > 0 THEN pnl_amount ELSE 0 END) as gross_profit,
                    SUM(CASE WHEN pnl_amount < 0 THEN pnl_amount ELSE 0 END) as gross_loss,
                    SUM(pnl_amount) as net_pnl,
                    AVG(pnl_percentage) as avg_pnl_pct,
                    MAX(pnl_amount) as best_trade,
                    MIN(pnl_amount) as worst_trade,
                    AVG(CAST((julianday(exit_timestamp) - julianday(entry_timestamp)) * 24 * 60 AS INTEGER)) as avg_duration_min
                FROM trade_records
                WHERE DATE(entry_timestamp) = ?
                AND exit_timestamp IS NOT NULL
            ''', (date_str,))
            
            trade_stats = cursor.fetchone()
            
            total_trades = trade_stats[0] or 0
            winners = trade_stats[1] or 0
            losers = trade_stats[2] or 0
            gross_profit = trade_stats[3] or 0
            gross_loss = abs(trade_stats[4] or 0)
            net_pnl = trade_stats[5] or 0
            
            # Calculate key metrics
            win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
            
            # Get catalyst performance
            cursor.execute('''
                SELECT 
                    catalyst_type,
                    COUNT(*) as count,
                    AVG(pnl_percentage) as avg_pnl,
                    SUM(CASE WHEN pnl_amount > 0 THEN 1 ELSE 0 END) as wins
                FROM trade_records
                WHERE DATE(entry_timestamp) = ?
                AND exit_timestamp IS NOT NULL
                AND catalyst_type IS NOT NULL
                GROUP BY catalyst_type
                ORDER BY avg_pnl DESC
            ''', (date_str,))
            
            catalyst_performance = {}
            best_catalyst = None
            best_catalyst_pnl = -float('inf')
            
            for row in cursor.fetchall():
                catalyst_performance[row[0]] = {
                    'trades': row[1],
                    'avg_pnl': round(row[2] or 0, 2),
                    'win_rate': round((row[3] / row[1] * 100) if row[1] > 0 else 0, 2)
                }
                if row[2] and row[2] > best_catalyst_pnl:
                    best_catalyst = row[0]
                    best_catalyst_pnl = row[2]
                    
            # Get pattern performance
            cursor.execute('''
                SELECT 
                    p.pattern_name,
                    COUNT(*) as count,
                    AVG(t.pnl_percentage) as avg_pnl
                FROM trade_records t
                JOIN pattern_analysis p ON t.symbol = p.symbol
                WHERE DATE(t.entry_timestamp) = ?
                AND t.exit_timestamp IS NOT NULL
                AND ABS(julianday(t.entry_timestamp) - julianday(p.detection_timestamp)) < 0.25
                GROUP BY p.pattern_name
                ORDER BY avg_pnl DESC
            ''', (date_str,))
            
            pattern_performance = {}
            best_pattern = None
            worst_pattern = None
            best_pattern_pnl = -float('inf')
            worst_pattern_pnl = float('inf')
            
            for row in cursor.fetchall():
                pattern_performance[row[0]] = {
                    'trades': row[1],
                    'avg_pnl': round(row[2] or 0, 2)
                }
                if row[2]:
                    if row[2] > best_pattern_pnl:
                        best_pattern = row[0]
                        best_pattern_pnl = row[2]
                    if row[2] < worst_pattern_pnl:
                        worst_pattern = row[0]
                        worst_pattern_pnl = row[2]
                        
            # Calculate Sharpe Ratio (simplified daily)
            if total_trades > 1:
                cursor.execute('''
                    SELECT pnl_percentage
                    FROM trade_records
                    WHERE DATE(entry_timestamp) = ?
                    AND exit_timestamp IS NOT NULL
                ''', (date_str,))
                
                returns = [row[0] for row in cursor.fetchall()]
                if returns:
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
                
            # Build summary
            summary = {
                'date': date_str,
                'overview': {
                    'total_trades': total_trades,
                    'winning_trades': winners,
                    'losing_trades': losers,
                    'win_rate': round(win_rate, 2),
                    'net_pnl': round(net_pnl, 2),
                    'gross_profit': round(gross_profit, 2),
                    'gross_loss': round(gross_loss, 2),
                    'profit_factor': round(profit_factor, 2),
                    'sharpe_ratio': round(sharpe_ratio, 2)
                },
                'trade_analysis': {
                    'avg_pnl_pct': round(trade_stats[6] or 0, 2),
                    'best_trade': round(trade_stats[7] or 0, 2),
                    'worst_trade': round(trade_stats[8] or 0, 2),
                    'avg_duration_minutes': round(trade_stats[9] or 0, 0)
                },
                'catalyst_performance': catalyst_performance,
                'pattern_performance': pattern_performance,
                'best_performers': {
                    'catalyst': best_catalyst,
                    'pattern': best_pattern
                },
                'worst_performers': {
                    'pattern': worst_pattern
                },
                'benchmarks': {
                    'met_win_rate': win_rate >= self.benchmarks['target_win_rate'],
                    'met_profit_factor': profit_factor >= self.benchmarks['target_profit_factor'],
                    'met_min_trades': total_trades >= self.benchmarks['min_trades_per_day']
                }
            }
            
            # Save to daily summary table
            self._save_daily_summary(date_str, summary)
            
            conn.close()
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating daily summary: {e}")
            return {'error': str(e)}
            
    def analyze_source_accuracy(self, days: int = 30) -> Dict:
        """
        Analyze news source reliability and accuracy
        Which sources actually lead to profitable trades?
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get source performance data
            cursor.execute('''
                SELECT 
                    n.source,
                    n.source_tier,
                    COUNT(DISTINCT n.news_id) as total_news,
                    COUNT(DISTINCT t.trade_id) as trades_from_source,
                    SUM(CASE WHEN t.pnl_amount > 0 THEN 1 ELSE 0 END) as profitable_trades,
                    SUM(t.pnl_amount) as total_pnl,
                    AVG(n.catalyst_score) as avg_catalyst_score,
                    AVG(t.pnl_percentage) as avg_trade_pnl_pct
                FROM news_raw n
                LEFT JOIN trade_records t ON n.news_id = t.entry_news_id
                WHERE n.published_timestamp > datetime('now', '-{} days')
                GROUP BY n.source, n.source_tier
                ORDER BY total_pnl DESC
            '''.format(days))
            
            source_analysis = []
            
            for row in cursor.fetchall():
                source = row[0]
                tier = row[1]
                total_news = row[2]
                trades = row[3] or 0
                profitable = row[4] or 0
                total_pnl = row[5] or 0
                
                # Calculate accuracy metrics
                trade_rate = (trades / total_news * 100) if total_news > 0 else 0
                win_rate = (profitable / trades * 100) if trades > 0 else 0
                
                source_analysis.append({
                    'source': source,
                    'tier': tier,
                    'total_news': total_news,
                    'news_to_trades': trades,
                    'trade_rate': round(trade_rate, 2),
                    'profitable_trades': profitable,
                    'win_rate': round(win_rate, 2),
                    'total_pnl': round(total_pnl, 2),
                    'avg_catalyst_score': round(row[6] or 0, 2),
                    'avg_pnl_per_trade': round(row[7] or 0, 2),
                    'reliability_score': round(win_rate * trade_rate / 100, 2)
                })
                
            # Separate by tiers
            tier_analysis = {}
            for tier in range(1, 6):
                tier_sources = [s for s in source_analysis if s['tier'] == tier]
                if tier_sources:
                    tier_analysis[f'tier_{tier}'] = {
                        'sources': len(tier_sources),
                        'avg_win_rate': round(
                            np.mean([s['win_rate'] for s in tier_sources]), 2
                        ),
                        'total_pnl': round(
                            sum(s['total_pnl'] for s in tier_sources), 2
                        )
                    }
                    
            # Find best and worst sources
            if source_analysis:
                best_by_pnl = max(source_analysis, key=lambda x: x['total_pnl'])
                best_by_accuracy = max(source_analysis, key=lambda x: x['win_rate'])
                worst_by_pnl = min(source_analysis, key=lambda x: x['total_pnl'])
                
                highlights = {
                    'best_by_profit': best_by_pnl['source'],
                    'best_by_accuracy': best_by_accuracy['source'],
                    'worst_source': worst_by_pnl['source']
                }
            else:
                highlights = {}
                
            conn.close()
            
            return {
                'period_days': days,
                'source_performance': source_analysis[:20],  # Top 20 sources
                'tier_analysis': tier_analysis,
                'highlights': highlights,
                'recommendation': self._generate_source_recommendations(source_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing source accuracy: {e}")
            return {'error': str(e)}
            
    def analyze_pattern_performance(self, days: int = 30, 
                                  pattern_filter: Optional[str] = None) -> Dict:
        """
        Analyze which patterns actually work
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Base query
            query = '''
                SELECT 
                    p.pattern_name,
                    p.pattern_type,
                    COUNT(DISTINCT p.id) as occurrences,
                    COUNT(DISTINCT t.trade_id) as trades,
                    SUM(CASE WHEN t.pnl_amount > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(t.pnl_amount) as total_pnl,
                    AVG(t.pnl_percentage) as avg_pnl_pct,
                    AVG(p.final_confidence) as avg_confidence,
                    SUM(CASE WHEN p.has_catalyst = 1 THEN 1 ELSE 0 END) as with_catalyst
                FROM pattern_analysis p
                LEFT JOIN trade_records t ON p.symbol = t.symbol
                    AND ABS(julianday(t.entry_timestamp) - julianday(p.detection_timestamp)) < 0.25
                WHERE p.detection_timestamp > datetime('now', '-{} days')
            '''.format(days)
            
            if pattern_filter:
                query += " AND p.pattern_name = ?"
                params = (pattern_filter,)
            else:
                params = ()
                
            query += '''
                GROUP BY p.pattern_name, p.pattern_type
                ORDER BY total_pnl DESC
            '''
            
            cursor.execute(query, params)
            
            pattern_analysis = []
            
            for row in cursor.fetchall():
                pattern = row[0]
                occurrences = row[2]
                trades = row[3] or 0
                wins = row[4] or 0
                
                # Calculate metrics
                trade_rate = (trades / occurrences * 100) if occurrences > 0 else 0
                win_rate = (wins / trades * 100) if trades > 0 else 0
                catalyst_rate = (row[8] / occurrences * 100) if occurrences > 0 else 0
                
                pattern_analysis.append({
                    'pattern_name': pattern,
                    'pattern_type': row[1],
                    'occurrences': occurrences,
                    'trades_executed': trades,
                    'trade_rate': round(trade_rate, 2),
                    'winning_trades': wins,
                    'win_rate': round(win_rate, 2),
                    'total_pnl': round(row[5] or 0, 2),
                    'avg_pnl_pct': round(row[6] or 0, 2),
                    'avg_confidence': round(row[7] or 0, 2),
                    'catalyst_present_pct': round(catalyst_rate, 2)
                })
                
            # Analyze pattern combinations
            cursor.execute('''
                SELECT 
                    p1.pattern_name || ' + ' || p2.pattern_name as combo,
                    COUNT(*) as occurrences,
                    AVG(t.pnl_percentage) as avg_pnl
                FROM pattern_analysis p1
                JOIN pattern_analysis p2 ON p1.symbol = p2.symbol
                    AND p1.pattern_name < p2.pattern_name
                    AND ABS(julianday(p1.detection_timestamp) - julianday(p2.detection_timestamp)) < 0.042
                LEFT JOIN trade_records t ON p1.symbol = t.symbol
                    AND ABS(julianday(t.entry_timestamp) - julianday(p1.detection_timestamp)) < 0.25
                WHERE p1.detection_timestamp > datetime('now', '-{} days')
                GROUP BY combo
                HAVING occurrences > 5
                ORDER BY avg_pnl DESC
                LIMIT 10
            '''.format(days))
            
            pattern_combos = []
            for row in cursor.fetchall():
                pattern_combos.append({
                    'combination': row[0],
                    'occurrences': row[1],
                    'avg_pnl': round(row[2] or 0, 2)
                })
                
            conn.close()
            
            return {
                'period_days': days,
                'pattern_performance': pattern_analysis,
                'best_combinations': pattern_combos,
                'summary': {
                    'total_patterns_analyzed': len(pattern_analysis),
                    'patterns_with_trades': sum(1 for p in pattern_analysis if p['trades_executed'] > 0),
                    'avg_pattern_win_rate': round(
                        np.mean([p['win_rate'] for p in pattern_analysis if p['trades_executed'] > 0]), 2
                    ) if pattern_analysis else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing pattern performance: {e}")
            return {'error': str(e)}
            
    def analyze_catalyst_effectiveness(self, days: int = 30) -> Dict:
        """
        Measure how catalysts impact trading performance
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Compare trades with and without catalysts
            cursor.execute('''
                SELECT 
                    CASE WHEN catalyst_type IS NOT NULL THEN 'With Catalyst' ELSE 'No Catalyst' END as category,
                    COUNT(*) as trades,
                    SUM(CASE WHEN pnl_amount > 0 THEN 1 ELSE 0 END) as wins,
                    AVG(pnl_percentage) as avg_pnl,
                    SUM(pnl_amount) as total_pnl,
                    AVG(ABS(pnl_percentage)) as avg_volatility
                FROM trade_records
                WHERE entry_timestamp > datetime('now', '-{} days')
                AND exit_timestamp IS NOT NULL
                GROUP BY category
            '''.format(days))
            
            catalyst_comparison = {}
            for row in cursor.fetchall():
                catalyst_comparison[row[0]] = {
                    'trades': row[1],
                    'win_rate': round((row[2] / row[1] * 100) if row[1] > 0 else 0, 2),
                    'avg_pnl': round(row[3] or 0, 2),
                    'total_pnl': round(row[4] or 0, 2),
                    'volatility': round(row[5] or 0, 2)
                }
                
            # Analyze by catalyst type
            cursor.execute('''
                SELECT 
                    catalyst_type,
                    COUNT(*) as trades,
                    SUM(CASE WHEN pnl_amount > 0 THEN 1 ELSE 0 END) as wins,
                    AVG(pnl_percentage) as avg_pnl,
                    MAX(pnl_percentage) as best_trade,
                    MIN(pnl_percentage) as worst_trade,
                    AVG(position_size_pct) as avg_position_size
                FROM trade_records
                WHERE catalyst_type IS NOT NULL
                AND entry_timestamp > datetime('now', '-{} days')
                AND exit_timestamp IS NOT NULL
                GROUP BY catalyst_type
                ORDER BY avg_pnl DESC
            '''.format(days))
            
            catalyst_types = []
            for row in cursor.fetchall():
                catalyst_types.append({
                    'type': row[0],
                    'trades': row[1],
                    'win_rate': round((row[2] / row[1] * 100) if row[1] > 0 else 0, 2),
                    'avg_pnl': round(row[3] or 0, 2),
                    'best_trade': round(row[4] or 0, 2),
                    'worst_trade': round(row[5] or 0, 2),
                    'avg_position_size': round(row[6] or 0, 2)
                })
                
            # Catalyst timing analysis
            cursor.execute('''
                SELECT 
                    market_state_at_entry,
                    COUNT(*) as trades,
                    AVG(pnl_percentage) as avg_pnl,
                    SUM(CASE WHEN pnl_amount > 0 THEN 1 ELSE 0 END) as wins
                FROM trade_records
                WHERE catalyst_type IS NOT NULL
                AND entry_timestamp > datetime('now', '-{} days')
                AND exit_timestamp IS NOT NULL
                GROUP BY market_state_at_entry
            '''.format(days))
            
            timing_analysis = {}
            for row in cursor.fetchall():
                timing_analysis[row[0]] = {
                    'trades': row[1],
                    'avg_pnl': round(row[2] or 0, 2),
                    'win_rate': round((row[3] / row[1] * 100) if row[1] > 0 else 0, 2)
                }
                
            conn.close()
            
            # Calculate catalyst effectiveness score
            if 'With Catalyst' in catalyst_comparison and 'No Catalyst' in catalyst_comparison:
                improvement = (
                    catalyst_comparison['With Catalyst']['win_rate'] - 
                    catalyst_comparison['No Catalyst']['win_rate']
                )
            else:
                improvement = 0
                
            return {
                'period_days': days,
                'catalyst_comparison': catalyst_comparison,
                'catalyst_types': catalyst_types,
                'timing_analysis': timing_analysis,
                'effectiveness_score': round(improvement, 2),
                'recommendation': 'Focus on catalyst-driven trades' if improvement > 5 else 'Catalysts not significantly improving performance'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing catalyst effectiveness: {e}")
            return {'error': str(e)}
            
    def assess_ml_data_quality(self) -> Dict:
        """
        Assess quality of data for ML training
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get basic counts
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN exit_timestamp IS NOT NULL THEN 1 END) as completed_trades,
                    COUNT(CASE WHEN catalyst_type IS NOT NULL THEN 1 END) as trades_with_catalyst,
                    COUNT(CASE WHEN entry_news_id IS NOT NULL THEN 1 END) as trades_with_news
                FROM trade_records
            ''')
            
            basic_stats = cursor.fetchone()
            total_trades = basic_stats[0] or 0
            
            # Calculate completeness scores
            completeness_scores = {
                'outcome_completeness': (basic_stats[1] / total_trades * 100) if total_trades > 0 else 0,
                'catalyst_completeness': (basic_stats[2] / total_trades * 100) if total_trades > 0 else 0,
                'news_linkage': (basic_stats[3] / total_trades * 100) if total_trades > 0 else 0
            }
            
            # Check pattern coverage
            cursor.execute('''
                SELECT COUNT(DISTINCT pattern_name) as unique_patterns
                FROM pattern_analysis
            ''')
            unique_patterns = cursor.fetchone()[0]
            
            # Check catalyst diversity
            cursor.execute('''
                SELECT COUNT(DISTINCT catalyst_type) as unique_catalysts
                FROM trade_records
                WHERE catalyst_type IS NOT NULL
            ''')
            unique_catalysts = cursor.fetchone()[0]
            
            # Check outcome distribution
            cursor.execute('''
                SELECT 
                    SUM(CASE WHEN pnl_amount > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl_amount < 0 THEN 1 ELSE 0 END) as losses
                FROM trade_records
                WHERE exit_timestamp IS NOT NULL
            ''')
            
            outcome_dist = cursor.fetchone()
            wins = outcome_dist[0] or 0
            losses = outcome_dist[1] or 0
            total_outcomes = wins + losses
            
            # Calculate balance score (closer to 50/50 is better for ML)
            if total_outcomes > 0:
                win_ratio = wins / total_outcomes
                balance_score = 100 * (1 - abs(win_ratio - 0.5) * 2)
            else:
                balance_score = 0
                
            # Check data recency
            cursor.execute('''
                SELECT 
                    COUNT(CASE WHEN entry_timestamp > datetime('now', '-7 days') THEN 1 END) as recent_trades,
                    COUNT(CASE WHEN entry_timestamp > datetime('now', '-30 days') THEN 1 END) as month_trades
                FROM trade_records
            ''')
            
            recency = cursor.fetchone()
            
            conn.close()
            
            # Calculate overall quality score
            quality_score = np.mean([
                completeness_scores['outcome_completeness'],
                completeness_scores['catalyst_completeness'],
                completeness_scores['news_linkage'],
                balance_score,
                min(100, unique_patterns * 5),  # Want at least 20 patterns
                min(100, unique_catalysts * 10)  # Want at least 10 catalyst types
            ])
            
            return {
                'data_statistics': {
                    'total_trades': total_trades,
                    'completed_trades': basic_stats[1],
                    'unique_patterns': unique_patterns,
                    'unique_catalysts': unique_catalysts,
                    'recent_trades_7d': recency[0],
                    'recent_trades_30d': recency[1]
                },
                'completeness_scores': completeness_scores,
                'outcome_balance': {
                    'wins': wins,
                    'losses': losses,
                    'balance_score': round(balance_score, 2)
                },
                'overall_quality_score': round(quality_score, 2),
                'ml_readiness': quality_score >= 70,
                'recommendations': self._generate_ml_recommendations(quality_score, completeness_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing ML data quality: {e}")
            return {'error': str(e)}
            
    def analyze_risk_metrics(self, days: int = 30) -> Dict:
        """
        Analyze risk management effectiveness
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get trades with risk data
            cursor.execute('''
                SELECT 
                    entry_timestamp,
                    pnl_amount,
                    pnl_percentage,
                    position_size_pct,
                    stop_loss_price,
                    entry_price,
                    exit_price,
                    exit_reason
                FROM trade_records
                WHERE entry_timestamp > datetime('now', '-{} days')
                AND exit_timestamp IS NOT NULL
                ORDER BY entry_timestamp
            '''.format(days))
            
            trades = cursor.fetchall()
            
            if not trades:
                return {'error': 'No trades found for analysis'}
                
            # Calculate max drawdown
            cumulative_pnl = []
            running_total = 0
            for trade in trades:
                running_total += trade[1]
                cumulative_pnl.append(running_total)
                
            peak = cumulative_pnl[0]
            max_drawdown = 0
            for value in cumulative_pnl:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100 if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
                
            # Analyze stop loss effectiveness
            stops_hit = sum(1 for t in trades if t[7] == 'stop_loss')
            total_exits = len(trades)
            stop_effectiveness = (stops_hit / total_exits * 100) if total_exits > 0 else 0
            
            # Position sizing analysis
            position_sizes = [t[3] for t in trades if t[3]]
            avg_position_size = np.mean(position_sizes) if position_sizes else 0
            
            # Risk/reward analysis
            risk_rewards = []
            for trade in trades:
                if trade[4] and trade[5]:  # Has stop loss and entry price
                    risk = abs(trade[5] - trade[4])
                    reward = abs(trade[6] - trade[5]) if trade[6] else 0
                    if risk > 0:
                        risk_rewards.append(reward / risk)
                        
            avg_risk_reward = np.mean(risk_rewards) if risk_rewards else 0
            
            conn.close()
            
            return {
                'period_days': days,
                'risk_metrics': {
                    'max_drawdown_pct': round(max_drawdown, 2),
                    'stop_loss_hit_rate': round(stop_effectiveness, 2),
                    'avg_position_size': round(avg_position_size, 2),
                    'avg_risk_reward_ratio': round(avg_risk_reward, 2)
                },
                'risk_assessment': {
                    'drawdown_acceptable': max_drawdown <= self.benchmarks['max_acceptable_drawdown'],
                    'position_sizing_discipline': avg_position_size <= 15,  # Conservative
                    'stop_loss_working': stop_effectiveness > 20 and stop_effectiveness < 40
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing risk metrics: {e}")
            return {'error': str(e)}
            
    def generate_performance_chart(self, chart_type: str, days: int) -> io.BytesIO:
        """
        Generate performance visualization
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            plt.figure(figsize=(12, 6))
            
            if chart_type == 'pnl_curve':
                # Cumulative P&L curve
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT entry_timestamp, pnl_amount
                    FROM trade_records
                    WHERE entry_timestamp > datetime('now', '-{} days')
                    AND exit_timestamp IS NOT NULL
                    ORDER BY entry_timestamp
                '''.format(days))
                
                trades = cursor.fetchall()
                if trades:
                    dates = [datetime.fromisoformat(t[0]) for t in trades]
                    pnl = [t[1] for t in trades]
                    cumulative_pnl = np.cumsum(pnl)
                    
                    plt.plot(dates, cumulative_pnl, linewidth=2)
                    plt.fill_between(dates, cumulative_pnl, alpha=0.3)
                    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                    plt.title('Cumulative P&L Curve')
                    plt.xlabel('Date')
                    plt.ylabel('Cumulative P&L ($)')
                    plt.grid(True, alpha=0.3)
                    
            elif chart_type == 'win_rate_by_catalyst':
                # Win rate by catalyst type
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        catalyst_type,
                        COUNT(*) as total,
                        SUM(CASE WHEN pnl_amount > 0 THEN 1 ELSE 0 END) as wins
                    FROM trade_records
                    WHERE catalyst_type IS NOT NULL
                    AND entry_timestamp > datetime('now', '-{} days')
                    AND exit_timestamp IS NOT NULL
                    GROUP BY catalyst_type
                    HAVING total > 3
                '''.format(days))
                
                data = cursor.fetchall()
                if data:
                    catalysts = [d[0] for d in data]
                    win_rates = [(d[2]/d[1]*100) for d in data]
                    
                    plt.bar(catalysts, win_rates)
                    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5)
                    plt.title('Win Rate by Catalyst Type')
                    plt.xlabel('Catalyst Type')
                    plt.ylabel('Win Rate (%)')
                    plt.xticks(rotation=45)
                    plt.grid(True, alpha=0.3, axis='y')
                    
            conn.close()
            
            plt.tight_layout()
            
            # Save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            self.logger.error(f"Error generating chart: {e}")
            plt.close()
            return io.BytesIO()
            
    def _save_daily_summary(self, date_str: str, summary: Dict):
        """Save daily summary to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO daily_performance_summary
                (trading_date, total_trades, winning_trades, losing_trades,
                 win_rate, gross_profit, gross_loss, net_pnl, profit_factor,
                 sharpe_ratio, best_catalyst_type, best_pattern, worst_pattern)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                date_str,
                summary['overview']['total_trades'],
                summary['overview']['winning_trades'],
                summary['overview']['losing_trades'],
                summary['overview']['win_rate'],
                summary['overview']['gross_profit'],
                summary['overview']['gross_loss'],
                summary['overview']['net_pnl'],
                summary['overview']['profit_factor'],
                summary['overview']['sharpe_ratio'],
                summary['best_performers'].get('catalyst'),
                summary['best_performers'].get('pattern'),
                summary['worst_performers'].get('pattern')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving daily summary: {e}")
            
    def _generate_source_recommendations(self, source_analysis: List[Dict]) -> List[str]:
        """Generate recommendations based on source analysis"""
        recommendations = []
        
        # Find underperforming sources
        poor_sources = [s for s in source_analysis if s['win_rate'] < 40 and s['news_to_trades'] > 5]
        if poor_sources:
            recommendations.append(
                f"Consider filtering out news from: {', '.join(s['source'] for s in poor_sources[:3])}"
            )
            
        # Find high-performing sources
        good_sources = [s for s in source_analysis if s['win_rate'] > 60 and s['news_to_trades'] > 10]
        if good_sources:
            recommendations.append(
                f"Prioritize news from: {', '.join(s['source'] for s in good_sources[:3])}"
            )
            
        # Check tier performance
        tier_1_sources = [s for s in source_analysis if s['tier'] == 1]
        if tier_1_sources and np.mean([s['win_rate'] for s in tier_1_sources]) < 50:
            recommendations.append(
                "Even Tier 1 sources showing low accuracy - review catalyst scoring"
            )
            
        return recommendations
        
    def _generate_ml_recommendations(self, quality_score: float, 
                                    completeness: Dict) -> List[str]:
        """Generate ML data quality recommendations"""
        recommendations = []
        
        if quality_score < 70:
            if completeness['outcome_completeness'] < 80:
                recommendations.append("Need more completed trades with outcomes")
                
            if completeness['catalyst_completeness'] < 60:
                recommendations.append("Increase catalyst coverage in news collection")
                
            if completeness['news_linkage'] < 50:
                recommendations.append("Improve news-to-trade linkage tracking")
                
        recommendations.append(
            f"Current ML readiness: {'Ready' if quality_score >= 70 else 'Not ready'} ({quality_score:.0f}% quality)"
        )
        
        return recommendations
        
    def generate_all_reports(self, date_str: str):
        """Generate all reports for a given date"""
        self.logger.info(f"Generating all reports for {date_str}")
        
        # Generate daily summary
        self.generate_daily_summary(date_str)
        
        # Update source accuracy
        self.analyze_source_accuracy(30)
        
        # Update pattern performance
        self.analyze_pattern_performance(30)
        
        # Assess ML data quality
        self.assess_ml_data_quality()
        
        self.logger.info("All reports generated successfully")
        
    def _register_with_coordination(self):
        """Register with coordination service"""
        try:
            import requests
            response = requests.post(
                f"{self.coordination_url}/register_service",
                json={
                    'service_name': 'reporting',
                    'service_info': {
                        'url': 'http://localhost:5009',
                        'port': 5009,
                        'version': '2.0.0',
                        'capabilities': ['accuracy_tracking', 'performance_analytics']
                    }
                }
            )
            if response.status_code == 200:
                self.logger.info("Successfully registered with coordination service")
        except Exception as e:
            self.logger.warning(f"Could not register with coordination: {e}")
            
    def run(self):
        """Start the Flask application"""
        self.logger.info("Starting Trading Accuracy Reporting v2.0.0 on port 5009")
        self.logger.info("Tracking: Performance, Source Accuracy, Pattern Success, ML Readiness")
        
        self.app.run(host='0.0.0.0', port=5009, debug=False)


if __name__ == "__main__":
    service = TradingAccuracyReporting()
    service.run()