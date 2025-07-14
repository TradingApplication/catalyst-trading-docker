# Catalyst Trading System - Functional Specification v2.0.0

**Version**: 2.0.0  
**Date**: June 28, 2025  
**Platform**: DigitalOcean  
**Status**: Implementation Ready  

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Business Logic](#2-core-business-logic)
3. [Service Specifications](#3-service-specifications)
4. [Data Flow Specifications](#4-data-flow-specifications)
5. [Integration Points](#5-integration-points)
6. [Performance Requirements](#6-performance-requirements)
7. [Security Requirements](#7-security-requirements)
8. [Error Handling](#8-error-handling)

---

## 1. System Overview

### 1.1 Purpose
The Catalyst Trading System is a news-driven algorithmic trading platform that identifies and executes day trading opportunities based on market catalysts. Unlike traditional systems that scan all securities randomly, this system focuses exclusively on securities with news events that create tradeable momentum.

### 1.2 Key Differentiators
- **News-First Selection**: Only trade securities with news catalysts
- **Source Intelligence**: Track source reliability and agenda patterns
- **Clean Architecture**: Raw data preserved, trading data optimized
- **ML-Ready**: Collect outcome data for future pattern discovery
- **Social Mission**: Profits fund homeless shelter operations

### 1.3 Operating Modes
- **Pre-Market Aggressive** (4:00 AM - 9:30 AM EST): 5-minute cycles
- **Market Hours Normal** (9:30 AM - 4:00 PM EST): 30-minute cycles
- **After-Hours Light** (4:00 PM - 8:00 PM EST): 60-minute cycles
- **Weekend Maintenance**: Data cleanup and analysis

---

## 2. Core Business Logic

### 2.1 News-Driven Selection Algorithm

#### 2.1.1 Catalyst Scoring Formula
```
Catalyst Score = (Source Tier Weight × Recency Weight × Keyword Weight × Market State Multiplier)

Where:
- Source Tier Weight: Tier 1 = 1.0, Tier 2 = 0.8, Tier 3 = 0.6, Tier 4 = 0.4, Tier 5 = 0.2
- Recency Weight: exp(-hours_old / 4) 
- Keyword Weight: Earnings = 1.2, FDA = 1.5, M&A = 1.3, Default = 1.0
- Market State Multiplier: Pre-market = 2.0, Regular = 1.0, After-hours = 0.8
```

#### 2.1.2 Multi-Stage Filtering
1. **Stage 1**: Collect all news (1000+ articles)
2. **Stage 2**: Filter by catalyst score > 30 (100 candidates)
3. **Stage 3**: Technical validation (20 candidates)
4. **Stage 4**: Pattern confirmation (5 final picks)

### 2.2 Source Alignment Tracking

#### 2.2.1 Source Tier Classification
- **Tier 1**: Bloomberg, Reuters, Dow Jones, AP
- **Tier 2**: WSJ, Financial Times, CNBC, MarketWatch
- **Tier 3**: Yahoo Finance, Seeking Alpha, Benzinga
- **Tier 4**: Zacks, TipRanks, Blog sources
- **Tier 5**: Unknown/Unverified sources

#### 2.2.2 Confirmation Tracking
- Track which Tier 1-2 sources confirm Tier 3-5 stories
- Measure confirmation delay (minutes/hours)
- Identify stories that move prices before confirmation
- Flag potential coordinated narratives

### 2.3 Pattern Detection Context

#### 2.3.1 Catalyst-Aware Pattern Weights
```python
pattern_weight_adjustments = {
    'bullish_patterns': {
        'with_positive_catalyst': 1.5,    # 50% boost
        'with_neutral_catalyst': 1.0,     # No change
        'with_negative_catalyst': 0.7     # 30% reduction
    },
    'bearish_patterns': {
        'with_negative_catalyst': 1.5,    # 50% boost
        'with_neutral_catalyst': 1.0,     # No change
        'with_positive_catalyst': 0.7     # 30% reduction
    }
}
```

### 2.4 Trading Signal Generation

#### 2.4.1 Signal Confidence Formula
```
Signal Confidence = (Catalyst Score × 0.35) + (Pattern Score × 0.35) + 
                   (Technical Score × 0.20) + (Volume Score × 0.10)

Trading Decision:
- Confidence > 70: Strong signal (full position)
- Confidence 50-70: Normal signal (half position)
- Confidence 30-50: Weak signal (quarter position)
- Confidence < 30: No trade
```

### 2.5 Risk Management Rules

#### 2.5.1 Position Sizing
- Maximum 20% of capital per position
- Maximum 5 concurrent positions
- Pre-market trades limited to 10% positions
- Scale based on signal confidence

#### 2.5.2 Stop Loss Rules
- Initial stop: 2% below entry (tight for day trading)
- Trailing stop: 1% below high water mark
- Time stop: Exit if no profit after 2 hours
- News reversal stop: Exit on contradicting news

---

## 3. Service Specifications

### 3.1 News Collection Service (Port 5008)

#### Purpose
Collect raw news from multiple sources without interpretation

#### Endpoints
- `POST /collect_news` - Trigger collection cycle
- `GET /search_news` - Search by criteria
- `GET /trending_news` - Get trending stories
- `POST /update_outcome` - Update with trading outcomes
- `GET /source_analysis` - Analyze source patterns
- `GET /coordinated_narratives` - Detect messaging patterns

#### Key Functions
```python
def collect_all_news(symbols=None, sources='all'):
    """Collect from NewsAPI, AlphaVantage, RSS feeds"""
    
def extract_sentiment_keywords(text):
    """Detect pump/dump/accumulation keywords"""
    
def check_story_confirmation(headline, symbol):
    """Check if Tier 1-2 sources confirmed"""
    
def detect_coordinated_narratives(hours=24):
    """Find suspiciously similar messaging"""
```

### 3.2 Security Scanner Service (Port 5001)

#### Purpose
Select top 5 trading candidates based on news catalysts

#### Endpoints
- `GET /scan` - Regular market scan
- `GET /scan_premarket` - Aggressive pre-market scan
- `POST /scan_symbols` - Scan specific symbols
- `GET /get_scan_results` - Retrieve latest results

#### Key Functions
```python
def calculate_catalyst_score(news_items, symbol):
    """Score based on source, recency, keywords"""
    
def validate_with_technicals(symbol):
    """Confirm with price/volume action"""
    
def narrow_to_top_picks(candidates, limit=5):
    """Multi-criteria selection"""
```

### 3.3 Pattern Analysis Service (Port 5002)

#### Purpose
Detect technical patterns with catalyst context awareness

#### Endpoints
- `POST /analyze_pattern` - Analyze single symbol
- `POST /batch_analyze` - Analyze multiple symbols
- `GET /pattern_statistics` - Historical accuracy

#### Key Functions
```python
def detect_patterns_with_context(symbol, price_data, catalyst_type):
    """Weight patterns based on news context"""
    
def calculate_pattern_confidence(pattern, has_catalyst):
    """Boost confidence for catalyst-aligned patterns"""
```

### 3.4 Technical Analysis Service (Port 5003)

#### Purpose
Generate trading signals combining catalysts, patterns, and indicators

#### Endpoints
- `POST /generate_signal` - Generate for single symbol
- `POST /batch_signals` - Generate multiple signals
- `GET /signal_performance` - Track accuracy

#### Key Functions
```python
def generate_catalyst_weighted_signal(symbol, patterns, catalyst_data):
    """Combine all factors into actionable signal"""
    
def calculate_entry_exit_stops(signal, volatility, catalyst_strength):
    """Dynamic levels based on catalyst"""
```

### 3.5 Paper Trading Service (Port 5005)

#### Purpose
Execute trades via Alpaca API and track outcomes

#### Endpoints
- `POST /execute_trade` - Place order
- `GET /positions` - Current positions
- `GET /orders` - Order status
- `POST /close_position` - Exit trade
- `GET /performance` - P&L metrics

#### Key Functions
```python
def execute_with_risk_management(signal):
    """Apply position sizing and stop rules"""
    
def track_outcome_for_ml(trade_id, entry_reason):
    """Record for pattern learning"""
```

### 3.6 Coordination Service (Port 5000)

#### Purpose
Orchestrate the complete workflow from news to trades

#### Endpoints
- `POST /start_trading_cycle` - Begin workflow
- `GET /current_cycle` - Status check
- `GET /service_health` - Monitor all services
- `POST /workflow_config` - Update settings

#### Workflow Steps
1. Trigger news collection
2. Wait for completion
3. Run security scanner
4. Analyze patterns for picks
5. Generate signals
6. Execute trades
7. Update outcomes

### 3.7 Reporting Service (Port 5009)

#### Purpose
Analytics, metrics, and performance tracking

#### Endpoints
- `GET /daily_summary` - Trading day overview
- `GET /source_accuracy` - News source metrics
- `GET /pattern_performance` - Pattern success rates
- `GET /pnl_report` - Profit/loss analysis

### 3.8 Web Dashboard Service (Port 5010)

#### Purpose
Real-time monitoring and control interface

#### Features
- Live news feed with catalyst scores
- Current picks and signals
- Position monitoring
- Performance charts
- Source reliability dashboard
- System health status

---

## 4. Data Flow Specifications

### 4.1 Primary Trading Flow

```
1. News Collection (Continuous)
   Input: API calls to news sources
   Output: news_raw records with metadata
   Frequency: Every 5 minutes (pre-market) to 60 minutes (after-hours)

2. Security Scanning (Scheduled)
   Input: news_raw from last 24 hours
   Process: Score → Filter → Validate
   Output: trading_candidates (5 symbols)

3. Pattern Analysis (Per Candidate)
   Input: Symbol + catalyst context
   Process: Detect patterns + weight by catalyst
   Output: pattern_results with confidence

4. Signal Generation (Per Pattern)
   Input: Patterns + indicators + catalyst
   Process: Calculate confidence + entry/exit
   Output: trading_signals

5. Trade Execution (Per Signal)
   Input: Signals above threshold
   Process: Risk checks + order placement
   Output: trade_records + positions

6. Outcome Tracking (Post-Trade)
   Input: Closed positions
   Process: Update news accuracy + patterns
   Output: ML training data
```

### 4.2 Feedback Loops

#### 4.2.1 News → Outcome
- Track which news led to profitable trades
- Update source reliability scores
- Identify most predictive keywords

#### 4.2.2 Pattern → Outcome  
- Measure pattern success with/without catalysts
- Adjust pattern weights dynamically
- Build ML training dataset

---

## 5. Integration Points

### 5.1 External APIs
- **NewsAPI**: General market news
- **AlphaVantage**: Financial news + sentiment
- **RSS Feeds**: Real-time updates
- **yfinance**: Price and volume data
- **Alpaca Markets**: Trade execution

### 5.2 Internal Communication
- REST APIs between services
- JSON message format
- HTTP status codes for errors
- Timeout: 30 seconds default

---

## 6. Performance Requirements

### 6.1 Response Times
- News collection: < 5 minutes full cycle
- Security scan: < 30 seconds
- Pattern analysis: < 5 seconds per symbol
- Signal generation: < 2 seconds
- Trade execution: < 1 second

### 6.2 Throughput
- Handle 1000+ news articles per hour
- Process 100 securities in screening
- Analyze 20 candidates in detail
- Execute up to 50 trades per day

### 6.3 Availability
- 99% uptime during market hours
- Graceful degradation on service failure
- Automatic recovery mechanisms

---

## 7. Security Requirements

### 7.1 API Security
- Environment variables for credentials
- No credentials in code
- API rate limiting
- Request validation

### 7.2 Trading Security
- Alpaca paper trading only (initially)
- Position limits enforced
- Stop losses mandatory
- Manual override capability

### 7.3 Data Security
- No PII collected
- Secure credential storage
- Audit trail for all trades
- Encrypted API communications

---

## 8. Error Handling

### 8.1 Service Level
- Automatic retry with backoff
- Circuit breaker pattern
- Graceful degradation
- Error logging and alerting

### 8.2 System Level
- Service health monitoring
- Automatic service restart
- Database lock handling
- Resource exhaustion prevention

### 8.3 Trading Level
- Order rejection handling
- Partial fill management
- Connection loss recovery
- Market halt detection

---

## Implementation Priority

### Phase 1: Core News-Driven Flow
1. News Collection Service
2. Security Scanner with catalyst scoring
3. Basic pattern + signal generation
4. Paper trading integration

### Phase 2: Intelligence Layer
1. Source alignment tracking
2. Confirmation monitoring
3. Narrative clustering
4. Outcome tracking

### Phase 3: ML Preparation
1. Training data collection
2. Pattern success tracking
3. Source accuracy metrics
4. Feedback loop completion

This specification provides the complete functional blueprint for implementing the Catalyst Trading System v2.0.0.