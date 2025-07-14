# Catalyst Trading System - Strategy & ML Integration Roadmap

**Version**: 1.0  
**Date**: July 9, 2025  
**Status**: Vision & Planning Document  
**Time Horizon**: 3 Years

## Executive Summary

This document outlines the strategic vision for evolving the Catalyst Trading System from a news-driven day trading platform into a sophisticated market intelligence system capable of tracking economic cascades, predicting cross-market impacts, and identifying hidden correlations. While we start with "baby steps" of paper trading at 55% accuracy, our architecture is designed to support Ray Dalio-level economic modeling—on a startup budget.

## Vision Statement

> "Build a trading system that not only reacts to news but understands the economic butterfly effect—where a lithium mine accident in Chile creates predictable ripples through EVs, traditional auto, and oil markets over weeks and months."

---

## Phase Overview

### Phase 1: Foundation (Months 1-6) ✅ [CURRENT]
**Budget**: $24-34/month (DigitalOcean)  
**Target**: 55% trading accuracy  
**Focus**: Get profitable paper trading working

### Phase 2: Intelligence Layer (Months 7-12)
**Budget**: $50-75/month  
**Target**: 60-65% accuracy  
**Focus**: Pattern recognition and source intelligence

### Phase 3: ML Integration (Year 2)
**Budget**: $100-150/month + Local GPU  
**Target**: 65-70% accuracy  
**Focus**: Predictive models and correlation discovery

### Phase 4: Cascade Modeling (Year 3)
**Budget**: $200-300/month  
**Target**: 70-75% accuracy  
**Focus**: Economic cascade prediction

---

## Phase 1: Foundation & Data Collection (Current)

### 1.1 Current State ✅
- News-driven security selection
- Recording 50 securities per scan → market_data
- Paper trading top 5 selections
- Basic pattern recognition
- Source reliability tracking

### 1.2 Immediate Goals (Months 1-3)
```python
# What we're building now
FOUNDATION_GOALS = {
    'trading_accuracy': 55,  # Percent win rate
    'daily_trades': 5,       # Focus on quality
    'data_collection': 50,   # Securities per scan
    'unique_securities': 1000  # After 3 months
}
```

### 1.3 Data Foundation Being Built
Every day we're collecting:
- Price/volume data for 50 securities
- News catalyst associations
- Pattern outcomes
- Source accuracy metrics

This creates our ML training dataset without additional cost.

### 1.4 Success Metrics
- [ ] Achieve 55% win rate consistently
- [ ] Build dataset of 1000+ unique securities
- [ ] Validate news → price correlation
- [ ] Establish baseline performance metrics

---

## Phase 2: Intelligence Layer (Months 7-12)

### 2.1 Enhanced Pattern Recognition

#### Technical Patterns 2.0
```python
ENHANCED_PATTERNS = {
    'traditional': [
        'double_bottom', 'head_shoulders', 'cup_handle',
        'ascending_triangle', 'bull_flag', 'morning_star'
    ],
    'catalyst_specific': [
        'earnings_gap_fade',      # Specific to earnings
        'fda_binary_setup',        # FDA approval patterns
        'merger_arb_spread',       # M&A specific
        'sympathy_momentum'        # Sector contagion
    ],
    'time_based': [
        'pre_market_breakout',
        'opening_range_break',
        'lunch_reversal',
        'closing_imbalance'
    ]
}
```

#### Pattern Context Awareness
- How patterns behave differently with catalysts
- Time-of-day pattern effectiveness
- Pattern success by market conditions

### 2.2 Source Intelligence Network

```python
class SourceIntelligence:
    """Track not just accuracy but agenda and timing"""
    
    def analyze_source_patterns(self, source):
        return {
            'accuracy_rate': 0.72,
            'average_early_warning': '3.5 hours',
            'agenda_bias': 'bullish_tech',
            'pump_detection_score': 0.85,
            'insider_correlation': 0.23,
            'best_for': ['tech', 'biotech'],
            'worst_for': ['energy', 'utilities']
        }
```

### 2.3 Narrative Cluster Analysis

Identify coordinated news campaigns:
- Multiple Tier-5 sources publish similar stories
- Detect pump & dump patterns
- Track narrative evolution
- Identify the "smart money" sources

### 2.4 Correlation Discovery

Start finding hidden relationships:
```python
CORRELATION_TARGETS = {
    'sector_correlations': {
        'semiconductor': ['tech', 'auto', 'defense'],
        'energy': ['airlines', 'shipping', 'chemicals'],
        'banking': ['real_estate', 'construction', 'retail']
    },
    'time_delays': {
        'immediate': '0-4 hours',
        'same_day': '4-8 hours',
        'next_day': '1-2 days',
        'weekly': '3-7 days'
    }
}
```

### 2.5 Infrastructure Upgrades
- Add dedicated pattern analysis server
- Implement Redis clustering
- Deploy time-series database (TimescaleDB)
- Add real-time correlation engine

---

## Phase 3: ML Integration (Year 2)

### 3.1 Local GPU Processing

#### Hardware Investment (~$2000)
```yaml
Local ML Server:
  GPU: NVIDIA RTX 4070 Ti (16GB)
  CPU: AMD Ryzen 9 5900X
  RAM: 64GB DDR4
  Storage: 2TB NVMe
  Purpose: Model training and batch inference
```

#### Cloud + Local Hybrid
- DigitalOcean: Real-time trading and data collection
- Local GPU: Nightly model training and analysis
- Sync: Model updates pushed to cloud daily

### 3.2 Initial ML Models

#### 3.2.1 Pattern Success Predictor
```python
class PatternSuccessModel:
    """Predict pattern outcome based on context"""
    
    features = [
        'pattern_type',
        'catalyst_present',
        'catalyst_type',
        'news_sentiment',
        'volume_ratio',
        'time_of_day',
        'market_conditions',
        'source_reliability',
        'sector_momentum'
    ]
    
    target = 'pattern_success_probability'
```

#### 3.2.2 Catalyst Impact Duration Model
```python
class CatalystDecayModel:
    """Predict how long a catalyst remains tradeable"""
    
    def predict_impact_duration(self, catalyst):
        # Returns expected hours of impact
        if catalyst.type == 'earnings':
            return 48  # 2 days
        elif catalyst.type == 'fda_approval':
            return 336  # 2 weeks
        elif catalyst.type == 'merger':
            return 720  # 30 days
```

#### 3.2.3 Source Reliability Predictor
ML model to predict which sources will be accurate:
- Track source historical accuracy
- Identify agenda patterns
- Predict pump & dump schemes
- Score source credibility in real-time

### 3.3 Advanced Correlation Discovery

#### Sector Rotation Model
```python
class SectorRotationPredictor:
    """Predict sector money flows"""
    
    def predict_rotation(self, trigger_event):
        if trigger_event == 'rate_hike':
            return {
                'out_of': ['tech', 'growth'],
                'into': ['banks', 'value'],
                'timeline': '2-5 days'
            }
```

#### Hidden Relationship Finder
- Use graph neural networks
- Discover non-obvious correlations
- Map supply chain impacts
- Identify substitution effects

### 3.4 Feature Engineering Pipeline

```python
FEATURE_CATEGORIES = {
    'price_features': [
        'returns_1h', 'returns_1d', 'returns_1w',
        'volatility_realized', 'volatility_implied',
        'volume_profile', 'price_levels'
    ],
    'news_features': [
        'catalyst_score', 'source_tier', 'sentiment',
        'similar_stories_count', 'confirmation_speed',
        'narrative_cluster_id'
    ],
    'market_features': [
        'vix_level', 'sector_performance',
        'market_breadth', 'correlation_breakdown'
    ],
    'temporal_features': [
        'time_of_day', 'day_of_week', 'days_to_earnings',
        'days_to_fomc', 'option_expiry_proximity'
    ]
}
```

---

## Phase 4: Economic Cascade Modeling (Year 3)

### 4.1 The Vision: Butterfly Effect Trading

#### Cascade Example: Lithium Mine Accident
```python
CASCADE_MODEL = {
    'trigger': 'Major lithium mine accident in Chile',
    'timeline': {
        'hours_0_4': {
            'direct': ['LAC +12%', 'ALB +8%', 'SQM +10%'],
            'futures': ['Lithium futures +15%']
        },
        'days_1_3': {
            'secondary': ['TSLA -3%', 'GM -2%', 'F -1.5%'],
            'suppliers': ['CATL -4%', 'BYD -3%']
        },
        'week_1_2': {
            'substitution': ['Traditional auto +1%', 'Oil majors +0.5%'],
            'speculation': ['Junior lithium miners +20-50%']
        },
        'month_1_2': {
            'policy': ['EV subsidies questioned', 'Mining regulations'],
            'strategic': ['Auto makers seek alternatives']
        }
    }
}
```

### 4.2 Cascade Detection Architecture

```python
class CascadeDetectionSystem:
    def __init__(self):
        self.monitors = {
            'commodities': CommodityMonitor(),
            'geopolitical': GeopoliticalMonitor(),
            'supply_chain': SupplyChainMonitor(),
            'policy': PolicyMonitor()
        }
        
    def detect_cascade_initiation(self, event):
        """Identify potential cascade triggers"""
        
        cascade_probability = self.assess_cascade_probability(event)
        if cascade_probability > 0.7:
            return self.predict_cascade_path(event)
```

### 4.3 Multi-Asset Correlation Matrix

Track relationships between:
- Commodities → Equities
- Currencies → Sectors  
- Geopolitical events → Markets
- Policy changes → Industries

### 4.4 Advanced Cascade Models

#### 4.4.1 Supply Chain Disruption Model
```python
SUPPLY_CHAIN_IMPACTS = {
    'semiconductor_shortage': {
        'immediate': ['NVDA', 'AMD', 'TSM'],
        '1_week': ['AAPL', 'Auto manufacturers'],
        '1_month': ['Consumer electronics', 'Industrial'],
        '3_month': ['Broad economic impact']
    }
}
```

#### 4.4.2 Geopolitical Event Model
```python
GEOPOLITICAL_CASCADES = {
    'middle_east_conflict': {
        'immediate': ['Oil futures', 'Defense stocks'],
        'days': ['Airlines', 'Shipping', 'EUR/USD'],
        'weeks': ['Inflation expectations', 'Central bank policy']
    },
    'trade_war_escalation': {
        'targets': ['Direct tariff targets'],
        'supply_chain': ['Alternative suppliers'],
        'currency': ['USD strength', 'EM weakness']
    }
}
```

#### 4.4.3 Policy Change Propagation
- Interest rate changes → Sector rotation
- Tax policy → Industry winners/losers
- Regulatory changes → Compliance costs
- Subsidy programs → Beneficiary identification

### 4.5 Implementation Architecture

```yaml
Cascade System Components:
  Data Ingestion:
    - Commodity feeds (metals, energy, agriculture)
    - Geopolitical news monitoring
    - Policy announcement tracking
    - Supply chain databases
    
  Processing:
    - Event classification engine
    - Cascade probability calculator
    - Impact propagation simulator
    - Timeline predictor
    
  Execution:
    - Multi-asset position manager
    - Cascade trade executor
    - Risk management overlay
    - Performance attribution
```

---

## Infrastructure Evolution

### Current (Phase 1)
```yaml
DigitalOcean Setup:
  Droplet: 4GB RAM ($24/month)
  Database: Basic PostgreSQL
  Storage: 80GB
  Services: 8 Python services
```

### Phase 2 (Months 7-12)
```yaml
Enhanced Setup:
  Droplet: 8GB RAM ($48/month)
  Database: PostgreSQL with read replica
  Cache: Redis cluster
  Storage: 200GB SSD
  ML: Basic sklearn models
```

### Phase 3 (Year 2)
```yaml
Hybrid Cloud + Local:
  Cloud: 
    - 8GB Droplet ($48/month)
    - Managed PostgreSQL ($15/month)
    - Redis ($15/month)
  Local:
    - GPU Server ($2000 one-time)
    - Model training nightly
```

### Phase 4 (Year 3)
```yaml
Full System:
  Cloud:
    - 16GB Droplet ($96/month)
    - PostgreSQL cluster ($60/month)
    - Redis cluster ($30/month)
    - TimescaleDB ($40/month)
  Local:
    - 2x GPU Server
    - Real-time inference
```

---

## Risk Management Evolution

### Phase 1: Basic Risk Controls
- Position sizing (max 20%)
- Stop losses (2%)
- Maximum daily loss (6%)

### Phase 2: Adaptive Risk
- Volatility-based position sizing
- Catalyst-specific risk limits
- Source reliability weighting

### Phase 3: ML Risk Models
- Predict drawdown probability
- Dynamic position sizing
- Correlation-based hedging

### Phase 4: Cascade Risk Management
- Multi-asset exposure limits
- Cascade scenario testing
- Automated hedging strategies

---

## Success Metrics by Phase

### Phase 1 (Current)
- Trading accuracy: 55%
- Sharpe ratio: 1.0
- Max drawdown: 15%
- Monthly return: 5-10%

### Phase 2
- Trading accuracy: 60-65%
- Sharpe ratio: 1.5
- Max drawdown: 12%
- Monthly return: 10-15%

### Phase 3
- Trading accuracy: 65-70%
- Sharpe ratio: 2.0
- Max drawdown: 10%
- Monthly return: 15-20%

### Phase 4
- Trading accuracy: 70-75%
- Sharpe ratio: 2.5+
- Max drawdown: 8%
- Monthly return: 20-30%

---

## Research & Development Topics

### Near-term R&D (Phase 1-2)
1. Optimal data aggregation strategies
2. Pattern detection improvements
3. Source reliability scoring
4. Basic correlation discovery

### Medium-term R&D (Phase 3)
1. Graph neural networks for market relationships
2. Transformer models for news analysis
3. Reinforcement learning for position sizing
4. Ensemble methods for signal generation

### Long-term R&D (Phase 4)
1. Economic cascade simulation
2. Multi-agent market modeling
3. Alternative data integration
4. Quantum computing applications

---

## Budget Projection

### 3-Year Total Investment
```
Infrastructure Costs:
- Year 1: $408 (cloud only)
- Year 2: $900 (cloud) + $2000 (GPU)
- Year 3: $2,640 (cloud) + $2000 (GPU upgrade)
- Total: $7,948

Potential Returns (Paper Trading → Live):
- Year 1: Validation only
- Year 2: $50K account → $80K (60% return)
- Year 3: $80K → $160K (100% return)
- ROI: 2,000%+
```

---

## Key Principles

1. **Start Small, Think Big**: Paper trade at 55% while building for 75%
2. **Data First**: Every trade contributes to future ML
3. **Incremental Progress**: Each phase builds on the last
4. **Budget Conscious**: Maximum impact per dollar spent
5. **Risk Aware**: Never risk the foundation for the dream

---

## Conclusion

The Catalyst Trading System is designed to evolve from a simple news-driven trader into a sophisticated market intelligence platform. By starting with focused execution on high-conviction trades while collecting comprehensive data, we build the foundation for advanced ML and cascade modeling—all on a startup budget.

The journey from 55% to 75% accuracy is not just about better algorithms; it's about understanding the interconnected nature of global markets and building systems that can see patterns humans miss.

**Remember**: Ray Dalio didn't build Bridgewater in a day, but he did start somewhere. This is our somewhere.