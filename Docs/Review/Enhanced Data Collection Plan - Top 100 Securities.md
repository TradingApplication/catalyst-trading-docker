# Enhanced Data Collection Plan - Top 100 Securities

## Executive Summary
Implement comprehensive data collection for top 100 securities (not just final 5) to enable advanced pattern detection between news, companies, and market movements. Use intelligent data aging to manage storage costs while maximizing ML training data.

## Architecture Overview

```
News Service → Identifies 500+ mentioned securities
     ↓
Scanner Service → Scores and ranks top 100
     ↓
Data Collector → Captures all 100 (not just top 5)
     ↓
Time-Based Aggregation → 15min → 1hr → Daily → Weekly
     ↓
ML Pattern Analysis → Find hidden correlations
```

## Implementation Plan

### Phase 1: Enhanced Scanner Service

```python
# scanner_service_enhanced.py
class EnhancedSecurityScanner:
    """
    Modified scanner that tracks top 100 instead of just top 5
    """
    
    def __init__(self):
        self.TOP_CANDIDATES_TO_TRACK = 100  # Up from 5
        self.TOP_CANDIDATES_TO_TRADE = 5    # Still only trade top 5
        self.tracked_securities = set()      # Currently tracking
        
    def scan_and_track_securities(self) -> Dict:
        """Enhanced scanning that keeps top 100"""
        
        # 1. Get all securities with news mentions
        news_mentions = self.get_news_mentioned_securities()  # ~500+ symbols
        
        # 2. Score all candidates
        scored_candidates = []
        for symbol in news_mentions:
            score = self.calculate_catalyst_score(symbol)
            scored_candidates.append({
                'symbol': symbol,
                'score': score,
                'news_count': score['news_count'],
                'catalyst_type': score['catalyst_type'],
                'first_seen': datetime.now()
            })
            
        # 3. Sort and get top 100
        scored_candidates.sort(key=lambda x: x['score']['total'], reverse=True)
        top_100 = scored_candidates[:self.TOP_CANDIDATES_TO_TRACK]
        top_5_trading = scored_candidates[:self.TOP_CANDIDATES_TO_TRADE]
        
        # 4. Start tracking all top 100
        for candidate in top_100:
            self.start_tracking_security(candidate)
            
        # 5. Return both sets
        return {
            'trading_candidates': top_5_trading,  # For immediate trading
            'tracking_candidates': top_100,        # For data collection
            'total_scanned': len(news_mentions),
            'timestamp': datetime.now()
        }
```

### Phase 2: Intelligent Data Collection Service

```python
# data_collection_service.py
class IntelligentDataCollector:
    """
    Collects and ages data for top 100 securities efficiently
    """
    
    def __init__(self):
        self.collection_intervals = {
            'ultra_high_freq': timedelta(minutes=1),   # Top movers
            'high_freq': timedelta(minutes=15),        # Active tracking
            'medium_freq': timedelta(hours=1),         # Aging data
            'low_freq': timedelta(hours=24),           # Historical
            'archive': timedelta(days=7)               # Weekly rollup
        }
        
        # Track security states
        self.security_tracking_state = {}  # symbol -> tracking_info
        
    def start_tracking_security(self, security_info: Dict):
        """Begin tracking a new security or update existing"""
        symbol = security_info['symbol']
        
        # Check if already tracking
        if symbol in self.security_tracking_state:
            # Update catalyst score, might need higher frequency
            self.update_tracking_priority(symbol, security_info)
        else:
            # New security to track
            self.security_tracking_state[symbol] = {
                'symbol': symbol,
                'first_seen': datetime.now(),
                'last_updated': datetime.now(),
                'collection_frequency': 'high_freq',  # Start aggressive
                'data_points_collected': 0,
                'catalyst_events': [security_info],
                'price_volatility': 0,
                'volume_profile': {},
                'aging_schedule': self.calculate_aging_schedule(security_info)
            }
            
            # Start immediate data collection
            self.collect_security_data(symbol, initial=True)
    
    def collect_security_data(self, symbol: str, initial: bool = False):
        """Collect comprehensive data for a security"""
        
        # Get current tracking info
        tracking_info = self.security_tracking_state.get(symbol)
        if not tracking_info:
            return
            
        # Collect based on current frequency
        data = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'frequency': tracking_info['collection_frequency'],
            
            # Price data
            'price_data': self.get_price_data(symbol),
            
            # Volume analysis
            'volume_data': self.get_volume_analysis(symbol),
            
            # Technical indicators
            'technical_indicators': self.calculate_indicators(symbol),
            
            # News sentiment
            'news_sentiment': self.get_current_sentiment(symbol),
            
            # Market microstructure
            'bid_ask_spread': self.get_spread_data(symbol),
            'order_flow': self.analyze_order_flow(symbol),
            
            # Correlation data
            'sector_correlation': self.calculate_sector_correlation(symbol),
            'market_correlation': self.calculate_market_correlation(symbol)
        }
        
        # Store in appropriate table based on frequency
        self.store_data_by_frequency(data, tracking_info['collection_frequency'])
        
        # Update tracking state
        tracking_info['last_updated'] = datetime.now()
        tracking_info['data_points_collected'] += 1
        
        # Check if we should age this security
        self.check_aging_criteria(symbol, tracking_info)
```

### Phase 3: Smart Aging System

```python
# data_aging_system.py
class SmartDataAging:
    """
    Intelligently ages data from high-freq to low-freq collection
    """
    
    def __init__(self):
        self.aging_criteria = {
            'volatility_threshold': 0.02,      # 2% movement triggers high-freq
            'volume_threshold': 2.0,           # 2x average volume
            'news_threshold': 5,               # 5+ news mentions
            'flatline_periods': 4,             # Consecutive periods of low activity
            'catalyst_decay_hours': 48         # How long catalyst effect lasts
        }
        
    def check_aging_criteria(self, symbol: str, tracking_info: Dict):
        """Determine if security should move to different frequency"""
        
        # Calculate activity metrics
        metrics = self.calculate_activity_metrics(symbol, tracking_info)
        
        # Current frequency
        current_freq = tracking_info['collection_frequency']
        
        # Promotion criteria (move to higher frequency)
        if self.should_promote_frequency(metrics):
            if current_freq == 'medium_freq':
                tracking_info['collection_frequency'] = 'high_freq'
                print(f"📈 {symbol} promoted to high frequency collection")
            elif current_freq == 'low_freq':
                tracking_info['collection_frequency'] = 'medium_freq'
                
        # Demotion criteria (move to lower frequency)
        elif self.should_demote_frequency(metrics):
            if current_freq == 'high_freq':
                tracking_info['collection_frequency'] = 'medium_freq'
                print(f"📉 {symbol} demoted to medium frequency")
            elif current_freq == 'medium_freq':
                tracking_info['collection_frequency'] = 'low_freq'
                
    def should_promote_frequency(self, metrics: Dict) -> bool:
        """Check if security needs more frequent monitoring"""
        
        # High volatility
        if metrics['volatility'] > self.aging_criteria['volatility_threshold']:
            return True
            
        # Volume surge
        if metrics['relative_volume'] > self.aging_criteria['volume_threshold']:
            return True
            
        # News catalyst
        if metrics['recent_news_count'] > self.aging_criteria['news_threshold']:
            return True
            
        # Technical breakout
        if metrics['technical_signal_strength'] > 0.7:
            return True
            
        return False
        
    def should_demote_frequency(self, metrics: Dict) -> bool:
        """Check if security can be monitored less frequently"""
        
        # Flatlined price action
        if metrics['flatline_score'] > self.aging_criteria['flatline_periods']:
            return True
            
        # Low volume
        if metrics['relative_volume'] < 0.5:
            return True
            
        # Old catalyst
        hours_since_catalyst = metrics['hours_since_last_catalyst']
        if hours_since_catalyst > self.aging_criteria['catalyst_decay_hours']:
            return True
            
        return False
```

### Phase 4: Database Schema for Tiered Storage

```sql
-- High-frequency data (15-min intervals)
CREATE TABLE security_data_high_freq (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Price data
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    volume BIGINT,
    
    -- Microstructure
    bid_ask_spread DECIMAL(8,4),
    order_imbalance DECIMAL(8,4),
    
    -- Indicators
    rsi_14 DECIMAL(5,2),
    vwap DECIMAL(10,2),
    
    -- Context
    news_count INTEGER,
    catalyst_active BOOLEAN,
    
    -- Indexes for performance
    INDEX idx_symbol_time (symbol, timestamp DESC),
    INDEX idx_timestamp (timestamp)
) PARTITION BY RANGE (timestamp);

-- Create partitions for each week
CREATE TABLE security_data_high_freq_2025_w27 
    PARTITION OF security_data_high_freq
    FOR VALUES FROM ('2025-07-07') TO ('2025-07-14');

-- Medium-frequency data (hourly)
CREATE TABLE security_data_hourly (
    -- Similar structure but hourly aggregates
    -- Includes OHLC for the hour, volume profile, etc.
);

-- Daily aggregates
CREATE TABLE security_data_daily (
    -- Daily OHLC, indicators, news summary
    -- Links to intraday data for drill-down
);

-- Pattern detection results
CREATE TABLE ml_pattern_discoveries (
    id SERIAL PRIMARY KEY,
    discovery_date TIMESTAMPTZ DEFAULT NOW(),
    pattern_type VARCHAR(100),
    
    -- Pattern details
    securities_involved JSONB,  -- Array of symbols
    pattern_confidence DECIMAL(5,2),
    
    -- What triggered this pattern
    trigger_conditions JSONB,
    news_correlation JSONB,
    
    -- Predictive value
    predicted_outcome JSONB,
    actual_outcome JSONB,
    
    -- For ML training
    feature_vector JSONB,
    model_version VARCHAR(50)
);
```

### Phase 5: Pattern Discovery Engine

```python
# pattern_discovery_engine.py
class PatternDiscoveryEngine:
    """
    Analyzes top 100 securities to find hidden patterns
    """
    
    def __init__(self):
        self.pattern_types = [
            'catalyst_sympathy',      # Securities that move together on news
            'institutional_rotation',  # Large player movements
            'pump_and_dump',          # Artificial price movements
            'news_arbitrage',         # Price disparities across news
            'sector_momentum',        # Sector-wide movements
            'correlation_breaks'      # When correlations fail
        ]
        
    def discover_patterns(self, lookback_hours: int = 24):
        """Run pattern discovery on collected data"""
        
        # Get all tracked securities data
        securities_data = self.load_tracked_securities_data(lookback_hours)
        
        patterns_found = []
        
        # 1. Catalyst Sympathy Patterns
        sympathy_patterns = self.find_catalyst_sympathy(securities_data)
        patterns_found.extend(sympathy_patterns)
        
        # 2. News-Price Correlation Patterns
        news_patterns = self.analyze_news_price_correlation(securities_data)
        patterns_found.extend(news_patterns)
        
        # 3. Hidden Market Maker Patterns
        mm_patterns = self.detect_market_maker_activity(securities_data)
        patterns_found.extend(mm_patterns)
        
        # 4. Cross-Security Arbitrage
        arb_patterns = self.find_arbitrage_opportunities(securities_data)
        patterns_found.extend(arb_patterns)
        
        # Store discoveries for ML training
        self.store_pattern_discoveries(patterns_found)
        
        # Alert on high-confidence patterns
        self.alert_on_actionable_patterns(patterns_found)
        
        return patterns_found
        
    def find_catalyst_sympathy(self, data: pd.DataFrame) -> List[Dict]:
        """Find securities that move together on similar news"""
        
        patterns = []
        
        # Group by time windows
        for window in pd.Grouper(freq='1H'):
            window_data = data.groupby(window)
            
            # Find securities with similar news
            news_clusters = self.cluster_by_news_similarity(window_data)
            
            for cluster in news_clusters:
                # Check if they moved together
                price_correlation = self.calculate_price_correlation(cluster)
                
                if price_correlation > 0.7:
                    pattern = {
                        'type': 'catalyst_sympathy',
                        'securities': cluster['symbols'],
                        'correlation': price_correlation,
                        'news_theme': cluster['common_theme'],
                        'timestamp': window,
                        'confidence': self.calculate_pattern_confidence(cluster)
                    }
                    patterns.append(pattern)
                    
        return patterns
```

### Implementation Timeline

#### Tonight (WSL Testing)
```bash
# Test on your laptop with WSL
1. Create enhanced scanner logic
2. Test with paper trading API
3. Verify data collection works
4. Check storage requirements
```

#### This Week
```python
# Deploy to production
1. Update scanner_service.py to track top 100
2. Create data_collection_service.py
3. Implement aging logic
4. Update database schema
5. Monitor storage growth
```

#### Next Week
```python
# Pattern discovery
1. Build pattern discovery engine
2. Start finding correlations
3. Feed discoveries to ML
4. Track pattern success rates
```

### Storage Cost Analysis

```python
# Estimated storage growth
storage_estimate = {
    'per_security_per_day': {
        'high_freq_15min': '96 records * 200 bytes = 19.2 KB',
        'hourly_after_24h': '24 records * 150 bytes = 3.6 KB',
        'daily_after_7d': '1 record * 500 bytes = 0.5 KB'
    },
    
    'top_100_securities': {
        'daily': '100 * 23.3 KB = 2.33 MB',
        'monthly': '2.33 MB * 30 = 70 MB',
        'yearly': '70 MB * 12 = 840 MB'
    },
    
    'with_ml_features': {
        'multiply_by': 3,  # Feature engineering triples size
        'yearly_total': '2.5 GB'
    },
    
    'digitalocean_costs': {
        'block_storage_100gb': '$10/month',
        'block_storage_500gb': '$50/month',
        'years_until_500gb': 5
    }
}
```

### Quick Start Code for Tonight

```python
# test_enhanced_collection.py
"""
Test enhanced data collection on WSL tonight
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json

class QuickDataCollectorTest:
    def __init__(self):
        self.top_100_file = 'top_100_tracking.json'
        self.load_tracking_state()
        
    def test_collection_loop(self):
        """Test collecting data for multiple securities"""
        
        # Test symbols
        test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        for symbol in test_symbols:
            print(f"\nCollecting data for {symbol}...")
            
            # Get data
            data = self.collect_security_data(symbol)
            
            # Check if we should track
            if self.should_track_security(data):
                self.add_to_tracking(symbol, data)
                print(f"✅ Added {symbol} to tracking")
            else:
                print(f"❌ {symbol} doesn't meet criteria")
                
    def collect_security_data(self, symbol):
        """Collect comprehensive data"""
        ticker = yf.Ticker(symbol)
        
        # Get various timeframes
        data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            '1d_data': ticker.history(period='1d', interval='15m'),
            '5d_data': ticker.history(period='5d', interval='1h'),
            'info': ticker.info,
            'news': ticker.news[:5] if hasattr(ticker, 'news') else []
        }
        
        # Calculate metrics
        if not data['1d_data'].empty:
            data['volatility'] = data['1d_data']['Close'].pct_change().std()
            data['volume_ratio'] = (
                data['1d_data']['Volume'].iloc[-1] / 
                data['1d_data']['Volume'].mean()
            )
        
        return data
        
    def should_track_security(self, data):
        """Determine if security should be tracked"""
        # High volatility
        if data.get('volatility', 0) > 0.02:
            return True
            
        # High volume
        if data.get('volume_ratio', 0) > 2:
            return True
            
        # Has news
        if len(data.get('news', [])) > 0:
            return True
            
        return False

# Run test
if __name__ == "__main__":
    collector = QuickDataCollectorTest()
    collector.test_collection_loop()
    collector.save_tracking_state()
    print("\nTest complete! Check top_100_tracking.json")
```

## Key Benefits

1. **Pattern Discovery**: With 100 securities, we can find:
   - Which stocks move together on similar news
   - Hidden market maker accumulation patterns
   - News arbitrage opportunities
   - Sector rotation patterns

2. **ML Training Data**: 20x more data points for:
   - Better feature engineering
   - More robust pattern detection
   - Correlation analysis
   - Anomaly detection

3. **Cost Efficiency**: 
   - Smart aging reduces storage needs
   - Only $10-50/month for massive data advantage
   - ROI through better pattern detection

4. **Competitive Edge**:
   - Most traders only watch their positions
   - We're building a market intelligence system
   - Find opportunities before they're obvious

## Next Steps

1. **Tonight**: Test enhanced collection in WSL
2. **Tomorrow**: Update scanner service in production
3. **This Week**: Deploy full tracking system
4. **Next Week**: Start pattern discovery

*"More data, more patterns, more profits!"* 🚀