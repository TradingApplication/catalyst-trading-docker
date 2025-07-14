# Catalyst Trading System - Development & Design Reconciliation Plan v1.0

**Date**: July 8, 2025  
**Purpose**: Align current implementation with enhanced design specifications for 100 securities tracking  
**Priority**: Critical - Required for AI/ML pattern discovery capabilities

---

## Executive Summary

The current implementation tracks only 5 securities for trading, while the enhanced design requires tracking 100 securities continuously to enable AI/ML pattern discovery. This reconciliation plan outlines the specific changes needed to align development with design specifications.

---

## 1. Critical Gaps Identified

### 1.1 Core Functionality Gaps
- ❌ **100 Securities Tracking**: System only tracks final 5, not top 100
- ❌ **Data Collection Service**: Missing entirely
- ❌ **Pattern Discovery Engine**: Not implemented
- ❌ **Intelligent Data Aging**: No aging system for efficient storage
- ❌ **Cross-Security Analysis**: No correlation tracking between securities

### 1.2 Database Schema Gaps
- ❌ Missing `security_tracking_state` table
- ❌ Missing `security_data_high_freq` table
- ❌ Missing `security_data_hourly` table
- ❌ Missing `security_data_daily` table
- ❌ Missing `ml_pattern_discoveries` table

### 1.3 Service Integration Gaps
- ❌ Coordination service missing critical database functions
- ❌ Scanner service not implementing enhanced tracking
- ❌ News service not optimized for 100 securities

---

## 2. Immediate Actions (Today - Critical)

### 2.1 Fix Coordination Service Crashes

**File**: `database_utils.py`  
**Version**: Update to v2.33  
**Action**: Add missing functions

```python
# Add these functions to database_utils.py:

def create_trading_cycle(cycle_data: Dict) -> int:
    """Create new trading cycle"""
    # Implementation provided in previous artifact

def update_trading_cycle(cycle_id: str, updates: Dict) -> bool:
    """Update trading cycle metrics"""
    # Implementation provided in previous artifact

def log_workflow_step(cycle_id: str, step: str, status: str, details: Dict = None) -> bool:
    """Log workflow execution steps"""
    # Implementation provided in previous artifact

def update_service_health(service_name: str, status: str, metrics: Dict = None) -> bool:
    """Update service health status"""
    # Implementation provided in previous artifact

def get_configuration(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    # Implementation provided in previous artifact
```

**Validation**: 
```bash
docker-compose build --no-cache coordination-service
docker-compose up -d coordination-service
docker logs -f catalyst-coordination  # Should start without errors
```

### 2.2 Create Database Tables

**Action**: Run table creation script

```sql
-- Add to create_tables() function in database_utils.py:

CREATE TABLE security_tracking_state (
    symbol VARCHAR(10) PRIMARY KEY,
    first_seen TIMESTAMP NOT NULL,
    last_updated TIMESTAMP,
    collection_frequency VARCHAR(20) DEFAULT 'high_freq',
    data_points_collected INTEGER DEFAULT 0,
    price_volatility DECIMAL(5,4),
    volume_profile JSONB,
    aging_schedule JSONB,
    tracking_reason VARCHAR(100),
    last_catalyst_score DECIMAL(5,2)
);

CREATE TABLE security_data_high_freq (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    price DECIMAL(10,2),
    volume BIGINT,
    bid DECIMAL(10,2),
    ask DECIMAL(10,2),
    spread DECIMAL(6,4),
    volatility_15min DECIMAL(5,4),
    relative_volume DECIMAL(5,2),
    news_mentions INTEGER DEFAULT 0,
    metadata JSONB,
    INDEX idx_symbol_time (symbol, timestamp DESC)
);

CREATE TABLE ml_pattern_discoveries (
    id SERIAL PRIMARY KEY,
    discovery_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    pattern_type VARCHAR(100),
    securities_involved JSONB,
    pattern_confidence DECIMAL(5,2),
    trigger_conditions JSONB,
    predicted_outcome JSONB,
    actual_outcome JSONB,
    backtest_results JSONB
);
```

---

## 3. Phase 1: Enhanced Scanner Service (This Week)

### 3.1 Update Scanner Service

**File**: `scanner_service.py`  
**Version**: Update to v3.1.0  
**Changes**:

```python
class EnhancedDynamicSecurityScanner:
    def __init__(self):
        # Existing initialization...
        
        # Add enhanced tracking parameters
        self.tracking_params = {
            'top_candidates_to_track': int(os.getenv('TOP_CANDIDATES_TO_TRACK', '100')),
            'top_candidates_to_trade': int(os.getenv('TOP_CANDIDATES_TO_TRADE', '5')),
            'tracking_cache_ttl': int(os.getenv('TRACKING_CACHE_TTL', '900')),  # 15 min
            'pattern_discovery_interval': int(os.getenv('PATTERN_DISCOVERY_INTERVAL', '3600'))  # 1 hour
        }
        
        # Tracking state
        self.tracked_securities = {}
        self.last_pattern_discovery = datetime.now()
    
    def perform_enhanced_scan(self, mode: str = 'normal') -> Dict:
        """Enhanced scan that tracks 100 but trades 5"""
        
        # Get initial universe
        universe = self._get_initial_universe()
        
        # Score all candidates
        scored_candidates = self._score_all_candidates(universe)
        
        # Get top 100 for tracking
        top_100 = scored_candidates[:self.tracking_params['top_candidates_to_track']]
        
        # Start/update tracking for all 100
        for candidate in top_100:
            self._update_tracking_state(candidate)
        
        # Get top 5 for trading
        top_5 = scored_candidates[:self.tracking_params['top_candidates_to_trade']]
        
        # Store tracking data
        self._store_tracking_data(top_100)
        
        # Return both sets
        return {
            'scan_id': self._generate_scan_id(mode),
            'timestamp': datetime.now().isoformat(),
            'mode': mode,
            'trading_candidates': top_5,      # For immediate trading
            'tracked_securities': top_100,    # For data collection
            'total_universe': len(universe),
            'execution_time': time.time() - start_time
        }
```

### 3.2 Update Environment Configuration

**File**: `.env`  
**Add**:

```bash
# Enhanced Tracking Configuration
TOP_CANDIDATES_TO_TRACK=100
TOP_CANDIDATES_TO_TRADE=5
TRACKING_CACHE_TTL=900
PATTERN_DISCOVERY_INTERVAL=3600

# Data Collection Intervals
HIGH_FREQ_INTERVAL=900          # 15 minutes
MEDIUM_FREQ_INTERVAL=3600       # 1 hour
LOW_FREQ_INTERVAL=86400         # 24 hours

# Data Aging Thresholds
HIGH_FREQ_AGE_HOURS=24
MEDIUM_FREQ_AGE_DAYS=7
LOW_FREQ_AGE_DAYS=30
```

---

## 4. Phase 2: Data Collection Service (This Week)

### 4.1 Create New Service

**File**: `data_collection_service.py`  
**Version**: v1.0.0

```python
#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: data_collection_service.py
Version: 1.0.0
Last Updated: 2025-07-08
Purpose: Intelligent data collection for 100 tracked securities

REVISION HISTORY:
v1.0.0 (2025-07-08) - Initial implementation
- Tracks 100 securities with intelligent aging
- High-frequency collection for active securities
- Automatic frequency reduction for inactive securities
- Pattern discovery preparation

Description of Service:
Collects comprehensive data for top 100 securities to enable ML pattern discovery
"""

# Implementation from design document...
```

### 4.2 Add to Docker Compose

**File**: `docker-compose.yml`  
**Add**:

```yaml
data-collection-service:
  build:
    context: .
    dockerfile: Dockerfile.datacollection
  container_name: catalyst-datacollection
  ports:
    - "5011:5011"
  env_file:
    - .env
  depends_on:
    - redis
    - postgres
    - scanner-service
  volumes:
    - ./logs:/app/logs
    - ./data:/app/data
  restart: unless-stopped
  networks:
    - catalyst-network
```

---

## 5. Phase 3: Pattern Discovery Engine (Next Week)

### 5.1 Create Pattern Discovery Service

**File**: `pattern_discovery_service.py`  
**Version**: v1.0.0

```python
class PatternDiscoveryEngine:
    """
    Analyzes 100 tracked securities to find hidden patterns
    """
    
    def __init__(self):
        self.pattern_types = [
            'catalyst_sympathy',
            'sector_rotation',
            'accumulation_patterns',
            'news_arbitrage',
            'correlation_breaks'
        ]
        
    def discover_patterns(self):
        # Implementation from design document...
```

---

## 6. Documentation Updates

### 6.1 Update Functional Specification

**File**: `Docs/Design/Functional Specification v2.1.0.md`  
**Changes**:

1. Add Section 2.1.3:
```markdown
### 2.1.3 Enhanced Tracking System

The system maintains two separate tracking levels:
- **Trading Candidates**: Top 5 securities for immediate trading
- **Tracking Universe**: Top 100 securities for pattern discovery

This dual approach enables:
- Immediate trading opportunities (top 5)
- Long-term pattern discovery (top 100)
- Cross-security correlation analysis
- Market manipulation detection
```

2. Update Section 3.2 (Scanner Service):
```markdown
#### Enhanced Endpoints
- `GET /tracked_securities` - Get all 100 tracked securities
- `GET /tracking_metrics` - Get tracking statistics
- `POST /update_tracking` - Force tracking update
```

### 6.2 Update Architecture Document

**File**: `Docs/Design/Architecture v2.1.0.md`  
**Add** new data flow diagram showing dual tracking

### 6.3 Update Database Schema

**File**: `Docs/Design/Database Schema v2.1.0.md`  
**Add** all new tables for enhanced tracking

---

## 7. Testing & Validation Plan

### 7.1 Unit Tests
```python
# test_enhanced_scanner.py
def test_tracks_100_securities():
    scanner = EnhancedDynamicSecurityScanner()
    result = scanner.perform_enhanced_scan()
    assert len(result['tracked_securities']) == 100
    assert len(result['trading_candidates']) == 5

def test_data_aging():
    collector = IntelligentDataCollector()
    # Test aging logic...
```

### 7.2 Integration Tests
1. Verify all 100 securities get tracked
2. Confirm data aging works correctly
3. Test pattern discovery on historical data
4. Validate storage growth matches estimates

### 7.3 Performance Tests
1. Measure scan time for 100 securities
2. Check database query performance
3. Monitor memory usage
4. Verify storage estimates

---

## 8. Rollout Timeline

### Week 1 (July 8-14)
- ✓ Day 1: Fix coordination service, add database tables
- ✓ Day 2: Update scanner service to v3.1.0
- ✓ Day 3: Create data collection service
- ✓ Day 4: Integration testing
- ✓ Day 5: Deploy to production

### Week 2 (July 15-21)
- ✓ Day 1-2: Create pattern discovery engine
- ✓ Day 3-4: Implement ML pattern detection
- ✓ Day 5: Production deployment

### Week 3 (July 22-28)
- ✓ Monitor and optimize
- ✓ Collect feedback
- ✓ Fine-tune patterns

---

## 9. Success Metrics

1. **Technical Metrics**
   - ✓ 100 securities tracked continuously
   - ✓ <2.5 GB storage per year
   - ✓ Pattern discovery runs hourly
   - ✓ 15-minute data collection intervals

2. **Business Metrics**
   - ✓ 20x more data for ML training
   - ✓ New patterns discovered weekly
   - ✓ Improved trading signals
   - ✓ Better risk management

---

## 10. Risk Mitigation

1. **Storage Growth**
   - Monitor daily
   - Implement aggressive aging
   - Add storage alerts at 80% capacity

2. **Performance Impact**
   - Use Redis caching extensively
   - Optimize database queries
   - Consider read replicas if needed

3. **Service Stability**
   - Gradual rollout
   - Feature flags for new functionality
   - Rollback plan ready

---

## Conclusion

This reconciliation plan bridges the gap between current implementation (5 securities) and enhanced design (100 securities). Following this plan will enable advanced AI/ML pattern discovery while maintaining system stability and performance.

**Next Immediate Action**: Update database_utils.py to v2.33 and restart coordination service.