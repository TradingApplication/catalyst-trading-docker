# Database Schema Implementation Plan

## Current Situation Analysis

### What the Design Document Says (v2.0.0)
The Database Schema v2.0.0 design document specifies a comprehensive schema for news-driven catalyst trading with:
- **12 main tables** for tracking news → candidates → signals → trades → outcomes
- **Full catalyst tracking** from news source to trade outcome
- **ML-ready structure** with feedback loops and outcome tracking
- **Source reliability metrics** to track which news sources are accurate

### What's Actually Implemented
Your current init.sql/init_database.py has:
- **8 simplified tables** with basic trading functionality
- **No catalyst tracking** - can't link trades back to news
- **Different table names** - `trades` instead of `trade_records`
- **Missing core tables** - no `news_raw`, `trading_candidates`, etc.

### Why Services Are Failing
The reporting service is looking for tables that don't exist:
- Looking for `trades` table (which exists)
- But the fields don't match what it expects
- No `trading_candidates` table for scanning results
- No proper `positions` table structure

## Implementation Options

### Option 1: Quick Fix - Make Services Work with Current Schema
**What This Means:**
- Keep your existing simple schema
- Update all service queries to match current tables
- Add missing fields to existing tables

**Pros:**
- ✅ Services start working immediately (1-2 hours)
- ✅ Minimal database changes
- ✅ Can start trading today

**Cons:**
- ❌ Loses catalyst tracking capability
- ❌ Can't track which news led to trades
- ❌ No source reliability tracking
- ❌ Will need complete rewrite later
- ❌ Not aligned with design vision

**When to Choose This:**
- You need to start trading immediately
- Catalyst tracking isn't critical right now
- You're okay with a major refactor later

---

### Option 2: Hybrid Approach - Add Compatibility Layer
**What This Means:**
- Implement full v2.0.0 schema
- Create views that map old names to new tables
- Services work without changes initially
- Gradually update services to use new schema

**Pros:**
- ✅ Services work immediately with views
- ✅ Full schema is implemented correctly
- ✅ Smooth transition path
- ✅ No data loss
- ✅ Can track catalysts from day one

**Cons:**
- ❌ More complex initial setup (3-4 hours)
- ❌ Views add small performance overhead
- ❌ Need to manage transition period

**When to Choose This:**
- You want the best of both worlds
- You need services running but want proper schema
- You have time for gradual migration

---

### Option 3: Full Implementation - Do It Right
**What This Means:**
- Implement complete v2.0.0 schema
- Update all services to use correct tables
- No compatibility layer

**Pros:**
- ✅ Clean, correct implementation
- ✅ No technical debt
- ✅ Best performance
- ✅ All features available immediately

**Cons:**
- ❌ Services down until updates complete (4-6 hours)
- ❌ More work upfront
- ❌ Risk of bugs during transition

**When to Choose This:**
- You can afford downtime
- You want the cleanest solution
- You have time to test thoroughly

## Recommended Approach: Option 2 (Hybrid)

### Why This Is Best:
1. **Immediate Functionality** - Services work right away
2. **Future Proof** - Correct schema from the start
3. **Risk Mitigation** - Can rollback if issues
4. **Gradual Migration** - Update services over time

### Implementation Steps:

#### Phase 1: Today (2-3 hours)
1. Run comprehensive schema creation (init_database_v2.sql)
2. Create compatibility views
3. Restart services
4. Verify everything works

#### Phase 2: This Week
1. Update reporting service to use `trade_records`
2. Update trading service to write catalyst data
3. Update scanner to use `trading_candidates`
4. Test each service thoroughly

#### Phase 3: Next Week
1. Remove compatibility views
2. Final testing
3. Document changes

## Decision Matrix

| Criteria | Option 1 (Quick) | Option 2 (Hybrid) | Option 3 (Full) |
|----------|------------------|-------------------|-----------------|
| Time to Working System | 1-2 hours | 2-3 hours | 4-6 hours |
| Catalyst Tracking | ❌ No | ✅ Yes | ✅ Yes |
| Technical Debt | High | Low | None |
| Risk Level | Low | Medium | High |
| Future Work Required | Major refactor | Minor updates | None |
| Recommended For | Emergency fix | **Most teams** | Greenfield |

## Next Steps

**If you choose Option 2 (Recommended):**
1. I'll provide the schema creation script
2. I'll provide the compatibility views
3. I'll create a migration guide for each service
4. You run the scripts and services work immediately

**What do you need to decide:**
- How critical is catalyst tracking? (Very → Option 2 or 3)
- Can you afford any downtime? (No → Option 1 or 2)
- How soon do you need this working? (Today → Option 1 or 2)

**My Recommendation:** Go with Option 2. You get working services today AND the correct schema for the future. The small overhead of views is worth the flexibility.

What option would you like to proceed with?