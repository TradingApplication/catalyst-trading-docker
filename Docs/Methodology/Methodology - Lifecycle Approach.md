# Software Development Lifecycle Approach
## Catalyst Trading System

**Version**: 1.0.0  
**Created**: 2025-01-03  
**Purpose**: Define our working approach to prevent jumping to solutions without proper planning

---

## Core Principle: STOP, THINK, DISCUSS, THEN ACT

### ❌ What NOT to do:
- Jump straight to creating code/scripts
- Make architectural decisions without checking documentation
- Add components that aren't in the architecture
- Create "solutions" before understanding the problem
- Assume what's needed without discussion

### ✅ What TO DO:
- Always check existing architecture documentation FIRST
- Discuss the problem and approach BEFORE coding
- Validate solutions against official documentation, Architecture, Database Schema, Database Services & Data Mgt, Function Specifications and application strategy

- Plan changes through discussion
- Respect the established design

---

## Our Development Lifecycle

### 1. DEVELOP Phase (Current Phase)
**Status**: Flexible, iterative, experimental

**Process**:
1. **Identify Need**
   - What problem are we solving?
   - Is this documented in our architecture?
   
2. **Check Documentation**
   - Review Architecture v2.0.0
   - Review Database Schema v2.0.0
   - Review Functional Specifications
   - Ensure alignment with existing design

3. **Discuss Approach**
   - "Here's what I found in the docs..."
   - "Should we approach it this way?"
   - "What are the implications?"
   - Get agreement BEFORE implementing

4. **Implement**
   - Code the agreed solution
   - Test locally
   - Iterate based on feedback

**Flexibility**: High - we can experiment and change freely

---

### 2. BUILD Phase
**Status**: Creating release candidate

**Process**:
1. **Pre-Build Review**
   - All features complete?
   - All tests passing?
   - Documentation updated?
   - Architecture compliance verified?

2. **Version Assignment**
   - Assign build version number
   - Document what's included
   - Create changelog

3. **Build Artifacts**
   - Docker images built
   - Database schema finalized
   - Configuration locked
   - Build verification completed

**Flexibility**: None - this is a snapshot

---

### 3. RELEASE Phase
**Status**: Production deployment

**Process**:
1. **Deploy Build**
   - Use exact build artifacts
   - No modifications during deployment
   - Follow deployment checklist

2. **Verify Deployment**
   - Health checks passing
   - All services running
   - Database migrated
   - No errors in logs

3. **Lock Version**
   - This is now the "production" version
   - Document deployment date/time
   - Tag in version control

**Flexibility**: Zero - production is immutable

---

### 4. CHANGE MANAGEMENT (Post-Release)
**Status**: Controlled changes only

**Process**:
1. **Change Request**
   - What needs to change and why?
   - What's the business justification?
   - What's the risk?

2. **Impact Assessment**
   - Which services affected?
   - Database changes needed?
   - Downtime required?
   - Rollback plan?

3. **Change Planning**
   - Detailed implementation plan
   - Testing approach
   - Deployment strategy
   - Communication plan

4. **Execute Change**
   - Return to DEVELOP phase
   - Follow full lifecycle
   - Extra testing due to production impact

**Flexibility**: Very low - all changes must be justified and planned

---

## Working Together Guidelines

### For Claude:
1. **ALWAYS** check architecture/documentation before suggesting solutions
2. **NEVER** create implementation without discussion
3. **ASK** "Should we..." instead of "Here's what I built..."
4. **VALIDATE** against official architecture
5. **WAIT** for agreement before creating artifacts

### For Human:
1. **REMIND** Claude when jumping ahead
2. **REDIRECT** to planning when needed
3. **APPROVE** approach before implementation
4. **VALIDATE** solutions match architecture

---

## Common Mistakes to Avoid

### 1. **Architecture Violations**
- ❌ Adding local PostgreSQL when architecture shows external database
- ❌ Creating tables not in Database Schema v2.0.0
- ❌ Adding services not in Architecture v2.0.0

### 2. **Process Violations**
- ❌ Creating build scripts during development
- ❌ Making production decisions while developing
- ❌ Implementing before discussing

### 3. **Scope Creep**
- ❌ Adding "nice to have" features not in spec
- ❌ Over-engineering solutions
- ❌ Solving problems we don't have yet

---

## Current Status Checkpoint

**Current Phase**: DEVELOP  
**Flexibility Level**: HIGH  
**Next Milestone**: Get all services running per architecture  

**Allowed Actions**:
- Experiment with code
- Test different approaches  
- Fix bugs and issues
- Update development environment

**NOT Allowed**:
- Finalize build processes
- Create production scripts
- Make architectural changes without discussion
- Add components not in official design

---

## Remember

> "The architecture is our contract. The schema is our blueprint. 
> Discussion is our method. Only then comes implementation."

This approach ensures we build what we designed, not what we imagine in the moment.