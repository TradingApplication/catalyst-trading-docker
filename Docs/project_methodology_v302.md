# Catalyst Trading System Project Methodology

**Document**: Catalyst Trading System PROJECT METHODOLOGY  
**Version**: 3.0.2  
**Last Updated**: 2025-06-19  
**Author**: Catalyst Trading System Development Team  

## REVISION HISTORY
- v3.0.2 (2025-06-19) - Clarified workflow from requirements to implementation using automated update process
- v3.0.1 (2025-06-19) - Added naming conventions for implementation plans and change diaries
- v3.0.0 (2025-06-19) - Added automated update process with change management lifecycle
- v2.0.1 (2025-06-19) - Simplified Google Drive documentation location
- v2.0.0 (2025-06-19) - Enhanced with testing protocols, rollback procedures, and success metrics
- v1.0.0 (2025-06-19) - Initial methodology framework

---

## 1.2 Complete Implementation Workflow

### Step 1: Requirements Definition
- User provides new requirements or changes
- Requirements should be clear and specific
- Include success criteria and constraints

### Step 2: Implementation Plan Creation
- Claude analyzes requirements against existing documentation
- Reviews existing implementation plans to avoid repetition
- Creates implementation plan with format: `Implementation Plan - [IDENTIFIER] - [DATE]`
- Plan includes phases, risks, and testing criteria

### Step 3: Plan Review and Approval
- User reviews implementation plan
- User either approves or requests modifications
- Once approved, plan is saved to Project Documentation

### Step 4: Execute Automated Update Process
- User runs: `python TRADING_SYSTEM_AUTOMATED_UPDATE_PROCESS_vXXX.py`
- Process reads the approved implementation plan
- Creates Change Diary: `Change Diary - [ImplementationID] - [DATE]`
- Begins Phase 1 of implementation

### Step 5: Phased Implementation
- Each phase executes according to implementation plan
- Change Diary updated in real-time
- After each phase, "What's Next Task List" provided
- User follows task list to continue or troubleshoot

### Step 6: Completion
- Implementation either completes successfully or rolls back
- Final Change Diary documents entire lifecycle
- Summary report generated
- System ready for next implementation

---

## 1. Objective

Release a fully functional Paper Trading Application that conforms to both the Architectural and Functional specifications located in our Project Documentation.

---

## 1.1 Implementation Workflow Overview

The complete workflow from requirements to implementation follows these steps:

1. **Requirements Supplied** → User provides new requirements
2. **Implementation Plan Created** → Claude creates implementation plan based on requirements
3. **Plan Approved** → User reviews and approves the implementation plan
4. **Automated Update Process** → User executes Catalyst Trading System Automated Update Process_vXXX.py
5. **Change Management** → Automated process creates and maintains Change Diary
6. **Phased Execution** → System executes phases defined in implementation plan
7. **What's Next Task List** → Each phase completion provides next steps
8. **Completion or Rollback** → Implementation completes successfully or rolls back

---

## 2. Requirement Delivery

**KEY PRINCIPLE**: Implementation Plans are created FIRST from requirements, THEN executed using the automated update process.

When I give you requirements, you will create an implementation plan. You will review existing implementation plans to make sure you don't repeat yourself. You will always make changes based on Architecture, Operational and Functional Specification documents.

### 2.1 Documentation Review
You will determine if Architecture, Functional Specification, Database Schema documents requires update. This is always one of the first parts of the implementation plan. The writing of py service files is secondary.

### 2.2 Implementation Planning
Claude will always write the new python file or files after I have approved implementation plan. So always use the current python file, and don't create one from scratch. Python files will be in Project Documentation too.
The current files are in project_index.txt located in root of repository.

#### 2.2.1 Implementation Plan Naming Convention
All implementation plans must follow this naming format:
- **Format**: `Implementation Plan - [IDENTIFIER] - [DATE]`
- **Example**: `Implementation Plan - AUTH-ENHANCEMENT - 2025-06-19`
- **Identifier**: A unique, descriptive identifier for the implementation (e.g., AUTH-ENHANCEMENT, DB-OPTIMIZATION, NEWS-FIX)
- **Date**: ISO format date (YYYY-MM-DD)

The implementation plan header must include:
```markdown
# Implementation Plan - [IDENTIFIER] - [DATE]

**Document**: Implementation Plan  
**Identifier**: [IDENTIFIER]  
**Date Created**: [DATE]  
**Author**: Catalyst Trading System Development Team  
**Status**: [Draft/Approved/In Progress/Completed]
```

### 2.3 Risk Assessment
Each implementation plan must identify:
- High-risk changes (database schema, service interfaces)
- Dependencies that might break
- Potential performance impacts
- Mitigation strategies for each risk

---

## 3. Implementation Plan Delivery

Once Implementation plan created and I have endorsed, the Catalyst Trading System Automated Update Process will execute it:

### 3.1 Pre-Implementation Steps
1. Implementation plan saved with proper naming convention
2. Required files placed in Google Drive Updates folder
3. User executes: `python TRADING_SYSTEM_AUTOMATED_UPDATE_PROCESS_vXXX.py`

### 3.2 Automated Execution
The automated process will:
- Create Change Diary with matching Implementation ID
- Execute phases defined in implementation plan
- Update Change Diary after each phase
- Provide What's Next Task List for guidance

### 3.3 Documentation and Code Updates
Before running the automated process:
- Claude supplies any documentation changes (Architecture, Functional Spec, etc.)
- Claude supplies updated service files (e.g., service_name_vXXX.py)
- User places all files in Google Drive Project Documentation
- User ensures service files are in Updates folder for automated discovery

#### 3.3.1 Pre-Implementation Backup
The automated process handles backups in Phase 3:
- Backup database: `python database_migration.py --backup`
- Backup service files: `cp -r /content/trading_system /content/backup_[timestamp]`
- Verify backup completion before proceeding

### 3.4 Phased Implementation
The automated process executes phases in order:
- If phase is successful → Continue to next phase
- If phase fails → Use diagnostic toolkit and follow What's Next Task List
- Process updates Change Diary after each phase

#### 3.4.1 Phase Success Criteria
- All services respond to health checks
- No critical errors in logs
- Core functionality tests pass
- Database operations complete without locking errors
- Performance meets baseline (response times < 2s)

### 3.5 Error Handling
If errors occur during phased delivery:
- Automated process logs errors in Change Diary
- What's Next Task List provides specific diagnostic commands:
  * Service integration report (diagnostic_service_integration.py)
  * Log analysis report (diagnostic_log_analysis.py)
  * Process/port status (diagnostic_process_ports.py)
- Include last 5 minutes of logs for failed services

The automated process will determine if:
- Minor fix needed → Update service file and retry phase
- Major change needed → Stop and request new implementation plan
- Rollback needed → Automatic rollback after 3 failed attempts

### 3.6 Testing and Validation
The automated process performs:
- **Unit Testing**: Test individual service endpoints
- **Integration Testing**: Test service-to-service communication
- **End-to-End Testing**: Complete trading cycle test
- Uses diagnostic toolkit reports as validation evidence
- Updates Change Diary with all test results

### 3.7 Rollback Procedures
Automated rollback triggers when:
- Phase fails after 3 attempts
- Critical services fail health checks
- Database integrity issues detected
- Manual rollback requested via: `python TRADING_SYSTEM_AUTOMATED_UPDATE_PROCESS_vXXX.py --rollback`

Rollback process:
- Stops all services
- Restores from timestamped backup
- Documents rollback reason in Change Diary
- Provides What's Next Task List for recovery

---

## 4. Automated Update Process

The Catalyst Trading System Automated Update Process_vXXX.py executes approved implementation plans and manages the complete implementation lifecycle with comprehensive change tracking.

### 4.1 Process Initiation
The automated update process is initiated AFTER an implementation plan has been created and approved:
1. User supplies requirements
2. Claude creates implementation plan following naming convention
3. User approves implementation plan
4. User executes: `python TRADING_SYSTEM_AUTOMATED_UPDATE_PROCESS_vXXX.py`
5. Process follows the phases defined in the implementation plan

### 4.2 Change Management Lifecycle
The automated process executes a 6-phase lifecycle with comprehensive documentation:

#### Phase 1: Discovery
- Read approved implementation plan
- Scan Google Drive Updates folder for specified files
- Identify new versions of service files (e.g., `_v105.py`)
- Compare against current system versions
- Create change manifest

#### Phase 2: Documentation
- Initialize Change Management Diary with Implementation ID
- Link to approved Implementation Plan
- Record all discovered updates
- Document current system state
- Begin lifecycle tracking

#### Phase 3: Preparation
- Stop affected services gracefully
- Create timestamped system backup
- Save current configuration state
- Verify backup integrity

#### Phase 4: Implementation
- Copy updated files from Google Drive
- Remove version suffixes (e.g., `news_service_v105.py` → `news_service.py`)
- Apply database migrations if needed
- Update configuration files

#### Phase 5: Testing
- Restart updated services
- Run health checks on all services
- Execute integration tests
- Verify core functionality
- Generate diagnostic reports

#### Phase 6: Completion or Rollback
**Success Path:**
- Update change diary with success status
- Archive backup (keep for 7 days)
- Generate summary report
- Provide "What's Next Task List" for any follow-up actions

**Failure Path:**
- Stop all services
- Restore from backup
- Document failure reasons
- Generate failure report
- Provide "What's Next Task List" for troubleshooting

### 4.3 What's Next Task List
After each phase, the automated process provides a "What's Next Task List" that includes:
- Current phase status
- Next phase to execute (if continuing)
- Any manual interventions required
- Diagnostic reports to run (if issues encountered)
- Rollback instructions (if needed)

Example What's Next Task List:
```
=== WHAT'S NEXT TASK LIST ===
Phase 3: Preparation - COMPLETED ✓
Next: Phase 4: Implementation

Actions Required:
1. Review backup completion in /content/backup_20250619_143022
2. Confirm all services have been stopped
3. Continue with: python TRADING_SYSTEM_AUTOMATED_UPDATE_PROCESS_vXXX.py --continue

If issues occur:
- Run: python diagnostic_toolkit.py --report
- Check logs: /content/logs/update_process.log
```

### 4.4 Change Management Diary Format
Each implementation creates a markdown diary tracking the complete lifecycle:

```markdown
# Change Management Implementation Diary

## Implementation ID: [ImplementationID]
## Related Implementation Plan: Implementation Plan - [ImplementationID] - [DATE]

### Overview
- **Start Time**: [ISO timestamp]
- **Implementation Plan**: [Name of approved plan]
- **Updates Required**: [count]
- **Files**: [list of files from implementation plan]

### Phase Progress
- [ ] Phase 1: Discovery - [timestamp]
- [ ] Phase 2: Documentation - [timestamp]
- [ ] Phase 3: Preparation - [timestamp]
- [ ] Phase 4: Implementation - [timestamp]
- [ ] Phase 5: Testing - [timestamp]
- [ ] Phase 6: Completion - [timestamp]

### Current Status
**Active Phase**: Phase 3 - Preparation
**Status**: IN PROGRESS
**Last Updated**: [timestamp]

### What's Next Task List
[Current task list based on phase status]

### Detailed Progress
[Phase-by-phase details with timestamps and results]
```

### 4.5 Automation Script Usage
```bash
# Execute approved implementation plan
python TRADING_SYSTEM_AUTOMATED_UPDATE_PROCESS_vXXX.py --plan "Implementation Plan - NEWS-FIX-105 - 2025-06-19"

# Continue from last checkpoint
python TRADING_SYSTEM_AUTOMATED_UPDATE_PROCESS_vXXX.py --continue

# Check implementation status
python TRADING_SYSTEM_AUTOMATED_UPDATE_PROCESS_vXXX.py --status

# Rollback current implementation
python TRADING_SYSTEM_AUTOMATED_UPDATE_PROCESS_vXXX.py --rollback
```

### 4.6 Update Safety Features
- **Pre-flight Checks**: Verify system health before updates
- **Incremental Updates**: Update one service at a time
- **Health Monitoring**: Continuous health checks during update
- **Automatic Rollback**: Triggered on test failures
- **Audit Trail**: Complete logging of all actions

### 4.7 Integration with Manual Process
The automated update process seamlessly integrates with the manual implementation planning:
- Executes phases defined in approved implementation plan
- Uses same testing criteria (Section 3.3.1)
- Follows same error handling (Section 3.4)
- Implements same rollback procedures (Section 3.6)
- Generates diagnostic reports as specified
- Updates Change Diary throughout lifecycle
- Provides What's Next Task List after each phase

---

## 5. Project Documentation and Code Locations

All project documentation is centrally located GitHub, and can be found in project_index.txt.

### 5.1 Architecture Document
Architecture document that shows application lives in Project Documentation.

### 5.2 Operation Document
The Operation Document that shows how to run the Trading application lives in Project Documentation.

### 5.3 Implementation Plans
The implementation plans are located in Project Documentation.

#### 5.3.1 Implementation Plan Storage
- All implementation plans must be saved with the naming convention: `Implementation Plan - [IDENTIFIER] - [DATE]`
- Store in the Project Documentation folder
- Cross-reference with related Change Diaries using the Implementation ID

### 5.4 Python Service Files
The python files used in Trading Application are in Project Documentation. Always take the file to update from Project Documentation.

### 5.5 Change Diary
Change document that summarises the implementation lives in Project Documentation.

#### 5.5.1 Change Diary Creation and Naming
- **Created by**: Catalyst Trading System Automated Update Process during implementation
- **When created**: Automatically at start of Phase 2 (Documentation)
- **Naming Format**: `Change Diary - [ImplementationID] - [DATE]`
- **Example**: `Change Diary - AUTH-ENHANCEMENT - 2025-06-19`
- **ImplementationID**: Must match the identifier used in the corresponding Implementation Plan
- **Date**: ISO format date (YYYY-MM-DD)

The change diary header must include:
```markdown
# Change Diary - [ImplementationID] - [DATE]

**Document**: Change Diary  
**Implementation ID**: [ImplementationID]  
**Related Implementation Plan**: Implementation Plan - [ImplementationID] - [DATE]  
**Date Created**: [DATE]  
**Author**: Catalyst Trading System Development Team  

## Implementation Summary
[Brief summary of what was implemented]

## Changes Made
[Detailed list of all changes]

## Issues Encountered
[Any issues and how they were resolved]

## Test Results
[Summary of testing outcomes]

## Rollback Information
[If applicable, any rollback procedures executed]
```

**NEW**: Automated Change Management Diaries are also created in `/content/trading_system/change_diary.md`

### 5.6 Functional Specification
The Functional Specification which gives details of services, classes, methods, interfaces used to deliver architecture design choices lives in Project Documentation.

### 5.7 Database Schema
The Database Schema give structure of the database lives in Project Documentation.

### 5.8 Database Schema & Database management
The Database Schema and database management lives in Project Documentation.
---

## 6. Constraints and Standards

### 6.0 General Constraints
- When requirements require multiple artifacts, present them in a way the chat does not overwrite the artifact. I never want you to create artifacts and overwrite them each time you present the next one!
- When Troubleshooting follow Implementation plan.
- When writing new service py files, the artifact name will be of format: 'service_name_vXXX.py'. You will ensure py header has the name of the file that the Trading Application uses.
- If a requirement is not catered for in current Architecture, Functional Specification or Operational documents the implementation plan will deliver the updated documents.

### 6.1 Version Control
- Increment minor version (X.Y) for feature additions
- Increment patch version (X.Y.Z) for bug fixes
- Major version (X.0.0) only for architecture changes
- Always preserve previous version before updates
- **NEW**: Automated updates handle version suffix removal

### 6.2 Documentation Standards
- All code changes must include inline comments for complex logic
- API endpoint changes must update Functional Specification
- New error codes must be documented
- Performance improvements must include before/after metrics
- **NEW**: Change Management Diaries auto-document all updates
- **NEW**: Implementation Plans and Change Diaries must follow naming conventions

### 6.3 Implementation Deliverables Checklist
Implementation Deliveries has to determine the following:
1. Architecture Document update if required? If required supply.
2. Operation Document update if required? If required supply.
3. Functional Specification update if required? If required supply.
4. Change Diary document always supply (following naming convention)
5. Service python files supplied
6. **NEW**: Change Management Implementation Diary (for automated updates)

### 6.4 Header Requirements
All documents and Code have the following Header Contents:
- Name of File: e.g. Catalyst Trading System PHASE 1
- Name of service file: e.g. pattern_analysis.py
- Version: e.g. 1.0.0
- Last Updated: 2025-06-11
- REVISION HISTORY:
  - v1.0.0 (2025-06-11) - Initial release with standardized authentication

#### 6.4.1 Additional Header Requirements
- **Implementation Plans**: Must include Implementation ID in header
- **Change Diaries**: Must include Implementation ID and reference to related Implementation Plan

---

## 7. Communication Protocol

### 7.1 Before Implementation
- Claude provides implementation plan with proper naming convention
- User reviews and approves/requests changes

### 7.2 During Implementation
- User reports phase completion status
- Claude analyzes results and provides next steps
- **NEW**: Automated updates provide real-time status in Change Diary

### 7.3 After Implementation
- User confirms full implementation success
- Claude provides summary and next recommendations
- **NEW**: Automated updates generate completion reports

---

## 8. Project Success Metrics

- System uptime > 99%
- All services maintain "healthy" status
- Trading cycles complete < 30 seconds
- Zero database locking errors
- Successful paper trades execution
- **NEW**: Automated updates complete < 5 minutes
- **NEW**: Zero-downtime updates for non-critical services

---

## 9. Glossary of Terms

- **WAL**: Write-Ahead Logging - SQLite journal mode for better concurrency
- **REST API**: Representational State Transfer Application Programming Interface
- **Service Mesh**: Network of microservices and their interactions
- **Health Check**: Endpoint that returns service operational status
- **Database Locking**: Condition where database access is blocked
- **Exponential Backoff**: Retry strategy with increasing delays
- **Paper Trading**: Simulated trading without real money
- **Trading Cycle**: Complete workflow from scanning to trade execution
- **Change Management Diary**: Document tracking complete implementation lifecycle
- **Version Suffix**: Version number appended to filename (e.g., `_v105`)
- **Zero-downtime Update**: Update process that maintains service availability
- **Implementation ID**: Unique identifier for tracking implementations and their changes
- **Catalyst Trading System Automated Update Process**: Python script that executes approved implementation plans
- **What's Next Task List**: Guidance provided after each implementation phase

---

## 10. Summary of Key Workflow Principles

1. **Requirements First**: All implementations start with clear requirements from the user
2. **Plan Before Execute**: Implementation plans are created and approved before any execution
3. **Automated Execution**: Catalyst Trading System Automated Update Process executes approved plans
4. **Change Tracking**: Change Diary created automatically to track entire lifecycle
5. **Guided Process**: What's Next Task List provides clear guidance at each step
6. **Safe Updates**: Automatic backups and rollback capabilities protect the system
7. **Continuous Feedback**: Real-time updates in Change Diary throughout implementation

Remember: The implementation plan defines WHAT to do, the automated process handles HOW to do it.

---

**Document Status**: Active Methodology with Automated Updates  
**Next Review**: Monthly or after major implementation