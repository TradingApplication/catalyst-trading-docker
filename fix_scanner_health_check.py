#!/usr/bin/env python3
"""
Fix for scanner_service.py health check endpoint
This fixes the health status check logic
"""

# Find this section in scanner_service.py around line 170:
# @self.app.route('/health', methods=['GET'])
# def health():
#     db_health = health_check()
#     return jsonify({
#         "status": "healthy" if db_health['database'] == 'healthy' else "degraded",
#         ...
#     })

# REPLACE WITH:

@self.app.route('/health', methods=['GET'])
def health():
    db_health = health_check()
    
    # Check both database and redis status properly
    db_status = db_health.get('database', {}).get('status', 'unknown')
    redis_status = db_health.get('redis', {}).get('status', 'unknown')
    
    # Overall status is healthy only if both are healthy
    overall_status = "healthy" if db_status == 'healthy' and redis_status == 'healthy' else "degraded"
    
    return jsonify({
        "status": overall_status,
        "service": "enhanced_security_scanner",
        "version": "3.0.2",
        "mode": "dynamic-catalyst",
        "database": db_status,
        "redis": redis_status,
        "tracking_size": len(self.performance_tracker),
        "cache_size": len(self.scan_cache),
        "timestamp": datetime.now().isoformat(),
        "details": db_health  # Include full health check details
    })
