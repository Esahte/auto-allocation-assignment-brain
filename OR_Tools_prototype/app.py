from flask import Flask, request, jsonify, render_template
import json
import os
import time
from datetime import datetime

app = Flask(__name__)


@app.route('/fleet-dashboard')
def fleet_dashboard():
    """Serve the fleet optimizer dashboard UI."""
    return render_template('fleet_dashboard.html')

# Performance monitoring
performance_stats = {
    "total_requests": 0,
    "batch_optimized_requests": 0,
    "fleet_optimizer_requests": 0,
    "average_response_time": 0.0,
    "cache_hits": 0,
    "algorithm_usage": {},
    "response_times": []
}

# Add import for batch optimization
try:
    from OR_tool_prototype_batch_optimized import recommend_agents_batch_optimized, clear_cache
    BATCH_AVAILABLE = True
except ImportError:
    BATCH_AVAILABLE = False
    print("Warning: Batch optimization not available")

# Add import for fleet optimization
try:
    from fleet_optimizer import optimize_fleet
    FLEET_AVAILABLE = True
except ImportError:
    FLEET_AVAILABLE = False
    print("Warning: Fleet optimization not available")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "performance_stats": performance_stats,
        "available_algorithms": ["batch_optimized", "fleet_optimizer"],
        "batch_optimizer_available": BATCH_AVAILABLE,
        "fleet_optimizer_available": FLEET_AVAILABLE
    }), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Single-task recommendation endpoint.
    Uses batch_optimized algorithm to recommend agents for a new task.
    """
    if not BATCH_AVAILABLE:
        return jsonify({
            "error": "Batch optimization not available",
            "message": "OR_tool_prototype_batch_optimized module not found"
        }), 503
    
    start_time = time.time()
    
    try:
        performance_stats["total_requests"] += 1
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['new_task', 'agents', 'current_tasks']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Get the data
        new_task = data['new_task']
        agents = data['agents']
        current_tasks = data['current_tasks']
        
        # Get optional parameters
        max_grace_period = data.get('max_grace_period', 3600)
        enable_debug = data.get('enable_debug', False)
        use_proximity = data.get('use_proximity', True)
        area_type = data.get('area_type', 'urban')
        max_distance_km = data.get('max_distance_km', None)
        optimization_mode = data.get('optimization_mode', 'tardiness_min')
        
        # Validate max_distance_km parameter
        if max_distance_km is not None:
            if not isinstance(max_distance_km, (int, float)):
                return jsonify({"error": "max_distance_km must be a number"}), 400
            if max_distance_km <= 0:
                return jsonify({"error": "max_distance_km must be positive"}), 400
            if max_distance_km > 200:
                return jsonify({"error": "max_distance_km cannot exceed 200km"}), 400
        
        # Run batch optimization
        recommendations = recommend_agents_batch_optimized(
            new_task=new_task,
            agents=agents,
            current_tasks=current_tasks,
            max_grace_period=max_grace_period,
            enable_debug=enable_debug,
            use_proximity=use_proximity,
            area_type=area_type,
            max_distance_km=max_distance_km,
            optimization_mode=optimization_mode
        )
        
        performance_stats["batch_optimized_requests"] += 1
        performance_stats["algorithm_usage"]["batch_optimized"] = performance_stats["algorithm_usage"].get("batch_optimized", 0) + 1
        
        # Update performance stats
        execution_time = time.time() - start_time
        performance_stats["average_response_time"] = (
            (performance_stats["average_response_time"] * (performance_stats["total_requests"] - 1) + execution_time) 
            / performance_stats["total_requests"]
        )
        
        # Ensure task_id is present
        if isinstance(recommendations, dict) and "task_id" not in recommendations:
            recommendations["task_id"] = new_task.get("id", "unknown")
        
        return jsonify(recommendations), 200
        
    except Exception as e:
        execution_time = time.time() - start_time
        app.logger.error(f"Error processing request in {execution_time:.3f}s: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/recommend/batch-optimized', methods=['POST'])
def recommend_batch_optimized():
    """
    Batch-optimized recommendation using native multi-vehicle VRP capabilities.
    Processes ALL agents simultaneously instead of sequential optimization.
    """
    if not BATCH_AVAILABLE:
        return jsonify({
            "error": "Batch optimization not available",
            "message": "OR_tool_prototype_batch_optimized module not found"
        }), 503
    
    try:
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        new_task = data.get('new_task')
        agents = data.get('agents', [])
        current_tasks = data.get('current_tasks', [])
        max_grace_period = data.get('max_grace_period', 3600)
        use_proximity = data.get('use_proximity', True)
        area_type = data.get('area_type', 'urban')
        max_distance_km = data.get('max_distance_km', None)
        optimization_mode = data.get('optimization_mode', 'tardiness_min')
        
        if not new_task or not agents:
            return jsonify({"error": "Missing required fields: new_task, agents"}), 400
        
        # Validate max_distance_km parameter
        if max_distance_km is not None:
            if not isinstance(max_distance_km, (int, float)):
                return jsonify({"error": "max_distance_km must be a number"}), 400
            if max_distance_km <= 0:
                return jsonify({"error": "max_distance_km must be positive"}), 400
            if max_distance_km > 200:
                return jsonify({"error": "max_distance_km cannot exceed 200km"}), 400
        
        enable_debug = data.get('debug', False)
        
        start_time = time.time()
        
        # Run batch optimization
        result = recommend_agents_batch_optimized(
            new_task=new_task,
            agents=agents,
            current_tasks=current_tasks,
            max_grace_period=max_grace_period,
            enable_debug=enable_debug,
            use_proximity=use_proximity,
            area_type=area_type,
            max_distance_km=max_distance_km,
            optimization_mode=optimization_mode
        )
        
        execution_time = time.time() - start_time
        
        # Update statistics
        performance_stats["total_requests"] += 1
        performance_stats["batch_optimized_requests"] += 1
        performance_stats["algorithm_usage"]["batch_optimized"] = performance_stats["algorithm_usage"].get("batch_optimized", 0) + 1
        performance_stats["response_times"].append(execution_time)
        if len(performance_stats["response_times"]) > 100:
            performance_stats["response_times"] = performance_stats["response_times"][-100:]
        
        # Add metadata
        result['metadata'] = {
            'algorithm': 'batch_optimized',
            'execution_time': execution_time,
            'agents_count': len(agents),
            'current_tasks_count': len(current_tasks),
            'optimization_method': 'native_multi_vehicle_vrp'
        }
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            "error": str(e),
            "algorithm": "batch_optimized"
        }), 500

@app.route('/optimize-fleet', methods=['POST'])
def optimize_fleet_endpoint():
    """
    Fleet-wide optimization endpoint (Many-to-Many).
    
    Fetches data from dashboard endpoints and returns optimized routes for all agents.
    
    Request body (optional):
    {
        "dashboard_url": "http://localhost:8000",  // Optional, defaults to localhost:8000
        "agents_data": {...},  // Optional, provide data directly instead of fetching
        "tasks_data": {...}    // Optional, provide data directly instead of fetching
    }
    """
    if not FLEET_AVAILABLE:
        return jsonify({
            "error": "Fleet optimization not available",
            "message": "fleet_optimizer module not found"
        }), 503
    
    import requests as req
    from concurrent.futures import ThreadPoolExecutor
    
    try:
        start_time = time.time()
        data = request.get_json() or {}
        
        # Get dashboard URL
        dashboard_url = data.get('dashboard_url', 'http://localhost:8000')
        
        # Check if data is provided directly or needs to be fetched
        if 'agents_data' in data and 'tasks_data' in data:
            # Use provided data
            agents_data = data['agents_data']
            tasks_data = data['tasks_data']
            print(f"[optimize-fleet] Using provided data")
        else:
            # Fetch from dashboard using PARALLEL requests
            print(f"[optimize-fleet] Fetching data from {dashboard_url} (parallel)")
            
            def fetch_agents():
                resp = req.get(f"{dashboard_url}/api/test/or-tools/agents", timeout=30)
                resp.raise_for_status()
                return resp.json()
            
            def fetch_tasks():
                resp = req.get(f"{dashboard_url}/api/test/or-tools/unassigned-tasks", timeout=30)
                resp.raise_for_status()
                return resp.json()
            
            agents_data = None
            tasks_data = None
            errors = []
            
            # Execute both fetches in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_agents = executor.submit(fetch_agents)
                future_tasks = executor.submit(fetch_tasks)
                
                try:
                    agents_data = future_agents.result(timeout=30)
                    print(f"[optimize-fleet] Got {agents_data.get('summary', {}).get('eligible_agents', 0)} agents")
                except Exception as e:
                    errors.append(f"agents: {str(e)}")
                
                try:
                    tasks_data = future_tasks.result(timeout=30)
                    print(f"[optimize-fleet] Got {tasks_data.get('summary', {}).get('total', 0)} tasks")
                except Exception as e:
                    errors.append(f"tasks: {str(e)}")
            
            if errors:
                return jsonify({
                    "error": f"Failed to fetch data from dashboard: {', '.join(errors)}",
                    "hint": "Ensure dashboard is running at the specified URL"
                }), 503
        
        # Run fleet optimization
        result = optimize_fleet(agents_data, tasks_data)
        
        execution_time = time.time() - start_time
        
        # Update stats
        performance_stats["total_requests"] += 1
        performance_stats["fleet_optimizer_requests"] += 1
        performance_stats["algorithm_usage"]["fleet_optimizer"] = performance_stats["algorithm_usage"].get("fleet_optimizer", 0) + 1
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        app.logger.error(f"Error in fleet optimization: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "algorithm": "fleet_optimizer"
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get detailed performance statistics."""
    try:
        from OR_tool_prototype_batch_optimized import _osrm_cache as batch_cache
        cache_info = {
            "batch_optimized_cache_size": len(batch_cache),
            "total_cache_entries": len(batch_cache)
        }
    except (ImportError, AttributeError):
        cache_info = {
            "batch_optimized_cache_size": 0,
            "total_cache_entries": 0
        }
    
    return jsonify({
        **performance_stats,
        **cache_info,
        "algorithm_usage": {
            "batch_optimized_percentage": round(performance_stats["batch_optimized_requests"] / max(performance_stats["total_requests"], 1) * 100, 1),
            "fleet_optimizer_percentage": round(performance_stats["fleet_optimizer_requests"] / max(performance_stats["total_requests"], 1) * 100, 1)
        }
    }), 200

@app.route('/cache/clear', methods=['POST'])
def clear_caches():
    """Clear all optimization caches."""
    try:
        cleared_caches = []
        
        # Clear batch optimization cache if available
        if BATCH_AVAILABLE:
            clear_cache()
            cleared_caches.append("batch_optimization_cache")
        
        return jsonify({
            "message": "Caches cleared successfully",
            "cleared_caches": cleared_caches,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to clear caches: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Get port from environment variable or default to 8080
    port = int(os.environ.get('PORT', 8080))
    # Use 0.0.0.0 to listen on all available network interfaces
    app.run(host='0.0.0.0', port=port, debug=False)
