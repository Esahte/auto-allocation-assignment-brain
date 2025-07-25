from flask import Flask, request, jsonify
import json
import os
import time
from datetime import datetime

app = Flask(__name__)

# Performance monitoring
performance_stats = {
    "total_requests": 0,
    "fixed_optimized_requests": 0,
    "light_optimized_requests": 0,
    "average_response_time": 0.0,
    "cache_hits": 0,
    "algorithm_usage": {},
    "response_times": []
}

# Add import for new batch optimization
try:
    from OR_tool_prototype_batch_optimized import recommend_agents_batch_optimized, clear_cache
    BATCH_AVAILABLE = True
except ImportError:
    BATCH_AVAILABLE = False
    print("Warning: Batch optimization not available")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "performance_stats": performance_stats,
        "available_algorithms": ["fixed_optimized", "light_optimized", "auto"]
    }), 200

@app.route('/recommend', methods=['POST'])
def recommend():
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
        algorithm = data.get('algorithm', 'auto')  # auto, fixed_optimized, light_optimized
        enable_debug = data.get('enable_debug', False)
        use_proximity = data.get('use_proximity', True)
        area_type = data.get('area_type', 'urban')  # 'urban' or 'rural'
        max_distance_km = data.get('max_distance_km', None)  # Maximum distance for agent selection
        
        # Validate max_distance_km parameter
        if max_distance_km is not None:
            if not isinstance(max_distance_km, (int, float)):
                return jsonify({"error": "max_distance_km must be a number"}), 400
            if max_distance_km <= 0:
                return jsonify({"error": "max_distance_km must be positive"}), 400
            if max_distance_km > 200:  # Reasonable upper limit
                return jsonify({"error": "max_distance_km cannot exceed 200km"}), 400
        
        # Auto-select algorithm based on dataset size and requirements
        if algorithm == 'auto':
            num_agents = len(agents)
            num_tasks = len(current_tasks)
            
            # Enhanced auto-selection with batch optimization preference
            if BATCH_AVAILABLE and num_agents >= 5 and num_tasks >= 2:
                algorithm = 'batch_optimized'   # Use batch optimization for multiple agents/tasks
            elif num_agents <= 20 and num_tasks <= 50:
                algorithm = 'light_optimized'    # Good for smaller datasets with caching benefits
            else:
                algorithm = 'fixed_optimized'    # Best for larger datasets with adaptive optimizations
        
        # Call the appropriate recommendation service
        if algorithm == 'batch_optimized' and BATCH_AVAILABLE:
            recommendations = recommend_agents_batch_optimized(
                new_task=new_task,
                agents=agents,
                current_tasks=current_tasks,
                max_grace_period=max_grace_period,
                enable_debug=enable_debug,
                use_proximity=use_proximity,
                area_type=area_type,
                max_distance_km=max_distance_km
            )
            if "algorithm_usage" not in performance_stats:
                performance_stats["algorithm_usage"] = {}
            performance_stats["algorithm_usage"]["batch_optimized"] = performance_stats["algorithm_usage"].get("batch_optimized", 0) + 1
            
        elif algorithm == 'fixed_optimized':
            from OR_tool_prototype_optimized_fixed import recommend_agents
            recommendations = recommend_agents(
                new_task, agents, current_tasks, max_grace_period
            )
            performance_stats["fixed_optimized_requests"] += 1
            
        elif algorithm == 'light_optimized':
            from OR_tool_prototype_light_optimized import recommend_agents
            recommendations = recommend_agents(
                new_task, agents, current_tasks, max_grace_period, 
                use_proximity, area_type, enable_debug, max_distance_km
            )
            performance_stats["light_optimized_requests"] += 1
            
        else:
            return jsonify({
                "error": f"Unknown algorithm: {algorithm}",
                "available_algorithms": ["batch_optimized", "fixed_optimized", "light_optimized", "auto"]
            }), 400
        
        # Handle detailed JSON response from optimized models
        if isinstance(recommendations, str):
            try:
                # Parse the JSON response from the models
                recommendations_dict = json.loads(recommendations)
            except json.JSONDecodeError:
                # Fallback for simple driver ID responses
                recommendations_dict = {
                    "task_id": new_task.get("id", "unknown"),
                    "recommendations": []
                }
        else:
            # Handle case where it's already a dict
            recommendations_dict = recommendations if isinstance(recommendations, dict) else {
                "task_id": new_task.get("id", "unknown"),
                "recommendations": []
            }
        
        # Update performance stats
        execution_time = time.time() - start_time
        performance_stats["average_response_time"] = (
            (performance_stats["average_response_time"] * (performance_stats["total_requests"] - 1) + execution_time) 
            / performance_stats["total_requests"]
        )
        
        # Ensure task_id is present
        if "task_id" not in recommendations_dict:
            recommendations_dict["task_id"] = new_task.get("id", "unknown")
        
        return jsonify(recommendations_dict), 200
        
    except Exception as e:
        execution_time = time.time() - start_time
        app.logger.error(f"Error processing request in {execution_time:.3f}s: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/recommend/fixed-optimized', methods=['POST'])
def recommend_fixed_optimized():
    """Endpoint specifically for fixed optimized recommendations."""
    data = request.get_json()
    data['algorithm'] = 'fixed_optimized'
    return recommend()

@app.route('/recommend/light-optimized', methods=['POST'])
def recommend_light_optimized():
    """Endpoint specifically for light optimized recommendations."""
    data = request.get_json()
    data['algorithm'] = 'light_optimized'
    return recommend()

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
        
        if not new_task or not agents:
            return jsonify({"error": "Missing required fields: new_task, agents"}), 400
        
        # Validate max_distance_km parameter
        if max_distance_km is not None:
            if not isinstance(max_distance_km, (int, float)):
                return jsonify({"error": "max_distance_km must be a number"}), 400
            if max_distance_km <= 0:
                return jsonify({"error": "max_distance_km must be positive"}), 400
            if max_distance_km > 200:  # Reasonable upper limit
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
            max_distance_km=max_distance_km
        )
        
        execution_time = time.time() - start_time
        
        # Update statistics
        performance_stats["total_requests"] += 1
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
        execution_time = time.time() - start_time
        app.logger.error(f"Error processing request in {execution_time:.3f}s: {str(e)}")
        return jsonify({
            "error": str(e),
            "algorithm": "batch_optimized"
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get detailed performance statistics."""
    try:
        from OR_tool_prototype_light_optimized import _osrm_cache as light_cache
        cache_info = {
            "light_optimized_cache_size": len(light_cache),
            "total_cache_entries": len(light_cache)
        }
    except (ImportError, AttributeError):
        cache_info = {
            "light_optimized_cache_size": 0,
            "total_cache_entries": 0
        }
    
    return jsonify({
        **performance_stats,
        **cache_info,
        "algorithm_usage": {
            "fixed_optimized_percentage": round(performance_stats["fixed_optimized_requests"] / max(performance_stats["total_requests"], 1) * 100, 1),
            "light_optimized_percentage": round(performance_stats["light_optimized_requests"] / max(performance_stats["total_requests"], 1) * 100, 1),
            "batch_optimized_percentage": round(performance_stats["algorithm_usage"].get("batch_optimized", 0) / max(performance_stats["total_requests"], 1) * 100, 1)
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
        
        # Clear any other caches here in the future
        
        return jsonify({
            "message": "Caches cleared successfully",
            "cleared_caches": cleared_caches,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to clear caches: {str(e)}"
        }), 500

@app.route('/benchmark', methods=['POST'])
def benchmark():
    """Run a quick benchmark comparing both available algorithms."""
    try:
        data = request.get_json()
        required_fields = ['new_task', 'agents', 'current_tasks']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        new_task = data['new_task']
        agents = data['agents']
        current_tasks = data['current_tasks']
        
        results = {}
        
        # Test fixed optimized
        try:
            start_time = time.time()
            from OR_tool_prototype_optimized_fixed import recommend_agents
            fixed_result = recommend_agents(new_task, agents, current_tasks)
            fixed_time = time.time() - start_time
            results["fixed_optimized"] = {
                "execution_time": round(fixed_time, 3),
                "success": True,
                "recommended_driver": fixed_result
            }
        except Exception as e:
            results["fixed_optimized"] = {"execution_time": 0, "success": False, "error": str(e)}
        
        # Test light optimized
        try:
            start_time = time.time()
            from OR_tool_prototype_light_optimized import recommend_agents
            light_result = recommend_agents(new_task, agents, current_tasks)
            light_time = time.time() - start_time
            results["light_optimized"] = {
                "execution_time": round(light_time, 3),
                "success": True,
                "recommended_driver": light_result
            }
        except Exception as e:
            results["light_optimized"] = {"execution_time": 0, "success": False, "error": str(e)}
        
        # Add comparison
        if results["fixed_optimized"]["success"] and results["light_optimized"]["success"]:
            fixed_time = results["fixed_optimized"]["execution_time"]
            light_time = results["light_optimized"]["execution_time"]
            
            if fixed_time < light_time:
                speedup = light_time / max(fixed_time, 0.001)
                faster_algorithm = "fixed_optimized"
            else:
                speedup = fixed_time / max(light_time, 0.001)
                faster_algorithm = "light_optimized"
                
            results["comparison"] = {
                "faster_algorithm": faster_algorithm,
                "speedup": f"{speedup:.1f}x faster",
                "time_difference_ms": round(abs(fixed_time - light_time) * 1000),
                "both_recommend_same_driver": results["fixed_optimized"]["recommended_driver"] == results["light_optimized"]["recommended_driver"]
            }
        
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable or default to 8080
    port = int(os.environ.get('PORT', 8080))
    # Use 0.0.0.0 to listen on all available network interfaces
    app.run(host='0.0.0.0', port=port, debug=False) 