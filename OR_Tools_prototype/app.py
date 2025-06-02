from flask import Flask, request, jsonify
import json
import os
import time

app = Flask(__name__)

# Performance monitoring
performance_stats = {
    "total_requests": 0,
    "fixed_optimized_requests": 0,
    "light_optimized_requests": 0,
    "average_response_time": 0.0,
    "cache_hits": 0
}

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
        
        # Auto-select algorithm based on dataset size and requirements
        if algorithm == 'auto':
            num_agents = len(agents)
            num_tasks = len(current_tasks)
            
            # Use light_optimized for smaller datasets (original solver settings with caching)
            # Use fixed_optimized for larger datasets (adaptive timeouts and algorithm selection)
            if num_agents <= 20 and num_tasks <= 50:
                algorithm = 'light_optimized'    # Good for smaller datasets with caching benefits
            else:
                algorithm = 'fixed_optimized'    # Best for larger datasets with adaptive optimizations
        
        # Call the appropriate recommendation service
        if algorithm == 'fixed_optimized':
            from OR_tool_prototype_optimized_fixed import recommend_agents
            recommendations = recommend_agents(
                new_task, agents, current_tasks, max_grace_period
            )
            performance_stats["fixed_optimized_requests"] += 1
            
        elif algorithm == 'light_optimized':
            from OR_tool_prototype_light_optimized import recommend_agents
            recommendations = recommend_agents(
                new_task, agents, current_tasks, max_grace_period
            )
            performance_stats["light_optimized_requests"] += 1
            
        else:
            return jsonify({
                "error": f"Unknown algorithm: {algorithm}",
                "available_algorithms": ["fixed_optimized", "light_optimized", "auto"]
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
            "light_optimized_percentage": round(performance_stats["light_optimized_requests"] / max(performance_stats["total_requests"], 1) * 100, 1)
        }
    }), 200

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear all caches."""
    try:
        from OR_tool_prototype_light_optimized import _osrm_cache as light_cache
        light_cache.clear()
        return jsonify({
            "message": "Light optimized cache cleared successfully",
            "caches_cleared": ["light_optimized"]
        }), 200
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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