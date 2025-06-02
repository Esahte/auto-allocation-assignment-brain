from flask import Flask, request, jsonify
import json
import os
import time

app = Flask(__name__)

# Performance monitoring
performance_stats = {
    "total_requests": 0,
    "ultra_fast_requests": 0,
    "optimized_requests": 0,
    "original_requests": 0,
    "average_response_time": 0.0,
    "cache_hits": 0
}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "performance_stats": performance_stats,
        "available_algorithms": ["ultra_fast", "optimized", "original"]
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
        algorithm = data.get('algorithm', 'auto')  # auto, ultra_fast, optimized, original
        enable_debug = data.get('enable_debug', False)
        
        # Auto-select algorithm based on dataset size and requirements
        if algorithm == 'auto':
            num_agents = len(agents)
            num_tasks = len(current_tasks)
            
            if num_agents <= 10 and num_tasks <= 20:
                algorithm = 'original'    # Best accuracy for small datasets
            elif num_agents <= 50 and num_tasks <= 100:
                algorithm = 'optimized'   # Good balance for medium datasets
            else:
                algorithm = 'ultra_fast'  # Only practical choice for large datasets
        
        # Call the appropriate recommendation service
        if algorithm == 'ultra_fast':
            from OR_tool_prototype_ultra_fast import recommend_agents_ultra_fast
            recommendations = recommend_agents_ultra_fast(
                new_task, agents, current_tasks, max_grace_period
            )
            performance_stats["ultra_fast_requests"] += 1
            
        elif algorithm == 'optimized':
            from OR_tool_prototype_optimized import recommend_agents_optimized
            recommendations = recommend_agents_optimized(
                new_task, agents, current_tasks, max_grace_period, enable_debug
            )
            performance_stats["optimized_requests"] += 1
            
        elif algorithm == 'original':
            from OR_tool_prototype import recommend_agents
            recommendations = recommend_agents(
                new_task, agents, current_tasks, max_grace_period
            )
            performance_stats["original_requests"] += 1
            
        else:
            return jsonify({
                "error": f"Unknown algorithm: {algorithm}",
                "available_algorithms": ["ultra_fast", "optimized", "original", "auto"]
            }), 400
        
        # Parse the JSON string back to a dict
        recommendations_dict = json.loads(recommendations)
        
        # Update performance stats
        execution_time = time.time() - start_time
        performance_stats["average_response_time"] = (
            (performance_stats["average_response_time"] * (performance_stats["total_requests"] - 1) + execution_time) 
            / performance_stats["total_requests"]
        )
        performance_stats["cache_hits"] = recommendations_dict.get("cache_hits", 0)
        
        # Add performance metadata to response
        recommendations_dict["api_response_time_seconds"] = round(execution_time, 3)
        recommendations_dict["algorithm_used"] = algorithm
        
        return jsonify(recommendations_dict), 200
        
    except Exception as e:
        execution_time = time.time() - start_time
        app.logger.error(f"Error processing request in {execution_time:.3f}s: {str(e)}")
        return jsonify({
            "error": str(e),
            "execution_time_seconds": round(execution_time, 3),
            "algorithm_used": algorithm if 'algorithm' in locals() else "unknown"
        }), 500

@app.route('/recommend/ultra-fast', methods=['POST'])
def recommend_ultra_fast():
    """Endpoint specifically for ultra-fast recommendations."""
    data = request.get_json()
    data['algorithm'] = 'ultra_fast'
    return recommend()

@app.route('/recommend/optimized', methods=['POST'])
def recommend_optimized():
    """Endpoint specifically for optimized recommendations."""
    data = request.get_json()
    data['algorithm'] = 'optimized'
    return recommend()

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get detailed performance statistics."""
    from OR_tool_prototype_ultra_fast import _osrm_cache as ultra_cache
    
    try:
        from OR_tool_prototype_optimized import _osrm_cache as opt_cache
        cache_info = {
            "ultra_fast_cache_size": len(ultra_cache),
            "optimized_cache_size": len(opt_cache),
            "total_cache_entries": len(ultra_cache) + len(opt_cache)
        }
    except ImportError:
        cache_info = {
            "ultra_fast_cache_size": len(ultra_cache),
            "total_cache_entries": len(ultra_cache)
        }
    
    return jsonify({
        **performance_stats,
        **cache_info,
        "algorithm_usage": {
            "ultra_fast_percentage": round(performance_stats["ultra_fast_requests"] / max(performance_stats["total_requests"], 1) * 100, 1),
            "optimized_percentage": round(performance_stats["optimized_requests"] / max(performance_stats["total_requests"], 1) * 100, 1),
            "original_percentage": round(performance_stats["original_requests"] / max(performance_stats["total_requests"], 1) * 100, 1)
        }
    }), 200

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear all caches."""
    try:
        from OR_tool_prototype_ultra_fast import _osrm_cache as ultra_cache
        ultra_cache.clear()
        
        try:
            from OR_tool_prototype_optimized import _osrm_cache as opt_cache
            opt_cache.clear()
            return jsonify({
                "message": "All caches cleared successfully",
                "caches_cleared": ["ultra_fast", "optimized"]
            }), 200
        except ImportError:
            return jsonify({
                "message": "Ultra-fast cache cleared",
                "caches_cleared": ["ultra_fast"]
            }), 200
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/benchmark', methods=['POST'])
def benchmark():
    """Run a quick benchmark comparing all available algorithms."""
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
        
        # Test ultra-fast
        try:
            start_time = time.time()
            from OR_tool_prototype_ultra_fast import recommend_agents_ultra_fast
            ultra_result = recommend_agents_ultra_fast(new_task, agents, current_tasks)
            ultra_time = time.time() - start_time
            results["ultra_fast"] = {
                "execution_time": round(ultra_time, 3),
                "success": True,
                "top_score": json.loads(ultra_result)["recommendations"][0]["score"] if ultra_result else 0
            }
        except Exception as e:
            results["ultra_fast"] = {"execution_time": 0, "success": False, "error": str(e)}
        
        # Test optimized (with short timeout)
        try:
            start_time = time.time()
            from OR_tool_prototype_optimized import recommend_agents_optimized
            opt_result = recommend_agents_optimized(new_task, agents, current_tasks)
            opt_time = time.time() - start_time
            results["optimized"] = {
                "execution_time": round(opt_time, 3),
                "success": True,
                "top_score": json.loads(opt_result)["recommendations"][0]["score"] if opt_result else 0
            }
        except Exception as e:
            results["optimized"] = {"execution_time": 0, "success": False, "error": str(e)}
        
        # Add comparison
        if results["ultra_fast"]["success"] and results["optimized"]["success"]:
            speedup = results["optimized"]["execution_time"] / max(results["ultra_fast"]["execution_time"], 0.001)
            results["comparison"] = {
                "ultra_fast_speedup": f"{speedup:.1f}x faster",
                "time_saved_ms": round((results["optimized"]["execution_time"] - results["ultra_fast"]["execution_time"]) * 1000)
            }
        
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable or default to 8080
    port = int(os.environ.get('PORT', 8080))
    # Use 0.0.0.0 to listen on all available network interfaces
    app.run(host='0.0.0.0', port=port, debug=False) 