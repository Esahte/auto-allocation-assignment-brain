from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import json
import os
import time
from datetime import datetime
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'fleet-optimizer-secret-key')

# Initialize Socket.IO with CORS support
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Track connected clients
connected_clients = {}

@app.route('/fleet-dashboard')
def fleet_dashboard():
    """Serve the fleet optimizer dashboard UI."""
    return render_template('fleet_dashboard.html')

@app.route('/ws-monitor')
def websocket_monitor():
    """Serve the WebSocket monitor/debug UI."""
    return render_template('websocket_monitor.html')

# Performance monitoring
performance_stats = {
    "total_requests": 0,
    "batch_optimized_requests": 0,
    "fleet_optimizer_requests": 0,
    "websocket_events": 0,
    "auto_optimizations": 0,
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


# =============================================================================
# CORE FUNCTIONS (shared by HTTP and WebSocket)
# =============================================================================

def process_batch_recommendation(data: dict) -> dict:
    """
    Core function for batch-optimized recommendations.
    Used by both HTTP API and WebSocket handlers.
    """
    if not BATCH_AVAILABLE:
        return {"error": "Batch optimization not available", "recommendations": []}
    
    new_task = data.get('new_task')
    agents = data.get('agents', [])
    current_tasks = data.get('current_tasks', [])
    
    if not new_task or not agents:
        return {"error": "Missing required fields: new_task, agents", "recommendations": []}
    
    result = recommend_agents_batch_optimized(
        new_task=new_task,
        agents=agents,
        current_tasks=current_tasks,
        max_grace_period=data.get('max_grace_period', 3600),
        enable_debug=data.get('enable_debug', False),
        use_proximity=data.get('use_proximity', True),
        area_type=data.get('area_type', 'urban'),
        max_distance_km=data.get('max_distance_km'),
        optimization_mode=data.get('optimization_mode', 'tardiness_min')
    )
    
    performance_stats["batch_optimized_requests"] += 1
    performance_stats["total_requests"] += 1
    performance_stats["algorithm_usage"]["batch_optimized"] = \
        performance_stats["algorithm_usage"].get("batch_optimized", 0) + 1
    
    return result


def process_fleet_optimization(data: dict) -> dict:
    """
    Core function for fleet-wide optimization.
    Used by both HTTP API and WebSocket handlers.
    """
    import requests as req
    from concurrent.futures import ThreadPoolExecutor
    
    if not FLEET_AVAILABLE:
        return {"error": "Fleet optimization not available", "success": False}
    
    # Check if data is provided directly
    if 'agents_data' in data and 'tasks_data' in data:
        agents_data = data['agents_data']
        tasks_data = data['tasks_data']
    else:
        # Fetch from dashboard
        # Priority: request data > environment variable > localhost default
        default_dashboard_url = os.environ.get('DASHBOARD_URL', 'http://localhost:8000')
        dashboard_url = data.get('dashboard_url', default_dashboard_url)
        
        def fetch_agents():
            resp = req.get(f"{dashboard_url}/api/test/or-tools/agents", timeout=30)
            resp.raise_for_status()
            return resp.json()
        
        def fetch_tasks():
            resp = req.get(f"{dashboard_url}/api/test/or-tools/unassigned-tasks", timeout=30)
            resp.raise_for_status()
            return resp.json()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_agents = executor.submit(fetch_agents)
            future_tasks = executor.submit(fetch_tasks)
            agents_data = future_agents.result(timeout=30)
            tasks_data = future_tasks.result(timeout=30)
    
    result = optimize_fleet(agents_data, tasks_data)
    
    performance_stats["fleet_optimizer_requests"] += 1
    performance_stats["total_requests"] += 1
    performance_stats["algorithm_usage"]["fleet_optimizer"] = \
        performance_stats["algorithm_usage"].get("fleet_optimizer", 0) + 1
    
    return result


def trigger_fleet_optimization(trigger_event: str, trigger_data: dict):
    """
    Trigger fleet optimization and emit updated routes.
    Called automatically when relevant events occur.
    """
    print(f"[WebSocket] Auto-triggering fleet optimization due to: {trigger_event}")
    performance_stats["auto_optimizations"] += 1
    
    try:
        start_time = time.time()
        
        # Use dashboard_url from trigger_data, env var, or default
        default_dashboard_url = os.environ.get('DASHBOARD_URL', 'http://localhost:8000')
        result = process_fleet_optimization({
            'dashboard_url': trigger_data.get('dashboard_url', default_dashboard_url)
        })
        
        execution_time = time.time() - start_time
        
        result['trigger_event'] = trigger_event
        result['trigger_data'] = trigger_data
        result['total_execution_time_seconds'] = round(execution_time, 3)
        
        # Emit updated routes to all connected clients
        emit('fleet:routes_updated', result, broadcast=True)
        
        assigned = result.get('metadata', {}).get('tasks_assigned', 0)
        print(f"[WebSocket] Auto-optimization complete: {assigned} tasks assigned in {execution_time:.3f}s")
        
    except Exception as e:
        print(f"[WebSocket] Auto-optimization failed: {e}")
        emit('fleet:routes_updated', {
            'trigger_event': trigger_event,
            'error': str(e),
            'success': False
        }, broadcast=True)


# =============================================================================
# SOCKET.IO EVENT HANDLERS
# =============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle new client connection."""
    client_id = request.sid
    connected_clients[client_id] = {
        'connected_at': datetime.now().isoformat(),
        'events_received': 0
    }
    print(f"[WebSocket] Client connected: {client_id}")
    emit('connection_established', {
        'client_id': client_id,
        'server_time': datetime.now().isoformat(),
        'available_events': [
            'task:get_recommendations',
            'task:created',
            'task:declined',
            'task:completed',
            'fleet:optimize_request',
            'agent:online',
            'agent:offline',
            'agent:location_update'
        ]
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    client_id = request.sid
    if client_id in connected_clients:
        del connected_clients[client_id]
    print(f"[WebSocket] Client disconnected: {client_id}")

@socketio.on('task:get_recommendations')
def handle_get_recommendations(data):
    """
    Handle single-task recommendation request.
    Returns agent recommendations for ONE specific task.
    """
    client_id = request.sid
    performance_stats["websocket_events"] += 1
    
    if client_id in connected_clients:
        connected_clients[client_id]['events_received'] += 1
    
    request_id = data.get('request_id', str(uuid.uuid4()))
    print(f"[WebSocket] task:get_recommendations from {client_id}, request_id={request_id}")
    
    try:
        start_time = time.time()
        result = process_batch_recommendation(data)
        execution_time = time.time() - start_time
        
        emit('task:recommendations', {
            'request_id': request_id,
            'task_id': result.get('task_id'),
            'recommendations': result.get('recommendations', []),
            'task_context': result.get('task_context', {}),
            'performance': {
                **result.get('performance', {}),
                'total_execution_time': round(execution_time, 3)
            },
            'error': result.get('error')
        })
        
        print(f"[WebSocket] Sent {len(result.get('recommendations', []))} recommendations in {execution_time:.3f}s")
        
    except Exception as e:
        print(f"[WebSocket] Error in task:get_recommendations: {e}")
        emit('task:recommendations', {
            'request_id': request_id,
            'error': str(e),
            'recommendations': []
        })

@socketio.on('fleet:optimize_request')
def handle_fleet_optimize(data):
    """
    Handle explicit fleet-wide optimization request.
    Returns optimized routes for ALL agents.
    """
    client_id = request.sid
    performance_stats["websocket_events"] += 1
    
    if client_id in connected_clients:
        connected_clients[client_id]['events_received'] += 1
    
    request_id = data.get('request_id', str(uuid.uuid4()))
    print(f"[WebSocket] fleet:optimize_request from {client_id}, request_id={request_id}")
    
    try:
        start_time = time.time()
        result = process_fleet_optimization(data)
        execution_time = time.time() - start_time
        
        result['request_id'] = request_id
        result['total_execution_time_seconds'] = round(execution_time, 3)
        
        emit('fleet:routes_updated', result)
        
        print(f"[WebSocket] Sent fleet routes ({result.get('metadata', {}).get('tasks_assigned', 0)} assigned) in {execution_time:.3f}s")
        
    except Exception as e:
        print(f"[WebSocket] Error in fleet:optimize_request: {e}")
        emit('fleet:routes_updated', {
            'request_id': request_id,
            'error': str(e),
            'success': False,
            'agent_routes': [],
            'unassigned_tasks': []
        })

# -----------------------------------------------------------------------------
# EVENTS THAT TRIGGER AUTOMATIC FLEET RE-OPTIMIZATION
# -----------------------------------------------------------------------------

@socketio.on('task:created')
def handle_task_created(data):
    """
    New task created → Re-optimize fleet to assign it.
    """
    performance_stats["websocket_events"] += 1
    task_id = data.get('task_id')
    print(f"[WebSocket] task:created: {task_id}")
    
    emit('task:created_ack', {
        'task_id': task_id,
        'received_at': datetime.now().isoformat(),
        'will_optimize': True
    })
    
    # Trigger fleet optimization
    trigger_fleet_optimization('task:created', {
        'task_id': task_id,
        'dashboard_url': data.get('dashboard_url', 'http://localhost:8000')
    })

@socketio.on('task:declined')
def handle_task_declined(data):
    """
    Agent declined task → Re-optimize to assign to someone else.
    """
    performance_stats["websocket_events"] += 1
    task_id = data.get('task_id')
    agent_id = data.get('agent_id')
    agent_name = data.get('agent_name', 'Unknown')
    print(f"[WebSocket] task:declined: {task_id} by {agent_name} ({agent_id})")
    
    emit('task:declined_ack', {
        'task_id': task_id,
        'agent_id': agent_id,
        'received_at': datetime.now().isoformat(),
        'will_optimize': True
    })
    
    # Trigger fleet optimization
    trigger_fleet_optimization('task:declined', {
        'task_id': task_id,
        'agent_id': agent_id,
        'agent_name': agent_name,
        'dashboard_url': data.get('dashboard_url', 'http://localhost:8000')
    })

@socketio.on('task:completed')
def handle_task_completed(data):
    """
    Task completed → Agent has capacity, re-optimize to give them more work.
    """
    performance_stats["websocket_events"] += 1
    task_id = data.get('task_id')
    agent_id = data.get('agent_id')
    agent_name = data.get('agent_name', 'Unknown')
    print(f"[WebSocket] task:completed: {task_id} by {agent_name} ({agent_id})")
    
    emit('task:completed_ack', {
        'task_id': task_id,
        'agent_id': agent_id,
        'received_at': datetime.now().isoformat(),
        'will_optimize': True
    })
    
    # Trigger fleet optimization
    trigger_fleet_optimization('task:completed', {
        'task_id': task_id,
        'agent_id': agent_id,
        'agent_name': agent_name,
        'dashboard_url': data.get('dashboard_url', 'http://localhost:8000')
    })

@socketio.on('task:cancelled')
def handle_task_cancelled(data):
    """
    Task cancelled → Remove from routes, re-optimize to redistribute work.
    """
    performance_stats["websocket_events"] += 1
    task_id = data.get('task_id')
    reason = data.get('reason', 'unknown')
    print(f"[WebSocket] task:cancelled: {task_id} (reason: {reason})")
    
    emit('task:cancelled_ack', {
        'task_id': task_id,
        'reason': reason,
        'received_at': datetime.now().isoformat(),
        'will_optimize': True
    })
    
    # Trigger fleet optimization - cancelled task is removed, redistribute remaining
    trigger_fleet_optimization('task:cancelled', {
        'task_id': task_id,
        'reason': reason,
        'dashboard_url': data.get('dashboard_url', 'http://localhost:8000')
    })

@socketio.on('task:updated')
def handle_task_updated(data):
    """
    Task updated → Details changed (times, locations, tags, etc.), re-optimize.
    Changes might affect agent compatibility or optimal routes.
    """
    performance_stats["websocket_events"] += 1
    task_id = data.get('task_id')
    updated_fields = data.get('updated_fields', [])
    print(f"[WebSocket] task:updated: {task_id} (fields: {', '.join(updated_fields) if updated_fields else 'unknown'})")
    
    emit('task:updated_ack', {
        'task_id': task_id,
        'updated_fields': updated_fields,
        'received_at': datetime.now().isoformat(),
        'will_optimize': True
    })
    
    # Trigger fleet optimization - task details changed, recalculate routes
    trigger_fleet_optimization('task:updated', {
        'task_id': task_id,
        'updated_fields': updated_fields,
        'dashboard_url': data.get('dashboard_url', 'http://localhost:8000')
    })

@socketio.on('agent:online')
def handle_agent_online(data):
    """
    Agent came online → Re-optimize to assign them tasks.
    """
    performance_stats["websocket_events"] += 1
    agent_id = data.get('agent_id')
    name = data.get('name', 'Unknown')
    print(f"[WebSocket] agent:online: {name} ({agent_id})")
    
    emit('agent:online_ack', {
        'agent_id': agent_id,
        'name': name,
        'received_at': datetime.now().isoformat(),
        'will_optimize': True
    })
    
    # Trigger fleet optimization
    trigger_fleet_optimization('agent:online', {
        'agent_id': agent_id,
        'agent_name': name,
        'dashboard_url': data.get('dashboard_url', 'http://localhost:8000')
    })

@socketio.on('agent:offline')
def handle_agent_offline(data):
    """
    Agent went offline → Re-optimize to reassign their tasks.
    """
    performance_stats["websocket_events"] += 1
    agent_id = data.get('agent_id')
    name = data.get('name', 'Unknown')
    print(f"[WebSocket] agent:offline: {name} ({agent_id})")
    
    emit('agent:offline_ack', {
        'agent_id': agent_id,
        'name': name,
        'received_at': datetime.now().isoformat(),
        'will_optimize': True
    })
    
    # Trigger fleet optimization
    trigger_fleet_optimization('agent:offline', {
        'agent_id': agent_id,
        'agent_name': name,
        'dashboard_url': data.get('dashboard_url', 'http://localhost:8000')
    })

# -----------------------------------------------------------------------------
# EVENTS THAT JUST TRACK (no auto-optimization - too frequent)
# -----------------------------------------------------------------------------

@socketio.on('agent:location_update')
def handle_agent_location_update(data):
    """
    Agent location updated → Just acknowledge (too frequent to trigger optimization).
    Location will be used in the next optimization request.
    """
    performance_stats["websocket_events"] += 1
    agent_id = data.get('agent_id')
    name = data.get('name', 'Unknown')
    location = data.get('location', [])
    print(f"[WebSocket] agent:location_update: {name} ({agent_id}) -> {location}")
    
    emit('agent:location_update_ack', {
        'agent_id': agent_id,
        'location': location,
        'received_at': datetime.now().isoformat()
    })


# =============================================================================
# HTTP REST ENDPOINTS
# =============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "websocket_enabled": True,
        "connected_clients": len(connected_clients),
        "performance_stats": performance_stats,
        "available_algorithms": ["batch_optimized", "fleet_optimizer"],
        "batch_optimizer_available": BATCH_AVAILABLE,
        "fleet_optimizer_available": FLEET_AVAILABLE
    }), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    """Single-task recommendation endpoint."""
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if 'new_task' not in data or 'agents' not in data:
            return jsonify({"error": "Missing required fields: new_task, agents"}), 400
        
        result = process_batch_recommendation(data)
        
        if 'error' in result and not result.get('recommendations'):
            return jsonify(result), 400
        
        execution_time = time.time() - start_time
        performance_stats["average_response_time"] = (
            (performance_stats["average_response_time"] * (performance_stats["total_requests"] - 1) + execution_time) 
            / max(performance_stats["total_requests"], 1)
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/recommend/batch-optimized', methods=['POST'])
def recommend_batch_optimized():
    """Batch-optimized recommendation endpoint."""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        result = process_batch_recommendation(data)
        
        if 'error' in result and not result.get('recommendations'):
            return jsonify(result), 400
        
        execution_time = time.time() - start_time
        
        performance_stats["response_times"].append(execution_time)
        if len(performance_stats["response_times"]) > 100:
            performance_stats["response_times"] = performance_stats["response_times"][-100:]
        
        result['metadata'] = {
            'algorithm': 'batch_optimized',
            'execution_time': execution_time,
            'agents_count': len(data.get('agents', [])),
            'current_tasks_count': len(data.get('current_tasks', [])),
            'optimization_method': 'native_multi_vehicle_vrp'
        }
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e), "algorithm": "batch_optimized"}), 500

@app.route('/optimize-fleet', methods=['POST'])
def optimize_fleet_endpoint():
    """Fleet-wide optimization endpoint."""
    start_time = time.time()
    
    try:
        data = request.get_json() or {}
        
        result = process_fleet_optimization(data)
        
        if 'error' in result and not result.get('success', True):
            return jsonify(result), 503
        
        execution_time = time.time() - start_time
        result['total_execution_time_seconds'] = round(execution_time, 3)
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        app.logger.error(f"Error in fleet optimization: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e), "algorithm": "fleet_optimizer"}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get detailed performance statistics."""
    try:
        from OR_tool_prototype_batch_optimized import _osrm_cache as batch_cache
        cache_info = {"batch_optimized_cache_size": len(batch_cache)}
    except (ImportError, AttributeError):
        cache_info = {"batch_optimized_cache_size": 0}
    
    return jsonify({
        **performance_stats,
        **cache_info,
        "connected_websocket_clients": len(connected_clients),
    }), 200

@app.route('/cache/clear', methods=['POST'])
def clear_caches():
    """Clear all optimization caches."""
    try:
        if BATCH_AVAILABLE:
            clear_cache()
        return jsonify({
            "message": "Caches cleared successfully",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": f"Failed to clear caches: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting server with WebSocket support on port {port}")
    print("")
    print("Events that TRIGGER fleet optimization:")
    print("  - task:created")
    print("  - task:declined")
    print("  - task:completed")
    print("  - task:cancelled")
    print("  - task:updated")
    print("  - agent:online")
    print("  - agent:offline")
    print("")
    print("Events that just track (no auto-optimization):")
    print("  - agent:location_update")
    print("")
    print("Request events:")
    print("  - task:get_recommendations (single task)")
    print("  - fleet:optimize_request (full fleet)")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
