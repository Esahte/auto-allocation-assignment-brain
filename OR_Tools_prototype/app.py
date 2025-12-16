from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import json
import os
import time
from datetime import datetime, timezone
import uuid
import threading

app = Flask(__name__)

# Import Fleet State (Abstract Map)
try:
    from fleet_state import fleet_state, AgentStatus, TaskStatus
    FLEET_STATE_AVAILABLE = True
    print("[FleetState] Abstract Map loaded successfully")
except ImportError as e:
    FLEET_STATE_AVAILABLE = False
    fleet_state = None
    print(f"[FleetState] Warning: Fleet state not available: {e}")
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'fleet-optimizer-secret-key')

# Initialize Socket.IO with CORS support
# Auto-detect async mode: use gevent in production (gunicorn/Cloud Run), threading locally
def get_async_mode():
    """Auto-detect the best async mode based on environment."""
    # Check if running under gunicorn or Cloud Run
    if 'gunicorn' in os.environ.get('SERVER_SOFTWARE', ''):
        return 'gevent'
    # Check if we're in Cloud Run (K_SERVICE is set)
    if os.environ.get('K_SERVICE'):
        return 'gevent'
    # Check if gevent is available
    try:
        import gevent
        # If explicitly requested
        if os.environ.get('ASYNC_MODE') == 'gevent':
            return 'gevent'
    except ImportError:
        pass
    # Default to threading for local development
    return 'threading'

async_mode = get_async_mode()
print(f"[SocketIO] Using async_mode: {async_mode}")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)

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


def trigger_incremental_optimization(
    agent_id: str,
    agent_name: str,
    task_id: str,
    task_name: str,
    distance_km: float,
    trigger_type: str,
    dashboard_url: str
):
    """
    Trigger incremental optimization for a SINGLE agent near a specific task.
    This is more efficient than full fleet optimization.
    
    Instead of optimizing the entire fleet, we:
    1. Get the agent's current state
    2. Get their nearby unassigned tasks
    3. Run single-agent optimization
    4. Emit assignment suggestion
    """
    print(f"[FleetState] 🚀 Incremental optimization: {agent_name} → {task_name}")
    performance_stats["auto_optimizations"] += 1
    
    try:
        start_time = time.time()
        
        # For now, we'll use the batch recommendation for single-agent optimization
        # This finds the best task(s) for this specific agent
        
        if FLEET_STATE_AVAILABLE and fleet_state:
            agent = fleet_state.get_agent(str(agent_id))
            
            if not agent:
                print(f"[FleetState] ❌ Agent {agent_id} not found in state")
                return
            
            # Find all tasks near this agent
            nearby_tasks = fleet_state.find_tasks_near_agent(
                str(agent_id),
                radius_km=fleet_state.assignment_radius_km,
                use_projected=False
            )
            
            if not nearby_tasks:
                print(f"[FleetState] No eligible tasks for {agent_name}")
                return
            
            # Build recommendation request for this single agent
            # We'll use the batch optimizer for this
            if BATCH_AVAILABLE:
                # Find the best task for this agent
                best_task, best_distance = nearby_tasks[0]
                
                # Format agent data
                agent_data = {
                    'id': agent.id,
                    'name': agent.name,
                    'location': [agent.current_location.lat, agent.current_location.lng]
                }
                
                # Format task data
                task_data = {
                    'id': best_task.id,
                    'job_type': best_task.job_type,
                    'restaurant_location': [best_task.restaurant_location.lat, best_task.restaurant_location.lng],
                    'delivery_location': [best_task.delivery_location.lat, best_task.delivery_location.lng],
                    'pickup_before': best_task.pickup_before.isoformat(),
                    'delivery_before': best_task.delivery_before.isoformat()
                }
                
                # Get agent's current tasks
                current_tasks_data = []
                for ct in agent.current_tasks:
                    current_tasks_data.append({
                        'id': ct.id,
                        'job_type': ct.job_type,
                        'restaurant_location': [ct.restaurant_location.lat, ct.restaurant_location.lng],
                        'delivery_location': [ct.delivery_location.lat, ct.delivery_location.lng],
                        'pickup_before': ct.pickup_before.isoformat(),
                        'delivery_before': ct.delivery_before.isoformat(),
                        'assigned_driver': agent.id,
                        'pickup_completed': ct.pickup_completed
                    })
                
                # Run single-agent optimization
                result = process_batch_recommendation({
                    'new_task': task_data,
                    'agents': [agent_data],
                    'current_tasks': current_tasks_data,
                    'max_distance_km': fleet_state.max_distance_km,
                    'optimization_mode': 'tardiness_min'
                })
                
                execution_time = time.time() - start_time
                
                recommendations = result.get('recommendations', [])
                
                if recommendations:
                    best_rec = recommendations[0]
                    
                    # Emit assignment suggestion
                    emit('task:assignment_suggested', {
                        'trigger_type': 'proximity',
                        'agent_id': agent.id,
                        'agent_name': agent.name,
                        'task_id': best_task.id,
                        'task_restaurant': best_task.restaurant_name,
                        'task_customer': best_task.customer_name,
                        'distance_km': round(best_distance, 2),
                        'score': best_rec.get('score', 0),
                        'lateness_seconds': best_rec.get('lateness_seconds', 0),
                        'recommendation': best_rec,
                        'other_nearby_tasks': len(nearby_tasks) - 1,
                        'execution_time_seconds': round(execution_time, 3)
                    }, broadcast=True)
                    
                    print(f"[FleetState] ✅ Suggested: {agent_name} → {best_task.restaurant_name} "
                          f"(score: {best_rec.get('score', 0)}, {execution_time:.3f}s)")
                else:
                    print(f"[FleetState] ⚠️ No valid recommendation for {agent_name} → {best_task.restaurant_name}")
            else:
                # Fallback: just emit that we detected proximity
                emit('task:assignment_suggested', {
                    'trigger_type': 'proximity',
                    'agent_id': agent_id,
                    'agent_name': agent_name,
                    'task_id': task_id,
                    'task_name': task_name,
                    'distance_km': round(distance_km, 2),
                    'note': 'Batch optimizer not available for scoring'
                }, broadcast=True)
        else:
            # Fallback to full fleet optimization if fleet state not available
            trigger_fleet_optimization('proximity_trigger', {
                'agent_id': agent_id,
                'task_id': task_id,
                'distance_km': distance_km,
                'dashboard_url': dashboard_url
            })
            
    except Exception as e:
        print(f"[FleetState] ❌ Incremental optimization failed: {e}")
        import traceback
        traceback.print_exc()


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
    New task created → Add to fleet state and check for nearby agents.
    """
    performance_stats["websocket_events"] += 1
    task_id = data.get('task_id')
    task_data = data.get('task', {})
    print(f"[WebSocket] task:created: {task_id}")
    
    # Update fleet state
    if FLEET_STATE_AVAILABLE and fleet_state:
        if task_data:
            task_data['id'] = task_id
            task = fleet_state.add_task(task_data)
            print(f"[FleetState] Added task: {task.restaurant_name} → {task.customer_name}")
            
            # Check if any agents are already near this task
            eligible_agents = fleet_state.find_eligible_agents_for_task(task_id)
            nearby_eligible = [
                (a, d, r) for a, d, r in eligible_agents 
                if r is None and d <= fleet_state.assignment_radius_km
            ]
            
            if nearby_eligible:
                best_agent, best_dist, _ = nearby_eligible[0]
                print(f"[FleetState] 🎯 Agent {best_agent.name} is already {best_dist:.2f}km from new task!")
    
    emit('task:created_ack', {
        'task_id': task_id,
        'received_at': datetime.now().isoformat(),
        'will_optimize': True
    })
    
    # Trigger fleet optimization
    trigger_fleet_optimization('task:created', {
        'task_id': task_id,
        'dashboard_url': data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    })

@socketio.on('task:declined')
def handle_task_declined(data):
    """
    Agent declined task → Record decline and re-optimize.
    """
    performance_stats["websocket_events"] += 1
    task_id = data.get('task_id')
    agent_id = data.get('agent_id')
    agent_name = data.get('agent_name', 'Unknown')
    print(f"[WebSocket] task:declined: {task_id} by {agent_name} ({agent_id})")
    
    # Update fleet state - record the decline
    if FLEET_STATE_AVAILABLE and fleet_state and agent_id and task_id:
        fleet_state.add_declined_task(str(agent_id), task_id)
    
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
        'dashboard_url': data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    })

@socketio.on('task:completed')
def handle_task_completed(data):
    """
    Task completed → Update state, agent has capacity for more work.
    """
    performance_stats["websocket_events"] += 1
    task_id = data.get('task_id')
    agent_id = data.get('agent_id')
    agent_name = data.get('agent_name', 'Unknown')
    print(f"[WebSocket] task:completed: {task_id} by {agent_name} ({agent_id})")
    
    # Update fleet state
    if FLEET_STATE_AVAILABLE and fleet_state and task_id:
        task = fleet_state.complete_task(task_id)
        if task:
            print(f"[FleetState] Task completed: {task.restaurant_name}")
        
        # Check if agent has nearby tasks to pick up
        if agent_id:
            agent = fleet_state.get_agent(str(agent_id))
            if agent and agent.has_capacity:
                nearby_tasks = fleet_state.find_tasks_near_agent(str(agent_id))
                if nearby_tasks:
                    nearest, dist = nearby_tasks[0]
                    print(f"[FleetState] 🎯 {agent_name} now idle, nearest task: {nearest.restaurant_name} ({dist:.2f}km)")
    
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
        'dashboard_url': data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    })

@socketio.on('task:cancelled')
def handle_task_cancelled(data):
    """
    Task cancelled → Update state, remove from routes.
    """
    performance_stats["websocket_events"] += 1
    task_id = data.get('task_id')
    reason = data.get('reason', 'unknown')
    print(f"[WebSocket] task:cancelled: {task_id} (reason: {reason})")
    
    # Update fleet state
    if FLEET_STATE_AVAILABLE and fleet_state and task_id:
        task = fleet_state.cancel_task(task_id)
        if task:
            print(f"[FleetState] Task cancelled: {task.restaurant_name}")
    
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
        'dashboard_url': data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
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
        'dashboard_url': data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    })

@socketio.on('agent:online')
def handle_agent_online(data):
    """
    Agent came online → Update state and check for nearby tasks.
    """
    performance_stats["websocket_events"] += 1
    agent_id = data.get('agent_id')
    name = data.get('name', 'Unknown')
    location = data.get('location')
    print(f"[WebSocket] agent:online: {name} ({agent_id})")
    
    # Update fleet state
    if FLEET_STATE_AVAILABLE and fleet_state and agent_id:
        loc_tuple = tuple(location) if location and len(location) >= 2 else None
        agent = fleet_state.set_agent_online(str(agent_id), name=name, location=loc_tuple)
        if agent:
            print(f"[FleetState] Agent online: {name} at {agent.current_location}")
            
            # Check for nearby tasks
            nearby_tasks = fleet_state.find_tasks_near_agent(str(agent_id))
            if nearby_tasks:
                nearest, dist = nearby_tasks[0]
                print(f"[FleetState] 🎯 {name} has {len(nearby_tasks)} nearby tasks, nearest: {nearest.restaurant_name} ({dist:.2f}km)")
    
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
        'dashboard_url': data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    })

@socketio.on('agent:offline')
def handle_agent_offline(data):
    """
    Agent went offline → Update state and reassign their tasks.
    """
    performance_stats["websocket_events"] += 1
    agent_id = data.get('agent_id')
    name = data.get('name', 'Unknown')
    print(f"[WebSocket] agent:offline: {name} ({agent_id})")
    
    # Update fleet state
    if FLEET_STATE_AVAILABLE and fleet_state and agent_id:
        agent = fleet_state.set_agent_offline(str(agent_id))
        if agent and agent.current_tasks:
            print(f"[FleetState] ⚠️ Agent offline with {len(agent.current_tasks)} tasks to reassign")
    
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
        'dashboard_url': data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    })

# -----------------------------------------------------------------------------
# EVENTS THAT JUST TRACK (no auto-optimization - too frequent)
# -----------------------------------------------------------------------------

@socketio.on('agent:location_update')
def handle_agent_location_update(data):
    """
    Agent location updated → Update fleet state and check for proximity triggers.
    Smart optimization: only triggers when agent enters assignment radius of a task.
    """
    performance_stats["websocket_events"] += 1
    agent_id = data.get('agent_id')
    name = data.get('name', 'Unknown')
    location = data.get('location', [])
    
    # Acknowledge receipt
    emit('agent:location_update_ack', {
        'agent_id': agent_id,
        'location': location,
        'received_at': datetime.now().isoformat()
    })
    
    # Update fleet state and check for proximity triggers
    if FLEET_STATE_AVAILABLE and fleet_state and len(location) >= 2:
        triggers = fleet_state.update_agent_location(
            agent_id=str(agent_id),
            lat=location[0],
            lng=location[1],
            name=name
        )
        
        # Check if we should trigger incremental optimization
        eligible_triggers = [t for t in triggers if t.is_eligible]
        
        if eligible_triggers:
            # Get the best trigger (closest task)
            best_trigger = min(eligible_triggers, key=lambda t: t.distance_km)
            
            # Check cooldown
            if fleet_state.should_trigger_optimization(agent_id):
                print(f"[FleetState] 🎯 Proximity trigger: {name} is {best_trigger.distance_km:.2f}km from {best_trigger.task.restaurant_name}")
                fleet_state.record_optimization(agent_id)
                
                # Trigger incremental optimization for this agent
                trigger_incremental_optimization(
                    agent_id=agent_id,
                    agent_name=name,
                    task_id=best_trigger.task.id,
                    task_name=best_trigger.task.restaurant_name,
                    distance_km=best_trigger.distance_km,
                    trigger_type=best_trigger.trigger_type,
                    dashboard_url=data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
                )
            else:
                print(f"[FleetState] ⏳ Skipped optimization for {name} (cooldown)")
        else:
            # No eligible triggers - just a regular update
            agent = fleet_state.get_agent(str(agent_id))
            if agent:
                tasks_near = fleet_state.find_tasks_near_agent(str(agent_id))
                if tasks_near:
                    task, dist = tasks_near[0]
                    print(f"[FleetState] 📍 {name} at {location}, nearest task: {task.restaurant_name} ({dist:.2f}km)")
                else:
                    print(f"[FleetState] 📍 {name} at {location}, no nearby tasks")
    else:
        print(f"[WebSocket] agent:location_update: {name} ({agent_id}) -> {location}")


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


# =============================================================================
# FLEET STATE ENDPOINTS (Abstract Map)
# =============================================================================

@app.route('/fleet-state', methods=['GET'])
def get_fleet_state():
    """Get current fleet state summary."""
    if not FLEET_STATE_AVAILABLE or not fleet_state:
        return jsonify({"error": "Fleet state not available"}), 503
    
    return jsonify(fleet_state.get_state_summary()), 200


@app.route('/fleet-state/agents', methods=['GET'])
def get_fleet_agents():
    """Get all agents in fleet state."""
    if not FLEET_STATE_AVAILABLE or not fleet_state:
        return jsonify({"error": "Fleet state not available"}), 503
    
    agents = []
    for agent in fleet_state.get_all_agents():
        agents.append({
            'id': agent.id,
            'name': agent.name,
            'status': agent.status.value,
            'current_location': {
                'lat': agent.current_location.lat,
                'lng': agent.current_location.lng
            },
            'projected_location': {
                'lat': agent.projected_location.lat,
                'lng': agent.projected_location.lng
            },
            'current_tasks': len(agent.current_tasks),
            'capacity': f"{len(agent.current_tasks)}/{agent.max_capacity}",
            'has_capacity': agent.has_capacity,
            'is_idle': agent.is_idle,
            'declined_tasks': len(agent.declined_task_ids),
            'last_update': agent.last_updated.isoformat()
        })
    
    return jsonify({
        'count': len(agents),
        'online': len([a for a in agents if a['status'] != 'offline']),
        'available': len([a for a in agents if a['has_capacity']]),
        'agents': agents
    }), 200


@app.route('/fleet-state/tasks', methods=['GET'])
def get_fleet_tasks():
    """Get all tasks in fleet state."""
    if not FLEET_STATE_AVAILABLE or not fleet_state:
        return jsonify({"error": "Fleet state not available"}), 503
    
    unassigned = []
    for task in fleet_state.get_unassigned_tasks():
        unassigned.append({
            'id': task.id,
            'restaurant_name': task.restaurant_name,
            'customer_name': task.customer_name,
            'restaurant_location': {
                'lat': task.restaurant_location.lat,
                'lng': task.restaurant_location.lng
            },
            'delivery_location': {
                'lat': task.delivery_location.lat,
                'lng': task.delivery_location.lng
            },
            'pickup_before': task.pickup_before.isoformat(),
            'delivery_before': task.delivery_before.isoformat(),
            'urgency_score': task.urgency_score(),
            'is_urgent': task.is_urgent,
            'is_overdue': task.is_overdue
        })
    
    # Sort by urgency
    unassigned.sort(key=lambda t: t['urgency_score'], reverse=True)
    
    return jsonify({
        'unassigned_count': len(unassigned),
        'urgent_count': len([t for t in unassigned if t['is_urgent']]),
        'overdue_count': len([t for t in unassigned if t['is_overdue']]),
        'tasks': unassigned
    }), 200


@app.route('/fleet-state/proximity', methods=['GET'])
def get_proximity_triggers():
    """Get current proximity triggers (agents near tasks)."""
    if not FLEET_STATE_AVAILABLE or not fleet_state:
        return jsonify({"error": "Fleet state not available"}), 503
    
    triggers = fleet_state.get_optimization_candidates()
    
    result = []
    for trigger in triggers:
        result.append({
            'agent_id': trigger.agent.id,
            'agent_name': trigger.agent.name,
            'task_id': trigger.task.id,
            'task_restaurant': trigger.task.restaurant_name,
            'task_customer': trigger.task.customer_name,
            'distance_km': round(trigger.distance_km, 2),
            'trigger_type': trigger.trigger_type,
            'is_eligible': trigger.is_eligible,
            'eligibility_reason': trigger.eligibility_reason,
            'task_urgency': trigger.task.urgency_score()
        })
    
    return jsonify({
        'count': len(result),
        'assignment_radius_km': fleet_state.assignment_radius_km,
        'triggers': result
    }), 200


@app.route('/fleet-state/config', methods=['GET', 'POST'])
def fleet_state_config():
    """Get or update fleet state configuration."""
    if not FLEET_STATE_AVAILABLE or not fleet_state:
        return jsonify({"error": "Fleet state not available"}), 503
    
    if request.method == 'GET':
        return jsonify({
            'assignment_radius_km': fleet_state.assignment_radius_km,
            'chain_lookahead_radius_km': fleet_state.chain_lookahead_radius_km,
            'max_distance_km': fleet_state.max_distance_km,
            'optimization_cooldown_seconds': fleet_state.optimization_cooldown_seconds
        }), 200
    
    # POST - update config
    data = request.get_json() or {}
    
    if 'assignment_radius_km' in data:
        fleet_state.assignment_radius_km = float(data['assignment_radius_km'])
    if 'chain_lookahead_radius_km' in data:
        fleet_state.chain_lookahead_radius_km = float(data['chain_lookahead_radius_km'])
    if 'max_distance_km' in data:
        fleet_state.max_distance_km = float(data['max_distance_km'])
    if 'optimization_cooldown_seconds' in data:
        fleet_state.optimization_cooldown_seconds = float(data['optimization_cooldown_seconds'])
    
    return jsonify({
        'message': 'Configuration updated',
        'assignment_radius_km': fleet_state.assignment_radius_km,
        'chain_lookahead_radius_km': fleet_state.chain_lookahead_radius_km,
        'max_distance_km': fleet_state.max_distance_km,
        'optimization_cooldown_seconds': fleet_state.optimization_cooldown_seconds
    }), 200


@app.route('/fleet-state/sync', methods=['POST'])
def sync_fleet_state():
    """Sync fleet state from dashboard data."""
    if not FLEET_STATE_AVAILABLE or not fleet_state:
        return jsonify({"error": "Fleet state not available"}), 503
    
    data = request.get_json() or {}
    
    if 'agents' in data:
        fleet_state.sync_agents(data['agents'])
    
    if 'tasks' in data:
        fleet_state.sync_tasks(data['tasks'])
    
    return jsonify({
        'message': 'Fleet state synced',
        'stats': fleet_state.get_stats()
    }), 200


@app.route('/fleet-state/clear', methods=['POST'])
def clear_fleet_state():
    """Clear all fleet state (for testing)."""
    if not FLEET_STATE_AVAILABLE or not fleet_state:
        return jsonify({"error": "Fleet state not available"}), 503
    
    fleet_state.clear()
    
    return jsonify({
        'message': 'Fleet state cleared',
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/fleet-map')
def fleet_map():
    """Serve the fleet map visualization UI."""
    return render_template('fleet_map.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 OR-Tools Fleet Optimizer Server                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Port: {port}                                                                  ║
║  Fleet State: {'✅ Enabled' if FLEET_STATE_AVAILABLE else '❌ Disabled'}                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

📡 Events that TRIGGER fleet optimization:
   • task:created      - New task → optimize to assign
   • task:declined     - Agent declined → reassign
   • task:completed    - Agent free → give more work
   • task:cancelled    - Task removed → redistribute
   • task:updated      - Task changed → recalculate
   • agent:online      - Agent available → assign tasks
   • agent:offline     - Agent gone → reassign tasks

🎯 Smart Location Events (Proximity-Based):
   • agent:location_update → Updates fleet state
     → Triggers optimization ONLY when agent enters task radius
     → Uses cooldown to prevent over-optimization

📋 Request Events:
   • task:get_recommendations - Single task recommendation
   • fleet:optimize_request   - Full fleet optimization

🗺️ Fleet State Endpoints:
   • GET  /fleet-state           - Summary of all agents & tasks
   • GET  /fleet-state/agents    - All agents with locations
   • GET  /fleet-state/tasks     - Unassigned tasks by urgency
   • GET  /fleet-state/proximity - Proximity triggers
   • GET  /fleet-state/config    - Current configuration
   • POST /fleet-state/config    - Update configuration
   • POST /fleet-state/sync      - Sync from dashboard
   • GET  /fleet-map             - Visual fleet map UI

""")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
