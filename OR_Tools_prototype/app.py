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

# Throttle location update logs (log every 60 seconds per agent, not every 5s)
LOCATION_LOG_INTERVAL_SECONDS = 60
_last_location_log_time = {}  # agent_id -> last log timestamp

def _should_log_location(agent_id: str) -> bool:
    """Check if we should log this agent's location update (throttled)"""
    import time
    now = time.time()
    last_log = _last_location_log_time.get(agent_id, 0)
    
    if now - last_log >= LOCATION_LOG_INTERVAL_SECONDS:
        _last_location_log_time[agent_id] = now
        return True
    return False

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
    
    Data source priority:
    1. Direct data in request (agents_data, tasks_data)
    2. FleetState (in-memory, real-time)
    3. HTTP API fallback (dashboard endpoints)
    """
    import requests as req
    from concurrent.futures import ThreadPoolExecutor
    
    if not FLEET_AVAILABLE:
        return {"error": "Fleet optimization not available", "success": False}
    
    # Check if data is provided directly
    if 'agents_data' in data and 'tasks_data' in data:
        agents_data = data['agents_data']
        tasks_data = data['tasks_data']
        print("[FleetOptimizer] Using directly provided data")
    
    # Try FleetState first (faster, already in memory)
    elif FLEET_STATE_AVAILABLE and fleet_state and fleet_state.get_stats().get('total_agents', 0) > 0:
        agents_data = fleet_state.export_agents_for_optimizer()
        tasks_data = fleet_state.export_tasks_for_optimizer()
        print(f"[FleetOptimizer] Using FleetState: {len(agents_data.get('agents', []))} agents, {len(tasks_data.get('tasks', []))} tasks")
    
    else:
        # Fallback: Fetch from dashboard HTTP API
        default_dashboard_url = os.environ.get('DASHBOARD_URL', 'http://localhost:8000')
        dashboard_url = data.get('dashboard_url', default_dashboard_url)
        
        print(f"[FleetOptimizer] FleetState empty, fetching from dashboard: {dashboard_url}")
        
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
        
        # EVENT-BASED: Full fleet optimization with unassigned_tasks + agents_considered
        result['trigger_type'] = 'event'  # Clear identifier: 'event' or 'proximity'
        result['trigger_event'] = trigger_event
        result['trigger_data'] = trigger_data
        result['total_execution_time_seconds'] = round(execution_time, 3)
        
        # Emit updated routes to all connected clients
        # Includes: agent_routes, unassigned_tasks (with agents_considered), metadata, compatibility
        emit('fleet:routes_updated', result, broadcast=True)
        
        assigned = result.get('metadata', {}).get('tasks_assigned', 0)
        unassigned = result.get('metadata', {}).get('tasks_unassigned', 0)
        print(f"[WebSocket] Auto-optimization complete: {assigned} assigned, {unassigned} unassigned in {execution_time:.3f}s")
        
    except Exception as e:
        print(f"[WebSocket] Auto-optimization failed: {e}")
        emit('fleet:routes_updated', {
            'trigger_type': 'event',
            'trigger_event': trigger_event,
            'trigger_data': trigger_data,
            'error': str(e),
            'success': False,
            'agent_routes': [],
            'unassigned_tasks': []
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
    Trigger optimization for a SINGLE agent who entered proximity of a task.
    
    Uses the full fleet optimizer but with only this ONE agent, so it can:
    1. Consider ALL unassigned tasks (not just the triggering one)
    2. Find optimal chains (Task A delivery near Task B pickup)
    3. Assign multiple tasks if they chain well
    
    Emits same format as event-based optimization: fleet:routes_updated
    """
    print(f"[FleetState] 🚀 Proximity optimization: {agent_name} triggered by {task_name}")
    performance_stats["auto_optimizations"] += 1
    
    try:
        start_time = time.time()
        
        if not FLEET_STATE_AVAILABLE or not fleet_state:
            # Fallback to full fleet optimization
            trigger_fleet_optimization('proximity_trigger', {
                'agent_id': agent_id,
                'task_id': task_id,
                'dashboard_url': dashboard_url
            })
            return
        
        agent = fleet_state.get_agent(str(agent_id))
        if not agent:
            print(f"[FleetState] ❌ Agent {agent_id} not found in state")
            return
        
        if not agent.has_capacity:
            print(f"[FleetState] ⚠️ {agent_name} is at capacity, skipping")
            return
        
        # Build single-agent optimization data
        # Format agent with current tasks
        current_tasks = []
        for ct in agent.current_tasks:
            current_tasks.append({
                'id': ct.id,
                'job_type': ct.job_type,
                'restaurant_location': [ct.restaurant_location.lat, ct.restaurant_location.lng],
                'delivery_location': [ct.delivery_location.lat, ct.delivery_location.lng],
                'pickup_before': ct.pickup_before.isoformat() if ct.pickup_before else None,
                'delivery_before': ct.delivery_before.isoformat() if ct.delivery_before else None,
                'assigned_driver': agent.id,
                'pickup_completed': ct.pickup_completed,
                '_meta': ct.meta
            })
        
        agents_data = {
            'agents': [{
                'id': agent.id,
                'name': agent.name,
                'location': [agent.current_location.lat, agent.current_location.lng],
                'current_tasks': current_tasks,
                'max_capacity': agent.max_capacity,
                'wallet_balance': agent.wallet_balance,
                'tags': agent.tags
            }],
            'geofence_data': [],
            'settings_used': {
                'walletNoCashThreshold': 500,
                'maxDistanceKm': fleet_state.max_distance_km
            }
        }
        
        # Get ALL unassigned tasks (not just the triggering one)
        # This allows OR-Tools to find chains
        tasks_data = fleet_state.export_tasks_for_optimizer()
        
        if not tasks_data.get('tasks'):
            print(f"[FleetState] No unassigned tasks available")
            return
        
        print(f"[FleetState] Running single-agent optimization: {agent_name} vs {len(tasks_data['tasks'])} tasks")
        
        # Run optimization for this single agent against all tasks
        result = optimize_fleet(agents_data, tasks_data)
        
        execution_time = time.time() - start_time
        
        # Add trigger context
        result['trigger_type'] = 'proximity'  # Clear identifier: 'event' or 'proximity'
        result['trigger_event'] = 'proximity_trigger'
        result['trigger_data'] = {
            'agent_id': agent_id,
            'agent_name': agent_name,
            'triggering_task_id': task_id,
            'triggering_task_name': task_name,
            'trigger_distance_km': round(distance_km, 2),
            'trigger_reason': trigger_type  # 'entered_radius', 'existing_task_near', etc.
        }
        result['total_execution_time_seconds'] = round(execution_time, 3)
        
        # PROXIMITY: Remove unassigned_tasks - this is a suggestion for ONE agent
        # Dashboard doesn't need to know about other unassigned tasks in proximity mode
        if 'unassigned_tasks' in result:
            del result['unassigned_tasks']
        
        # Check if any tasks were assigned
        assigned_count = result.get('metadata', {}).get('tasks_assigned', 0)
        
        if assigned_count > 0:
            # Emit result (same base format as event-based, minus unassigned_tasks)
            emit('fleet:routes_updated', result, broadcast=True)
            
            # Log what was assigned
            for route in result.get('agent_routes', []):
                new_tasks = route.get('assigned_new_tasks', [])
                if new_tasks:
                    print(f"[FleetState] ✅ Proximity assigned {len(new_tasks)} task(s) to {agent_name}: {new_tasks}")
        else:
            print(f"[FleetState] ⚠️ Proximity trigger but no assignment possible for {agent_name}")
            # Don't emit for proximity if nothing assigned - no need to spam dashboard
    
    except Exception as e:
        print(f"[FleetState] ❌ Proximity optimization failed: {e}")
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
        'fleet_state_enabled': FLEET_STATE_AVAILABLE,
        'available_events': [
            # Request events
            'task:get_recommendations',
            'fleet:optimize_request',
            'fleet:sync',  # Initial state sync
            # Task events
            'task:created',
            'task:assigned',
            'task:accepted',
            'task:declined',
            'task:completed',
            'task:cancelled',
            'task:updated',
            # Agent events
            'agent:online',
            'agent:offline',
            'agent:location_update'
        ]
    })

@socketio.on('fleet:sync')
def handle_fleet_sync(data):
    """
    Initial fleet state sync from dashboard.
    Payload: {
        agents: [...],
        unassigned_tasks: [...],
        in_progress_tasks: [...],
        geofences: [...],
        config: {...},
        dashboard_url
    }
    """
    performance_stats["websocket_events"] += 1
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
    print(f"[WebSocket] fleet:sync received from dashboard")
    
    if not FLEET_STATE_AVAILABLE or not fleet_state:
        emit('fleet:sync_ack', {
            'success': False,
            'error': 'Fleet state not available',
            'received_at': datetime.now().isoformat()
        })
        return
    
    try:
        # Sync agents
        agents_data = data.get('agents', [])
        if agents_data:
            fleet_state.sync_agents(agents_data)
            print(f"[FleetState] Synced {len(agents_data)} agents")
        
        # Clear existing tasks before full sync
        fleet_state.clear_tasks()
        
        # Sync unassigned tasks
        unassigned_tasks = data.get('unassigned_tasks', [])
        if unassigned_tasks:
            fleet_state.sync_tasks(unassigned_tasks)
            print(f"[FleetState] Synced {len(unassigned_tasks)} unassigned tasks")
        
        # Sync in-progress tasks
        in_progress_tasks = data.get('in_progress_tasks', [])
        if in_progress_tasks:
            fleet_state.sync_tasks(in_progress_tasks)
            print(f"[FleetState] Synced {len(in_progress_tasks)} in-progress tasks")
        
        # Store geofences (if provided)
        geofences = data.get('geofences', [])
        if geofences:
            # TODO: Implement geofence storage and checking
            print(f"[FleetState] Received {len(geofences)} geofences (not yet implemented)")
        
        # Apply config (if provided)
        config = data.get('config', {})
        if config:
            if 'default_max_distance_km' in config:
                fleet_state.max_distance_km = float(config['default_max_distance_km'])
            if 'max_tasks_per_agent' in config:
                # Update agent capacities
                pass
            print(f"[FleetState] Applied config: {config}")
        
        stats = fleet_state.get_stats()
        
        emit('fleet:sync_ack', {
            'success': True,
            'received_at': datetime.now().isoformat(),
            'synced': {
                'agents': len(agents_data),
                'unassigned_tasks': len(unassigned_tasks),
                'in_progress_tasks': len(in_progress_tasks),
                'geofences': len(geofences)
            },
            'fleet_stats': stats
        })
        
        print(f"[FleetState] ✅ Initial sync complete: {stats['online_agents']} agents, {stats['unassigned_tasks']} tasks")
        
    except Exception as e:
        print(f"[FleetState] ❌ Sync failed: {e}")
        import traceback
        traceback.print_exc()
        emit('fleet:sync_ack', {
            'success': False,
            'error': str(e),
            'received_at': datetime.now().isoformat()
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
        
        # ON-DEMAND: Manual request from dashboard
        result['trigger_type'] = 'manual'  # Clear identifier: 'event', 'proximity', or 'manual'
        result['trigger_event'] = 'fleet:optimize_request'
        result['request_id'] = request_id
        result['total_execution_time_seconds'] = round(execution_time, 3)
        
        emit('fleet:routes_updated', result)
        
        assigned = result.get('metadata', {}).get('tasks_assigned', 0)
        unassigned = result.get('metadata', {}).get('tasks_unassigned', 0)
        print(f"[WebSocket] Sent fleet routes: {assigned} assigned, {unassigned} unassigned in {execution_time:.3f}s")
        
    except Exception as e:
        print(f"[WebSocket] Error in fleet:optimize_request: {e}")
        emit('fleet:routes_updated', {
            'trigger_type': 'manual',
            'trigger_event': 'fleet:optimize_request',
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
    Payload: { task: { id, ... }, dashboard_url }
    """
    performance_stats["websocket_events"] += 1
    task_data = data.get('task', {})
    task_id = task_data.get('id', data.get('id', ''))
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
    print(f"[WebSocket] task:created: {task_id[:20]}...")
    
    # Update fleet state
    triggered_optimization = False
    if FLEET_STATE_AVAILABLE and fleet_state and task_data:
        task = fleet_state.add_task(task_data)
        print(f"[FleetState] Added task: {task.restaurant_name} → {task.customer_name}")
        
        # PROACTIVE CHECK: Are any agents already near this task?
        eligible_agents = fleet_state.find_eligible_agents_for_task(task_id)
        nearby_eligible = [
            (agent, dist, reason) for agent, dist, reason in eligible_agents 
            if reason is None and dist <= fleet_state.assignment_radius_km
        ]
        
        if nearby_eligible:
            # Sort by distance to find best agent
            nearby_eligible.sort(key=lambda x: x[1])
            best_agent, best_dist, _ = nearby_eligible[0]
            
            # Determine trigger type based on agent's current tasks
            if best_agent.current_tasks:
                trigger_type = "projected_location"  # Agent will be near after current tasks
            else:
                trigger_type = "current_location"  # Agent is idle and near
            
            print(f"[FleetState] 🎯 NEW TASK: {best_agent.name} is {best_dist:.2f}km from {task.restaurant_name} ({trigger_type})")
            
            # Check cooldown and trigger optimization
            if fleet_state.should_trigger_optimization(best_agent.id):
                fleet_state.record_optimization(best_agent.id)
                triggered_optimization = True
                
                # Trigger incremental optimization
                trigger_incremental_optimization(
                    agent_id=best_agent.id,
                    agent_name=best_agent.name,
                    task_id=task_id,
                    task_name=task.restaurant_name,
                    distance_km=best_dist,
                    trigger_type=trigger_type,
                    dashboard_url=dashboard_url
                )
            else:
                print(f"[FleetState] ⏳ {best_agent.name} on cooldown, skipping immediate optimization")
        else:
            print(f"[FleetState] 📋 No eligible agents near {task.restaurant_name} - waiting for proximity trigger")
    
    emit('task:created_ack', {
        'id': task_id,
        'received_at': datetime.now().isoformat(),
        'added_to_fleet_state': True,
        'triggered_optimization': True
    })
    
    # EVENT-BASED: Immediately try to assign the new task
    trigger_fleet_optimization('task:created', {
        'id': task_id,
        'dashboard_url': dashboard_url
    })

@socketio.on('task:declined')
def handle_task_declined(data):
    """
    Agent declined task → Record decline and find another eligible agent.
    Payload: { id, declined_by: [...], latest_decline: { agent_id, agent_name, declined_at }, dashboard_url }
    """
    performance_stats["websocket_events"] += 1
    task_id = data.get('id', '')
    declined_by = data.get('declined_by', [])
    latest_decline = data.get('latest_decline', {})
    latest_agent_id = latest_decline.get('agent_id')
    latest_agent_name = latest_decline.get('agent_name', 'Unknown')
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
    print(f"[WebSocket] task:declined: {task_id[:20]}... by {latest_agent_name} (total: {len(declined_by)} declines)")
    
    # Update fleet state - record the declines
    triggered_optimization = False
    if FLEET_STATE_AVAILABLE and fleet_state and task_id:
        fleet_state.record_task_decline(task_id, declined_by, latest_agent_id)
        
        # Get the task to check its restaurant name
        task = fleet_state.get_task(task_id)
        
        if task:
            # PROACTIVE CHECK: Find another eligible agent near this task
            eligible_agents = fleet_state.find_eligible_agents_for_task(task_id)
            nearby_eligible = [
                (agent, dist, reason) for agent, dist, reason in eligible_agents 
                if reason is None and dist <= fleet_state.assignment_radius_km
            ]
            
            if nearby_eligible:
                # Sort by distance to find best agent
                nearby_eligible.sort(key=lambda x: x[1])
                best_agent, best_dist, _ = nearby_eligible[0]
                
                # Determine trigger type
                trigger_type = "projected_location" if best_agent.current_tasks else "current_location"
                
                print(f"[FleetState] 🔄 DECLINED: {best_agent.name} is {best_dist:.2f}km from {task.restaurant_name} ({trigger_type})")
                
                # Check cooldown and trigger optimization
                if fleet_state.should_trigger_optimization(best_agent.id):
                    fleet_state.record_optimization(best_agent.id)
                    triggered_optimization = True
                    
                    # Trigger incremental optimization
                    trigger_incremental_optimization(
                        agent_id=best_agent.id,
                        agent_name=best_agent.name,
                        task_id=task_id,
                        task_name=task.restaurant_name,
                        distance_km=best_dist,
                        trigger_type=trigger_type,
                        dashboard_url=dashboard_url
                    )
                else:
                    print(f"[FleetState] ⏳ {best_agent.name} on cooldown, skipping immediate optimization")
            else:
                print(f"[FleetState] 📋 No other eligible agents near {task.restaurant_name} - waiting for proximity trigger")
    
    emit('task:declined_ack', {
        'id': task_id,
        'declined_by': declined_by,
        'received_at': datetime.now().isoformat(),
        'decline_recorded': True,
        'triggered_optimization': True
    })
    
    # EVENT-BASED: Immediately try to reassign to another agent
    trigger_fleet_optimization('task:declined', {
        'id': task_id,
        'declined_by': declined_by,
        'dashboard_url': dashboard_url
    })

@socketio.on('task:completed')
def handle_task_completed(data):
    """
    Task completed → Update state, agent has capacity for more work.
    Payload: { id, agent_id, agent_name, completed_at, dashboard_url }
    """
    performance_stats["websocket_events"] += 1
    task_id = data.get('id', '')
    agent_id = data.get('agent_id')
    agent_name = data.get('agent_name', 'Unknown')
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
    print(f"[WebSocket] task:completed: {task_id[:20]}... by {agent_name} ({agent_id})")
    
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
        'id': task_id,
        'agent_id': agent_id,
        'received_at': datetime.now().isoformat(),
        'task_removed': True
    })
    
    # No auto-optimization - proximity triggers will handle next assignment

@socketio.on('task:cancelled')
def handle_task_cancelled(data):
    """
    Task cancelled → Update state, remove from routes.
    Payload: { id, reason, dashboard_url }
    """
    performance_stats["websocket_events"] += 1
    task_id = data.get('id', '')
    reason = data.get('reason', 'unknown')
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
    print(f"[WebSocket] task:cancelled: {task_id[:20]}... (reason: {reason})")
    
    # Update fleet state
    if FLEET_STATE_AVAILABLE and fleet_state and task_id:
        task = fleet_state.cancel_task(task_id)
        if task:
            print(f"[FleetState] Task cancelled: {task.restaurant_name}")
    
    emit('task:cancelled_ack', {
        'id': task_id,
        'reason': reason,
        'received_at': datetime.now().isoformat(),
        'task_removed': True
    })
    
    # No auto-optimization - task removed from state

@socketio.on('task:updated')
def handle_task_updated(data):
    """
    Task updated → Details changed (times, locations, tags, etc.), re-optimize.
    Payload: { id, updated_fields, dashboard_url }
    """
    performance_stats["websocket_events"] += 1
    task_id = data.get('id', '')
    updated_fields = data.get('updated_fields', [])
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
    print(f"[WebSocket] task:updated: {task_id[:20]}... (fields: {', '.join(updated_fields) if updated_fields else 'unknown'})")
    
    emit('task:updated_ack', {
        'id': task_id,
        'updated_fields': updated_fields,
        'received_at': datetime.now().isoformat(),
        'task_updated': True
    })
    
    # No auto-optimization - task details updated in state

@socketio.on('task:assigned')
def handle_task_assigned(data):
    """
    Task assigned to agent (manually or via optimization).
    Payload: { id, agent_id, agent_name, assigned_at, dashboard_url }
    """
    performance_stats["websocket_events"] += 1
    task_id = data.get('id', '')
    agent_id = data.get('agent_id')
    agent_name = data.get('agent_name', 'Unknown')
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
    print(f"[WebSocket] task:assigned: {task_id[:20]}... → {agent_name} ({agent_id})")
    
    # Update fleet state
    if FLEET_STATE_AVAILABLE and fleet_state and task_id and agent_id:
        task = fleet_state.assign_task(task_id, str(agent_id), agent_name)
        if task:
            print(f"[FleetState] Task {task.restaurant_name} assigned to {agent_name}")
    
    emit('task:assigned_ack', {
        'id': task_id,
        'agent_id': agent_id,
        'agent_name': agent_name,
        'received_at': datetime.now().isoformat()
    })

@socketio.on('task:accepted')
def handle_task_accepted(data):
    """
    Agent accepted the task assignment.
    Payload: { id, agent_id, agent_name, accepted_at, dashboard_url }
    """
    performance_stats["websocket_events"] += 1
    task_id = data.get('id', '')
    agent_id = data.get('agent_id')
    agent_name = data.get('agent_name', 'Unknown')
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
    print(f"[WebSocket] task:accepted: {task_id[:20]}... by {agent_name} ({agent_id})")
    
    # Update fleet state
    if FLEET_STATE_AVAILABLE and fleet_state and task_id and agent_id:
        task = fleet_state.accept_task(task_id, str(agent_id))
        if task:
            print(f"[FleetState] Task {task.restaurant_name} accepted by {agent_name}")
    
    emit('task:accepted_ack', {
        'id': task_id,
        'agent_id': agent_id,
        'agent_name': agent_name,
        'received_at': datetime.now().isoformat()
    })

@socketio.on('agent:online')
def handle_agent_online(data):
    """
    Agent came online → Update state and check for nearby tasks.
    Payload: { agent_id, name, location, dashboard_url }
    """
    performance_stats["websocket_events"] += 1
    agent_id = data.get('agent_id')
    name = data.get('name', 'Unknown')
    location = data.get('location')
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
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
        'agent_added': True,
        'triggered_optimization': True
    })
    
    # EVENT-BASED: Immediately try to give this agent work
    trigger_fleet_optimization('agent:online', {
        'agent_id': agent_id,
        'agent_name': name,
        'dashboard_url': dashboard_url
    })

@socketio.on('agent:offline')
def handle_agent_offline(data):
    """
    Agent went offline → Update state and reassign their tasks.
    Payload: { agent_id, name, dashboard_url }
    """
    performance_stats["websocket_events"] += 1
    agent_id = data.get('agent_id')
    name = data.get('name', 'Unknown')
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
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
        'agent_removed': True
    })
    
    # No auto-optimization - tasks will remain unassigned until manually reassigned or proximity trigger

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
                # Throttled log for cooldown skips
                _should_log = _should_log_location(agent_id)
                if _should_log:
                    print(f"[FleetState] ⏳ {name} near tasks but on cooldown")
        else:
            # No eligible triggers - throttled logging for regular updates
            if _should_log_location(agent_id):
                agent = fleet_state.get_agent(str(agent_id))
                if agent:
                    tasks_near = fleet_state.find_tasks_near_agent(str(agent_id))
                    if tasks_near:
                        task, dist = tasks_near[0]
                        print(f"[FleetState] 📍 {name} at {location}, nearest task: {task.restaurant_name} ({dist:.2f}km)")
                    else:
                        print(f"[FleetState] 📍 {name} at {location}, no nearby tasks")


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
            'tags': agent.tags,
            'wallet_balance': agent.wallet_balance,
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
        fleet_state.clear_tasks()
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
