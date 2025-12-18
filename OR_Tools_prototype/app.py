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

# Debouncing for event-based optimization
# Prevents redundant runs when multiple events fire in quick succession
import threading
_pending_optimization_timer = None
_optimization_epoch = 0
_optimization_lock = threading.Lock()
DEBOUNCE_DELAY_SECONDS = 0.5  # Wait 500ms before running optimization

# Sync lock - prevents optimizations during fleet sync
_sync_in_progress = False
_sync_lock = threading.Lock()
_optimization_pending_after_sync = False  # Queue optimization to run after sync

# Task-level lock - prevents multiple proximity optimizations for the same task
_tasks_being_optimized = set()  # Set of task IDs currently being optimized
_task_optimization_lock = threading.Lock()
_pending_trigger_type = None

def is_sync_in_progress() -> bool:
    """Check if a fleet sync is currently in progress."""
    with _sync_lock:
        return _sync_in_progress

def queue_optimization_after_sync(trigger_type: str):
    """Queue an optimization to run after sync completes."""
    global _optimization_pending_after_sync, _pending_trigger_type
    with _sync_lock:
        _optimization_pending_after_sync = True
        _pending_trigger_type = trigger_type
        print(f"[FleetState] 📋 Queued optimization ({trigger_type}) to run after sync")

def set_sync_in_progress(value: bool):
    """Set the sync in progress flag."""
    global _sync_in_progress, _optimization_pending_after_sync, _pending_trigger_type
    with _sync_lock:
        _sync_in_progress = value
        if value:
            print("[FleetState] 🔒 Sync started - blocking optimizations")
            _optimization_pending_after_sync = False  # Reset queue
            _pending_trigger_type = None
        else:
            print("[FleetState] 🔓 Sync complete - optimizations unblocked")
            # Check if optimization was queued during sync
            if _optimization_pending_after_sync:
                trigger = _pending_trigger_type or 'post_sync'
                _optimization_pending_after_sync = False
                _pending_trigger_type = None
                print(f"[FleetState] ▶️ Running queued optimization ({trigger})")
                # Schedule with small delay to let sync fully complete
                timer = threading.Timer(0.2, lambda: trigger_debounced_optimization(f'{trigger}_post_sync'))
                timer.start()

def trigger_debounced_optimization(trigger_type: str, dashboard_url: str = None, agent_id: str = None):
    """
    Schedule a debounced fleet optimization.
    
    If called multiple times in quick succession, only the last call
    will actually trigger the optimization (after DEBOUNCE_DELAY_SECONDS).
    
    This prevents:
    - Duplicate optimizations when task:created + proximity trigger fire together
    - Stale optimizations from older events
    - Optimizations during sync (would see incomplete state)
    """
    global _pending_optimization_timer, _optimization_epoch
    
    # Block if sync is in progress - queue to run after sync
    if is_sync_in_progress():
        print(f"[Debounce] ⚠️ Sync in progress - queuing {trigger_type} for after sync")
        queue_optimization_after_sync(trigger_type)
        return
    
    with _optimization_lock:
        # Cancel any pending optimization
        if _pending_optimization_timer is not None:
            _pending_optimization_timer.cancel()
            print(f"[Debounce] Cancelled pending optimization (new trigger: {trigger_type})")
        
        # Increment epoch to invalidate any in-flight optimizations
        _optimization_epoch += 1
        current_epoch = _optimization_epoch
        
        # Get default dashboard URL
        if dashboard_url is None:
            dashboard_url = os.environ.get('DASHBOARD_URL', 'http://localhost:8000')
        
        def run_optimization():
            global _pending_optimization_timer
            with _optimization_lock:
                # Check if this optimization is still valid (epoch hasn't changed)
                if current_epoch != _optimization_epoch:
                    print(f"[Debounce] Skipping stale optimization (epoch {current_epoch} != {_optimization_epoch})")
                    return
                _pending_optimization_timer = None
            
            # Run the actual optimization
            print(f"[Debounce] Running optimization for: {trigger_type}")
            trigger_fleet_optimization(trigger_type, {
                'dashboard_url': dashboard_url,
                'agent_id': agent_id
            })
        
        # Schedule the optimization
        _pending_optimization_timer = threading.Timer(DEBOUNCE_DELAY_SECONDS, run_optimization)
        _pending_optimization_timer.start()
        print(f"[Debounce] Scheduled optimization in {DEBOUNCE_DELAY_SECONDS}s for: {trigger_type}")

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


def process_fleet_optimization(data: dict, prefilter_distance: bool = True) -> dict:
    """
    Core function for fleet-wide optimization.
    Used by both HTTP API and WebSocket handlers.
    
    Data source priority:
    1. Direct data in request (agents_data, tasks_data)
    2. FleetState (in-memory, real-time)
    3. HTTP API fallback (dashboard endpoints)
    
    Args:
        data: Request data with optional dashboard_url, agents_data, tasks_data
        prefilter_distance: Whether to pre-filter by distance before solver
            - True (default): FAST mode for proximity triggers
            - False: THOROUGH mode for event-based/on-demand, solver handles distance
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
    
    result = optimize_fleet(agents_data, tasks_data, prefilter_distance=prefilter_distance)
    
    performance_stats["fleet_optimizer_requests"] += 1
    performance_stats["total_requests"] += 1
    performance_stats["algorithm_usage"]["fleet_optimizer"] = \
        performance_stats["algorithm_usage"].get("fleet_optimizer", 0) + 1
    
    return result


def trigger_fleet_optimization(trigger_event: str, trigger_data: dict):
    """
    Trigger fleet optimization and emit updated routes.
    Called automatically when relevant events occur.
    
    Uses THOROUGH mode (prefilter_distance=False) - lets solver handle distance
    for comprehensive fleet-wide optimization.
    """
    # Block if sync is in progress - queue to run after sync
    if is_sync_in_progress():
        print(f"[WebSocket] ⚠️ Sync in progress - queuing auto-optimization for {trigger_event}")
        queue_optimization_after_sync(trigger_event)
        return
    
    print(f"[WebSocket] Auto-triggering fleet optimization due to: {trigger_event}")
    print(f"[WebSocket] Mode: THOROUGH (solver handles distance)")
    performance_stats["auto_optimizations"] += 1
    
    try:
        start_time = time.time()
        
        # Use dashboard_url from trigger_data, env var, or default
        default_dashboard_url = os.environ.get('DASHBOARD_URL', 'http://localhost:8000')
        # EVENT-BASED: Use THOROUGH mode - solver handles distance constraints
        result = process_fleet_optimization({
            'dashboard_url': trigger_data.get('dashboard_url', default_dashboard_url)
        }, prefilter_distance=False)  # <-- THOROUGH: solver handles distance
        
        execution_time = time.time() - start_time
        
        # EVENT-BASED: Full fleet optimization with unassigned_tasks + agents_considered
        result['trigger_type'] = 'event'  # Clear identifier: 'event' or 'proximity'
        result['trigger_event'] = trigger_event
        result['trigger_data'] = trigger_data
        result['total_execution_time_seconds'] = round(execution_time, 3)
        
        # OPTIMISTIC UPDATE: Mark tasks as assigned immediately to prevent race conditions
        for route in result.get('agent_routes', []):
            agent_id = route.get('driver_id')
            agent_name = route.get('driver_name', 'Unknown')
            new_tasks = route.get('assigned_new_tasks', [])
            if new_tasks and agent_id and fleet_state:
                for assigned_task_id in new_tasks:
                    fleet_state.assign_task(assigned_task_id, str(agent_id), agent_name)
        
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
    global _tasks_being_optimized
    
    # TASK-LEVEL LOCK: Prevent concurrent optimizations for the same task
    with _task_optimization_lock:
        if task_id in _tasks_being_optimized:
            print(f"[FleetState] ⏳ Task {task_name} already being optimized - skipping {agent_name}")
            return
        # Mark this task as being optimized
        _tasks_being_optimized.add(task_id)
    
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
        
        # EARLY CHECK: Verify task is still unassigned
        task = fleet_state.get_task(str(task_id))
        if task and task.status != TaskStatus.UNASSIGNED:
            print(f"[FleetState] ⚠️ Task {task_name} already {task.status.name} - skipping optimization")
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
                'driver_id': agent.id,  # Agent.from_dict expects 'driver_id'
                'name': agent.name,
                'current_location': [agent.current_location.lat, agent.current_location.lng],  # Agent.from_dict expects 'current_location'
                'current_tasks': current_tasks,
                'wallet_balance': agent.wallet_balance,
                '_meta': {
                    'max_tasks': agent.max_capacity,
                    'available_capacity': agent.available_capacity,
                    'tags': agent.tags,
                    'has_no_cash_tag': 'NoCash' in agent.tags,
                    'is_scooter_agent': 'scooter' in [t.lower() for t in agent.tags]
                }
            }],
            'geofence_data': [],
            'settings_used': {
                'walletNoCashThreshold': fleet_state.wallet_threshold,
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
        print(f"[FleetState] Mode: FAST (distance pre-filtered)")
        
        # Log task details for debugging
        from datetime import datetime as dt
        now = dt.now().strftime('%H:%M:%S')
        print(f"[FleetState] Current time: {now}")
        for t in tasks_data['tasks'][:3]:
            pickup_before = t.get('pickup_before', 'none')
            delivery_before = t.get('delivery_before', 'none')
            task_name = t.get('_meta', {}).get('restaurant_name', t.get('id', 'unknown')[:15])
            # Parse and show time difference
            try:
                if pickup_before and pickup_before != 'none':
                    pickup_dt = dt.fromisoformat(pickup_before.replace('Z', '+00:00'))
                    time_to_pickup = (pickup_dt.replace(tzinfo=None) - dt.utcnow()).total_seconds() / 60
                    print(f"[FleetState] Task: {task_name} | pickup in {time_to_pickup:.1f}min | deadline: {delivery_before}")
                else:
                    print(f"[FleetState] Task: {task_name} | pickup: {pickup_before} | delivery: {delivery_before}")
            except Exception as e:
                print(f"[FleetState] Task: {task_name} | pickup: {pickup_before} | delivery: {delivery_before}")
        
        # PROXIMITY: Use FAST mode - pre-filter by distance for quick response
        result = optimize_fleet(agents_data, tasks_data, prefilter_distance=True)
        
        execution_time = time.time() - start_time
        
        # DEBUG: Log the full result structure
        print(f"[FleetState] Solver result: success={result.get('success')}, tasks_assigned={result.get('metadata', {}).get('tasks_assigned', 0)}")
        if not result.get('success'):
            print(f"[FleetState] Solver failure reason: {result.get('error', 'unknown')}")
        
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
        
        # Log unassigned task reasons BEFORE removing them
        unassigned_tasks = result.get('unassigned_tasks', [])
        if unassigned_tasks and assigned_count == 0:
            print(f"[FleetState] 📋 Why tasks weren't assigned to {agent_name}:")
            for ut in unassigned_tasks[:3]:  # Log first 3 reasons
                reason = ut.get('reason', 'unknown')
                reason_detail = ut.get('reason_detail', '')
                task_name = ut.get('_meta', {}).get('restaurant_name', ut.get('id', 'unknown')[:15])
                agents_considered = ut.get('agents_considered', [])
                
                if agents_considered:
                    found = False
                    for ac in agents_considered:
                        if ac.get('agent_name') == agent_name:
                            print(f"  → {task_name}: {ac.get('reason', reason)} - {ac.get('reason_detail', reason_detail)}")
                            found = True
                            break
                    if not found:
                        print(f"  → {task_name}: {reason} - {reason_detail}")
                else:
                    print(f"  → {task_name}: {reason} - {reason_detail}")
        
        if assigned_count > 0:
            # OPTIMISTIC UPDATE: Mark tasks as assigned immediately to prevent race conditions
            # This prevents other triggers from picking up the same task while dashboard processes
            for route in result.get('agent_routes', []):
                new_tasks = route.get('assigned_new_tasks', [])
                if new_tasks:
                    for assigned_task_id in new_tasks:
                        fleet_state.assign_task(assigned_task_id, agent_id, agent_name)
                    print(f"[FleetState] ✅ Proximity assigned {len(new_tasks)} task(s) to {agent_name}: {new_tasks}")
            
            # Emit result (same base format as event-based, minus unassigned_tasks)
            emit('fleet:routes_updated', result, broadcast=True)
        else:
            print(f"[FleetState] ⚠️ Proximity trigger but no assignment possible for {agent_name}")
            # Don't emit for proximity if nothing assigned - no need to spam dashboard
    
    except Exception as e:
        print(f"[FleetState] ❌ Proximity optimization failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # ALWAYS release the task lock
        with _task_optimization_lock:
            _tasks_being_optimized.discard(task_id)


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
            'pickup:completed',
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
    
    # Block optimizations during sync
    set_sync_in_progress(True)
    
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
            fleet_state.sync_geofences(geofences)
            print(f"[FleetState] Synced {len(geofences)} geofences")
        
        # Apply config (if provided)
        config = data.get('config', {})
        if config:
            print(f"[FleetState] Config received: {config}")
            
            if 'default_max_distance_km' in config:
                max_dist = float(config['default_max_distance_km'])
                fleet_state.max_distance_km = max_dist
                # Use max_distance as assignment radius too (consolidated)
                fleet_state.assignment_radius_km = max_dist
                print(f"[FleetState] → max_distance_km = {max_dist}km")
            
            if 'chain_lookahead_radius_km' in config:
                fleet_state.chain_lookahead_radius_km = float(config['chain_lookahead_radius_km'])
                print(f"[FleetState] → chain_lookahead_radius_km = {fleet_state.chain_lookahead_radius_km}km")
            
            if 'max_lateness_minutes' in config:
                fleet_state.max_lateness_minutes = int(config['max_lateness_minutes'])
                print(f"[FleetState] → max_lateness_minutes = {fleet_state.max_lateness_minutes}min")
            
            if 'max_pickup_delay_minutes' in config:
                fleet_state.max_pickup_delay_minutes = int(config['max_pickup_delay_minutes'])
                print(f"[FleetState] → max_pickup_delay_minutes = {fleet_state.max_pickup_delay_minutes}min")
            
            if 'wallet_threshold' in config:
                fleet_state.wallet_threshold = float(config['wallet_threshold'])
                print(f"[FleetState] → wallet_threshold = ${fleet_state.wallet_threshold}")
            
            if 'max_tasks_per_agent' in config:
                new_capacity = int(config['max_tasks_per_agent'])
                old_capacity = fleet_state.default_max_capacity
                fleet_state.default_max_capacity = new_capacity
                print(f"[FleetState] → max_tasks_per_agent = {new_capacity}")
                # Update all existing agents that still have the old default
                for agent in fleet_state.get_all_agents():
                    if agent.max_capacity == old_capacity:
                        agent.max_capacity = new_capacity
            if 'wallet_threshold' in config:
                fleet_state.wallet_threshold = float(config['wallet_threshold'])
            print(f"[FleetState] Applied config: max_dist={fleet_state.max_distance_km}km, chain_lookahead={fleet_state.chain_lookahead_radius_km}km, capacity={fleet_state.default_max_capacity}, wallet={fleet_state.wallet_threshold}")
        
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
            'config_applied': {
                'max_distance_km': fleet_state.max_distance_km,
                'max_lateness_minutes': fleet_state.max_lateness_minutes,
                'max_pickup_delay_minutes': fleet_state.max_pickup_delay_minutes,
                'wallet_threshold': fleet_state.wallet_threshold,
                'max_tasks_per_agent': fleet_state.default_max_capacity,
                'chain_lookahead_radius_km': fleet_state.chain_lookahead_radius_km
            },
            'fleet_stats': stats
        })
        
        print(f"[FleetState] ✅ Initial sync complete: {stats['online_agents']} agents, {stats['unassigned_tasks']} tasks")
        print(f"[FleetState] Config: max_dist={fleet_state.max_distance_km}km, max_lateness={fleet_state.max_lateness_minutes}min, wallet=${fleet_state.wallet_threshold}")
        
    except Exception as e:
        print(f"[FleetState] ❌ Sync failed: {e}")
        import traceback
        traceback.print_exc()
        emit('fleet:sync_ack', {
            'success': False,
            'error': str(e),
            'received_at': datetime.now().isoformat()
        })
    finally:
        # Always unblock optimizations when sync is done (success or failure)
        set_sync_in_progress(False)

@socketio.on('config:update')
def handle_config_update(data):
    """
    Update fleet configuration dynamically.
    
    Payload:
    {
        "config": {
            "default_max_distance_km": 3.0,
            "max_lateness_minutes": 45,
            "max_pickup_delay_minutes": 35,
            "wallet_threshold": 2500,
            "max_tasks_per_agent": 2,
            "chain_lookahead_radius_km": 5.0
        },
        "dashboard_url": "https://..."
    }
    """
    performance_stats["websocket_events"] += 1
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
    print(f"[WebSocket] config:update received from dashboard")
    
    if not FLEET_STATE_AVAILABLE or not fleet_state:
        emit('config:update_ack', {
            'success': False,
            'error': 'Fleet state not available',
            'received_at': datetime.now().isoformat()
        })
        return
    
    config = data.get('config', {})
    if not config:
        emit('config:update_ack', {
            'success': False,
            'error': 'No config provided',
            'received_at': datetime.now().isoformat()
        })
        return
    
    try:
        changes = []
        
        if 'default_max_distance_km' in config:
            old_val = fleet_state.max_distance_km
            max_dist = float(config['default_max_distance_km'])
            fleet_state.max_distance_km = max_dist
            fleet_state.assignment_radius_km = max_dist  # Keep them in sync
            changes.append(f"max_distance_km: {old_val} → {max_dist}")
        
        if 'max_lateness_minutes' in config:
            old_val = fleet_state.max_lateness_minutes
            fleet_state.max_lateness_minutes = int(config['max_lateness_minutes'])
            changes.append(f"max_lateness_minutes: {old_val} → {fleet_state.max_lateness_minutes}")
        
        if 'max_pickup_delay_minutes' in config:
            old_val = fleet_state.max_pickup_delay_minutes
            fleet_state.max_pickup_delay_minutes = int(config['max_pickup_delay_minutes'])
            changes.append(f"max_pickup_delay_minutes: {old_val} → {fleet_state.max_pickup_delay_minutes}")
        
        if 'wallet_threshold' in config:
            old_val = fleet_state.wallet_threshold
            fleet_state.wallet_threshold = float(config['wallet_threshold'])
            changes.append(f"wallet_threshold: {old_val} → {fleet_state.wallet_threshold}")
        
        if 'max_tasks_per_agent' in config:
            old_val = fleet_state.default_max_capacity
            new_capacity = int(config['max_tasks_per_agent'])
            fleet_state.default_max_capacity = new_capacity
            # Update all existing agents that still have the old default
            for agent in fleet_state.get_all_agents():
                if agent.max_capacity == old_val:
                    agent.max_capacity = new_capacity
            changes.append(f"max_tasks_per_agent: {old_val} → {new_capacity}")
        
        if 'chain_lookahead_radius_km' in config:
            old_val = fleet_state.chain_lookahead_radius_km
            fleet_state.chain_lookahead_radius_km = float(config['chain_lookahead_radius_km'])
            changes.append(f"chain_lookahead_radius_km: {old_val} → {fleet_state.chain_lookahead_radius_km}")
        
        # Log changes
        if changes:
            print(f"[FleetState] ✅ Config updated:")
            for change in changes:
                print(f"  → {change}")
        else:
            print(f"[FleetState] ℹ️ No config changes applied")
        
        emit('config:update_ack', {
            'success': True,
            'received_at': datetime.now().isoformat(),
            'changes': changes,
            'config_applied': {
                'max_distance_km': fleet_state.max_distance_km,
                'max_lateness_minutes': fleet_state.max_lateness_minutes,
                'max_pickup_delay_minutes': fleet_state.max_pickup_delay_minutes,
                'wallet_threshold': fleet_state.wallet_threshold,
                'max_tasks_per_agent': fleet_state.default_max_capacity,
                'chain_lookahead_radius_km': fleet_state.chain_lookahead_radius_km
            }
        })
        
    except Exception as e:
        print(f"[FleetState] ❌ Config update failed: {e}")
        import traceback
        traceback.print_exc()
        emit('config:update_ack', {
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
    
    Uses THOROUGH mode (prefilter_distance=False) for comprehensive optimization.
    """
    client_id = request.sid
    performance_stats["websocket_events"] += 1
    
    if client_id in connected_clients:
        connected_clients[client_id]['events_received'] += 1
    
    request_id = data.get('request_id', str(uuid.uuid4()))
    print(f"[WebSocket] fleet:optimize_request from {client_id}, request_id={request_id}")
    
    # Block if sync is in progress
    if is_sync_in_progress():
        print(f"[WebSocket] ⚠️ Sync in progress - deferring optimize request")
        emit('fleet:routes_updated', {
            'trigger_type': 'manual',
            'trigger_event': 'fleet:optimize_request',
            'request_id': request_id,
            'error': 'Sync in progress - please wait and retry',
            'success': False,
            'sync_in_progress': True,
            'agent_routes': [],
            'unassigned_tasks': []
        })
        return
    
    print(f"[WebSocket] Mode: THOROUGH (solver handles distance)")
    
    try:
        start_time = time.time()
        # ON-DEMAND: Use THOROUGH mode - solver handles distance constraints
        result = process_fleet_optimization(data, prefilter_distance=False)
        execution_time = time.time() - start_time
        
        # ON-DEMAND: Manual request from dashboard
        result['trigger_type'] = 'manual'  # Clear identifier: 'event', 'proximity', or 'manual'
        result['trigger_event'] = 'fleet:optimize_request'
        result['request_id'] = request_id
        result['total_execution_time_seconds'] = round(execution_time, 3)
        
        # OPTIMISTIC UPDATE: Mark tasks as assigned immediately to prevent race conditions
        for route in result.get('agent_routes', []):
            agent_id = route.get('driver_id')
            agent_name = route.get('driver_name', 'Unknown')
            new_tasks = route.get('assigned_new_tasks', [])
            if new_tasks and agent_id and fleet_state:
                for assigned_task_id in new_tasks:
                    fleet_state.assign_task(assigned_task_id, str(agent_id), agent_name)
        
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
    
    print(f"[WebSocket] task:created: {str(task_id)[:20]}...")
    
    # Update fleet state
    triggered_optimization = False
    if FLEET_STATE_AVAILABLE and fleet_state and task_data:
        # Check for duplicate event - if task already exists, skip optimization
        existing_task = fleet_state.get_task(str(task_id))
        if existing_task:
            print(f"[FleetState] ⚠️ Duplicate task:created event for {existing_task.restaurant_name} - skipping")
            emit('task:created_ack', {'id': task_id, 'duplicate': True})
            return
        
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
        'triggered_optimization': triggered_optimization
    })
    
    # EVENT-BASED: Only trigger if proximity didn't already assign the task
    # Check if task is still unassigned after proximity optimization
    task_after = fleet_state.get_task(str(task_id)) if fleet_state else None
    if task_after and task_after.status == TaskStatus.UNASSIGNED:
        trigger_fleet_optimization('task:created', {
            'id': task_id,
            'dashboard_url': dashboard_url
        })
    elif task_after:
        print(f"[FleetState] ⏩ Skipping event-based optimization - task already {task_after.status.value}")

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
    
    print(f"[WebSocket] task:declined: {str(task_id)[:20]}... by {latest_agent_name} (total: {len(declined_by)} declines)")
    
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
    
    # EVENT-BASED: Only try to reassign if proximity didn't already assign it
    task_after_proximity = fleet_state.get_task(str(task_id)) if (FLEET_STATE_AVAILABLE and fleet_state and task_id) else None
    if task_after_proximity and task_after_proximity.status == TaskStatus.UNASSIGNED:
        trigger_fleet_optimization('task:declined', {
            'id': task_id,
            'declined_by': declined_by,
            'dashboard_url': dashboard_url
        })
    elif task_after_proximity:
        print(f"[FleetState] ℹ️ Task {str(task_id)[:20]}... already assigned by proximity, skipping event-based optimization.")

@socketio.on('task:completed')
def handle_task_completed(data):
    """
    Task completed → Update state, agent has capacity for more work.
    Payload: { id, agent_id, agent_name, completed_at, job_type, dashboard_url }
    
    IMPORTANT: job_type determines what "completed" means:
    - job_type=0 (PICKUP): Just pickup done - DON'T remove task, update pickup_completed
    - job_type=1 (DELIVERY): Full task done - remove from agent, free up capacity
    - job_type not provided: Assume DELIVERY for backward compatibility
    """
    performance_stats["websocket_events"] += 1
    task_id = data.get('id', '')
    agent_id = data.get('agent_id')
    agent_name = data.get('agent_name', 'Unknown')
    job_type = data.get('job_type')  # 0=pickup, 1=delivery, None=assume delivery
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
    # Determine if this is a pickup or delivery completion
    is_pickup_completion = (job_type == 0 or job_type == '0')
    
    if is_pickup_completion:
        print(f"[WebSocket] task:pickup_completed: {str(task_id)[:20]}... by {agent_name} ({agent_id})")
        
        # Just mark pickup as complete - DON'T remove task from agent
        if FLEET_STATE_AVAILABLE and fleet_state and task_id:
            task = fleet_state.mark_pickup_complete(str(task_id))
            if task:
                print(f"[FleetState] Pickup completed: {task.restaurant_name} (agent still busy with delivery)")
        
        emit('task:completed_ack', {
            'id': task_id,
            'agent_id': agent_id,
            'received_at': datetime.now().isoformat(),
            'task_removed': False,
            'pickup_completed': True
        })
    else:
        print(f"[WebSocket] task:completed: {str(task_id)[:20]}... by {agent_name} ({agent_id})")
        
        # Full task completion - remove from agent
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
        
        # EVENT-BASED: Trigger optimization if there are unassigned tasks and agent has capacity
        # Note: Proximity triggers may also fire from agent:location_update - debouncing handles dedup
        unassigned_tasks = fleet_state.get_unassigned_tasks()
        unassigned_count = len(unassigned_tasks)
        if unassigned_count > 0:
            agent = fleet_state.get_agent(str(agent_id)) if agent_id else None
            if agent and agent.has_capacity:
                print(f"[FleetState] {agent_name} completed task, has capacity, {unassigned_count} unassigned tasks - triggering optimization")
                # Use debounced optimization - will cancel if proximity triggers shortly after
                trigger_debounced_optimization(
                    trigger_type='task:completed',
                    dashboard_url=dashboard_url,
                    agent_id=str(agent_id)
                )
            else:
                print(f"[FleetState] {agent_name} completed task but at capacity - skipping optimization")
        else:
            print(f"[FleetState] {agent_name} completed task - no unassigned tasks to assign")

@socketio.on('pickup:completed')
def handle_pickup_completed(data):
    """
    Pickup completed → Update task, but agent still needs to do delivery.
    Payload: { id, agent_id, agent_name, completed_at, dashboard_url }
    
    This is important for route optimization - once pickup is done,
    the optimizer only needs to route to the delivery location.
    """
    performance_stats["websocket_events"] += 1
    task_id = data.get('id', '')
    agent_id = data.get('agent_id')
    agent_name = data.get('agent_name', 'Unknown')
    
    print(f"[WebSocket] pickup:completed: {str(task_id)[:20]}... by {agent_name} ({agent_id})")
    
    # Mark pickup as complete - agent still busy with delivery
    if FLEET_STATE_AVAILABLE and fleet_state and task_id:
        task = fleet_state.mark_pickup_complete(str(task_id))
        if task:
            print(f"[FleetState] ✓ Pickup done for {task.restaurant_name} → now delivering")
    
    emit('pickup:completed_ack', {
        'id': task_id,
        'agent_id': agent_id,
        'received_at': datetime.now().isoformat(),
        'pickup_completed': True,
        'task_removed': False
    })

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
    
    print(f"[WebSocket] task:cancelled: {str(task_id)[:20]}... (reason: {reason})")
    
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
    Task updated → Full task object with updated values.
    
    Payload (full task object):
    {
        "id": "task-order-id",
        "job_type": "PAIRED",
        "restaurant_location": [17.123, -61.845],
        "delivery_location": [17.130, -61.850],
        "pickup_before": "2025-12-18T17:00:00.000Z",
        "delivery_before": "2025-12-18T17:30:00.000Z",
        "tags": ["PRIORITY"],
        "payment_method": "prepaid",
        "delivery_fee": 5.00,
        "tips": 2.00,
        "max_distance_km": 3,
        "declined_by": ["agent-123"],
        "status": "Unassigned",
        "assigned_agent_id": null,
        "pickup_completed": false,
        "_meta": { "restaurant_name": "Hey Pizza", ... },
        "dashboard_url": "https://..."
    }
    """
    performance_stats["websocket_events"] += 1
    task_id = data.get('id', '')
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
    if not FLEET_STATE_AVAILABLE or not fleet_state:
        emit('task:updated_ack', {
            'id': task_id,
            'success': False,
            'error': 'Fleet state not available',
            'received_at': datetime.now().isoformat()
        })
        return
    
    # Get existing task to compare what changed
    existing_task = fleet_state.get_task(task_id)
    if not existing_task:
        print(f"[WebSocket] task:updated: {str(task_id)[:20]}... (task not found in fleet state)")
        emit('task:updated_ack', {
            'id': task_id,
            'success': False,
            'error': 'Task not found',
            'received_at': datetime.now().isoformat()
        })
        return
    
    # Track what fields changed
    changes = []
    status_changed_to_unassigned = False
    
    # Update task fields
    meta = data.get('_meta', {})
    restaurant_name = meta.get('restaurant_name', existing_task.restaurant_name)
    
    # CRITICAL: Handle status changes first (before other field updates)
    # Status changes can unassign tasks or mark them as declined
    new_status = data.get('status', '').lower() if data.get('status') else None
    new_assigned_agent = data.get('assigned_agent_id')
    old_assigned_agent = existing_task.assigned_agent_id
    
    if new_status:
        # Check if task is being unassigned (status changed to unassigned/declined)
        if new_status in ['unassigned', 'declined']:
            # If task was previously assigned, remove it from that agent
            if old_assigned_agent:
                old_agent = fleet_state.get_agent(str(old_assigned_agent))
                if old_agent:
                    # Remove task from agent's current_tasks
                    old_agent.current_tasks = [t for t in old_agent.current_tasks if t.task_id != task_id]
                    # Recalculate agent status
                    if not old_agent.current_tasks:
                        old_agent.status = AgentStatus.IDLE
                    print(f"[FleetState] Task unassigned from {old_agent.name} (status → {new_status})")
                
                # Clear the assignment on the task
                existing_task.assigned_agent_id = None
                existing_task.status = TaskStatus.UNASSIGNED
                changes.append(f'status:{new_status}')
                status_changed_to_unassigned = True
        
        # Check if task is being reassigned to a different agent
        elif new_assigned_agent and str(new_assigned_agent) != str(old_assigned_agent or ''):
            # First unassign from old agent if any
            if old_assigned_agent:
                old_agent = fleet_state.get_agent(str(old_assigned_agent))
                if old_agent:
                    old_agent.current_tasks = [t for t in old_agent.current_tasks if t.task_id != task_id]
                    if not old_agent.current_tasks:
                        old_agent.status = AgentStatus.IDLE
                    print(f"[FleetState] Task removed from {old_agent.name} (reassigning)")
            
            # Then assign to new agent
            new_agent = fleet_state.get_agent(str(new_assigned_agent))
            if new_agent:
                fleet_state.assign_task(task_id, str(new_assigned_agent), new_agent.name)
                changes.append(f'reassigned_to:{new_agent.name}')
            else:
                # Agent not in fleet state, just update the task's assigned_agent_id
                existing_task.assigned_agent_id = str(new_assigned_agent)
                existing_task.status = TaskStatus.ASSIGNED
                changes.append(f'assigned_agent_id:{new_assigned_agent}')
    
    # Location updates
    if 'restaurant_location' in data and data['restaurant_location']:
        loc = data['restaurant_location']
        if [existing_task.restaurant_location.lat, existing_task.restaurant_location.lng] != loc:
            existing_task.restaurant_location.lat = loc[0]
            existing_task.restaurant_location.lng = loc[1]
            changes.append('restaurant_location')
    
    if 'delivery_location' in data and data['delivery_location']:
        loc = data['delivery_location']
        if [existing_task.delivery_location.lat, existing_task.delivery_location.lng] != loc:
            existing_task.delivery_location.lat = loc[0]
            existing_task.delivery_location.lng = loc[1]
            changes.append('delivery_location')
    
    # Time window updates
    if 'pickup_before' in data and data['pickup_before']:
        try:
            new_time = fleet_state._parse_datetime(data['pickup_before'])
            if new_time and existing_task.pickup_before != new_time:
                existing_task.pickup_before = new_time
                changes.append('pickup_before')
        except Exception:
            pass
    
    if 'delivery_before' in data and data['delivery_before']:
        try:
            new_time = fleet_state._parse_datetime(data['delivery_before'])
            if new_time and existing_task.delivery_before != new_time:
                existing_task.delivery_before = new_time
                changes.append('delivery_before')
        except Exception:
            pass
    
    # Business rule updates
    if 'tags' in data:
        new_tags = data['tags'] or []
        if existing_task.tags != new_tags:
            existing_task.tags = new_tags
            changes.append('tags')
    
    if 'payment_method' in data:
        new_payment = data['payment_method'] or 'card'
        if existing_task.payment_method != new_payment:
            existing_task.payment_method = new_payment
            changes.append('payment_method')
    
    if 'delivery_fee' in data:
        new_fee = float(data['delivery_fee'] or 0)
        if existing_task.delivery_fee != new_fee:
            existing_task.delivery_fee = new_fee
            changes.append('delivery_fee')
    
    if 'tips' in data:
        new_tips = float(data['tips'] or 0)
        if existing_task.tips != new_tips:
            existing_task.tips = new_tips
            changes.append('tips')
    
    if 'max_distance_km' in data and data['max_distance_km'] is not None:
        new_dist = float(data['max_distance_km'])
        if existing_task.max_distance_km != new_dist:
            existing_task.max_distance_km = new_dist
            changes.append('max_distance_km')
    
    if 'declined_by' in data:
        new_declined = set(str(d) for d in (data['declined_by'] or []))
        if existing_task.declined_by != new_declined:
            existing_task.declined_by = new_declined
            changes.append('declined_by')
    
    if 'pickup_completed' in data:
        new_pickup = bool(data['pickup_completed'])
        if existing_task.pickup_completed != new_pickup:
            existing_task.pickup_completed = new_pickup
            changes.append('pickup_completed')
    
    # Meta updates
    if meta.get('restaurant_name') and existing_task.restaurant_name != meta['restaurant_name']:
        existing_task.restaurant_name = meta['restaurant_name']
        changes.append('restaurant_name')
    
    if meta.get('customer_name') and existing_task.customer_name != meta['customer_name']:
        existing_task.customer_name = meta['customer_name']
        changes.append('customer_name')
    
    # Update timestamp
    existing_task.last_updated = datetime.now(timezone.utc)
    
    # Log the update
    changes_str = ', '.join(changes) if changes else 'no changes'
    print(f"[WebSocket] task:updated: {restaurant_name} ({str(task_id)[:20]}...) → {changes_str}")
    
    emit('task:updated_ack', {
        'id': task_id,
        'success': True,
        'changes': changes,
        'status_changed_to_unassigned': status_changed_to_unassigned,
        'received_at': datetime.now().isoformat()
    })
    
    # If task was unassigned, trigger optimization to reassign it
    if status_changed_to_unassigned:
        print(f"[FleetState] Task {restaurant_name} now unassigned - triggering optimization")
        # Use the same pattern as task:declined - debounced event-based optimization
        trigger_debounced_optimization(
            trigger_type='task:updated_unassigned',
            dashboard_url=dashboard_url,
            task_id=task_id
        )

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
    
    # Update fleet state (returns None if already assigned to this agent)
    task = None
    if FLEET_STATE_AVAILABLE and fleet_state and task_id and agent_id:
        task = fleet_state.assign_task(task_id, str(agent_id), agent_name)
    
    # Only log if this was a new assignment (not a duplicate)
    if task:
        print(f"[WebSocket] task:assigned: {str(task_id)[:20]}... → {agent_name} ({agent_id})")
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
    
    print(f"[WebSocket] task:accepted: {str(task_id)[:20]}... by {agent_name} ({agent_id})")
    
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
    
    Payload (same format as sync agents):
    {
        "agent_id": "123456",       // or "id"
        "name": "John Smith",
        "location": [17.125, -61.827],
        "max_capacity": 2,
        "tags": ["NoCash", "scooter"],
        "wallet_balance": 1500.00,
        "priority": 1,              // Optional: only for priority agents
        "dashboard_url": "..."
    }
    """
    performance_stats["websocket_events"] += 1
    
    # Parse agent data (support both 'agent_id' and 'id' keys)
    agent_id = data.get('agent_id') or data.get('id')
    name = data.get('name', 'Unknown')
    location = data.get('location')
    priority = data.get('priority')
    max_capacity = data.get('max_capacity')
    tags = data.get('tags', [])
    wallet_balance = data.get('wallet_balance')
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
    # Build log string with key info
    priority_str = f" [Priority {priority}]" if priority else ""
    tags_str = f" tags={tags}" if tags else ""
    print(f"[WebSocket] agent:online: {name} ({agent_id}){priority_str}{tags_str}")
    
    # Update fleet state with full agent profile
    if FLEET_STATE_AVAILABLE and fleet_state and agent_id:
        loc_tuple = tuple(location) if location and len(location) >= 2 else None
        agent = fleet_state.set_agent_online(
            str(agent_id),
            name=name,
            location=loc_tuple,
            priority=priority,
            max_capacity=int(max_capacity) if max_capacity else None,
            tags=tags,
            wallet_balance=float(wallet_balance) if wallet_balance is not None else None
        )
        if agent:
            print(f"[FleetState] Agent online: {name} at {agent.current_location}{priority_str} capacity={agent.max_capacity}")
            
            # Check for nearby tasks
            nearby_tasks = fleet_state.find_tasks_near_agent(str(agent_id))
            if nearby_tasks:
                nearest, dist = nearby_tasks[0]
                print(f"[FleetState] 🎯 {name} has {len(nearby_tasks)} nearby tasks, nearest: {nearest.restaurant_name} ({dist:.2f}km)")
    
    # Only trigger optimization if there are unassigned tasks
    has_unassigned = False
    if FLEET_STATE_AVAILABLE and fleet_state:
        unassigned_count = fleet_state.get_unassigned_task_count()
        has_unassigned = unassigned_count > 0
        if not has_unassigned:
            print(f"[FleetState] No unassigned tasks - skipping optimization for {name}")
    
    emit('agent:online_ack', {
        'agent_id': agent_id,
        'name': name,
        'received_at': datetime.now().isoformat(),
        'agent_added': True,
        'triggered_optimization': has_unassigned
    })
    
    # EVENT-BASED: Only try to give this agent work if there are unassigned tasks
    if has_unassigned:
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


@socketio.on('agent:update')
def handle_agent_update(data):
    """
    Agent profile updated → Update their settings in fleet state.
    
    Payload:
    {
        "agent_id": "2068032",
        "name": "Esahtengang Asonganyi",
        "max_capacity": 3,
        "tags": ["INTERNAL", "HEAVY"],
        "priority": 1,
        "wallet_balance": 500.00,
        "dashboard_url": "https://..."
    }
    """
    performance_stats["websocket_events"] += 1
    
    agent_id = data.get('agent_id') or data.get('id')
    name = data.get('name')
    max_capacity = data.get('max_capacity')
    tags = data.get('tags')
    priority = data.get('priority')
    wallet_balance = data.get('wallet_balance')
    
    # Build update summary for logging
    updates = []
    if max_capacity is not None:
        updates.append(f"capacity={max_capacity}")
    if tags is not None:
        updates.append(f"tags={tags}")
    if priority is not None:
        updates.append(f"priority={priority}")
    if wallet_balance is not None:
        updates.append(f"wallet=${wallet_balance}")
    
    update_str = ", ".join(updates) if updates else "no changes"
    print(f"[WebSocket] agent:update: {name or agent_id} → {update_str}")
    
    # Update fleet state
    updated = False
    if FLEET_STATE_AVAILABLE and fleet_state and agent_id:
        agent = fleet_state.update_agent(
            str(agent_id),
            name=name,
            max_capacity=int(max_capacity) if max_capacity is not None else None,
            tags=tags,
            priority=priority,
            wallet_balance=float(wallet_balance) if wallet_balance is not None else None
        )
        if agent:
            updated = True
            print(f"[FleetState] ✓ Updated {agent.name}: capacity={agent.max_capacity}, tags={agent.tags}")
        else:
            print(f"[FleetState] ⚠️ Agent {agent_id} not found in fleet state")
    
    emit('agent:update_ack', {
        'agent_id': agent_id,
        'received_at': datetime.now().isoformat(),
        'updated': updated
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
    print(f"[DEBUG] *** RECEIVED agent:location_update event ***", flush=True)
    performance_stats["websocket_events"] += 1
    agent_id = data.get('agent_id')
    name = data.get('name', 'Unknown')
    location = data.get('location', [])
    print(f"[DEBUG] Agent: {name} ({agent_id}), Location: {location}", flush=True)
    
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
        ineligible_triggers = [t for t in triggers if not t.is_eligible]
        
        # DEBUG: Log trigger details
        if triggers:
            print(f"[DEBUG] {name}: {len(triggers)} triggers, {len(eligible_triggers)} eligible, {len(ineligible_triggers)} ineligible")
            if ineligible_triggers:
                reasons = set(t.eligibility_reason for t in ineligible_triggers)
                print(f"[DEBUG] Ineligibility reasons: {reasons}")
        
        if eligible_triggers:
            # Get the best trigger (closest task)
            best_trigger = min(eligible_triggers, key=lambda t: t.distance_km)
            print(f"[DEBUG] Best trigger: {name} → {best_trigger.task.restaurant_name} ({best_trigger.distance_km:.2f}km)")
            
            # Check cooldown
            can_trigger = fleet_state.should_trigger_optimization(agent_id)
            print(f"[DEBUG] Cooldown check for {name}: {'PASSED' if can_trigger else 'ON COOLDOWN'}")
            
            if can_trigger:
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
    try:
        start_time = time.time()
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
        'max_distance_km': fleet_state.max_distance_km,  # Assignment trigger radius
        'triggers': result
    }), 200


@app.route('/fleet-state/config', methods=['GET', 'POST'])
def fleet_state_config():
    """Get or update fleet state configuration."""
    if not FLEET_STATE_AVAILABLE or not fleet_state:
        return jsonify({"error": "Fleet state not available"}), 503
    
    if request.method == 'GET':
        return jsonify({
            'max_distance_km': fleet_state.max_distance_km,  # Also used as assignment trigger radius
            'max_lateness_minutes': fleet_state.max_lateness_minutes,  # Max allowed delivery lateness
            'max_pickup_delay_minutes': fleet_state.max_pickup_delay_minutes,  # Max delay after food ready
            'wallet_threshold': fleet_state.wallet_threshold,  # Max wallet balance for cash orders
            'max_tasks_per_agent': fleet_state.default_max_capacity,  # Agent capacity limit
            'chain_lookahead_radius_km': fleet_state.chain_lookahead_radius_km,
            'optimization_cooldown_seconds': fleet_state.optimization_cooldown_seconds
        }), 200
    
    # POST - update config
    data = request.get_json() or {}
    
    # max_distance and assignment_radius are now consolidated
    if 'max_distance_km' in data:
        max_dist = float(data['max_distance_km'])
        fleet_state.max_distance_km = max_dist
        fleet_state.assignment_radius_km = max_dist  # Keep them in sync
    if 'assignment_radius_km' in data:
        # Legacy support - also updates max_distance
        radius = float(data['assignment_radius_km'])
        fleet_state.assignment_radius_km = radius
        fleet_state.max_distance_km = radius
    if 'max_lateness_minutes' in data:
        fleet_state.max_lateness_minutes = int(data['max_lateness_minutes'])
    if 'max_pickup_delay_minutes' in data:
        fleet_state.max_pickup_delay_minutes = int(data['max_pickup_delay_minutes'])
    if 'wallet_threshold' in data:
        fleet_state.wallet_threshold = float(data['wallet_threshold'])
    if 'max_tasks_per_agent' in data:
        fleet_state.default_max_capacity = int(data['max_tasks_per_agent'])
    if 'chain_lookahead_radius_km' in data:
        fleet_state.chain_lookahead_radius_km = float(data['chain_lookahead_radius_km'])
    if 'optimization_cooldown_seconds' in data:
        fleet_state.optimization_cooldown_seconds = float(data['optimization_cooldown_seconds'])
    
    return jsonify({
        'message': 'Configuration updated',
        'max_distance_km': fleet_state.max_distance_km,
        'max_lateness_minutes': fleet_state.max_lateness_minutes,
        'max_pickup_delay_minutes': fleet_state.max_pickup_delay_minutes,
        'wallet_threshold': fleet_state.wallet_threshold,
        'max_tasks_per_agent': fleet_state.default_max_capacity,
        'chain_lookahead_radius_km': fleet_state.chain_lookahead_radius_km,
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
