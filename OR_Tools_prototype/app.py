from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from typing import List, Optional
import json
import os
import time
from datetime import datetime, timezone
import uuid
import threading
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# =============================================================================
# PERSISTENT FILE LOGGING
# =============================================================================
# Create logs directory if it doesn't exist
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Configure file handler for ALL logs (rotating, max 10MB, keep 5 backups)
all_log_file = os.path.join(LOG_DIR, 'fleet_optimizer.log')
file_handler = RotatingFileHandler(all_log_file, maxBytes=10*1024*1024, backupCount=5)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

# Configure file handler for IMPORTANT events only (assignments, errors, syncs)
important_log_file = os.path.join(LOG_DIR, 'important_events.log')
important_handler = RotatingFileHandler(important_log_file, maxBytes=5*1024*1024, backupCount=3)
important_handler.setLevel(logging.INFO)
important_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

# Create loggers
logger = logging.getLogger('fleet_optimizer')
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(important_handler)

# Also add console handler so we still see logs in terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(console_handler)

# Helper function for logging (replaces print statements for important events)
def log_event(message: str, level: str = 'info'):
    """Log an event to both file and console."""
    if level == 'debug':
        logger.debug(message)
    elif level == 'warning':
        logger.warning(message)
    elif level == 'error':
        logger.error(message)
    else:
        logger.info(message)

def log_payload(event_name: str, data: dict, max_items: int = 5):
    """
    Log WebSocket event payload to persistent logs.
    Truncates large arrays and long strings to keep logs readable.
    """
    import json
    
    def truncate_value(v, max_len=100):
        """Truncate long strings"""
        if isinstance(v, str) and len(v) > max_len:
            return v[:max_len] + '...'
        return v
    
    def summarize_payload(payload, depth=0):
        """Create a summary of the payload, truncating large arrays"""
        if depth > 2:  # Don't go too deep
            return str(payload)[:100] + '...' if len(str(payload)) > 100 else payload
            
        if isinstance(payload, dict):
            result = {}
            for k, v in payload.items():
                if isinstance(v, list):
                    if len(v) > max_items:
                        result[k] = f"[{len(v)} items: {summarize_payload(v[:2], depth+1)}...]"
                    else:
                        result[k] = [summarize_payload(item, depth+1) for item in v]
                elif isinstance(v, dict):
                    result[k] = summarize_payload(v, depth+1)
                else:
                    result[k] = truncate_value(v)
            return result
        elif isinstance(payload, list):
            if len(payload) > max_items:
                return [summarize_payload(item, depth+1) for item in payload[:max_items]] + [f'...+{len(payload)-max_items} more']
            return [summarize_payload(item, depth+1) for item in payload]
        else:
            return truncate_value(payload)
    
    try:
        summary = summarize_payload(data)
        payload_str = json.dumps(summary, default=str, ensure_ascii=False)
        logger.info(f"[Payload] {event_name}: {payload_str}")
    except Exception as e:
        logger.warning(f"[Payload] {event_name}: Could not serialize payload: {e}")

print(f"[Logging] Logs saved to: {LOG_DIR}")
print(f"[Logging] All logs: fleet_optimizer.log")
print(f"[Logging] Important events: important_events.log")

# Import Fleet State (Abstract Map)
try:
    from fleet_state import fleet_state, AgentStatus, TaskStatus
    FLEET_STATE_AVAILABLE = True
    print("[FleetState] Abstract Map loaded successfully")
    
    # Connect fleet_state logger to our file handlers for persistent logging
    fleet_state_logger = logging.getLogger('fleet_state')
    fleet_state_logger.setLevel(logging.DEBUG)  # Enable DEBUG level for decline tracking
    fleet_state_logger.addHandler(file_handler)
    fleet_state_logger.addHandler(important_handler)
    
    # Connect fleet_optimizer logger to file handlers as well
    fleet_optimizer_logger = logging.getLogger('fleet_optimizer')
    fleet_optimizer_logger.setLevel(logging.DEBUG)  # Enable DEBUG level for decline tracking
    fleet_optimizer_logger.addHandler(file_handler)
    fleet_optimizer_logger.addHandler(important_handler)
    
    print("[FleetState] Loggers connected to file handlers")
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

# GLOBAL optimization lock - prevents ANY optimization from running concurrently
# This includes both event-based (full fleet) and proximity-based (single agent)
_optimization_running = False
_optimization_running_lock = threading.Lock()
_events_queued_during_optimization = []  # List of event types that arrived during optimization
_queued_dashboard_url = None  # Dashboard URL to use for queued optimization
_queued_request_id = None  # Request ID for queued manual optimization

# =============================================================================
# MARKETPLACE BROADCAST SYSTEM
# =============================================================================
# Fleet-wide task marketplace: ALL unassigned tasks are broadcast to ALL 
# eligible agents simultaneously. Each agent sees all tasks they can take.
# First to accept wins (Tookan handles the race condition).
#
# Mode can be toggled from dashboard via fleet:sync settings.
# Tasks have countdown timers - if not accepted, marketplace re-broadcasts.
# =============================================================================

# Proximity Broadcast mode tracking
_proximity_lock = threading.Lock()
_last_proximity_broadcast = {}  # Track last broadcast per task: {task_id: timestamp}
_task_offer_times = {}  # Track when each task was first offered: {task_id: timestamp}
_task_expanded_radius = {}  # Track expanded radius per task: {task_id: radius_km}
_task_current_agents = {}  # Track current feasible agents per task: {task_id: set(agent_ids)}
_agent_pending_broadcasts = {}  # Track pending broadcasts per agent: {agent_id: set(task_ids)}
_agent_pending_directions = {}  # Track pending broadcast directions: {agent_id: {task_id: bearing_degrees}}
_task_agent_broadcast_counts = {}  # Track broadcast counts: {task_id: {agent_id: count}}


def get_agent_broadcast_capacity(agent_id: str) -> int:
    """
    Get how many more tasks can be broadcast to this agent.
    Returns: max_tasks - current_tasks - pending_broadcasts
    """
    if not fleet_state:
        return 0
    
    agent = fleet_state.get_agent(agent_id)
    if not agent:
        return 0
    
    with _proximity_lock:
        pending = len(_agent_pending_broadcasts.get(agent_id, set()))
    
    capacity = agent.max_capacity - len(agent.current_tasks) - pending
    return max(0, capacity)


def add_pending_broadcast(agent_id: str, task_id: str, bearing: float = None):
    """
    Track that a task has been broadcast to an agent (pending acceptance).
    Also stores the task's delivery bearing for directional compatibility checks.
    """
    with _proximity_lock:
        if agent_id not in _agent_pending_broadcasts:
            _agent_pending_broadcasts[agent_id] = set()
        _agent_pending_broadcasts[agent_id].add(task_id)
        
        # Store direction if provided
        if bearing is not None:
            if agent_id not in _agent_pending_directions:
                _agent_pending_directions[agent_id] = {}
            _agent_pending_directions[agent_id][task_id] = bearing


# Track pending proactive checks to prevent duplicate checks
_pending_proactive_checks = set()  # Set of (agent_id, task_id) tuples currently being checked


def trigger_perceived_projected_check(agent_id: str, broadcast_task_id: str, dashboard_url: str):
    """
    PROACTIVE CHECK: After broadcasting a task to an agent, check if there are
    EXISTING unassigned tasks near the agent's NEW perceived projected location
    (the delivery location of the broadcast task).
    
    This catches the edge case where:
    1. Task B already exists (no agents nearby at creation)
    2. Task A is broadcast to Agent (delivers near Task B's pickup)
    3. Now we should add Task B to the agent's batch
    
    Args:
        agent_id: The agent who received the broadcast
        broadcast_task_id: The task that was just broadcast
        dashboard_url: Dashboard URL for callbacks
    """
    global _pending_proactive_checks
    
    # Prevent duplicate checks for same agent+task
    check_key = (agent_id, broadcast_task_id)
    if check_key in _pending_proactive_checks:
        return
    _pending_proactive_checks.add(check_key)
    
    try:
        if not fleet_state:
            return
        
        agent = fleet_state.get_agent(agent_id)
        if not agent or not agent.is_online:
            return
        
        # Check broadcast capacity
        broadcast_capacity = get_agent_broadcast_capacity(agent_id)
        if broadcast_capacity <= 0:
            return  # Agent is at capacity, no room for more tasks
        
        # Get the broadcast task to find its delivery location
        broadcast_task = fleet_state.get_task(broadcast_task_id)
        if not broadcast_task or not broadcast_task.delivery_location:
            return
        
        # This is the agent's new perceived projected location
        perceived_loc = broadcast_task.delivery_location
        search_radius = fleet_state.chain_lookahead_radius_km or 5.0
        
        # Find unassigned tasks near perceived projected location
        tasks_near_perceived = []
        pending_task_ids = get_agent_pending_tasks(agent_id)
        
        for unassigned_task in fleet_state.get_unassigned_tasks():
            # Skip tasks already pending for this agent
            if unassigned_task.id in pending_task_ids:
                continue
            # Skip the task we just broadcast
            if unassigned_task.id == broadcast_task_id:
                continue
            # Skip tasks THIS AGENT already declined (Tookan won't push to them)
            task_declined_by = set(str(d) for d in (unassigned_task.declined_by or []))
            if str(agent_id) in task_declined_by:
                continue
            
            dist = perceived_loc.distance_to(unassigned_task.restaurant_location)
            if dist <= search_radius:
                tasks_near_perceived.append((unassigned_task, dist))
        
        if not tasks_near_perceived:
            return  # No nearby tasks found
        
        # Sort by distance
        tasks_near_perceived.sort(key=lambda x: x[1])
        
        # Check directional compatibility with the broadcast task
        broadcast_bearing = get_task_delivery_bearing(broadcast_task)
        compatible_tasks = []
        
        for task, dist in tasks_near_perceived:
            task_bearing = get_task_delivery_bearing(task)
            diff = angular_difference(broadcast_bearing, task_bearing)
            
            if diff <= PROXIMITY_BATCH_DIRECTION_TOLERANCE_DEGREES:
                # Check per-task broadcast limit
                bcast_count = get_task_broadcast_count_for_agent(task.id, agent_id)
                if bcast_count < PROXIMITY_MAX_BROADCASTS_PER_AGENT:
                    compatible_tasks.append(task)
                    # Limit to remaining capacity
                    if len(compatible_tasks) >= broadcast_capacity:
                        break
        
        if not compatible_tasks:
            return  # No compatible tasks found
        
        log_event(f"[ProximityBroadcast] 🔮 PROACTIVE: Found {len(compatible_tasks)} existing tasks near {agent.name}'s perceived projected location")
        
        # Trigger batched broadcast for these additional tasks
        task_ids = [t.id for t in compatible_tasks]
        result = trigger_batched_proximity_broadcast(
            agent_id=agent_id,
            agent_name=agent.name,
            task_ids=task_ids,
            dashboard_url=dashboard_url,
            force=True
        )
        
        if result.get('success'):
            log_event(f"[ProximityBroadcast] 🔮 PROACTIVE: Added {result.get('broadcast_count', 0)} tasks to {agent.name}'s batch")
        
    finally:
        # Clean up check tracking
        _pending_proactive_checks.discard(check_key)


def remove_pending_broadcast(agent_id: str, task_id: str):
    """Remove a task from agent's pending broadcasts (accepted, declined, or assigned)."""
    with _proximity_lock:
        if agent_id in _agent_pending_broadcasts:
            _agent_pending_broadcasts[agent_id].discard(task_id)
            # Clean up empty sets
            if not _agent_pending_broadcasts[agent_id]:
                del _agent_pending_broadcasts[agent_id]
        
        # Also clean up direction tracking
        if agent_id in _agent_pending_directions:
            _agent_pending_directions[agent_id].pop(task_id, None)
            if not _agent_pending_directions[agent_id]:
                del _agent_pending_directions[agent_id]


def clear_task_from_all_pending(task_id: str):
    """Remove a task from ALL agents' pending broadcasts (task assigned/cancelled)."""
    with _proximity_lock:
        for agent_id in list(_agent_pending_broadcasts.keys()):
            _agent_pending_broadcasts[agent_id].discard(task_id)
            if not _agent_pending_broadcasts[agent_id]:
                del _agent_pending_broadcasts[agent_id]
        
        # Also clean up direction tracking
        for agent_id in list(_agent_pending_directions.keys()):
            _agent_pending_directions[agent_id].pop(task_id, None)
            if not _agent_pending_directions[agent_id]:
                del _agent_pending_directions[agent_id]


def get_agent_pending_tasks(agent_id: str) -> set:
    """Get the set of task IDs currently pending for an agent."""
    with _proximity_lock:
        return _agent_pending_broadcasts.get(agent_id, set()).copy()


def get_agent_pending_directions(agent_id: str) -> dict:
    """Get the pending broadcast directions for an agent: {task_id: bearing}."""
    with _proximity_lock:
        return _agent_pending_directions.get(agent_id, {}).copy()


def get_agent_perceived_projected_location(agent_id: str):
    """
    Calculate where an agent will be if they accept ALL pending broadcasts.
    
    This is the "perceived projected location" - the delivery location of the
    last pending broadcast task (furthest in the chain).
    
    Returns:
        Location object or None if no pending broadcasts
    """
    if not fleet_state:
        return None
    
    pending_task_ids = get_agent_pending_tasks(agent_id)
    if not pending_task_ids:
        return None
    
    # Get all pending tasks and find the one with the latest delivery time
    # (this represents where they'll end up after accepting all pending)
    latest_delivery_time = None
    perceived_location = None
    
    for task_id in pending_task_ids:
        task = fleet_state.get_task(task_id)
        if task and task.delivery_location:
            # Use delivery_before as proxy for "order in chain"
            if latest_delivery_time is None or (task.delivery_before and task.delivery_before > latest_delivery_time):
                latest_delivery_time = task.delivery_before
                perceived_location = task.delivery_location
    
    return perceived_location


def get_all_agent_proximity_locations(agent_id: str) -> list:
    """
    Get ALL locations to check for proximity eligibility:
    1. Current location (where agent is now)
    2. Next destination (where they're heading for current task)
    3. Projected location (where they'll be after current accepted tasks)
    4. Perceived projected location (where they'll be if they accept pending broadcasts)
    
    Returns:
        List of (location, location_type) tuples
    """
    if not fleet_state:
        return []
    
    agent = fleet_state.get_agent(agent_id)
    if not agent:
        return []
    
    locations = []
    
    # 1. Current location (always)
    if agent.current_location and agent.current_location.lat and agent.current_location.lng:
        locations.append((agent.current_location, "current"))
    
    # 2. Next destination (if agent has current tasks)
    if agent.current_tasks and agent.next_destination:
        if agent.next_destination.lat and agent.next_destination.lng:
            locations.append((agent.next_destination, "next_destination"))
    
    # 3. Projected location (after current tasks)
    if agent.current_tasks and agent.projected_location:
        if agent.projected_location.lat and agent.projected_location.lng:
            locations.append((agent.projected_location, "projected"))
    
    # 4. Perceived projected location (after pending broadcasts)
    perceived_loc = get_agent_perceived_projected_location(agent_id)
    if perceived_loc and perceived_loc.lat and perceived_loc.lng:
        locations.append((perceived_loc, "perceived_projected"))
    
    return locations


def is_task_directionally_compatible_with_pending(task_bearing: float, agent_id: str, tolerance_degrees: float = None) -> tuple:
    """
    Check if a task's direction is compatible with an agent's pending broadcasts.
    
    Args:
        task_bearing: The bearing of the new task (restaurant → delivery)
        agent_id: The agent to check
        tolerance_degrees: Max angular difference (default: PROXIMITY_BATCH_DIRECTION_TOLERANCE_DEGREES)
    
    Returns:
        (is_compatible: bool, reason: str or None)
        - If no pending broadcasts, returns (True, None)
        - If compatible, returns (True, None)
        - If not compatible, returns (False, reason_string)
    """
    if tolerance_degrees is None:
        tolerance_degrees = PROXIMITY_BATCH_DIRECTION_TOLERANCE_DEGREES
    
    pending_directions = get_agent_pending_directions(agent_id)
    
    if not pending_directions:
        return (True, None)  # No pending broadcasts, always compatible
    
    # Check against each pending broadcast's direction
    # Use the average direction of pending broadcasts as reference
    bearings = list(pending_directions.values())
    
    # Calculate circular mean for bearings (handles wrap-around at 0/360)
    import math
    sin_sum = sum(math.sin(math.radians(b)) for b in bearings)
    cos_sum = sum(math.cos(math.radians(b)) for b in bearings)
    avg_bearing = math.degrees(math.atan2(sin_sum, cos_sum))
    avg_bearing = (avg_bearing + 360) % 360
    
    # Check if new task is within tolerance of average pending direction
    diff = angular_difference(task_bearing, avg_bearing)
    
    if diff <= tolerance_degrees:
        return (True, None)
    else:
        return (False, f"bearing {round(task_bearing)}° differs {round(diff)}° from pending avg {round(avg_bearing)}°")


def get_task_broadcast_count_for_agent(task_id: str, agent_id: str) -> int:
    """Get how many times a task has been broadcast to a specific agent."""
    with _proximity_lock:
        task_counts = _task_agent_broadcast_counts.get(task_id, {})
        return task_counts.get(agent_id, 0)


def increment_task_broadcast_count(task_id: str, agent_id: str) -> int:
    """Increment and return the broadcast count for a task->agent pair."""
    with _proximity_lock:
        if task_id not in _task_agent_broadcast_counts:
            _task_agent_broadcast_counts[task_id] = {}
        if agent_id not in _task_agent_broadcast_counts[task_id]:
            _task_agent_broadcast_counts[task_id][agent_id] = 0
        _task_agent_broadcast_counts[task_id][agent_id] += 1
        return _task_agent_broadcast_counts[task_id][agent_id]


def clear_task_broadcast_counts(task_id: str):
    """Clear all broadcast counts for a task (when task is assigned/completed/cancelled)."""
    with _proximity_lock:
        _task_agent_broadcast_counts.pop(task_id, None)


def get_agents_at_broadcast_limit(task_id: str) -> set:
    """Get set of agent IDs that have hit the broadcast limit for this task."""
    with _proximity_lock:
        task_counts = _task_agent_broadcast_counts.get(task_id, {})
        return {
            agent_id for agent_id, count in task_counts.items()
            if count >= PROXIMITY_MAX_BROADCASTS_PER_AGENT
        }


# Configuration (can be updated via dashboard settings)
PROXIMITY_BROADCAST_ENABLED = True  # Toggle between broadcast (multi-agent) and auto-assign (single agent)
PROXIMITY_DEBOUNCE_MS = 1000  # Debounce rapid broadcasts for same task (1 second)
PROXIMITY_TASK_TIMEOUT_SECONDS = 120  # How long before task re-broadcasts (default 2 min)
PROXIMITY_DEFAULT_RADIUS_KM = 3.0  # Default search radius
PROXIMITY_MAX_RADIUS_KM = 10.0  # Maximum expanded radius
PROXIMITY_MAX_BROADCASTS_PER_AGENT = 3  # Max times a task can be broadcast to the same agent
PROXIMITY_BATCH_DIRECTION_TOLERANCE_DEGREES = 90  # Max angular difference for tasks to be batched together


import math

def calculate_bearing(from_lat: float, from_lng: float, to_lat: float, to_lng: float) -> float:
    """
    Calculate the bearing (compass direction) from one point to another.
    
    Returns bearing in degrees (0-360 where 0=North, 90=East, 180=South, 270=West).
    """
    lat1 = math.radians(from_lat)
    lat2 = math.radians(to_lat)
    diff_lng = math.radians(to_lng - from_lng)
    
    x = math.sin(diff_lng) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(diff_lng)
    
    bearing = math.atan2(x, y)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360  # Normalize to 0-360
    
    return bearing


def get_task_delivery_bearing(task) -> float:
    """
    Calculate the bearing from restaurant to delivery for a task.
    This represents the "direction" the task is going.
    """
    return calculate_bearing(
        task.restaurant_location.lat, task.restaurant_location.lng,
        task.delivery_location.lat, task.delivery_location.lng
    )


def angular_difference(bearing1: float, bearing2: float) -> float:
    """
    Calculate the smallest angular difference between two bearings.
    Returns value between 0 and 180 degrees.
    """
    diff = abs(bearing1 - bearing2)
    if diff > 180:
        diff = 360 - diff
    return diff


def filter_directionally_complementary_tasks(tasks: list, tolerance_degrees: float = None) -> list:
    """
    Filter tasks to only include those going in similar directions.
    
    Takes the first task's direction as the "primary" direction,
    then filters out tasks that deviate more than tolerance_degrees.
    
    Args:
        tasks: List of task objects
        tolerance_degrees: Max angular difference (default: PROXIMITY_BATCH_DIRECTION_TOLERANCE_DEGREES)
    
    Returns:
        List of directionally complementary tasks
    """
    if not tasks:
        return []
    
    if len(tasks) == 1:
        return tasks
    
    if tolerance_degrees is None:
        tolerance_degrees = PROXIMITY_BATCH_DIRECTION_TOLERANCE_DEGREES
    
    # Calculate bearing for each task
    task_bearings = []
    for task in tasks:
        bearing = get_task_delivery_bearing(task)
        task_bearings.append((task, bearing))
    
    # Use first task's bearing as the reference
    primary_task, primary_bearing = task_bearings[0]
    complementary = [primary_task]
    excluded = []
    
    for task, bearing in task_bearings[1:]:
        diff = angular_difference(primary_bearing, bearing)
        if diff <= tolerance_degrees:
            complementary.append(task)
        else:
            excluded.append((task.restaurant_name, round(bearing), round(diff)))
    
    if excluded:
        log_event(
            f"[ProximityBroadcast] 🧭 Direction filter: Primary bearing={round(primary_bearing)}° ({primary_task.restaurant_name}). "
            f"Excluded {len(excluded)} tasks: {excluded}"
        )
    
    return complementary


def trigger_proximity_broadcast(
    task_id: str,
    triggered_by_agent: str,
    dashboard_url: str,
    radius_km: float = None,
    force: bool = False
) -> dict:
    """
    Run solver for a SINGLE task with all nearby agents in BROADCAST mode.
    
    Instead of assigning to one agent, this:
    1. Finds all agents near the task (using both current and projected locations)
    2. Runs the solver to verify time feasibility for each agent
    3. Broadcasts task:proximity to all FEASIBLE agents
    
    OPTIMIZATION: Only broadcasts if:
    - force=True (timeout, expand, decline events)
    - OR a NEW agent enters the feasible set
    - Skips broadcast if same agents are still feasible (no spam)
    
    Args:
        task_id: The task to broadcast
        triggered_by_agent: Name of agent who triggered proximity (for logging)
        dashboard_url: Dashboard URL for callbacks
        radius_km: Search radius (uses expanded radius if set, else default)
        force: If True, always broadcast (used for timeout/expand/decline)
    
    Returns: {
        'success': bool,
        'task_id': str,
        'feasible_agents': [...],
        'infeasible_agents': [...],
        'broadcast_count': int
    }
    """
    global _task_offer_times, _last_proximity_broadcast, _task_current_agents
    
    if not FLEET_STATE_AVAILABLE or not fleet_state:
        return {'success': False, 'error': 'Fleet state not available'}
    
    task = fleet_state.get_task(task_id)
    if not task:
        return {'success': False, 'error': 'Task not found'}
    
    if task.status != TaskStatus.UNASSIGNED:
        return {'success': False, 'error': 'Task already assigned'}
    
    # NOTE: declined_by is NOT a blocker - we just filter out those specific agents
    # Tookan won't push to agents who declined, but WILL push to other agents
    task_declined_by = set(str(d) for d in (task.declined_by or []))
    
    # Debounce per-task broadcasts
    now = time.time()
    with _proximity_lock:
        last_broadcast = _last_proximity_broadcast.get(task_id, 0)
        if (now - last_broadcast) * 1000 < PROXIMITY_DEBOUNCE_MS:
            return {'success': True, 'debounced': True}
        _last_proximity_broadcast[task_id] = now
    
    # Determine search radius
    search_radius = radius_km or _task_expanded_radius.get(task_id, PROXIMITY_DEFAULT_RADIUS_KM)
    
    # Check if this is a premium task (P1 agents get unlimited distance)
    is_premium_task = task.is_premium_task
    
    # Find all nearby agents (eligible or not)
    all_agents = fleet_state.get_all_agents()
    nearby_agents = []
    p1_agents_included = []
    
    for agent in all_agents:
        if not agent.is_online or not agent.has_capacity:
            continue
        
        # Check ALL proximity locations:
        # 1. Current location
        # 2. Next destination (where heading for current task)
        # 3. Projected location (after current tasks)
        # 4. Perceived projected location (after pending broadcasts)
        all_locations = get_all_agent_proximity_locations(agent.id)
        
        if not all_locations:
            # Fallback to just current location if function fails
            if agent.current_location and agent.current_location.lat and agent.current_location.lng:
                all_locations = [(agent.current_location, "current")]
            else:
                continue  # Skip agent with no valid location
        
        # Calculate minimum distance from any of these locations
        # Skip locations with None values to prevent TypeError
        distances = []
        best_location_type = "current"
        for loc, loc_type in all_locations:
            if loc and loc.lat is not None and loc.lng is not None:
                try:
                    dist = loc.distance_to(task.restaurant_location)
                    distances.append((dist, loc_type))
                except (TypeError, AttributeError):
                    continue  # Skip invalid locations
        
        if not distances:
            continue  # Skip agent if no valid distances calculated
        
        # Find minimum distance and which location type it came from
        min_dist, best_location_type = min(distances, key=lambda x: x[0])
        
        # P1 agents get UNLIMITED distance for premium tasks
        is_p1_agent = agent.priority == 1
        include_agent = False
        
        if is_premium_task and is_p1_agent:
            # Premium task + P1 agent = include regardless of distance
            include_agent = True
            if min_dist > search_radius:
                p1_agents_included.append(f"{agent.name}({min_dist:.1f}km via {best_location_type})")
        elif min_dist <= search_radius:
            # Regular distance check for everyone else
            include_agent = True
        
        if include_agent:
            # Check business rule eligibility (pass expanded radius to override task's default)
            # For P1 agents on premium tasks, use their actual distance for eligibility check
            eligibility = fleet_state._check_eligibility(agent, task, override_max_distance_km=max(search_radius, min_dist + 1))
            nearby_agents.append({
                'agent': agent,
                'distance_km': min_dist,
                'eligibility_reason': eligibility,
                'is_p1_premium': is_premium_task and is_p1_agent,
                'proximity_type': best_location_type  # Track how we found this agent
            })
    
    # Log P1 agents included beyond normal radius
    if p1_agents_included:
        log_event(f"[ProximityBroadcast] ⭐ PREMIUM TASK: Including {len(p1_agents_included)} P1 agents beyond radius: {p1_agents_included}")
    
    # Log agents found via perceived projected location (pending broadcasts)
    perceived_projected_agents = [a for a in nearby_agents if a.get('proximity_type') == 'perceived_projected']
    if perceived_projected_agents:
        names = [f"{a['agent'].name}({a['distance_km']:.1f}km)" for a in perceived_projected_agents]
        log_event(f"[ProximityBroadcast] 🔮 PERCEIVED PROJECTION: {len(perceived_projected_agents)} agents near via pending broadcasts: {names}")
    
    if not nearby_agents:
        log_event(f"[ProximityBroadcast] ⚠️ No nearby agents for {task.restaurant_name} within {search_radius}km")
        return {'success': False, 'error': 'No nearby agents', 'radius_km': search_radius}
    
    # Filter to only eligible agents
    eligible_agents = [a for a in nearby_agents if a['eligibility_reason'] is None]
    ineligible_agents = [a for a in nearby_agents if a['eligibility_reason'] is not None]
    
    if not eligible_agents:
        reasons = {a['eligibility_reason'] for a in ineligible_agents}
        log_event(f"[ProximityBroadcast] ⚠️ No eligible agents for {task.restaurant_name}. Reasons: {reasons}")
        return {'success': False, 'error': 'No eligible agents', 'reasons': list(reasons), 'radius_km': search_radius}
    
    # Filter agents by broadcast capacity (current_tasks + pending_broadcasts < max_tasks)
    agents_with_capacity = []
    agents_at_capacity = []
    for agent_info in eligible_agents:
        agent = agent_info['agent']
        broadcast_capacity = get_agent_broadcast_capacity(agent.id)
        if broadcast_capacity > 0:
            agents_with_capacity.append(agent_info)
        else:
            pending_tasks = get_agent_pending_tasks(agent.id)
            agents_at_capacity.append({
                'agent_id': agent.id,
                'agent_name': agent.name,
                'current_tasks': len(agent.current_tasks),
                'pending_broadcasts': len(pending_tasks),
                'max_capacity': agent.max_capacity
            })
    
    if agents_at_capacity:
        names = [a['agent_name'] for a in agents_at_capacity]
        log_event(f"[ProximityBroadcast] 🚫 Skipping {len(agents_at_capacity)} agents at broadcast capacity: {names}")
    
    if not agents_with_capacity:
        log_event(f"[ProximityBroadcast] ⚠️ All eligible agents at broadcast capacity for {task.restaurant_name}")
        return {'success': False, 'error': 'All agents at broadcast capacity', 'agents_at_capacity': agents_at_capacity, 'radius_km': search_radius}
    
    # Filter agents by per-task broadcast limit (max times this task can be broadcast to same agent)
    agents_under_limit = []
    agents_at_limit = []
    for agent_info in agents_with_capacity:
        agent = agent_info['agent']
        broadcast_count = get_task_broadcast_count_for_agent(task_id, agent.id)
        if broadcast_count < PROXIMITY_MAX_BROADCASTS_PER_AGENT:
            agents_under_limit.append(agent_info)
        else:
            agents_at_limit.append({
                'agent_id': agent.id,
                'agent_name': agent.name,
                'broadcast_count': broadcast_count,
                'limit': PROXIMITY_MAX_BROADCASTS_PER_AGENT
            })
    
    if agents_at_limit:
        names = [f"{a['agent_name']}({a['broadcast_count']}x)" for a in agents_at_limit]
        log_event(f"[ProximityBroadcast] 🔁 Skipping {len(agents_at_limit)} agents at broadcast limit for {task.restaurant_name}: {names}")
    
    if not agents_under_limit:
        log_event(f"[ProximityBroadcast] ⚠️ All agents hit broadcast limit ({PROXIMITY_MAX_BROADCASTS_PER_AGENT}x) for {task.restaurant_name}")
        return {'success': False, 'error': f'All agents hit broadcast limit ({PROXIMITY_MAX_BROADCASTS_PER_AGENT}x)', 'agents_at_limit': agents_at_limit, 'radius_km': search_radius}
    
    # Calculate this task's bearing for directional compatibility check
    task_bearing = get_task_delivery_bearing(task)
    
    # Filter agents by directional compatibility with their pending broadcasts
    agents_compatible = []
    agents_incompatible = []
    for agent_info in agents_under_limit:
        agent = agent_info['agent']
        is_compatible, reason = is_task_directionally_compatible_with_pending(task_bearing, agent.id)
        if is_compatible:
            agents_compatible.append(agent_info)
        else:
            agents_incompatible.append({
                'agent_id': agent.id,
                'agent_name': agent.name,
                'reason': reason
            })
    
    if agents_incompatible:
        names = [f"{a['agent_name']}({a['reason']})" for a in agents_incompatible]
        log_event(f"[ProximityBroadcast] 🧭 Skipping {len(agents_incompatible)} agents with incompatible pending direction for {task.restaurant_name}: {names}")
    
    if not agents_compatible:
        log_event(f"[ProximityBroadcast] ⚠️ All agents have incompatible pending directions for {task.restaurant_name} (bearing {round(task_bearing)}°)")
        return {'success': False, 'error': 'All agents have incompatible pending directions', 'task_bearing': round(task_bearing), 'agents_incompatible': agents_incompatible, 'radius_km': search_radius}
    
    # Filter out agents who already declined this task (Tookan won't push to them)
    agents_not_declined = []
    agents_declined = []
    for agent_info in agents_compatible:
        agent = agent_info['agent']
        if str(agent.id) in task_declined_by:
            agents_declined.append({
                'agent_id': agent.id,
                'agent_name': agent.name
            })
        else:
            agents_not_declined.append(agent_info)
    
    if agents_declined:
        names = [a['agent_name'] for a in agents_declined]
        log_event(f"[ProximityBroadcast] 🚫 Skipping {len(agents_declined)} agents who declined {task.restaurant_name}: {names}")
    
    if not agents_not_declined:
        log_event(f"[ProximityBroadcast] ⚠️ All nearby agents have declined {task.restaurant_name} - needs manual assignment")
        return {
            'success': False, 
            'error': 'All nearby agents have declined this task',
            'declined_by': list(task_declined_by),
            'radius_km': search_radius,
            'blocked': True,
            'blocked_reason': 'all_declined'
        }
    
    # Run solver for single task with agents who haven't declined
    feasible_agents = []
    infeasible_agents = []
    
    for agent_info in agents_not_declined:
        agent = agent_info['agent']
        
        # Build single-agent data for solver
        # Include BOTH current tasks AND pending broadcasts to check true feasibility
        current_tasks = []
        
        # Add agent's actual current tasks
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
        
        # Add pending broadcasts as "virtual" current tasks for feasibility check
        # This ensures we check if new task can be done AFTER all pending tasks
        pending_task_ids = get_agent_pending_tasks(agent.id)
        pending_count = 0
        for pending_id in pending_task_ids:
            pending_task = fleet_state.get_task(pending_id)
            if pending_task and pending_task.id != task_id:  # Don't include the task we're checking
                current_tasks.append({
                    'id': pending_task.id,
                    'job_type': pending_task.job_type,
                    'restaurant_location': [pending_task.restaurant_location.lat, pending_task.restaurant_location.lng],
                    'delivery_location': [pending_task.delivery_location.lat, pending_task.delivery_location.lng],
                    'pickup_before': pending_task.pickup_before.isoformat() if pending_task.pickup_before else None,
                    'delivery_before': pending_task.delivery_before.isoformat() if pending_task.delivery_before else None,
                    'assigned_driver': agent.id,
                    'pickup_completed': False,  # Pending tasks haven't started pickup
                    '_meta': pending_task.meta
                })
                pending_count += 1
        
        if pending_count > 0:
            log_event(f"[ProximityBroadcast] 🔄 Including {pending_count} pending tasks in feasibility check for {agent.name}")
        
        agents_data = {
            'agents': [{
                'driver_id': agent.id,
                'name': agent.name,
                'current_location': [agent.current_location.lat, agent.current_location.lng],
                'current_tasks': current_tasks,
                'wallet_balance': agent.wallet_balance,
                '_meta': {
                    'max_tasks': agent.max_capacity,
                    'available_capacity': agent.available_capacity,
                    'tags': agent.tags,
                    'priority': agent.priority
                }
            }],
            'geofence_data': [],
            'settings_used': {
                'walletNoCashThreshold': fleet_state.wallet_threshold,
                'maxDistanceKm': search_radius  # Use current search radius
            }
        }
        
        # Get just this one task
        task_export = {
            'tasks': [{
                'id': task.id,
                'restaurant_location': [task.restaurant_location.lat, task.restaurant_location.lng],
                'delivery_location': [task.delivery_location.lat, task.delivery_location.lng],
                'pickup_before': task.pickup_before.isoformat() if task.pickup_before else None,
                'delivery_before': task.delivery_before.isoformat() if task.delivery_before else None,
                'payment_method': task.payment_method,
                'tags': task.tags,
                '_meta': {
                    'restaurant_name': task.restaurant_name,
                    'customer_name': task.meta.get('customer_name', 'Unknown'),
                    'delivery_fee': task.delivery_fee,
                    'tips': task.tips
                }
            }]
        }
        
        try:
            # Run solver for this agent + this task
            result = optimize_fleet(agents_data, task_export, prefilter_distance=False)
            
            if result.get('success') and result.get('metadata', {}).get('tasks_assigned', 0) > 0:
                # Solver says this agent can complete the task on time
                route = result.get('agent_routes', [{}])[0]
                eta_info = route.get('stops', [])
                
                feasible_agents.append({
                    'agent_id': agent.id,
                    'agent_name': agent.name,
                    'distance_km': round(agent_info['distance_km'], 2),
                    'location': [agent.current_location.lat, agent.current_location.lng] if agent.current_location else None,
                    'current_task_count': len(agent.current_tasks),
                    'available_capacity': agent.available_capacity,
                    'priority': agent.priority,
                    'solver_verified': True,
                    'eta_stops': eta_info
                })
            else:
                # Solver says not feasible
                reason = result.get('error', 'time_window_infeasible')
                infeasible_agents.append({
                    'agent_id': agent.id,
                    'agent_name': agent.name,
                    'reason': reason
                })
        except Exception as e:
            infeasible_agents.append({
                'agent_id': agent.id,
                'agent_name': agent.name,
                'reason': f'solver_error: {str(e)}'
            })
    
    if not feasible_agents:
        log_event(f"[ProximityBroadcast] ⚠️ No feasible agents for {task.restaurant_name} (solver rejected all)")
        return {
            'success': False, 
            'error': 'No feasible agents (time windows)', 
            'infeasible': infeasible_agents
        }
    
    # Check if we should skip broadcast (same agents, not forced)
    current_feasible_ids = set(a['agent_id'] for a in feasible_agents)
    with _proximity_lock:
        previous_agents = _task_current_agents.get(task_id, set())
        new_agents = current_feasible_ids - previous_agents
        
        if not force and not new_agents and previous_agents:
            # Same agents as before, skip broadcast
            return {
                'success': True,
                'skipped': True,
                'reason': 'same_agents',
                'current_agents': list(current_feasible_ids)
            }
        
        # Update tracked agents
        _task_current_agents[task_id] = current_feasible_ids
        
        # Log if new agent entered
        if new_agents and not force:
            new_names = [a['agent_name'] for a in feasible_agents if a['agent_id'] in new_agents]
            log_event(f"[ProximityBroadcast] 🆕 New agent(s) for {task.restaurant_name}: {new_names}")
    
    # Track first_offered_at for timer
    with _proximity_lock:
        if task_id not in _task_offer_times:
            _task_offer_times[task_id] = now
        first_offered_at = _task_offer_times[task_id]
    
    # Build and emit task:proximity payload
    proximity_payload = {
        'event': 'task:proximity',
        'task': {
            'id': task.id,
            'restaurant_name': task.restaurant_name,
            'customer_name': task.customer_name,
            'locations': {
                'restaurant': [task.restaurant_location.lat, task.restaurant_location.lng],
                'delivery': [task.delivery_location.lat, task.delivery_location.lng]
            },
            'times': {
                'pickup_before': task.pickup_before.isoformat() if task.pickup_before else None,
                'delivery_before': task.delivery_before.isoformat() if task.delivery_before else None,
                'created_at': task.created_at.isoformat() if task.created_at else None
            },
            'payment': {
                'delivery_fee': task.delivery_fee,
                'tips': task.tips,
                'type': task.payment_method,
                'total': (task.delivery_fee or 0) + (task.tips or 0)
            },
            'tags': task.tags,
            'is_premium': 'premium' in (task.tags or [])
        },
        'feasible_agents': feasible_agents,
        'competition': len(feasible_agents),
        'first_offered_at': first_offered_at,
        'timeout_seconds': PROXIMITY_TASK_TIMEOUT_SECONDS,
        'search_radius_km': search_radius,
        'triggered_by': triggered_by_agent,
        'timestamp': now,
        'dashboard_url': dashboard_url
    }
    
    socketio.emit('task:proximity', proximity_payload)
    
    # Track pending broadcasts and increment broadcast counts for each agent
    # Pass task_bearing so future broadcasts can check directional compatibility
    for agent in feasible_agents:
        add_pending_broadcast(agent['agent_id'], task_id, task_bearing)
        increment_task_broadcast_count(task_id, agent['agent_id'])
    
    # Build detailed log with agent capacity and broadcast count info
    agent_details = []
    for a in feasible_agents:
        pending = len(get_agent_pending_tasks(a['agent_id']))
        bcast_count = get_task_broadcast_count_for_agent(task_id, a['agent_id'])
        agent_details.append(f"{a['agent_name']}(pending:{pending},bcast:{bcast_count}/{PROXIMITY_MAX_BROADCASTS_PER_AGENT})")
    
    log_event(f"[ProximityBroadcast] 📡 {task.restaurant_name} → {len(feasible_agents)} agents: {agent_details} (radius: {search_radius}km)")
    
    # PROACTIVE CHECK: After broadcasting, check for existing tasks near each agent's
    # new perceived projected location (this task's delivery)
    for agent in feasible_agents:
        trigger_perceived_projected_check(agent['agent_id'], task_id, dashboard_url)
    
    return {
        'success': True,
        'task_id': task_id,
        'task_name': task.restaurant_name,
        'feasible_agents': feasible_agents,
        'infeasible_agents': infeasible_agents,
        'broadcast_count': len(feasible_agents),
        'radius_km': search_radius
    }


def trigger_batched_proximity_broadcast(
    agent_id: str,
    agent_name: str,
    task_ids: List[str],
    dashboard_url: str,
    force: bool = False
) -> dict:
    """
    Run solver for MULTIPLE tasks with a SINGLE agent - BATCHED BROADCAST.
    
    Instead of broadcasting each task individually, this:
    1. Takes all nearby tasks for an agent
    2. Runs the solver with ALL tasks together
    3. Broadcasts ALL feasible tasks to the agent in ONE payload
    
    This enables smart chaining where an agent can pick up multiple nearby tasks.
    
    Args:
        agent_id: The agent to check
        agent_name: Agent's name (for logging)
        task_ids: List of task IDs to consider
        dashboard_url: Dashboard URL for callbacks
        force: If True, always broadcast
    
    Returns: {
        'success': bool,
        'agent_id': str,
        'feasible_tasks': [...],
        'infeasible_tasks': [...],
        'broadcast_count': int
    }
    """
    global _task_offer_times, _last_proximity_broadcast
    
    if not FLEET_STATE_AVAILABLE or not fleet_state:
        return {'success': False, 'error': 'Fleet state not available'}
    
    agent = fleet_state.get_agent(str(agent_id))
    if not agent:
        return {'success': False, 'error': 'Agent not found'}
    
    if not agent.is_online or not agent.has_capacity:
        return {'success': False, 'error': 'Agent not available'}
    
    # Check agent's broadcast capacity (current + pending must be < max)
    broadcast_capacity = get_agent_broadcast_capacity(agent_id)
    if broadcast_capacity <= 0:
        pending_tasks = get_agent_pending_tasks(agent_id)
        log_event(f"[ProximityBroadcast] 🚫 {agent_name} at broadcast capacity (tasks: {len(agent.current_tasks)}, pending: {len(pending_tasks)}, max: {agent.max_capacity})")
        return {
            'success': False, 
            'error': 'Agent at broadcast capacity',
            'agent_id': agent_id,
            'current_tasks': len(agent.current_tasks),
            'pending_broadcasts': len(pending_tasks),
            'max_capacity': agent.max_capacity
        }
    
    # Filter to valid unassigned tasks (limited by broadcast capacity and per-task limit)
    valid_tasks = []
    skipped_at_limit = []
    skipped_declined = []
    for task_id in task_ids:
        task = fleet_state.get_task(task_id)
        if task and task.status == TaskStatus.UNASSIGNED:
            # Skip if THIS AGENT already declined this task (Tookan won't push to them)
            task_declined_by = set(str(d) for d in (task.declined_by or []))
            if str(agent_id) in task_declined_by:
                skipped_declined.append(task.restaurant_name)
                continue
            
            # Check if this task has already been broadcast to this agent too many times
            bcast_count = get_task_broadcast_count_for_agent(task_id, agent_id)
            if bcast_count >= PROXIMITY_MAX_BROADCASTS_PER_AGENT:
                skipped_at_limit.append((task.restaurant_name, bcast_count))
                continue
            
            valid_tasks.append(task)
            # Stop when we hit broadcast capacity limit
            if len(valid_tasks) >= broadcast_capacity:
                break
    
    if skipped_declined:
        log_event(f"[ProximityBroadcast] 🚫 Batched: Skipping {len(skipped_declined)} declined tasks for {agent_name}: {skipped_declined}")
    
    if skipped_at_limit:
        log_event(f"[ProximityBroadcast] 🔁 Batched: Skipping {len(skipped_at_limit)} tasks at limit for {agent_name}: {skipped_at_limit}")
    
    if not valid_tasks:
        return {'success': False, 'error': 'No valid tasks (all at broadcast limit)'}
    
    # Apply directional filtering - only batch tasks going in similar directions
    if len(valid_tasks) > 1:
        pre_filter_count = len(valid_tasks)
        valid_tasks = filter_directionally_complementary_tasks(valid_tasks)
        if len(valid_tasks) < pre_filter_count:
            log_event(f"[ProximityBroadcast] 🧭 Direction filter: {pre_filter_count} → {len(valid_tasks)} tasks for {agent_name}")
    
    if not valid_tasks:
        return {'success': False, 'error': 'No directionally compatible tasks'}
    
    # Check if batch direction is compatible with agent's existing pending broadcasts
    # Use the first task's bearing as the batch direction
    primary_task = valid_tasks[0]
    batch_bearing = get_task_delivery_bearing(primary_task)
    is_compatible, reason = is_task_directionally_compatible_with_pending(batch_bearing, agent_id)
    
    if not is_compatible:
        log_event(f"[ProximityBroadcast] 🧭 Batched: {agent_name} has pending broadcasts in different direction. {primary_task.restaurant_name} {reason}")
        return {
            'success': False, 
            'error': 'Batch incompatible with pending broadcasts',
            'agent_id': agent_id,
            'batch_bearing': round(batch_bearing),
            'reason': reason
        }
    
    log_event(f"[ProximityBroadcast] 📦 Batched check for {agent_name}: {len(valid_tasks)} tasks (capacity: {broadcast_capacity})")
    
    # Debounce batched broadcast per agent
    now = time.time()
    batch_key = f"batch_{agent_id}"
    with _proximity_lock:
        last_broadcast = _last_proximity_broadcast.get(batch_key, 0)
        if (now - last_broadcast) * 1000 < PROXIMITY_DEBOUNCE_MS:
            return {'success': True, 'debounced': True}
        _last_proximity_broadcast[batch_key] = now
    
    # Build agent data for solver
    # Include BOTH current tasks AND pending broadcasts (not in this batch) for true feasibility
    current_tasks = []
    valid_task_ids = {t.id for t in valid_tasks}  # IDs of tasks we're checking in this batch
    
    # Add agent's actual current tasks
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
    
    # Add pending broadcasts (not in current batch) as "virtual" current tasks
    pending_task_ids = get_agent_pending_tasks(agent_id)
    pending_count = 0
    for pending_id in pending_task_ids:
        if pending_id not in valid_task_ids:  # Don't include tasks we're checking
            pending_task = fleet_state.get_task(pending_id)
            if pending_task:
                current_tasks.append({
                    'id': pending_task.id,
                    'job_type': pending_task.job_type,
                    'restaurant_location': [pending_task.restaurant_location.lat, pending_task.restaurant_location.lng],
                    'delivery_location': [pending_task.delivery_location.lat, pending_task.delivery_location.lng],
                    'pickup_before': pending_task.pickup_before.isoformat() if pending_task.pickup_before else None,
                    'delivery_before': pending_task.delivery_before.isoformat() if pending_task.delivery_before else None,
                    'assigned_driver': agent.id,
                    'pickup_completed': False,
                    '_meta': pending_task.meta
                })
                pending_count += 1
    
    if pending_count > 0:
        log_event(f"[ProximityBroadcast] 🔄 Batched: Including {pending_count} pending tasks in feasibility check for {agent_name}")
    
    agents_data = {
        'agents': [{
            'driver_id': agent.id,
            'name': agent.name,
            'current_location': [agent.current_location.lat, agent.current_location.lng],
            'current_tasks': current_tasks,
            'wallet_balance': agent.wallet_balance,
            '_meta': {
                'max_tasks': agent.max_capacity,
                'available_capacity': agent.available_capacity,
                'tags': agent.tags,
                'priority': agent.priority
            }
        }],
        'geofence_data': [],
        'settings_used': {
            'walletNoCashThreshold': fleet_state.wallet_threshold,
            'maxDistanceKm': fleet_state.max_distance_km
        }
    }
    
    # Build task data for ALL tasks
    tasks_export = []
    for task in valid_tasks:
        # Get search radius for this task (expanded or default)
        search_radius = _task_expanded_radius.get(task.id, PROXIMITY_DEFAULT_RADIUS_KM)
        tasks_export.append({
            'id': task.id,
            'restaurant_location': [task.restaurant_location.lat, task.restaurant_location.lng],
            'delivery_location': [task.delivery_location.lat, task.delivery_location.lng],
            'pickup_before': task.pickup_before.isoformat() if task.pickup_before else None,
            'delivery_before': task.delivery_before.isoformat() if task.delivery_before else None,
            'payment_method': task.payment_method,
            'tags': task.tags,
            '_meta': {
                'restaurant_name': task.restaurant_name,
                'customer_name': task.meta.get('customer_name', 'Unknown'),
                'delivery_fee': task.delivery_fee,
                'tips': task.tips,
                'search_radius_km': search_radius
            }
        })
    
    task_export = {'tasks': tasks_export}
    
    try:
        # Run solver with ALL tasks at once
        result = optimize_fleet(agents_data, task_export, prefilter_distance=False)
        
        if not result.get('success'):
            return {
                'success': False,
                'error': result.get('error', 'solver_failed'),
                'agent_id': agent_id
            }
        
        # Extract which tasks were assigned by the solver
        feasible_tasks = []
        infeasible_task_ids = set(t.id for t in valid_tasks)
        
        routes = result.get('agent_routes', [])
        if routes:
            route = routes[0]
            assigned_task_ids = route.get('assigned_new_tasks', [])
            stops = route.get('stops', [])
            
            for task in valid_tasks:
                if task.id in assigned_task_ids:
                    infeasible_task_ids.discard(task.id)
                    
                    # Get search radius for display
                    search_radius = _task_expanded_radius.get(task.id, PROXIMITY_DEFAULT_RADIUS_KM)
                    
                    # Calculate distance using all proximity locations
                    all_locations = get_all_agent_proximity_locations(agent_id)
                    min_dist = None
                    if all_locations:
                        distances = []
                        for loc, _ in all_locations:
                            if loc and loc.lat is not None and loc.lng is not None:
                                try:
                                    distances.append(loc.distance_to(task.restaurant_location))
                                except (TypeError, AttributeError):
                                    continue
                        if distances:
                            min_dist = min(distances)
                    
                    if min_dist is None:
                        # Fallback with null checks
                        try:
                            if agent.current_location and agent.current_location.lat is not None:
                                min_dist = agent.current_location.distance_to(task.restaurant_location)
                            elif agent.projected_location and agent.projected_location.lat is not None:
                                min_dist = agent.projected_location.distance_to(task.restaurant_location)
                            else:
                                min_dist = 0  # Default if no valid location
                        except (TypeError, AttributeError):
                            min_dist = 0
                    
                    # Find ETA from stops
                    task_stops = [s for s in stops if s.get('task_id') == task.id]
                    
                    # Calculate task bearing for directional tracking
                    task_bearing = get_task_delivery_bearing(task)
                    
                    feasible_tasks.append({
                        'task_id': task.id,
                        'restaurant_name': task.restaurant_name,
                        'customer_name': task.meta.get('customer_name', 'Unknown'),
                        'distance_km': round(min_dist, 2),
                        'search_radius_km': search_radius,
                        'is_premium': task.is_premium_task,
                        'delivery_fee': task.delivery_fee,
                        'tips': task.tips,
                        'payment_method': task.payment_method,
                        'locations': {
                            'restaurant': [task.restaurant_location.lat, task.restaurant_location.lng],
                            'delivery': [task.delivery_location.lat, task.delivery_location.lng]
                        },
                        'times': {
                            'pickup_before': task.pickup_before.isoformat() if task.pickup_before else None,
                            'delivery_before': task.delivery_before.isoformat() if task.delivery_before else None
                        },
                        'eta_stops': task_stops,
                        'bearing': task_bearing  # Store bearing for directional tracking
                    })
        
        if not feasible_tasks:
            log_event(f"[ProximityBroadcast] ⚠️ Batched: No feasible tasks for {agent_name} (solver rejected all {len(valid_tasks)})")
            return {
                'success': False,
                'error': 'No feasible tasks',
                'agent_id': agent_id,
                'tasks_checked': len(valid_tasks)
            }
        
        # Track first_offered_at for each task
        with _proximity_lock:
            for ft in feasible_tasks:
                if ft['task_id'] not in _task_offer_times:
                    _task_offer_times[ft['task_id']] = now
        
        # Build and emit BATCHED proximity payload
        batched_payload = {
            'event': 'proximity:batched_broadcast',
            'agent': {
                'id': agent.id,
                'name': agent.name,
                'location': [agent.current_location.lat, agent.current_location.lng],
                'current_task_count': len(agent.current_tasks),
                'available_capacity': agent.available_capacity
            },
            'tasks': feasible_tasks,
            'task_count': len(feasible_tasks),
            'first_offered_at': now,
            'timeout_seconds': PROXIMITY_TASK_TIMEOUT_SECONDS,
            'triggered_by': 'batched_proximity',
            'timestamp': now,
            'dashboard_url': dashboard_url
        }
        
        socketio.emit('proximity:batched_broadcast', batched_payload)
        
        task_names = [t['restaurant_name'] for t in feasible_tasks]
        log_event(f"[ProximityBroadcast] 📦 BATCHED: {agent_name} can do {len(feasible_tasks)} tasks: {task_names}")
        
        # ALSO emit individual task:proximity for each task (for backward compatibility)
        # This ensures the existing dashboard UI still works
        for task_info in feasible_tasks:
            task = fleet_state.get_task(task_info['task_id'])
            if task:
                first_offered = _task_offer_times.get(task.id, now)
                search_radius = task_info['search_radius_km']
                
                individual_payload = {
                    'event': 'task:proximity',
                    'task': {
                        'id': task.id,
                        'restaurant_name': task.restaurant_name,
                        'customer_name': task_info['customer_name'],
                        'locations': task_info['locations'],
                        'times': task_info['times'],
                        'payment': {
                            'delivery_fee': task.delivery_fee,
                            'tips': task.tips,
                            'type': task.payment_method,
                            'total': (task.delivery_fee or 0) + (task.tips or 0)
                        },
                        'tags': task.tags,
                        'is_premium': task_info['is_premium']
                    },
                    'feasible_agents': [{
                        'agent_id': agent.id,
                        'agent_name': agent.name,
                        'distance_km': task_info['distance_km'],
                        'location': [agent.current_location.lat, agent.current_location.lng],
                        'current_task_count': len(agent.current_tasks),
                        'available_capacity': agent.available_capacity,
                        'priority': agent.priority,
                        'solver_verified': True,
                        'batched_with': len(feasible_tasks) - 1  # Number of OTHER tasks in batch
                    }],
                    'competition': 1,
                    'first_offered_at': first_offered,
                    'timeout_seconds': PROXIMITY_TASK_TIMEOUT_SECONDS,
                    'search_radius_km': search_radius,
                    'triggered_by': f'batched ({len(feasible_tasks)} tasks)',
                    'timestamp': now,
                    'dashboard_url': dashboard_url,
                    'is_batched': True,
                    'batch_size': len(feasible_tasks)
                }
                
                socketio.emit('task:proximity', individual_payload)
                log_event(f"[ProximityBroadcast] 📡 {task.restaurant_name} → {agent_name} (batched with {len(feasible_tasks)-1} others)")
        
        # Track pending broadcasts and increment broadcast counts for this agent
        # Pass bearing so future broadcasts can check directional compatibility
        for task_info in feasible_tasks:
            add_pending_broadcast(agent_id, task_info['task_id'], task_info.get('bearing'))
            increment_task_broadcast_count(task_info['task_id'], agent_id)
        
        # PROACTIVE CHECK: After batched broadcast, check for existing tasks near
        # the agent's new perceived projected location (last task's delivery)
        # Only trigger for the LAST task in the batch (represents final destination)
        if feasible_tasks:
            last_task_id = feasible_tasks[-1]['task_id']
            trigger_perceived_projected_check(agent_id, last_task_id, dashboard_url)
        
        return {
            'success': True,
            'agent_id': agent_id,
            'agent_name': agent_name,
            'feasible_tasks': feasible_tasks,
            'infeasible_task_ids': list(infeasible_task_ids),
            'broadcast_count': len(feasible_tasks)
        }
        
    except Exception as e:
        log_event(f"[ProximityBroadcast] ❌ Batched solver error for {agent_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': f'solver_error: {str(e)}',
            'agent_id': agent_id
        }


def update_proximity_broadcast_settings(settings: dict):
    """
    Update proximity broadcast settings from dashboard.
    Called during fleet:sync or config update.
    """
    global PROXIMITY_BROADCAST_ENABLED, PROXIMITY_TASK_TIMEOUT_SECONDS, PROXIMITY_DEFAULT_RADIUS_KM, PROXIMITY_MAX_BROADCASTS_PER_AGENT
    
    if 'proximity_broadcast_enabled' in settings:
        PROXIMITY_BROADCAST_ENABLED = bool(settings['proximity_broadcast_enabled'])
        log_event(f"[ProximityBroadcast] Mode {'ENABLED' if PROXIMITY_BROADCAST_ENABLED else 'DISABLED'}")
    
    if 'proximity_task_timeout_seconds' in settings:
        PROXIMITY_TASK_TIMEOUT_SECONDS = int(settings['proximity_task_timeout_seconds'])
        log_event(f"[ProximityBroadcast] Task timeout set to {PROXIMITY_TASK_TIMEOUT_SECONDS}s")
    
    if 'proximity_default_radius_km' in settings:
        PROXIMITY_DEFAULT_RADIUS_KM = float(settings['proximity_default_radius_km'])
        log_event(f"[ProximityBroadcast] Default radius set to {PROXIMITY_DEFAULT_RADIUS_KM}km")
    
    if 'proximity_max_broadcasts_per_agent' in settings:
        PROXIMITY_MAX_BROADCASTS_PER_AGENT = int(settings['proximity_max_broadcasts_per_agent'])
        log_event(f"[ProximityBroadcast] Max broadcasts per agent set to {PROXIMITY_MAX_BROADCASTS_PER_AGENT}")


def expand_task_radius(task_id: str, new_radius_km: float, dashboard_url: str) -> dict:
    """
    Expand the search radius for a task and re-run proximity broadcast.
    Called when dashboard timer expires or manually expanded.
    
    Args:
        task_id: Task to expand radius for
        new_radius_km: New search radius
        dashboard_url: For callbacks
    
    Returns: Result of trigger_proximity_broadcast with expanded radius
    """
    global _task_expanded_radius, _task_offer_times
    
    # Cap at max radius
    new_radius_km = min(new_radius_km, PROXIMITY_MAX_RADIUS_KM)
    
    # Store expanded radius in app.py tracking
    with _proximity_lock:
        old_radius = _task_expanded_radius.get(task_id, PROXIMITY_DEFAULT_RADIUS_KM)
        _task_expanded_radius[task_id] = new_radius_km
        
        # Reset timer for this task
        _task_offer_times[task_id] = time.time()
    
    # CRITICAL: Also update FleetState so proximity triggers use expanded radius
    if FLEET_STATE_AVAILABLE and fleet_state:
        fleet_state.set_task_expanded_radius(task_id, new_radius_km)
    
    # Clear this task from ALL agents' pending broadcasts before re-broadcasting
    # This frees up their broadcast capacity for the expanded search
    clear_task_from_all_pending(task_id)
    
    log_event(f"[ProximityBroadcast] 📏 Expanded radius for task {task_id}: {old_radius}km → {new_radius_km}km")
    
    # Re-run broadcast with expanded radius (force=True to always broadcast on expand)
    return trigger_proximity_broadcast(
        task_id=task_id,
        triggered_by_agent="radius_expansion",
        dashboard_url=dashboard_url,
        radius_km=new_radius_km,
        force=True
    )


def clean_task_tracking(task_id: str):
    """Clean up tracking data when a task is assigned/completed."""
    global _task_offer_times, _task_expanded_radius, _last_proximity_broadcast, _task_current_agents
    
    with _proximity_lock:
        _task_offer_times.pop(task_id, None)
        _task_expanded_radius.pop(task_id, None)
        _last_proximity_broadcast.pop(task_id, None)
        _task_current_agents.pop(task_id, None)
    
    # Clear from ALL agents' pending broadcasts
    clear_task_from_all_pending(task_id)
    
    # Clear broadcast counts for this task
    clear_task_broadcast_counts(task_id)
    
    # Also clear from FleetState
    if FLEET_STATE_AVAILABLE and fleet_state:
        fleet_state.clear_task_expanded_radius(task_id)


def handle_proximity_acceptance(task_id: str, agent_id: str, agent_name: str, dashboard_url: str) -> dict:
    """
    Handle when an agent accepts a task from proximity broadcast.
    Dashboard has already handled the Tookan assignment - we just clean up tracking.
    
    After acceptance, check if there are unassigned tasks near the agent's
    NEW projected delivery location (proactive chaining).
    
    Returns: {'success': bool}
    """
    if not FLEET_STATE_AVAILABLE or not fleet_state:
        return {'success': False}
    
    task = fleet_state.get_task(task_id)
    if not task:
        return {'success': False}
    
    # Clean up tracking for this task
    clean_task_tracking(task_id)
    
    log_event(f"[ProximityBroadcast] ✅ Task accepted: {task.restaurant_name} → {agent_name}")
    
    # PROJECTED PROXIMITY: After acceptance, agent's projected location changes
    # Check if there are unassigned tasks near the agent's new delivery destination
    agent = fleet_state.get_agent(str(agent_id))
    if agent and agent.has_capacity:
        projected_loc = agent.projected_location
        if projected_loc and projected_loc.lat and projected_loc.lng:
            # Find unassigned tasks near projected location
            search_radius = fleet_state.chain_lookahead_radius_km or 5.0
            tasks_near_projected = []
            for unassigned_task in fleet_state.get_unassigned_tasks():
                if unassigned_task.id == task_id:
                    continue  # Skip the task that was just accepted
                dist = projected_loc.distance_to(unassigned_task.restaurant_location)
                if dist <= search_radius:
                    tasks_near_projected.append((unassigned_task, dist))
            
            if tasks_near_projected:
                tasks_near_projected.sort(key=lambda x: x[1])
                nearest_task, nearest_dist = tasks_near_projected[0]
                print(f"[ProximityBroadcast] 🔮 PROJECTED: {agent_name}'s delivery ends near {nearest_task.restaurant_name} ({nearest_dist:.2f}km)")
                
                # Trigger proximity broadcast for nearest task
                trigger_proximity_broadcast(
                    task_id=nearest_task.id,
                    triggered_by_agent=f"{agent_name} (projected)",
                    dashboard_url=dashboard_url
                )
    
    return {'success': True}


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
    from fleet_optimizer import optimize_fleet, optimize_fleet_with_retry
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
            - False: THOROUGH mode for event-based/on-demand (max_distance_km still enforced)
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
    
    # Use retry version for THOROUGH mode (event-based) to handle ROUTING_FAIL gracefully
    # PROXIMITY mode (prefilter_distance=True) uses regular optimize_fleet for speed
    if prefilter_distance:
        result = optimize_fleet(agents_data, tasks_data, prefilter_distance=True)
    else:
        # THOROUGH mode: Use gradual elimination on ROUTING_FAIL
        result = optimize_fleet_with_retry(agents_data, tasks_data, prefilter_distance=False)
    
    performance_stats["fleet_optimizer_requests"] += 1
    performance_stats["total_requests"] += 1
    performance_stats["algorithm_usage"]["fleet_optimizer"] = \
        performance_stats["algorithm_usage"].get("fleet_optimizer", 0) + 1
    
    return result


def trigger_fleet_optimization(trigger_event: str, trigger_data: dict):
    """
    Trigger fleet optimization and emit updated routes.
    Called automatically when relevant events occur.
    
    Uses THOROUGH mode (prefilter_distance=False) - max_distance_km still enforced
    for comprehensive fleet-wide optimization.
    
    If optimization is already running, queues the event and re-runs after completion
    to ensure the new agent/task is considered.
    """
    global _optimization_running, _events_queued_during_optimization, _queued_dashboard_url
    
    # Block if sync is in progress - queue to run after sync
    if is_sync_in_progress():
        print(f"[WebSocket] ⚠️ Sync in progress - queuing auto-optimization for {trigger_event}")
        queue_optimization_after_sync(trigger_event)
        return
    
    # Check global optimization lock
    with _optimization_running_lock:
        if _optimization_running:
            # QUEUE the event instead of skipping - we'll re-run after current optimization
            _events_queued_during_optimization.append(trigger_event)
            _queued_dashboard_url = trigger_data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
            log_event(f"[WebSocket] 📋 Queued {trigger_event} (will re-run after current optimization)")
            return
        _optimization_running = True
        _events_queued_during_optimization = []  # Clear queue when starting new optimization
    
    print(f"[WebSocket] Auto-triggering fleet optimization due to: {trigger_event}")
    print(f"[WebSocket] Mode: THOROUGH (max_distance_km enforced)")
    performance_stats["auto_optimizations"] += 1
    
    try:
        start_time = time.time()
        
        # Use dashboard_url from trigger_data, env var, or default
        default_dashboard_url = os.environ.get('DASHBOARD_URL', 'http://localhost:8000')
        # EVENT-BASED: Use THOROUGH mode - max_distance_km still enforced as hard constraint
        result = process_fleet_optimization({
            'dashboard_url': trigger_data.get('dashboard_url', default_dashboard_url)
        }, prefilter_distance=False)  # <-- THOROUGH: distance still enforced
        
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
        # Use socketio.emit() instead of emit() to work from background threads
        socketio.emit('fleet:routes_updated', result, namespace='/')
        
        assigned = result.get('metadata', {}).get('tasks_assigned', 0)
        unassigned = result.get('metadata', {}).get('tasks_unassigned', 0)
        log_event(f"[WebSocket] Auto-optimization complete: {assigned} assigned, {unassigned} unassigned in {execution_time:.3f}s")
        
        # Log reasons for unassigned tasks
        unassigned_tasks = result.get('unassigned_tasks', [])
        if unassigned_tasks:
            log_event(f"[WebSocket] 📋 {len(unassigned_tasks)} unassigned task(s) with reasons:")
            for ut in unassigned_tasks:
                task_id = ut.get('task_id', 'unknown')[:20]
                reason = ut.get('reason', 'unknown')
                reason_detail = ut.get('reason_detail', '')
                restaurant = ut.get('restaurant_name', 'Unknown')
                log_event(f"  • {restaurant} ({task_id}...): {reason} - {reason_detail}")
        
        # NOTE: Gradual elimination is now handled inside optimize_fleet_with_retry
        # No need for separate fallback logic here
        
    except Exception as e:
        log_event(f"[WebSocket] Auto-optimization failed: {e}", 'error')
        socketio.emit('fleet:routes_updated', {
            'trigger_type': 'event',
            'trigger_event': trigger_event,
            'trigger_data': trigger_data,
            'error': str(e),
            'success': False,
            'agent_routes': [],
            'unassigned_tasks': []
        }, namespace='/')
    finally:
        # Check if events were queued during this optimization
        _run_queued_optimization_if_needed()


def _run_queued_optimization_if_needed():
    """
    Helper to check for queued events and run follow-up optimization.
    
    IMPORTANT: This function KEEPS _optimization_running = True if there are queued events,
    then runs the follow-up optimization directly (not in a Timer thread). This prevents
    race conditions where new events could start optimizations between releasing the lock
    and the Timer firing.
    """
    global _optimization_running, _events_queued_during_optimization, _queued_dashboard_url
    
    queued_events = []
    dashboard_url = None
    
    with _optimization_running_lock:
        if _events_queued_during_optimization:
            # There are queued events - keep _optimization_running = True
            # to prevent new optimizations from starting
            queued_events = _events_queued_during_optimization.copy()
            dashboard_url = _queued_dashboard_url
            _events_queued_during_optimization = []
            _queued_dashboard_url = None
            # DO NOT set _optimization_running = False yet!
        else:
            # No queued events - safe to release
            _optimization_running = False
    
    # If events were queued, run follow-up optimization DIRECTLY (not in Timer)
    # This ensures the lock is held until the follow-up completes
    if queued_events:
        merged_trigger = '+'.join(set(queued_events))  # e.g., "task:created+agent:online"
        log_event(f"[WebSocket] ▶️ Re-running optimization for queued events: {merged_trigger}")
        
        # Small delay to let state settle (optimistic updates fully visible)
        time.sleep(0.1)
        
        # Run directly - this will call process_fleet_optimization and then
        # call _run_queued_optimization_if_needed again in its finally block
        _run_follow_up_optimization(merged_trigger, dashboard_url)


def _run_follow_up_optimization(trigger_event: str, dashboard_url: str):
    """
    Internal function to run a follow-up optimization for queued events.
    
    This function is called when _optimization_running is already True,
    so it bypasses the lock check and runs the optimization directly.
    """
    global _optimization_running, _events_queued_during_optimization, _queued_dashboard_url
    
    log_event(f"[WebSocket] Follow-up optimization for: {trigger_event}")
    performance_stats["auto_optimizations"] += 1
    
    try:
        start_time = time.time()
        
        # Run optimization with THOROUGH mode
        result = process_fleet_optimization({
            'dashboard_url': dashboard_url
        }, prefilter_distance=False)
        
        execution_time = time.time() - start_time
        
        result['trigger_type'] = 'event'
        result['trigger_event'] = trigger_event
        result['trigger_data'] = {'dashboard_url': dashboard_url}
        result['total_execution_time_seconds'] = round(execution_time, 3)
        
        # OPTIMISTIC UPDATE: Mark tasks as assigned immediately
        for route in result.get('agent_routes', []):
            agent_id = route.get('driver_id')
            agent_name = route.get('driver_name', 'Unknown')
            new_tasks = route.get('assigned_new_tasks', [])
            if new_tasks and agent_id and fleet_state:
                for assigned_task_id in new_tasks:
                    fleet_state.assign_task(assigned_task_id, str(agent_id), agent_name)
        
        # Emit updated routes
        socketio.emit('fleet:routes_updated', result, namespace='/')
        
        assigned = result.get('metadata', {}).get('tasks_assigned', 0)
        unassigned = result.get('metadata', {}).get('tasks_unassigned', 0)
        log_event(f"[WebSocket] Follow-up optimization complete: {assigned} assigned, {unassigned} unassigned in {execution_time:.3f}s")
        
        # Log reasons for unassigned tasks
        unassigned_tasks = result.get('unassigned_tasks', [])
        if unassigned_tasks:
            log_event(f"[WebSocket] 📋 {len(unassigned_tasks)} unassigned task(s) with reasons:")
            for ut in unassigned_tasks:
                task_id = ut.get('task_id', 'unknown')[:20]
                reason = ut.get('reason', 'unknown')
                reason_detail = ut.get('reason_detail', '')
                restaurant = ut.get('restaurant_name', 'Unknown')
                log_event(f"  • {restaurant} ({task_id}...): {reason} - {reason_detail}")
        
    except Exception as e:
        log_event(f"[WebSocket] Follow-up optimization failed: {e}", 'error')
        socketio.emit('fleet:routes_updated', {
            'trigger_type': 'event',
            'trigger_event': trigger_event,
            'error': str(e),
            'success': False,
            'agent_routes': [],
            'unassigned_tasks': []
        }, namespace='/')
    finally:
        # Check for more queued events (chain continues if needed)
        _run_queued_optimization_if_needed()


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
    global _tasks_being_optimized, _optimization_running
    
    # GLOBAL LOCK: Check if any optimization is already running
    with _optimization_running_lock:
        if _optimization_running:
            print(f"[FleetState] ⏳ Another optimization running - skipping proximity for {agent_name}")
            return
    
    # TASK-LEVEL LOCK: Prevent concurrent optimizations for the same task
    with _task_optimization_lock:
        if task_id in _tasks_being_optimized:
            print(f"[FleetState] ⏳ Task {task_name} already being optimized - skipping {agent_name}")
            return
        # Mark this task as being optimized
        _tasks_being_optimized.add(task_id)
    
    # Now acquire the global lock for actual optimization
    with _optimization_running_lock:
        if _optimization_running:
            # Another optimization started between our checks - release task lock and exit
            with _task_optimization_lock:
                _tasks_being_optimized.discard(task_id)
            print(f"[FleetState] ⏳ Another optimization started - skipping proximity for {agent_name}")
            return
        _optimization_running = True
    
    log_event(f"[FleetState] 🚀 Proximity optimization: {agent_name} triggered by {task_name}")
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
                    'has_no_cash_tag': any('nocash' in t.lower().replace('-', '').replace('_', '').replace(' ', '') for t in agent.tags),
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
                    log_event(f"[FleetState] ✅ ASSIGNED: {len(new_tasks)} task(s) to {agent_name}: {new_tasks}")
            
            # Emit result (same base format as event-based, minus unassigned_tasks)
            # Use socketio.emit() instead of emit() to work from background threads
            socketio.emit('fleet:routes_updated', result, namespace='/')
        else:
            print(f"[FleetState] ⚠️ Proximity trigger but no assignment possible for {agent_name}")
            # Don't emit for proximity if nothing assigned - no need to spam dashboard
    
    except Exception as e:
        log_event(f"[FleetState] ❌ Proximity optimization failed: {e}", 'error')
        import traceback
        traceback.print_exc()
    
    finally:
        # Release task lock first
        with _task_optimization_lock:
            _tasks_being_optimized.discard(task_id)
        
        # Check if events were queued during this optimization and run them
        _run_queued_optimization_if_needed()


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

@socketio.on('ping')
def handle_ping(data):
    """
    Respond to dashboard pings to keep Cloud Run connection alive.
    Cloud Run disconnects idle WebSocket connections after 5 minutes.
    """
    emit('pong', {
        'timestamp': data.get('timestamp'),
        'sent_at': data.get('sent_at'),
        'server_time': datetime.now().isoformat()
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
    
    log_event(f"[WebSocket] fleet:sync received from dashboard")
    log_payload('fleet:sync', data)
    
    # Track sync time for debugging stale data
    update_last_sync_time(source=data.get('dashboard_url', 'unknown'))
    
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
            # DEBUG: Log premium fields for first task to verify dashboard is sending them
            first_task = unassigned_tasks[0]
            fee = first_task.get('delivery_fee', 'NOT_SENT')
            tips = first_task.get('tips', 'NOT_SENT')
            print(f"[DEBUG] First unassigned task fields: delivery_fee={fee}, tips={tips}")
            if fee == 'NOT_SENT' or tips == 'NOT_SENT':
                print(f"[DEBUG] ⚠️ Dashboard is NOT sending delivery_fee/tips! Keys present: {list(first_task.keys())}")
            
            # Mark these task IDs as explicitly unassigned by dashboard
            # so we DON'T restore optimistic assignments for them
            unassigned_task_ids = {t.get('id') for t in unassigned_tasks if t.get('id')}
            fleet_state.set_dashboard_unassigned_tasks(unassigned_task_ids)
            
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
            
            # Set chain_lookahead_radius_km to match max_distance_km (they should be the same)
            if 'default_max_distance_km' in config:
                fleet_state.chain_lookahead_radius_km = float(config['default_max_distance_km'])
                print(f"[FleetState] → chain_lookahead_radius_km = {fleet_state.chain_lookahead_radius_km}km (synced with max_distance)")
            
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
            
            # Proximity broadcast settings - ONLY update timeout/radius/limit, NOT the mode
            # Mode should only change via explicit config:update to prevent sync from resetting user's toggle
            proximity_config_keys = ['proximity_task_timeout_seconds', 'proximity_default_radius_km', 'proximity_max_broadcasts_per_agent']
            if any(k in config for k in proximity_config_keys):
                # Filter out proximity_broadcast_enabled to preserve current mode during sync
                safe_config = {k: v for k, v in config.items() if k != 'proximity_broadcast_enabled'}
                if safe_config:
                    update_proximity_broadcast_settings(safe_config)
                print(f"[FleetState] → proximity_broadcast = {'ENABLED' if PROXIMITY_BROADCAST_ENABLED else 'DISABLED'} (preserved), timeout = {PROXIMITY_TASK_TIMEOUT_SECONDS}s, maxBcast = {PROXIMITY_MAX_BROADCASTS_PER_AGENT}")
            
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
                'chain_lookahead_radius_km': fleet_state.chain_lookahead_radius_km,
                'proximity_broadcast_enabled': PROXIMITY_BROADCAST_ENABLED,
                'proximity_task_timeout_seconds': PROXIMITY_TASK_TIMEOUT_SECONDS,
                'proximity_max_broadcasts_per_agent': PROXIMITY_MAX_BROADCASTS_PER_AGENT
            },
            'fleet_stats': stats
        })
        
        # Broadcast to ALL clients (debug dashboards) that fleet state was updated
        socketio.emit('fleet:state_updated', {
            'reason': 'sync',
            'agents': stats['online_agents'],
            'unassigned_tasks': stats['unassigned_tasks'],
            'timestamp': datetime.now().isoformat()
        })
        
        log_event(f"[FleetState] ✅ SYNC COMPLETE: {stats['online_agents']} agents, {stats['unassigned_tasks']} unassigned tasks")
        print(f"[FleetState] Config: max_dist={fleet_state.max_distance_km}km, max_lateness={fleet_state.max_lateness_minutes}min, wallet=${fleet_state.wallet_threshold}")
        
    except Exception as e:
        log_event(f"[FleetState] ❌ SYNC FAILED: {e}", 'error')
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
            fleet_state.chain_lookahead_radius_km = max_dist  # Keep chain lookahead in sync too
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
        
        # chain_lookahead_radius_km is now synced with max_distance_km automatically
        # No need to update it separately in config updates
        
        # Proximity broadcast settings
        if 'proximity_broadcast_enabled' in config:
            old_val = PROXIMITY_BROADCAST_ENABLED
            update_proximity_broadcast_settings({'proximity_broadcast_enabled': config['proximity_broadcast_enabled']})
            changes.append(f"proximity_broadcast_enabled: {old_val} → {PROXIMITY_BROADCAST_ENABLED}")
        
        if 'proximity_task_timeout_seconds' in config:
            old_val = PROXIMITY_TASK_TIMEOUT_SECONDS
            update_proximity_broadcast_settings({'proximity_task_timeout_seconds': config['proximity_task_timeout_seconds']})
            changes.append(f"proximity_task_timeout_seconds: {old_val} → {PROXIMITY_TASK_TIMEOUT_SECONDS}")
        
        if 'proximity_default_radius_km' in config:
            old_val = PROXIMITY_DEFAULT_RADIUS_KM
            update_proximity_broadcast_settings({'proximity_default_radius_km': config['proximity_default_radius_km']})
            changes.append(f"proximity_default_radius_km: {old_val} → {PROXIMITY_DEFAULT_RADIUS_KM}")
        
        if 'proximity_max_broadcasts_per_agent' in config:
            old_val = PROXIMITY_MAX_BROADCASTS_PER_AGENT
            update_proximity_broadcast_settings({'proximity_max_broadcasts_per_agent': config['proximity_max_broadcasts_per_agent']})
            changes.append(f"proximity_max_broadcasts_per_agent: {old_val} → {PROXIMITY_MAX_BROADCASTS_PER_AGENT}")
        
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
                'chain_lookahead_radius_km': fleet_state.chain_lookahead_radius_km,
                'proximity_broadcast_enabled': PROXIMITY_BROADCAST_ENABLED,
                'proximity_task_timeout_seconds': PROXIMITY_TASK_TIMEOUT_SECONDS,
                'proximity_default_radius_km': PROXIMITY_DEFAULT_RADIUS_KM,
                'proximity_max_broadcasts_per_agent': PROXIMITY_MAX_BROADCASTS_PER_AGENT
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

@socketio.on('debug:request_sync')
def handle_debug_request_sync(data):
    """
    Debug dashboard requesting a state sync.
    Broadcasts to all clients to request the main dashboard to send fleet:sync.
    Also returns current state status.
    """
    source = data.get('source', 'unknown')
    client_id = request.sid
    
    log_event(f"[WebSocket] Debug sync requested from {source} (client: {client_id})")
    
    # Check current state
    state_status = {
        'source': source,
        'timestamp': datetime.now().isoformat(),
        'fleet_state_available': FLEET_STATE_AVAILABLE
    }
    
    if FLEET_STATE_AVAILABLE and fleet_state:
        stats = fleet_state.get_stats()
        state_status['current_state'] = {
            'agents': stats.get('total_agents', 0),
            'online_agents': stats.get('online_agents', 0),
            'unassigned_tasks': stats.get('unassigned_tasks', 0),
            'in_progress_tasks': stats.get('in_progress_tasks', 0)
        }
        state_status['is_empty'] = stats.get('total_agents', 0) == 0
        
        if state_status['is_empty']:
            state_status['message'] = 'Fleet state is EMPTY - main dashboard needs to send fleet:sync'
            log_event("[WebSocket] ⚠️ Fleet state is EMPTY - requesting sync from main dashboard", 'warning')
        else:
            state_status['message'] = f"Fleet state has data: {stats.get('online_agents', 0)} agents, {stats.get('unassigned_tasks', 0)} tasks"
    else:
        state_status['is_empty'] = True
        state_status['message'] = 'Fleet state not available'
    
    # Broadcast to ALL connected clients to request sync
    # The main dashboard should listen for this and respond with fleet:sync
    socketio.emit('debug:sync_requested', state_status, namespace='/')
    
    # Also emit acknowledgment to the requesting client
    emit('debug:request_sync_ack', state_status)

@socketio.on('task:get_recommendations')
def handle_get_recommendations(data):
    """
    Handle single-task recommendation request.
    Returns agent recommendations for ONE specific task.
    
    When PROXIMITY_BROADCAST_ENABLED, uses proximity logic for consistency.
    Otherwise, uses batch optimizer.
    """
    client_id = request.sid
    performance_stats["websocket_events"] += 1
    
    if client_id in connected_clients:
        connected_clients[client_id]['events_received'] += 1
    
    request_id = data.get('request_id', str(uuid.uuid4()))
    task_id = data.get('new_task', {}).get('id') or data.get('task_id')
    print(f"[WebSocket] task:get_recommendations from {client_id}, request_id={request_id}, task_id={task_id}")
    
    try:
        start_time = time.time()
        
        # When proximity mode is enabled, use proximity logic for consistency
        if PROXIMITY_BROADCAST_ENABLED and task_id and fleet_state:
            print(f"[WebSocket] Using proximity logic for recommendations (PROXIMITY_BROADCAST_ENABLED=True)")
            
            task = fleet_state.get_task(str(task_id))
            if not task:
                emit('task:recommendations', {
                    'request_id': request_id,
                    'task_id': task_id,
                    'recommendations': [],
                    'error': 'Task not found in fleet state',
                    'proximity_mode': True
                })
                return
            
            # Use proximity broadcast logic to find eligible agents
            dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
            search_radius = _task_expanded_radius.get(task_id, PROXIMITY_DEFAULT_RADIUS_KM)
            
            # Get all online agents and check eligibility
            all_agents = fleet_state.get_online_agents()
            recommendations = []
            agents_checked = 0
            
            for agent in all_agents:
                if not agent.is_online or not agent.has_capacity:
                    continue
                
                agents_checked += 1
                
                # Check distance (use both current and projected)
                current_dist = agent.current_location.distance_to(task.restaurant_location)
                projected_dist = agent.projected_location.distance_to(task.restaurant_location)
                min_dist = min(current_dist, projected_dist)
                
                # Check business rule eligibility
                eligibility = fleet_state._check_eligibility(agent, task, override_max_distance_km=search_radius)
                
                if eligibility is None and min_dist <= search_radius:
                    # Agent is eligible - add to recommendations
                    recommendations.append({
                        'driver_id': agent.id,
                        'name': agent.name,
                        'distance_km': round(min_dist, 2),
                        'current_task_count': len(agent.current_tasks),
                        'available_capacity': agent.available_capacity,
                        'priority': agent.priority,
                        'score': 100 - int(min_dist * 10),  # Simple distance-based score
                        'proximity_eligible': True
                    })
            
            # Sort by distance
            recommendations.sort(key=lambda x: x['distance_km'])
            
            execution_time = time.time() - start_time
            
            emit('task:recommendations', {
                'request_id': request_id,
                'task_id': task_id,
                'recommendations': recommendations,
                'task_context': {
                    'restaurant_name': task.restaurant_name,
                    'customer_name': task.customer_name,
                    'search_radius_km': search_radius
                },
                'performance': {
                    'agents_evaluated': agents_checked,
                    'total_agents': len(all_agents),
                    'execution_time_seconds': round(execution_time, 3),
                    'proximity_mode': True
                },
                'error': None if recommendations else 'No eligible agents within radius'
            })
            
            print(f"[WebSocket] Proximity recommendations: {len(recommendations)} agents in {execution_time:.3f}s")
            return
        
        # Fallback: Use batch optimizer (PROXIMITY_BROADCAST_ENABLED = False)
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
        import traceback
        traceback.print_exc()
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
    
    Uses THOROUGH mode (prefilter_distance=False) - max_distance_km still enforced.
    """
    global _optimization_running
    client_id = request.sid
    performance_stats["websocket_events"] += 1
    
    if client_id in connected_clients:
        connected_clients[client_id]['events_received'] += 1
    
    request_id = data.get('request_id', str(uuid.uuid4()))
    log_payload('fleet:optimize_request', data)
    log_event(f"[WebSocket] fleet:optimize_request from {client_id}, request_id={request_id}")
    
    # Block if sync is in progress - queue to run after sync
    if is_sync_in_progress():
        print(f"[WebSocket] ⚠️ Sync in progress - queuing manual optimize to run after sync")
        queue_optimization_after_sync('fleet:optimize_request')
        # Store the request data for when optimization runs
        global _queued_dashboard_url, _queued_request_id
        _queued_dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
        _queued_request_id = request_id
        emit('fleet:routes_updated', {
            'trigger_type': 'manual',
            'trigger_event': 'fleet:optimize_request',
            'request_id': request_id,
            'message': 'Optimization queued - will run after sync completes',
            'success': True,
            'queued': True,
            'agent_routes': [],
            'unassigned_tasks': []
        })
        return
    
    # Check global optimization lock - manual requests will wait
    with _optimization_running_lock:
        if _optimization_running:
            print(f"[WebSocket] ⏳ Another optimization running - please wait")
            emit('fleet:routes_updated', {
                'trigger_type': 'manual',
                'trigger_event': 'fleet:optimize_request',
                'request_id': request_id,
                'error': 'Another optimization is currently running - please wait',
                'success': False,
                'optimization_in_progress': True,
                'agent_routes': [],
                'unassigned_tasks': []
            })
            return
        _optimization_running = True
    
    print(f"[WebSocket] Mode: THOROUGH (max_distance_km enforced)")
    
    try:
        start_time = time.time()
        # ON-DEMAND: Use THOROUGH mode - max_distance_km still enforced as hard constraint
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
        log_event(f"[WebSocket] Manual optimize complete: {assigned} assigned, {unassigned} unassigned in {execution_time:.3f}s")
        
        # Log reasons for unassigned tasks (PERSISTENT LOG)
        unassigned_tasks = result.get('unassigned_tasks', [])
        if unassigned_tasks:
            log_event(f"[WebSocket] 📋 {len(unassigned_tasks)} unassigned task(s) with reasons:")
            for ut in unassigned_tasks:
                task_id = ut.get('task_id', 'unknown')[:20]
                reason = ut.get('reason', 'unknown')
                reason_detail = ut.get('reason_detail', '')
                restaurant = ut.get('restaurant_name', 'Unknown')
                log_event(f"  • {restaurant} ({task_id}...): {reason} - {reason_detail}")
        
    except Exception as e:
        log_event(f"[WebSocket] Error in fleet:optimize_request: {e}", 'error')
        emit('fleet:routes_updated', {
            'trigger_type': 'manual',
            'trigger_event': 'fleet:optimize_request',
            'request_id': request_id,
            'error': str(e),
            'success': False,
            'agent_routes': [],
            'unassigned_tasks': []
        })
    finally:
        # Check if events were queued during this optimization and run them
        # This ensures task:created, task:declined, etc. events that arrived
        # during an on-demand optimization are not lost
        _run_queued_optimization_if_needed()

# -----------------------------------------------------------------------------
# EVENTS THAT TRIGGER AUTOMATIC FLEET RE-OPTIMIZATION
# -----------------------------------------------------------------------------

@socketio.on('task:created')
def handle_task_created(data):
    """
    New task created → Add to fleet state and check for nearby agents.
    Payload: { task: { id, ... }, dashboard_url }
    """
    update_last_event_time('task:created')
    performance_stats["websocket_events"] += 1
    task_data = data.get('task', {})
    task_id = task_data.get('id', data.get('id', ''))
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
    # Log payload and premium-related fields (these go to persistent logs)
    log_payload('task:created', data)
    delivery_fee = task_data.get('delivery_fee', 0)
    tips = task_data.get('tips', 0)
    is_premium = (float(delivery_fee) >= 18.0 or float(tips) >= 5.0)
    premium_indicator = "⭐ PREMIUM" if is_premium else "regular"
    log_event(f"[WebSocket] task:created: {str(task_id)[:20]}... | fee=${delivery_fee}, tips=${tips} [{premium_indicator}]")
    if is_premium:
        log_event(f"[WebSocket] ⭐ NEW PREMIUM TASK: {task_data.get('restaurant_name', 'Unknown')} | fee=${delivery_fee}, tips=${tips}")
    
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
        print(f"[FleetState] Added task: {task.restaurant_name} → {task.customer_name} | fee=${task.delivery_fee:.2f}, tips=${task.tips:.2f} [{premium_indicator}]")
        
        # Broadcast to all clients so dashboards can update their task lists
        socketio.emit('task:created', {
            'id': task_id,
            'restaurant_name': task.restaurant_name,
            'customer_name': task.customer_name,
            'delivery_fee': task.delivery_fee,
            'tips': task.tips,
            'is_premium': task.is_premium_task,
            'created_at': datetime.now(timezone.utc).isoformat()
        })
        
        # =========================================================================
        # PROXIMITY BROADCAST MODE: Immediately try to find nearby agents
        # =========================================================================
        if PROXIMITY_BROADCAST_ENABLED:
            # Emit a notification that a new task is available
            socketio.emit('task:available', {
                'id': task_id,
                'restaurant_name': task.restaurant_name,
                'customer_name': task.customer_name,
                'location': [task.restaurant_location.lat, task.restaurant_location.lng] if task.restaurant_location else None,
                'pickup_before': task.pickup_before.isoformat() if task.pickup_before else None,
                'delivery_fee': task.delivery_fee,
                'tips': task.tips,
                'is_premium': task.is_premium_task,
                'message': 'New task - checking for nearby agents'
            })
            
            # IMMEDIATELY try to broadcast to nearby agents (don't wait for location updates)
            print(f"[ProximityBroadcast] 📋 New task: {task.restaurant_name} - checking for nearby agents")
            broadcast_result = trigger_proximity_broadcast(
                task_id=task_id,
                triggered_by_agent='task_created',
                dashboard_url=dashboard_url,
                force=True
            )
            
            if broadcast_result.get('success') and not broadcast_result.get('debounced'):
                agents_count = len(broadcast_result.get('agents', []))
                print(f"[ProximityBroadcast] ✅ Immediate broadcast: {task.restaurant_name} → {agents_count} agents")
            elif broadcast_result.get('error'):
                print(f"[ProximityBroadcast] ⚠️ No agents nearby for {task.restaurant_name}: {broadcast_result.get('error')}")
            
            emit('task:created_ack', {
                'id': task_id,
                'received_at': datetime.now().isoformat(),
                'added_to_fleet_state': True,
                'proximity_mode': True,
                'immediate_broadcast': broadcast_result.get('success', False),
                'agents_found': len(broadcast_result.get('agents', []))
            })
            return  # EXIT EARLY - proximity broadcast handles assignment
        
        # =========================================================================
        # FLEET OPTIMIZATION MODE (PROXIMITY_BROADCAST_ENABLED = False)
        # =========================================================================
        # PROACTIVE CHECK: Are any agents already near this task?
        eligible_agents = fleet_state.find_eligible_agents_for_task(task_id)
        nearby_eligible = [
            (agent, dist, reason) for agent, dist, reason in eligible_agents 
            if reason is None and dist <= fleet_state.assignment_radius_km
        ]
        
        if nearby_eligible:
            # Sort by distance (closest first)
            # Note: P1 agents are now included regardless of distance in trigger_proximity_broadcast
            # for premium tasks, so we just sort here
            nearby_eligible.sort(key=lambda x: x[1])
        
        if nearby_eligible:
            best_agent, best_dist, _ = nearby_eligible[0]
            
            # Determine trigger type based on agent's current tasks
            if best_agent.current_tasks:
                trigger_type = "projected_location"  # Agent will be near after current tasks
            else:
                trigger_type = "current_location"  # Agent is idle and near
            
            priority_indicator = " ⭐" if task.is_premium_task and best_agent.priority == 1 else ""
            print(f"[FleetState] 🎯 NEW TASK: {best_agent.name}{priority_indicator} is {best_dist:.2f}km from {task.restaurant_name} ({trigger_type})")
            
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
    
    # EVENT-BASED optimization (only when PROXIMITY_BROADCAST_ENABLED = False)
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
    update_last_event_time('task:declined')
    performance_stats["websocket_events"] += 1
    task_id = data.get('id', '')
    declined_by = data.get('declined_by', [])
    latest_decline = data.get('latest_decline', {})
    latest_agent_id = latest_decline.get('agent_id')
    latest_agent_name = latest_decline.get('agent_name', 'Unknown')
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
    log_payload('task:declined', data)
    log_event(f"[WebSocket] task:declined: {str(task_id)[:20]}... by {latest_agent_name} (total: {len(declined_by)} declines)")
    
    # Clear this task from the declining agent's pending broadcasts
    if latest_agent_id and task_id:
        remove_pending_broadcast(str(latest_agent_id), task_id)
        log_event(f"[ProximityBroadcast] 🔄 Cleared {task_id[:15]}... from {latest_agent_name}'s pending broadcasts")
    
    # Update fleet state - record the declines
    triggered_optimization = False
    if FLEET_STATE_AVAILABLE and fleet_state and task_id:
        fleet_state.record_task_decline(task_id, declined_by, latest_agent_id)
        
        # Get the task to check its restaurant name
        task = fleet_state.get_task(task_id)
        
        # =========================================================================
        # PROXIMITY BROADCAST MODE: Record decline, wait for task:updated to re-broadcast
        # =========================================================================
        if PROXIMITY_BROADCAST_ENABLED and task:
            # NOTE: Tookan won't push to the agent who declined, but WILL push to other agents
            # The task:updated event that follows will trigger a new proximity search
            # which will correctly filter out agents in declined_by list
            log_event(f"[ProximityBroadcast] 📝 Decline recorded: {task.restaurant_name} - {latest_agent_name} (total: {len(declined_by)} declines)")
            
            emit('task:declined_ack', {
                'id': task_id,
                'declined_by': declined_by,
                'received_at': datetime.now().isoformat(),
                'decline_recorded': True,
                'proximity_mode': True,
                'blocked': False,  # NOT blocked - will try other agents via task:updated
                'message': 'Decline recorded, waiting for task:updated to find other agents',
                'triggered_optimization': False
            })
            
            return  # Let task:updated handle re-broadcasting to other agents
        
        # =========================================================================
        # FLEET OPTIMIZATION MODE (PROXIMITY_BROADCAST_ENABLED = False)
        # =========================================================================
        if task:
            # PROACTIVE CHECK: Find another eligible agent near this task
            eligible_agents = fleet_state.find_eligible_agents_for_task(task_id)
            nearby_eligible = [
                (agent, dist, reason) for agent, dist, reason in eligible_agents 
                if reason is None and dist <= fleet_state.assignment_radius_km
            ]
            
            if nearby_eligible:
                # Sort by distance (closest first)
                # Note: P1 agents are now included regardless of distance in trigger_proximity_broadcast
                # for premium tasks, so we just sort here
                nearby_eligible.sort(key=lambda x: x[1])
            
            if nearby_eligible:
                best_agent, best_dist, _ = nearby_eligible[0]
                
                # Determine trigger type
                trigger_type = "projected_location" if best_agent.current_tasks else "current_location"
                
                priority_indicator = " ⭐" if task.is_premium_task and best_agent.priority == 1 else ""
                print(f"[FleetState] 🔄 DECLINED: {best_agent.name}{priority_indicator} is {best_dist:.2f}km from {task.restaurant_name} ({trigger_type})")
                
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
        'triggered_optimization': triggered_optimization
    })


@socketio.on('proximity:timeout')
def handle_proximity_timeout(data):
    """
    Dashboard timer expired for a task - re-run proximity solver.
    
    Payload: { 
        task_id: str,  # Task that timed out
        dashboard_url: str 
    }
    """
    global _task_offer_times
    
    performance_stats["websocket_events"] += 1
    task_id = data.get('task_id')
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
    if not task_id:
        log_event(f"[ProximityBroadcast] ⏰ Timeout with no task_id - ignoring")
        return
    
    # Reset timer for this task
    now = time.time()
    with _proximity_lock:
        _task_offer_times[task_id] = now
    
    # Clear this task from ALL agents' pending broadcasts (they didn't accept in time)
    # This frees up their broadcast capacity for the re-broadcast
    clear_task_from_all_pending(task_id)
    
    log_event(f"[ProximityBroadcast] ⏰ Task {task_id[:20]}... timed out - re-running solver")
    
    # Re-run proximity broadcast (force=True to always broadcast on timeout)
    result = trigger_proximity_broadcast(
        task_id=task_id,
        triggered_by_agent="timeout",
        dashboard_url=dashboard_url,
        force=True
    )
    
    # Emit result
    emit('proximity:timeout_result', {
        'task_id': task_id,
        'success': result.get('success', False),
        'broadcast_count': result.get('broadcast_count', 0),
        'error': result.get('error')
    })


@socketio.on('proximity:expand_radius')
def handle_proximity_expand_radius(data):
    """
    Dashboard requests expanded search radius for a task.
    Called when task has no feasible agents at current radius.
    
    Payload: {
        task_id: str,
        new_radius_km: float,  # Requested new radius
        dashboard_url: str
    }
    """
    performance_stats["websocket_events"] += 1
    task_id = data.get('task_id')
    new_radius_km = data.get('new_radius_km', PROXIMITY_DEFAULT_RADIUS_KM + 2.0)
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
    if not task_id:
        emit('proximity:expand_result', {'success': False, 'error': 'No task_id provided'})
        return
    
    log_event(f"[ProximityBroadcast] 📏 Expanding radius for task {task_id} to {new_radius_km}km")
    
    # Expand radius and re-run solver
    result = expand_task_radius(task_id, new_radius_km, dashboard_url)
    
    emit('proximity:expand_result', {
        'task_id': task_id,
        'new_radius_km': result.get('radius_km', new_radius_km),
        'success': result.get('success', False),
        'broadcast_count': result.get('broadcast_count', 0),
        'error': result.get('error')
    })


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
    update_last_event_time('task:completed')
    performance_stats["websocket_events"] += 1
    task_id = data.get('id', '')
    agent_id = data.get('agent_id')
    agent_name = data.get('agent_name', 'Unknown')
    job_type = data.get('job_type')  # 0=pickup, 1=delivery, None=assume delivery
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
    log_payload('task:completed', data)
    
    # Determine if this is a pickup or delivery completion
    is_pickup_completion = (job_type == 0 or job_type == '0')
    
    if is_pickup_completion:
        log_event(f"[WebSocket] task:pickup_completed: {str(task_id)[:20]}... by {agent_name} ({agent_id})")
        
        # Just mark pickup as complete - DON'T remove task from agent
        restaurant_name = None
        if FLEET_STATE_AVAILABLE and fleet_state and task_id:
            task = fleet_state.mark_pickup_complete(str(task_id))
            if task:
                restaurant_name = task.restaurant_name
                print(f"[FleetState] Pickup completed: {task.restaurant_name} (agent still busy with delivery)")
        
        emit('task:completed_ack', {
            'id': task_id,
            'agent_id': agent_id,
            'received_at': datetime.now().isoformat(),
            'task_removed': False,
            'pickup_completed': True
        })
        
        # Broadcast pickup completion to all clients
        socketio.emit('pickup:completed', {
            'id': task_id,
            'agent_id': agent_id,
            'agent_name': agent_name,
            'restaurant_name': restaurant_name,
            'completed_at': datetime.now(timezone.utc).isoformat()
        })
    else:
        log_event(f"[WebSocket] task:completed: {str(task_id)[:20]}... by {agent_name} ({agent_id})")
        
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
        
        # Broadcast to all clients so dashboards can update
        socketio.emit('task:completed', {
            'id': task_id,
            'agent_id': agent_id,
            'agent_name': agent_name,
            'completed_at': datetime.now(timezone.utc).isoformat()
        })
        
        # Clean up any tracking for this task (including pending broadcasts)
        clean_task_tracking(task_id)
        
        # PROXIMITY BROADCAST: Agent now has capacity - proximity will trigger when they're near tasks
        if PROXIMITY_BROADCAST_ENABLED:
            unassigned_tasks = fleet_state.get_unassigned_tasks()
            if len(unassigned_tasks) > 0:
                print(f"[ProximityBroadcast] {agent_name} completed task, {len(unassigned_tasks)} unassigned tasks - awaiting proximity trigger")
            return  # Skip legacy optimization - proximity handles assignments
        
        # LEGACY: EVENT-BASED optimization (only if marketplace mode disabled)
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
    
    # Broadcast to all clients so dashboards can update
    socketio.emit('task:cancelled', {
        'id': task_id,
        'reason': reason,
        'cancelled_at': datetime.now(timezone.utc).isoformat()
    })
    
    # Clean up any tracking for this task (including pending broadcasts)
    clean_task_tracking(task_id)
    
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
    
    log_payload('task:updated', data)
    
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
        log_event(f"[WebSocket] task:updated: {str(task_id)[:20]}... (task not found in fleet state)")
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
    declined_by_cleared_by_admin = False  # Track if we cleared declined_by due to admin reset
    
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
                    old_agent.current_tasks = [t for t in old_agent.current_tasks if t.id != task_id]
                    # Recalculate agent status
                    if not old_agent.current_tasks:
                        old_agent.status = AgentStatus.IDLE
                    print(f"[FleetState] Task unassigned from {old_agent.name} (status → {new_status})")
                
                # Clear the assignment on the task
                existing_task.assigned_agent_id = None
                changes.append(f'status:{new_status}')
                status_changed_to_unassigned = True
            
            # Always update status to unassigned (even if it was already unassigned)
            existing_task.status = TaskStatus.UNASSIGNED
            
            # IMPORTANT: If admin explicitly sets status to "Unassigned", clear declined_by
            # This unblocks the task for proximity broadcast - admin is resetting it
            # This runs REGARDLESS of whether task had a previous assignment
            if existing_task.declined_by:
                print(f"[FleetState] 🔓 Clearing declined_by for {existing_task.restaurant_name} (admin reset to {new_status})")
                existing_task.declined_by = set()
                changes.append('declined_by_cleared')
                status_changed_to_unassigned = True
            
            # CRITICAL: Always block declined_by field updates when status is "unassigned"
            # This prevents the SECOND update (delivery job) from re-adding declines
            # after the FIRST update (pickup job) already cleared them
            declined_by_cleared_by_admin = True
        
        # Check if task is being reassigned to a different agent
        elif new_assigned_agent and str(new_assigned_agent) != str(old_assigned_agent or ''):
            # First unassign from old agent if any
            if old_assigned_agent:
                old_agent = fleet_state.get_agent(str(old_assigned_agent))
                if old_agent:
                    old_agent.current_tasks = [t for t in old_agent.current_tasks if t.id != task_id]
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
        # IMPORTANT: Don't let incoming max_distance_km reset an expanded radius!
        # Check BOTH tracking systems for expanded radius
        expanded_radius_fleet = fleet_state.get_task_expanded_radius(task_id) if fleet_state else None
        expanded_radius_app = _task_expanded_radius.get(task_id)
        current_expanded = max(expanded_radius_fleet or 0, expanded_radius_app or 0)
        
        if current_expanded and new_dist < current_expanded:
            # Keep the expanded radius, don't downgrade
            app.logger.info(f"[FleetState] 🔒 Preserving expanded radius {current_expanded}km (dashboard sent {new_dist}km)")
        else:
            # Dashboard is expanding radius OR no previous expansion - accept the new value
            # Handle None values safely - use 0 as fallback for comparison
            comparison_value = current_expanded or existing_task.max_distance_km or 0
            if new_dist > comparison_value:
                # This is a radius EXPANSION from dashboard
                app.logger.info(f"[FleetState] 📏 Dashboard expanding radius: {existing_task.max_distance_km}km → {new_dist}km")
                with _proximity_lock:
                    _task_expanded_radius[task_id] = new_dist
                if fleet_state:
                    fleet_state.set_task_expanded_radius(task_id, new_dist)
            
            if existing_task.max_distance_km != new_dist:
                existing_task.max_distance_km = new_dist
                changes.append('max_distance_km')
    
    # CRITICAL: Only process declined_by from data if we didn't just clear it due to admin reset
    # Otherwise the dashboard's stale declined_by data would overwrite our clearing
    if 'declined_by' in data and not declined_by_cleared_by_admin:
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
    
    # Note: customer_name is a read-only property derived from _meta, skip update
    # The customer_name comes from _meta which we can update via update_task()
    
    # Update timestamp
    existing_task.last_updated = datetime.now(timezone.utc)
    
    # Log the update
    changes_str = ', '.join(changes) if changes else 'no changes'
    app.logger.info(f"[WebSocket] task:updated: {restaurant_name} ({str(task_id)[:20]}...) → {changes_str}")
    
    # Determine if optimization should be triggered
    # IMPORTANT: We trigger optimization if:
    # 1. Task status changed to unassigned (needs reassignment)
    # 2. Location changed (might affect which agents are eligible/nearby)
    # 3. Time windows changed (might make previously infeasible assignments feasible)
    # 4. Tags changed (might affect agent eligibility)
    # 5. Declined_by changed (new agents might be eligible)
    
    should_optimize = False
    optimization_reason = None
    
    # Check for changes that warrant optimization
    routing_affecting_changes = {
        'restaurant_location', 'delivery_location',
        'pickup_before', 'delivery_before',
        'tags', 'declined_by', 'max_distance_km'
    }
    
    changed_routing_fields = routing_affecting_changes.intersection(set(changes))
    
    if status_changed_to_unassigned:
        should_optimize = True
        optimization_reason = 'task_unassigned'
    elif changed_routing_fields and existing_task.status == TaskStatus.UNASSIGNED:
        # Only optimize for field changes if task is still unassigned
        should_optimize = True
        optimization_reason = f'fields_changed:{",".join(changed_routing_fields)}'
    
    # Emit acknowledgment to sender
    emit('task:updated_ack', {
        'id': task_id,
        'success': True,
        'changes': changes,
        'status_changed_to_unassigned': status_changed_to_unassigned,
        'triggered_optimization': should_optimize,
        'received_at': datetime.now().isoformat()
    })
    
    # Broadcast to ALL clients (debug dashboards) so they can update
    socketio.emit('task:updated', {
        'id': task_id,
        'restaurant_name': restaurant_name,
        'changes': changes,
        'declined_by': list(existing_task.declined_by) if existing_task.declined_by else [],
        'status': existing_task.status.value if hasattr(existing_task.status, 'value') else str(existing_task.status),
        'timestamp': datetime.now().isoformat()
    })
    
    # If declined_by was cleared (either directly or via status reset), emit unblocked event
    was_unblocked = (
        ('declined_by' in changes and len(existing_task.declined_by) == 0) or
        'declined_by_cleared' in changes
    )
    if was_unblocked:
        app.logger.info(f"[ProximityBroadcast] Task {restaurant_name} unblocked - declined_by cleared")
        socketio.emit('task:unblocked', {
            'task_id': task_id,
            'restaurant_name': restaurant_name,
            'reason': 'admin_reset_to_unassigned' if 'declined_by_cleared' in changes else 'declined_by_cleared',
            'timestamp': datetime.now().isoformat()
        })
    
    # Trigger optimization with all safety nets (debouncing, sync lock, global lock)
    if should_optimize:
        app.logger.info(f"[FleetState] Task {restaurant_name} updated ({optimization_reason}) - triggering optimization")
        
        # Check if proximity broadcast mode is enabled
        if PROXIMITY_BROADCAST_ENABLED:
            # Use proximity broadcast instead of fleet optimization
            app.logger.info(f"[ProximityBroadcast] Task {restaurant_name} updated - triggering proximity broadcast")
            trigger_proximity_broadcast(
                task_id=task_id,
                triggered_by_agent='task_updated',
                dashboard_url=dashboard_url,
                force=True
            )
        else:
            # Use debounced optimization (same safety nets as task:created, task:declined)
            trigger_debounced_optimization(
                trigger_type=f'task:updated:{optimization_reason}',
                dashboard_url=dashboard_url
            )

@socketio.on('task:assigned')
def handle_task_assigned(data):
    """
    Task assigned to agent (manually or via optimization).
    Payload: { id, agent_id, agent_name, assigned_at, dashboard_url }
    """
    update_last_event_time('task:assigned')
    performance_stats["websocket_events"] += 1
    task_id = data.get('id', '')
    agent_id = data.get('agent_id')
    agent_name = data.get('agent_name', 'Unknown')
    dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
    
    log_payload('task:assigned', data)
    
    # Update fleet state (returns None if already assigned to this agent or task doesn't exist)
    task = None
    task_name = 'Unknown'
    if FLEET_STATE_AVAILABLE and fleet_state and task_id and agent_id:
        task = fleet_state.assign_task(task_id, str(agent_id), agent_name)
        if task:
            task_name = task.restaurant_name
    
    log_event(f"[WebSocket] task:assigned: {str(task_id)[:20]}... → {agent_name} ({agent_id})")
    
    if task:
        print(f"[FleetState] Task {task.restaurant_name} assigned to {agent_name}")
    else:
        print(f"[FleetState] Task {task_id[:20]}... assigned to {agent_name} (not in fleet state or duplicate)")
    
    # PROXIMITY BROADCAST: Always clean up tracking for this task
    if PROXIMITY_BROADCAST_ENABLED:
        clean_task_tracking(task_id)
        
        # PROJECTED PROXIMITY: After assignment, agent's projected location changes
        # Check if there are unassigned tasks near the agent's new delivery destination
        if FLEET_STATE_AVAILABLE and fleet_state and agent_id:
            agent = fleet_state.get_agent(str(agent_id))
            if agent and agent.has_capacity:
                # Agent's projected_location is now the delivery location of their last task
                projected_loc = agent.projected_location
                if projected_loc and projected_loc.lat and projected_loc.lng:
                    # Find unassigned tasks near projected location
                    search_radius = fleet_state.chain_lookahead_radius_km or 5.0
                    tasks_near_projected = []
                    for unassigned_task in fleet_state.get_unassigned_tasks():
                        if unassigned_task.id == task_id:
                            continue  # Skip the task we just assigned
                        dist = projected_loc.distance_to(unassigned_task.restaurant_location)
                        if dist <= search_radius:
                            tasks_near_projected.append((unassigned_task, dist))
                    
                    if tasks_near_projected:
                        tasks_near_projected.sort(key=lambda x: x[1])
                        nearest_task, nearest_dist = tasks_near_projected[0]
                        print(f"[ProximityBroadcast] 🔮 PROJECTED: {agent_name}'s delivery ends near {nearest_task.restaurant_name} ({nearest_dist:.2f}km)")
                        
                        # Trigger proximity broadcast for nearest task
                        trigger_proximity_broadcast(
                            task_id=nearest_task.id,
                            triggered_by_agent=f"{agent_name} (projected)",
                            dashboard_url=dashboard_url
                        )
    
    # ALWAYS broadcast assignment to ALL clients (debug pages need to clear Active Broadcasts)
    # Even if task wasn't in our state, the debug page might have it
    socketio.emit('task:assigned', {
        'id': task_id,
        'agent_id': agent_id,
        'agent_name': agent_name,
        'task_name': task_name,
        'assigned_at': datetime.now().isoformat()
    })
    
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
    
    In MARKETPLACE MODE: Agent accepted a task from the marketplace.
    Dashboard has already assigned via Tookan - we update our state and re-broadcast.
    
    Payload: { id, agent_id, agent_name, accepted_at, dashboard_url }
    """
    update_last_event_time('task:accepted')
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
            
            # PROXIMITY BROADCAST: Clean up tracking after acceptance
            if PROXIMITY_BROADCAST_ENABLED:
                handle_proximity_acceptance(task_id, str(agent_id), agent_name, dashboard_url)
            
            # Broadcast acceptance to ALL clients (debug pages, etc.)
            socketio.emit('task:accepted', {
                'id': task_id,
                'agent_id': agent_id,
                'agent_name': agent_name,
                'task_name': task.restaurant_name,
                'accepted_at': datetime.now().isoformat()
            })
    
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
    update_last_event_time('agent:online')
    performance_stats["websocket_events"] += 1
    
    log_payload('agent:online', data)
    
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
    log_event(f"[WebSocket] agent:online: {name} ({agent_id}){priority_str}{tags_str}")
    
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
    
    # Broadcast to ALL clients (debug dashboards) so they can update UI
    agent = fleet_state.get_agent(str(agent_id)) if agent_id and fleet_state else None
    socketio.emit('agent:status_changed', {
        'agent_id': agent_id,
        'name': name,
        'status': 'online',
        'location': location,
        'max_capacity': int(max_capacity) if max_capacity else (agent.max_capacity if agent else 0),
        'current_tasks': len(agent.current_tasks) if agent else 0,
        'tags': tags,
        'priority': priority,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })
    
    # PROXIMITY BROADCAST: Agent online - check if they're already near tasks
    if PROXIMITY_BROADCAST_ENABLED and has_unassigned:
        agent = fleet_state.get_agent(str(agent_id)) if agent_id else None
        if agent and agent.current_location and agent.current_location.lat:
            # Agent came online with a location - trigger proximity check immediately
            nearby_tasks = fleet_state.find_tasks_near_agent(str(agent_id))
            if nearby_tasks:
                log_event(f"[ProximityBroadcast] 🆕 {name} online near {len(nearby_tasks)} tasks - triggering proximity check")
                
                # Get task IDs for batched broadcast
                task_ids = [task.id for task, dist in nearby_tasks]
                
                # Trigger batched broadcast if multiple tasks, single otherwise
                if len(task_ids) > 1:
                    trigger_batched_proximity_broadcast(
                        agent_id=str(agent_id),
                        agent_name=name,
                        task_ids=task_ids,
                        dashboard_url=dashboard_url,
                        force=True
                    )
                elif task_ids:
                    trigger_proximity_broadcast(
                        task_id=task_ids[0],
                        triggered_by_agent=name,
                        dashboard_url=dashboard_url,
                        force=True
                    )
            else:
                print(f"[ProximityBroadcast] 📋 {name} online, no tasks in range - awaiting proximity trigger")
        else:
            print(f"[ProximityBroadcast] 📋 {name} online (no location yet) - awaiting proximity trigger")
        return  # Skip legacy optimization - proximity handles assignments
    
    # LEGACY: EVENT-BASED optimization (only if marketplace mode disabled)
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
    
    log_payload('agent:offline', data)
    log_event(f"[WebSocket] agent:offline: {name} ({agent_id})")
    
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
    
    # Broadcast to ALL clients (debug dashboards) so they can update UI
    socketio.emit('agent:status_changed', {
        'agent_id': agent_id,
        'name': name,
        'status': 'offline',
        'timestamp': datetime.now(timezone.utc).isoformat()
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
            wallet_balance=float(wallet_balance) if wallet_balance is not None else None,
            priority_explicitly_set=True  # agent:update is SOURCE OF TRUTH for priority
        )
        if agent:
            updated = True
            priority_str = f", priority={agent.priority}" if agent.priority else ""
            print(f"[FleetState] ✓ Updated {agent.name}: capacity={agent.max_capacity}, tags={agent.tags}{priority_str}")
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
    update_last_event_time('agent:location_update')
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
            dashboard_url = data.get('dashboard_url', os.environ.get('DASHBOARD_URL', 'http://localhost:8000'))
            
            # PROXIMITY BROADCAST MODE: Run solver and broadcast
            if PROXIMITY_BROADCAST_ENABLED:
                # BATCHED BROADCAST: If agent is near multiple tasks, try to batch them
                if len(eligible_triggers) > 1:
                    # Sort by distance and get all unique task IDs
                    eligible_triggers.sort(key=lambda t: t.distance_km)
                    task_ids = list(set(t.task.id for t in eligible_triggers))
                    task_names = [t.task.restaurant_name for t in eligible_triggers[:3]]  # Log first 3
                    
                    print(f"[ProximityBroadcast] 📦 {name} near {len(task_ids)} tasks: {task_names}{'...' if len(task_ids) > 3 else ''}")
                    
                    # Try batched broadcast - runs solver with ALL tasks
                    result = trigger_batched_proximity_broadcast(
                        agent_id=str(agent_id),
                        agent_name=name,
                        task_ids=task_ids,
                        dashboard_url=dashboard_url
                    )
                    
                    if result.get('success') and result.get('broadcast_count', 0) > 0:
                        print(f"[ProximityBroadcast] ✅ Batched {result.get('broadcast_count')} tasks to {name}")
                    elif not result.get('debounced'):
                        # Batched failed, fall back to single-task broadcast for closest
                        print(f"[ProximityBroadcast] ⚠️ Batch failed, falling back to single task")
                        best_trigger = eligible_triggers[0]
                        trigger_proximity_broadcast(
                            task_id=best_trigger.task.id,
                            triggered_by_agent=name,
                            dashboard_url=dashboard_url
                        )
                else:
                    # Single task - use original single-task broadcast
                    best_trigger = eligible_triggers[0]
                    print(f"[ProximityBroadcast] 📍 {name} is {best_trigger.distance_km:.2f}km from {best_trigger.task.restaurant_name}")
                    
                    trigger_proximity_broadcast(
                        task_id=best_trigger.task.id,
                        triggered_by_agent=name,
                        dashboard_url=dashboard_url
                    )
            else:
                # FLEET OPTIMIZATION MODE: Trigger optimization
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
                    dashboard_url=dashboard_url
                )
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

# Track last event time for debugging stale data issues
_last_sync_time = None
_last_sync_from = None
_last_event_time = None
_last_event_type = None

def update_last_sync_time(source="unknown"):
    """Update the last sync timestamp."""
    global _last_sync_time, _last_sync_from, _last_event_time, _last_event_type
    _last_sync_time = datetime.now()
    _last_sync_from = source
    _last_event_time = _last_sync_time
    _last_event_type = "fleet:sync"

def update_last_event_time(event_type="unknown"):
    """Update the last event timestamp (for any WebSocket event)."""
    global _last_event_time, _last_event_type
    _last_event_time = datetime.now()
    _last_event_type = event_type

@app.route('/fleet-state', methods=['GET'])
def get_fleet_state():
    """Get current fleet state summary."""
    if not FLEET_STATE_AVAILABLE or not fleet_state:
        return jsonify({"error": "Fleet state not available"}), 503
    
    return jsonify(fleet_state.get_state_summary()), 200

@app.route('/fleet-state/health', methods=['GET'])
def get_fleet_state_health():
    """
    Get fleet state health status - useful for debugging stale data.
    Shows when last event occurred and connection status.
    
    Note: This system is EVENT-DRIVEN. fleet:sync only happens on mode switch,
    on-demand, or server start. Real-time updates come via individual events.
    """
    now = datetime.now()
    
    health = {
        'timestamp': now.isoformat(),
        'fleet_state_available': FLEET_STATE_AVAILABLE,
        'connected_clients': len(connected_clients),
        'last_sync_time': _last_sync_time.isoformat() if _last_sync_time else None,
        'last_sync_from': _last_sync_from,
        'seconds_since_last_sync': (now - _last_sync_time).total_seconds() if _last_sync_time else None,
        # Event-driven tracking (more relevant than sync time)
        'last_event_time': _last_event_time.isoformat() if _last_event_time else None,
        'last_event_type': _last_event_type,
        'seconds_since_last_event': (now - _last_event_time).total_seconds() if _last_event_time else None
    }
    
    if FLEET_STATE_AVAILABLE and fleet_state:
        stats = fleet_state.get_stats()
        health['state'] = {
            'agents': stats.get('total_agents', 0),
            'online_agents': stats.get('online_agents', 0),
            'unassigned_tasks': stats.get('unassigned_tasks', 0),
            'in_progress_tasks': stats.get('in_progress_tasks', 0)
        }
        health['is_empty'] = stats.get('total_agents', 0) == 0
        
        # Warn based on EVENT time, not sync time (event-driven architecture)
        if _last_event_time:
            event_age = (now - _last_event_time).total_seconds()
            if event_age > 300:  # 5 minutes without ANY event
                health['warning'] = f'No events received for {int(event_age / 60)} minutes - connection may be broken'
            # No warning if we're receiving events - that's normal for event-driven system
        elif _last_sync_time is None:
            # Never received any sync or event
            health['warning'] = 'No events received since server started - waiting for initial sync'
        # If we have a sync but no recent events, that's fine - events happen when things change
    else:
        health['is_empty'] = True
        health['warning'] = 'Fleet state not available'
    
    return jsonify(health), 200


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
            'max_capacity': agent.max_capacity,  # Individual agent's max tasks
            'capacity': f"{len(agent.current_tasks)}/{agent.max_capacity}",
            'has_capacity': agent.has_capacity,
            'is_idle': agent.is_idle,
            'tags': agent.tags,
            'wallet_balance': agent.wallet_balance,
            'priority': agent.priority,
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
            'is_overdue': task.is_overdue,
            'delivery_fee': task.delivery_fee,
            'tips': task.tips,
            'is_premium': task.is_premium_task,
            'payment_method': task.payment_method,
            'declined_by': list(task.declined_by) if task.declined_by else []
        })
    
    # Sort by urgency
    unassigned.sort(key=lambda t: t['urgency_score'], reverse=True)
    
    declined_count = len([t for t in unassigned if t['declined_by']])
    
    return jsonify({
        'unassigned_count': len(unassigned),
        'urgent_count': len([t for t in unassigned if t['is_urgent']]),
        'overdue_count': len([t for t in unassigned if t['is_overdue']]),
        'declined_count': declined_count,
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
            'optimization_cooldown_seconds': fleet_state.optimization_cooldown_seconds,
            # Proximity broadcast settings
            'proximity_broadcast_enabled': PROXIMITY_BROADCAST_ENABLED,
            'proximity_task_timeout_seconds': PROXIMITY_TASK_TIMEOUT_SECONDS,
            'proximity_default_radius_km': PROXIMITY_DEFAULT_RADIUS_KM,
            'proximity_max_radius_km': PROXIMITY_MAX_RADIUS_KM,
            'proximity_max_broadcasts_per_agent': PROXIMITY_MAX_BROADCASTS_PER_AGENT
        }), 200
    
    # POST - update config
    data = request.get_json() or {}
    
    # max_distance and assignment_radius are now consolidated
    if 'max_distance_km' in data:
        max_dist = float(data['max_distance_km'])
        fleet_state.max_distance_km = max_dist
        fleet_state.assignment_radius_km = max_dist  # Keep them in sync
        fleet_state.chain_lookahead_radius_km = max_dist  # Keep chain lookahead in sync too
    if 'assignment_radius_km' in data:
        # Legacy support - also updates max_distance
        radius = float(data['assignment_radius_km'])
        fleet_state.assignment_radius_km = radius
        fleet_state.max_distance_km = radius
        fleet_state.chain_lookahead_radius_km = radius  # Keep chain lookahead in sync too
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
    
    # Proximity broadcast settings
    if 'proximity_broadcast_enabled' in data or 'proximity_task_timeout_seconds' in data:
        update_proximity_broadcast_settings(data)
    
    return jsonify({
        'message': 'Configuration updated',
        'max_distance_km': fleet_state.max_distance_km,
        'max_lateness_minutes': fleet_state.max_lateness_minutes,
        'max_pickup_delay_minutes': fleet_state.max_pickup_delay_minutes,
        'wallet_threshold': fleet_state.wallet_threshold,
        'max_tasks_per_agent': fleet_state.default_max_capacity,
        'chain_lookahead_radius_km': fleet_state.chain_lookahead_radius_km,
        'optimization_cooldown_seconds': fleet_state.optimization_cooldown_seconds,
        'proximity_broadcast_enabled': PROXIMITY_BROADCAST_ENABLED,
        'proximity_task_timeout_seconds': PROXIMITY_TASK_TIMEOUT_SECONDS,
        'proximity_default_radius_km': PROXIMITY_DEFAULT_RADIUS_KM,
        'proximity_max_radius_km': PROXIMITY_MAX_RADIUS_KM,
        'proximity_max_broadcasts_per_agent': PROXIMITY_MAX_BROADCASTS_PER_AGENT
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


@app.route('/proximity')
def proximity_debug():
    """Serve the proximity broadcast debug UI."""
    return render_template('proximity_broadcast.html')


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
