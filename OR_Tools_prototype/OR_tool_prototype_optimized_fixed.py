"""
Fixed optimized OR-Tools prototype with proper performance improvements.
This version addresses the issues found in the debug analysis.
"""

INITIAL_GRACE_PERIOD = 600  # 10 minutes
DEFAULT_MAX_GRACE_PERIOD = 3600  # 60 minutes
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import hashlib
import time
from functools import lru_cache

# In-memory cache for OSRM results
_osrm_cache = {}
_cache_timeout = 300  # 5 minutes cache timeout

def get_cache_key(locations: List[Tuple[float, float]]) -> str:
    """Generate a cache key for locations list."""
    locations_str = json.dumps(sorted(locations), sort_keys=True)
    return hashlib.md5(locations_str.encode()).hexdigest()

def is_cache_valid(timestamp: float) -> bool:
    """Check if cache entry is still valid."""
    return time.time() - timestamp < _cache_timeout

@lru_cache(maxsize=1000)
def parse_iso_to_seconds_from_now_cached(iso_time: str, grace_period: int = INITIAL_GRACE_PERIOD):
    """Cached version of time parsing to avoid repeated datetime operations."""
    if iso_time is None:
        return None, 0
    dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00")).astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    diff = (dt - now).total_seconds()
    is_late = diff < 0
    if is_late:
        grace_used = abs(diff)
        return (0, grace_period), grace_used
    return max(0, int(diff)), 0

def build_osrm_time_matrix_cached(locations: List[Tuple[float, float]], use_cache: bool = True) -> List[List[int]]:
    """
    Cached version of OSRM matrix building with option to disable cache for small datasets.
    """
    # For small datasets (< 5 locations), skip caching overhead
    if len(locations) < 5:
        use_cache = False
    
    if not use_cache:
        from osrm_tables_test import build_osrm_time_matrix
        return build_osrm_time_matrix(locations)
    
    cache_key = get_cache_key(locations)
    
    # Check cache first
    if cache_key in _osrm_cache:
        cached_data, timestamp = _osrm_cache[cache_key]
        if is_cache_valid(timestamp):
            return cached_data
        else:
            # Remove expired entry
            del _osrm_cache[cache_key]
    
    # If not in cache or expired, fetch from OSRM
    from osrm_tables_test import build_osrm_time_matrix
    matrix = build_osrm_time_matrix(locations)
    
    # Cache the result
    _osrm_cache[cache_key] = (matrix, time.time())
    
    # Clean up old cache entries (keep cache size manageable)
    if len(_osrm_cache) > 100:
        current_time = time.time()
        expired_keys = [k for k, (_, ts) in _osrm_cache.items() if not is_cache_valid(ts)]
        for k in expired_keys:
            del _osrm_cache[k]
    
    return matrix

def get_adaptive_time_limit(num_agents: int, num_tasks: int) -> int:
    """
    Calculate adaptive time limit based on problem size.
    Smaller problems get less time, larger problems get more.
    """
    if num_agents <= 2 and num_tasks <= 2:
        return 30  # 30 seconds for very small problems
    elif num_agents <= 5 and num_tasks <= 5:
        return 15  # 15 seconds for small problems  
    elif num_agents <= 10 and num_tasks <= 10:
        return 20  # 20 seconds for medium problems
    else:
        return 30  # 30 seconds for large problems

def solve_single_agent_routing_fixed(data, agent_id, assigned_tasks, new_task, adaptive_time_limit: bool = True):
    """
    Fixed version with adaptive time limits and early termination.
    """
    coordinates = data['coordinates']
    time_matrix = data['time_matrix']
    time_windows = data['time_windows']
    pickups_deliveries = data['pickups_deliveries']
    
    num_locations = len(coordinates)
    
    # Adaptive time limit based on problem size
    if adaptive_time_limit:
        num_tasks = len(assigned_tasks) + (1 if new_task else 0)
        time_limit = get_adaptive_time_limit(1, num_tasks)
    else:
        time_limit = 10
    
    # Create routing index manager
    manager = pywrapcp.RoutingIndexManager(num_locations, 1, agent_id)
    
    # Create routing model
    routing = pywrapcp.RoutingModel(manager)
    
    # Create time callback
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add time window constraints
    time = 'Time'
    max_time = max([max(tw) for tw in time_windows.values()]) + 1000
    routing.AddDimension(
        transit_callback_index,
        max_time,  # allow waiting time
        max_time,  # maximum time per vehicle
        False,     # Don't force start cumul to zero
        time)
    time_dimension = routing.GetDimensionOrDie(time)
    
    # Set time windows
    for location_idx, (start_time, end_time) in time_windows.items():
        if location_idx != agent_id:  # Don't set window for depot
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(start_time, end_time)
    
    # Add pickup and delivery constraints
    for pickup_index, delivery_index in pickups_deliveries:
        pickup_node = manager.NodeToIndex(pickup_index)
        delivery_node = manager.NodeToIndex(delivery_index)
        routing.AddPickupAndDelivery(pickup_node, delivery_node)
        routing.solver().Add(
            routing.VehicleVar(pickup_node) == routing.VehicleVar(delivery_node))
        routing.solver().Add(
            time_dimension.CumulVar(pickup_node) <= time_dimension.CumulVar(delivery_node))
    
    # Optimized search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    
    # Use different strategies based on problem size
    if num_locations <= 6:  # Small problems - use faster search
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)
    else:  # Larger problems - use more thorough search
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    
    search_parameters.time_limit.seconds = time_limit
    search_parameters.log_search = False  # Disable logging for speed
    
    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        # Extract solution efficiently
        route_time = solution.ObjectiveValue()
        
        # Build detailed route
        detailed_route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != agent_id and node_index in data['node_metadata']:
                metadata = data['node_metadata'][node_index]
                time_var = time_dimension.CumulVar(index)
                route_time_at_node = solution.Value(time_var)
                detailed_route.append({
                    "task_id": metadata["task_id"],
                    "type": metadata["type"],
                    "location": data['coordinates'][node_index],
                    "deadline_iso": metadata["deadline_iso"],
                    "route_time_seconds": route_time_at_node,
                    "is_new_task": metadata["is_new_task"]
                })
            index = solution.Value(routing.NextVar(index))
        
        return {
            "feasible": True,
            "route_time": route_time,
            "detailed_route": detailed_route
        }
    else:
        return {
            "feasible": False,
            "route_time": float('inf'),
            "detailed_route": []
        }

def build_data_model_fast(new_task: Dict[str, Any], agents: List[Dict[str, Any]], 
                         current_tasks: List[Dict[str, Any]], grace_period: int = INITIAL_GRACE_PERIOD):
    """
    Streamlined data model building for better performance.
    """
    data = {}

    # Build coordinates list and map agent_id to index
    coordinates = []
    agent_starts = []
    agent_id_to_index = {}
    for idx, agent in enumerate(agents):
        loc = agent.get("current_location", None)
        if loc is None or len(loc) != 2:
            raise ValueError(f"Agent {agent.get('driver_id', idx)} missing or invalid location")
        coordinates.append(tuple(loc))
        agent_starts.append(idx)
        agent_id_to_index[agent.get("driver_id", str(idx))] = idx

    # Fast location processing
    location_index = len(coordinates)
    location_map = {}
    pickups_deliveries = []
    time_windows = {}
    node_metadata = {}
    
    def add_location(loc_tuple):
        nonlocal location_index
        if loc_tuple not in location_map:
            location_map[loc_tuple] = location_index
            coordinates.append(loc_tuple)
            location_index += 1
        return location_map[loc_tuple]

    # Process tasks efficiently - avoid copying
    all_tasks = current_tasks[:]
    if all(t.get("id") != new_task.get("id") for t in current_tasks):
        all_tasks.append(new_task)

    task_id_to_indices = {}
    MAX_TIME = 5000
    late_flags = {}
    grace_penalties = {}

    # Simplified task processing
    for task in all_tasks:
        tid = task.get("id")
        ttype = task.get("job_type", "PAIRED")
        pickup_loc = task.get("restaurant_location")
        delivery_loc = task.get("delivery_location")
        pickup_before = task.get("pickup_before")
        delivery_before = task.get("delivery_before")
        is_new_task = tid == new_task.get("id")

        if ttype == "DELIVERY_ONLY":
            delivery_index = add_location(tuple(delivery_loc))
            node_metadata[delivery_index] = {
                "task_id": tid,
                "type": "delivery",
                "deadline_iso": delivery_before,
                "is_new_task": is_new_task
            }
            
            if delivery_before:
                delivery_tw_end, grace_used = parse_iso_to_seconds_from_now_cached(delivery_before, grace_period)
                grace_penalties[delivery_index] = grace_used
                if isinstance(delivery_tw_end, tuple):
                    delivery_tw = delivery_tw_end
                    dt = datetime.fromisoformat(delivery_before.replace("Z", "+00:00")).astimezone(timezone.utc)
                    now = datetime.now(timezone.utc)
                    deadline_seconds = int((dt - now).total_seconds())
                    late_flags[delivery_index] = max(deadline_seconds, 0)
                else:
                    delivery_tw = (0, delivery_tw_end if delivery_tw_end is not None else MAX_TIME)
                    late_flags[delivery_index] = delivery_tw_end if delivery_tw_end is not None else MAX_TIME
            else:
                delivery_tw = (0, MAX_TIME)
                late_flags[delivery_index] = MAX_TIME
                grace_penalties[delivery_index] = 0
            time_windows[delivery_index] = delivery_tw
            task_id_to_indices[tid] = (None, delivery_index)
        else:
            pickup_index = add_location(tuple(pickup_loc))
            delivery_index = add_location(tuple(delivery_loc))

            node_metadata[pickup_index] = {
                "task_id": tid,
                "type": "pickup",
                "deadline_iso": pickup_before,
                "is_new_task": is_new_task
            }
            node_metadata[delivery_index] = {
                "task_id": tid,
                "type": "delivery", 
                "deadline_iso": delivery_before,
                "is_new_task": is_new_task
            }

            # Fast time window processing
            for loc_idx, deadline in [(pickup_index, pickup_before), (delivery_index, delivery_before)]:
                if deadline:
                    tw_end, grace_used = parse_iso_to_seconds_from_now_cached(deadline, grace_period)
                    grace_penalties[loc_idx] = grace_used
                    if isinstance(tw_end, tuple):
                        tw = tw_end
                        dt = datetime.fromisoformat(deadline.replace("Z", "+00:00")).astimezone(timezone.utc)
                        now = datetime.now(timezone.utc)
                        deadline_seconds = int((dt - now).total_seconds())
                        late_flags[loc_idx] = max(deadline_seconds, 0)
                    else:
                        tw = (0, tw_end if tw_end is not None else MAX_TIME)
                        late_flags[loc_idx] = tw_end if tw_end is not None else MAX_TIME
                else:
                    tw = (0, MAX_TIME)
                    late_flags[loc_idx] = MAX_TIME
                    grace_penalties[loc_idx] = 0
                time_windows[loc_idx] = tw

            task_id_to_indices[tid] = (pickup_index, delivery_index)

    # Build pickups_deliveries list
    for tid, (pidx, didx) in task_id_to_indices.items():
        if pidx is not None and didx is not None:
            pickups_deliveries.append((pidx, didx))

    # Use cached OSRM matrix with smart caching
    time_matrix_raw = build_osrm_time_matrix_cached(coordinates, use_cache=(len(coordinates) >= 5))
    time_matrix = [[int(round(cell)) for cell in row] for row in time_matrix_raw]

    # Determine assigned tasks per agent
    assigned_tasks_per_agent = {i: [] for i in range(len(agents))}
    new_task_indices = None
    
    for task in current_tasks:
        assigned_driver = task.get("assigned_driver")
        if assigned_driver and assigned_driver in agent_id_to_index:
            agent_idx = agent_id_to_index[assigned_driver]
            task_id = task.get("id")
            if task_id in task_id_to_indices:
                assigned_tasks_per_agent[agent_idx].append(task_id_to_indices[task_id])
    
    # Get new task indices
    new_task_id = new_task.get("id")
    if new_task_id in task_id_to_indices:
        new_task_indices = task_id_to_indices[new_task_id]

    data.update({
        'coordinates': coordinates,
        'time_matrix': time_matrix,
        'time_windows': time_windows,
        'pickups_deliveries': pickups_deliveries,
        'node_metadata': node_metadata,
        'late_flags': late_flags,
        'grace_penalties': grace_penalties,
        'num_agents': len(agents),
        'agent_id_to_index': agent_id_to_index,
        'assigned_tasks_per_agent': assigned_tasks_per_agent,
        'new_task_indices': new_task_indices
    })
    
    return data, assigned_tasks_per_agent, new_task_indices

def calculate_agent_score_fast(grace_penalty_seconds: float, additional_time_minutes: float, 
                              current_task_count: int, already_late_stops: int,
                              total_route_time: int, max_grace_period: int) -> int:
    """Fast scoring calculation without debug prints."""
    score = 100.0
    
    if grace_penalty_seconds > 0:
        grace_penalty_ratio = min(grace_penalty_seconds / max_grace_period, 1.0)
        score -= grace_penalty_ratio * 25
    
    max_reasonable_additional_time = 60.0
    additional_time_penalty = min(additional_time_minutes / max_reasonable_additional_time, 1.0)
    score -= additional_time_penalty * 30
    
    max_reasonable_tasks = 8
    workload_penalty = min(current_task_count / max_reasonable_tasks, 1.0)
    score -= workload_penalty * 20
    
    max_reasonable_route_time = 28800
    route_efficiency_penalty = min(total_route_time / max_reasonable_route_time, 1.0)
    score -= route_efficiency_penalty * 15
    
    max_reasonable_late_stops = 3
    late_stops_penalty = min(already_late_stops / max_reasonable_late_stops, 1.0)
    score -= late_stops_penalty * 10
    
    return max(0, round(score))

def recommend_agents_optimized_fixed(new_task: Dict[str, Any], agents: List[Dict[str, Any]], 
                                   current_tasks: List[Dict[str, Any]], 
                                   max_grace_period: int = DEFAULT_MAX_GRACE_PERIOD,
                                   enable_debug: bool = False):
    """
    Fixed optimized recommendation engine with proper performance tuning.
    """
    start_time = time.time()
    cache_hits = 0
    
    # For small problems, use simpler approach with single grace period
    is_small_problem = len(agents) <= 3 and len(current_tasks) <= 2
    
    if is_small_problem:
        grace_periods = [INITIAL_GRACE_PERIOD]
    else:
        grace_periods = [INITIAL_GRACE_PERIOD, INITIAL_GRACE_PERIOD * 2, max_grace_period]
    
    best_recommendations = []
    
    for grace_period in grace_periods:
        data, assigned_tasks_per_agent, new_task_indices = build_data_model_fast(
            new_task, agents, current_tasks, grace_period)

        if enable_debug:
            print(f"=== Processing with grace period: {grace_period}s ===")

        # Calculate base route times efficiently
        base_route_times = {}
        for agent in range(data['num_agents']):
            res = solve_single_agent_routing_fixed(
                data, agent, assigned_tasks_per_agent.get(agent, []), None, adaptive_time_limit=True)
            base_route_times[agent] = res['route_time'] if res['feasible'] else float('inf')

        recommendations = []
        feasible_found = False

        # Process agents with adaptive time limits
        for agent in range(data['num_agents']):
            assigned = assigned_tasks_per_agent.get(agent, [])
            res = solve_single_agent_routing_fixed(
                data, agent, assigned, new_task_indices, adaptive_time_limit=True)
            
            if not res['feasible']:
                score = 0
                additional_time_minutes = 0
                grace_penalty_seconds = 0
                already_late_stops = 0
                route = []
            else:
                feasible_found = True
                additional_time = res['route_time'] - base_route_times[agent]
                additional_time_minutes = round(additional_time / 60.0, 1)
                
                # Convert route to old API format
                route = convert_route_to_old_format(res['detailed_route'], agent, data)

                # Fast penalty calculation
                grace_penalty_seconds = 0
                already_late_stops = 0
                agent_tasks = assigned + ([new_task_indices] if new_task_indices else [])
                
                for task_tuple in agent_tasks:
                    if isinstance(task_tuple, tuple):
                        pickup_idx, delivery_idx = task_tuple
                    else:
                        pickup_idx, delivery_idx = task_tuple, None
                    
                    for idx in [pickup_idx, delivery_idx]:
                        if idx and idx in data['grace_penalties']:
                            grace_used = data['grace_penalties'][idx]
                            if grace_used > 0:
                                grace_penalty_seconds += grace_used
                                already_late_stops += 1

                score = calculate_agent_score_fast(
                    grace_penalty_seconds=grace_penalty_seconds,
                    additional_time_minutes=additional_time_minutes,
                    current_task_count=len(assigned),
                    already_late_stops=already_late_stops,
                    total_route_time=res['route_time'],
                    max_grace_period=max_grace_period
                )

            agent_data = agents[agent]
            recommendations.append({
                "driver_id": agent_data.get("driver_id", f"Agent{agent}"),
                "name": agent_data.get("name", f"Agent {agent}"),
                "score": score,
                "additional_time_minutes": additional_time_minutes,
                "grace_penalty_seconds": grace_penalty_seconds,
                "already_late_stops": already_late_stops,
                "route": route
            })

        if feasible_found:
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            best_recommendations = recommendations
            break  # Early exit for small problems

    execution_time = time.time() - start_time
    
    task_id = new_task.get("id", "unknown")
    return {
        "task_id": task_id,
        "recommendations": best_recommendations
    }

def recommend_agents(new_task: Dict[str, Any], agents: List[Dict[str, Any]], 
                    current_tasks: List[Dict[str, Any]], 
                    max_grace_period: int = DEFAULT_MAX_GRACE_PERIOD):
    """
    Main entry point - fixed optimized version.
    """
    result = recommend_agents_optimized_fixed(new_task, agents, current_tasks, max_grace_period)
    return json.dumps(result, indent=2)

def convert_route_to_old_format(detailed_route, agent_index, data):
    """
    Convert our internal route format to the old API format with proper indices and timestamps.
    """
    if not detailed_route:
        return []
    
    converted_route = []
    current_time = datetime.now(timezone.utc)
    
    # Add start entry
    converted_route.append({
        "type": "start",
        "index": agent_index
    })
    
    # Process each route entry
    for entry in detailed_route:
        task_id = entry.get("task_id")
        entry_type = entry.get("type")
        location_index = None
        
        # Find the location index by searching coordinates
        route_location = entry.get("location")
        if route_location:
            for idx, coord in enumerate(data['coordinates']):
                if coord == tuple(route_location):
                    location_index = idx
                    break
        
        # Calculate arrival time from route_time_seconds
        route_time_seconds = entry.get("route_time_seconds", 0)
        arrival_time = current_time + timedelta(seconds=route_time_seconds)
        arrival_time_str = arrival_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Prepare route entry
        is_new_task = entry.get("is_new_task", False)
        
        if entry_type == "pickup":
            if is_new_task:
                route_entry = {
                    "type": "new_task_pickup",
                    "task_id": task_id,
                    "pickup_index": location_index,
                    "arrival_time": arrival_time_str,
                    "lateness": 0
                }
            else:
                route_entry = {
                    "type": "existing_task_pickup", 
                    "task_id": task_id,
                    "pickup_index": location_index,
                    "arrival_time": arrival_time_str,
                    "lateness": 0
                }
        elif entry_type == "delivery":
            if is_new_task:
                route_entry = {
                    "type": "new_task_delivery",
                    "task_id": task_id,
                    "delivery_index": location_index,
                    "arrival_time": arrival_time_str,
                    "lateness": 0
                }
            else:
                route_entry = {
                    "type": "existing_task_delivery",
                    "task_id": task_id, 
                    "delivery_index": location_index,
                    "arrival_time": arrival_time_str,
                    "lateness": 0
                }
        else:
            continue  # Skip unknown types
            
        # Add deadline if available
        deadline_iso = entry.get("deadline_iso")
        if deadline_iso:
            deadline_dt = datetime.fromisoformat(deadline_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
            deadline_str = deadline_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            route_entry["deadline"] = deadline_str
            
        converted_route.append(route_entry)
    
    # Add end entry
    converted_route.append({
        "type": "end",
        "index": agent_index
    })
    
    return converted_route 