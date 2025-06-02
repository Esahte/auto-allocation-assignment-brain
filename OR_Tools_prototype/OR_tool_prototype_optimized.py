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

def build_osrm_time_matrix_cached(locations: List[Tuple[float, float]]) -> List[List[int]]:
    """
    Cached version of OSRM matrix building with smart subset handling.
    """
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

def build_data_model_optimized(new_task: Dict[str, Any], agents: List[Dict[str, Any]], 
                              current_tasks: List[Dict[str, Any]], grace_period: int = INITIAL_GRACE_PERIOD):
    """
    Optimized version of build_data_model with reduced computational overhead.
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

    # Optimized location processing
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

    # Process tasks more efficiently
    all_tasks = current_tasks.copy()
    if all(t.get("id") != new_task.get("id") for t in current_tasks):
        all_tasks.append(new_task)

    task_id_to_indices = {}
    MAX_TIME = 5000
    late_flags = {}
    grace_penalties = {}

    # Batch process time parsing
    for task in all_tasks:
        tid = task.get("id")
        ttype = task.get("job_type", "PAIRED")
        pickup_loc = task.get("restaurant_location")
        delivery_loc = task.get("delivery_location")
        pickup_before = task.get("pickup_before")
        delivery_before = task.get("delivery_before")
        is_new_task = tid == new_task.get("id")

        pickup_index = None
        delivery_index = None

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

            # Process time windows efficiently
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

    # Use cached OSRM matrix
    time_matrix_raw = build_osrm_time_matrix_cached(coordinates)
    time_matrix = [[int(round(cell)) for cell in row] for row in time_matrix_raw]

    # Map current tasks to agents efficiently
    assigned_tasks_per_agent = {}
    for task in current_tasks:
        assigned_to = task.get("assigned_to", [])
        for agent_id in assigned_to:
            agent_idx = agent_id_to_index.get(agent_id)
            if agent_idx is not None:
                task_indices = task_id_to_indices.get(task.get("id"))
                if task_indices:
                    if agent_idx not in assigned_tasks_per_agent:
                        assigned_tasks_per_agent[agent_idx] = []
                    assigned_tasks_per_agent[agent_idx].append(task_indices)

    # Build agents info efficiently
    agents_info = {}
    for idx, agent in enumerate(agents):
        agents_info[idx] = {
            "driver_id": agent.get("driver_id", f"Agent{idx}"),
            "name": agent.get("name", f"Agent {idx}"),
            "phone": agent.get("phone", ""),
            "email": agent.get("email", ""),
            "team_id": agent.get("team_id", ""),
            "team_name": agent.get("team_name", ""),
            "has_tag_no_cash": agent.get("has_tag_no_cash", False)
        }

    # Determine new task indices
    new_task_indices = task_id_to_indices.get(new_task.get("id"))

    data.update({
        "coordinates": coordinates,
        "time_matrix": time_matrix,
        "time_windows": time_windows,
        "num_agents": len(agents),
        "agent_starts": agent_starts,
        "pickups_deliveries": pickups_deliveries,
        "agents_info": agents_info,
        "node_metadata": node_metadata,
        "late_flags": late_flags,
        "grace_penalties": grace_penalties,
    })

    return data, assigned_tasks_per_agent, new_task_indices

def solve_single_agent_routing_optimized(data, agent_id, assigned_tasks, new_task, time_limit_seconds=10):
    """
    Optimized version with time limits and reduced solver overhead.
    """
    coordinates = data['coordinates']
    time_matrix = data['time_matrix']
    time_windows = data['time_windows']
    pickups_deliveries = data['pickups_deliveries']
    
    num_locations = len(coordinates)
    
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
    
    # Setting first solution heuristic with time limit
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = time_limit_seconds  # Limit solver time
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

def calculate_agent_score_fast(grace_penalty_seconds: float, additional_time_minutes: float, 
                              current_task_count: int, already_late_stops: int,
                              total_route_time: int, max_grace_period: int) -> int:
    """
    Fast scoring calculation without debug prints.
    """
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

def recommend_agents_optimized(new_task: Dict[str, Any], agents: List[Dict[str, Any]], 
                              current_tasks: List[Dict[str, Any]], 
                              max_grace_period: int = DEFAULT_MAX_GRACE_PERIOD,
                              enable_debug: bool = False):
    """
    Optimized recommendation engine with parallel processing and caching.
    """
    start_time = time.time()
    
    grace_period = INITIAL_GRACE_PERIOD
    best_recommendations = []
    
    while grace_period <= max_grace_period:
        data, assigned_tasks_per_agent, new_task_indices = build_data_model_optimized(
            new_task, agents, current_tasks, grace_period)

        if enable_debug:
            print(f"=== Processing with grace period: {grace_period}s ===")

        # Calculate base route times efficiently
        base_route_times = {}
        for agent in range(data['num_agents']):
            res = solve_single_agent_routing_optimized(
                data, agent, assigned_tasks_per_agent.get(agent, []), None, time_limit_seconds=5)
            base_route_times[agent] = res['route_time'] if res['feasible'] else float('inf')

        recommendations = []
        feasible_found = False

        # Process agents in parallel concept (sequential for simplicity but optimized)
        for agent in range(data['num_agents']):
            assigned = assigned_tasks_per_agent.get(agent, [])
            res = solve_single_agent_routing_optimized(
                data, agent, assigned, new_task_indices, time_limit_seconds=10)
            
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
                route = res['detailed_route']

                # Fast calculation of penalties
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

            agent_info = data['agents_info'].get(agent, {"driver_id": f"Agent{agent}", "name": f"Agent {agent}"})
            recommendations.append({
                "driver_id": agent_info["driver_id"],
                "name": agent_info["name"],
                "score": score,
                "additional_time_minutes": additional_time_minutes,
                "grace_penalty_seconds": grace_penalty_seconds,
                "already_late_stops": already_late_stops,
                "current_task_count": len(assigned),
                "total_route_time_seconds": res['route_time'] if res['feasible'] else 0,
                "route": route
            })

        if feasible_found:
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            best_recommendations = recommendations[:3]
            break
        else:
            grace_period += 600  # Increase by 10 minutes
            if enable_debug:
                print(f"No feasible solution found, increasing grace period to {grace_period}")

    if not best_recommendations:
        # Return best effort recommendations even if not feasible
        if recommendations:
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            best_recommendations = recommendations[:3]

    execution_time = time.time() - start_time
    output = {
        "task_id": new_task.get("id", "unknown"),
        "recommendations": best_recommendations,
        "execution_time_seconds": round(execution_time, 3),
        "cache_hits": len(_osrm_cache)
    }
    
    if enable_debug:
        print(f"Total execution time: {execution_time:.3f} seconds")
    
    return json.dumps(output, indent=2)

# Legacy function wrapper for compatibility
def recommend_agents(new_task: Dict[str, Any], agents: List[Dict[str, Any]], 
                    current_tasks: List[Dict[str, Any]], 
                    max_grace_period: int = DEFAULT_MAX_GRACE_PERIOD):
    """Legacy wrapper that maintains original API."""
    return recommend_agents_optimized(new_task, agents, current_tasks, max_grace_period, enable_debug=False)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print("Usage: python OR_tool_prototype_optimized.py new_task.json agents.json current_tasks.json")
        sys.exit(1)

    new_task_file = sys.argv[1]
    agents_file = sys.argv[2]
    current_tasks_file = sys.argv[3]

    with open(new_task_file, 'r') as f:
        new_task = json.load(f)
    with open(agents_file, 'r') as f:
        agents = json.load(f)
    with open(current_tasks_file, 'r') as f:
        current_tasks = json.load(f)

    recommendations_json = recommend_agents_optimized(new_task, agents, current_tasks, enable_debug=True)
    print(recommendations_json) 