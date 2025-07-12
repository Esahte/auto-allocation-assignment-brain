"""
Lightly optimized OR-Tools prototype with targeted improvements:
1. OSRM API call caching (especially for grace period loops)
2. Coordinate deduplication and reuse
3. Batch processing and efficient loops
4. Incremental data model building instead of rebuilding from scratch
"""

INITIAL_GRACE_PERIOD = 600  # 10 minutes
DEFAULT_MAX_GRACE_PERIOD = 3600  # 60 minutes
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone, timedelta
import hashlib
import time
import math

# OSRM cache for API calls - especially useful for grace period loops
_osrm_cache = {}
_cache_timeout = 300  # 5 minutes

# Global coordinate cache to avoid rebuilding same coordinate sets
_coordinate_cache = {}

def get_osrm_cache_key(coordinates: List[Tuple[float, float]]) -> str:
    """Generate cache key for OSRM matrix requests."""
    # Sort coordinates to handle same set in different order
    sorted_coords = sorted(coordinates)
    coords_str = json.dumps(sorted_coords, sort_keys=True)
    return hashlib.md5(coords_str.encode()).hexdigest()

def is_cache_valid(timestamp: float) -> bool:
    """Check if cache entry is still valid."""
    return time.time() - timestamp < _cache_timeout

def build_osrm_time_matrix_cached(coordinates: List[Tuple[float, float]]) -> List[List[float]]:
    """
    Cached OSRM matrix building - avoids repeated API calls for same coordinate sets.
    This is especially beneficial during grace period loops.
    """
    cache_key = get_osrm_cache_key(coordinates)
    
    # Check cache first
    if cache_key in _osrm_cache:
        cached_data, timestamp = _osrm_cache[cache_key]
        if is_cache_valid(timestamp):
            return cached_data
        else:
            # Remove expired entry
            del _osrm_cache[cache_key]
    
    # If not cached or expired, make API call
    from osrm_tables_test import build_osrm_time_matrix
    matrix = build_osrm_time_matrix(coordinates)
    
    # Cache the result
    _osrm_cache[cache_key] = (matrix, time.time())
    
    # Clean up expired entries periodically
    if len(_osrm_cache) > 50:  # Keep cache manageable
        current_time = time.time()
        expired_keys = [k for k, (_, ts) in _osrm_cache.items() if not is_cache_valid(ts)]
        for k in expired_keys:
            del _osrm_cache[k]
    
    return matrix

def parse_iso_to_seconds_from_now(iso_time: str, grace_period=INITIAL_GRACE_PERIOD):
    """Convert ISO time string to seconds from now, and flag if already late."""
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

class DataModelBuilder:
    """
    Incremental data model builder that reuses coordinate mappings and avoids rebuilding from scratch.
    """
    
    def __init__(self):
        self.coordinate_map = {}  # coordinate -> index mapping
        self.coordinates = []     # list of coordinates
        self.next_index = 0
        self.last_agents_signature = None
        self.agent_coordinate_count = 0
    
    def get_coordinate_signature(self, agents: List[Dict[str, Any]]) -> str:
        """Generate signature for agent coordinate set to detect changes."""
        agent_locations = [tuple(agent.get("current_location", [])) for agent in agents]
        return str(sorted(agent_locations))
    
    def reset_if_agents_changed(self, agents: List[Dict[str, Any]]):
        """Reset coordinate mapping if agent set has changed."""
        current_signature = self.get_coordinate_signature(agents)
        if current_signature != self.last_agents_signature:
            self.coordinate_map.clear()
            self.coordinates.clear()
            self.next_index = 0
            self.agent_coordinate_count = 0
            self.last_agents_signature = current_signature
    
    def add_coordinate(self, coord_tuple: Tuple[float, float]) -> int:
        """Add coordinate if not seen before, return its index."""
        if coord_tuple in self.coordinate_map:
            return self.coordinate_map[coord_tuple]
        
        index = self.next_index
        self.coordinate_map[coord_tuple] = index
        self.coordinates.append(coord_tuple)
        self.next_index += 1
        return index
    
    def initialize_agent_coordinates(self, agents: List[Dict[str, Any]]) -> Tuple[List[int], Dict[str, int]]:
        """Initialize agent coordinates and return starts and ID mapping."""
        self.reset_if_agents_changed(agents)
        
        agent_starts = []
        agent_id_to_index = {}
        
        for idx, agent in enumerate(agents):
            loc = agent.get("current_location", None)
            if loc is None or len(loc) != 2:
                raise ValueError(f"Agent {agent.get('driver_id', idx)} missing or invalid location")
            
            coord_index = self.add_coordinate(tuple(loc))
            agent_starts.append(coord_index)
            agent_id_to_index[agent.get("driver_id", str(idx))] = idx
        
        self.agent_coordinate_count = len(agents)
        return agent_starts, agent_id_to_index
    
    def get_current_coordinates(self) -> List[Tuple[float, float]]:
        """Get current coordinate list."""
        return self.coordinates.copy()

# Global data model builder instance
_data_builder = DataModelBuilder()

def build_data_model_incremental(new_task: Dict[str, Any], agents: List[Dict[str, Any]], 
                                current_tasks: List[Dict[str, Any]], grace_period=INITIAL_GRACE_PERIOD):
    """
    Incrementally build data model with coordinate reuse and batch processing.
    """
    data = {}
    
    # Initialize agent coordinates (reuses existing if agents haven't changed)
    agent_starts, agent_id_to_index = _data_builder.initialize_agent_coordinates(agents)
    
    # Batch process all tasks to extract unique locations
    all_tasks = current_tasks.copy()
    if all(t.get("id") != new_task.get("id") for t in current_tasks):
        all_tasks.append(new_task)
    
    # Extract all unique locations from tasks in batch
    task_locations = []
    for task in all_tasks:
        pickup_loc = task.get("restaurant_location")
        delivery_loc = task.get("delivery_location")
        if pickup_loc and len(pickup_loc) == 2:
            task_locations.append(tuple(pickup_loc))
        if delivery_loc and len(delivery_loc) == 2:
            task_locations.append(tuple(delivery_loc))
    
    # Deduplicate and add locations
    unique_locations = list(set(task_locations))
    location_indices = {}
    for loc in unique_locations:
        location_indices[loc] = _data_builder.add_coordinate(loc)
    
    # Build task processing data structures
    pickups_deliveries = []
    time_windows = {}
    node_metadata = {}
    task_id_to_indices = {}
    late_flags = {}
    grace_penalties = {}
    MAX_TIME = 5000
    
    # Batch process time parsing for efficiency
    time_strings = []
    time_to_task_mapping = {}
    
    for task in all_tasks:
        pickup_before = task.get("pickup_before")
        delivery_before = task.get("delivery_before")
        tid = task.get("id")
        
        if pickup_before:
            time_strings.append((pickup_before, tid, "pickup"))
        if delivery_before:
            time_strings.append((delivery_before, tid, "delivery"))
    
    # Parse all times in batch
    parsed_times = {}
    for time_str, tid, ttype in time_strings:
        if time_str not in parsed_times:
            parsed_times[time_str] = parse_iso_to_seconds_from_now(time_str, grace_period)
    
    # Process tasks with pre-parsed times
    for task in all_tasks:
        tid = task.get("id")
        job_type = task.get("job_type", "PAIRED")
        pickup_loc = task.get("restaurant_location")
        delivery_loc = task.get("delivery_location")
        pickup_before = task.get("pickup_before")
        delivery_before = task.get("delivery_before")
        is_new_task = tid == new_task.get("id")
        
        pickup_index = None
        delivery_index = None
        
        if job_type == "DELIVERY_ONLY":
            if delivery_loc is None or len(delivery_loc) != 2:
                raise ValueError(f"Task {tid} missing valid delivery_location")
            delivery_index = location_indices[tuple(delivery_loc)]
            
            node_metadata[delivery_index] = {
                "task_id": tid,
                "type": "delivery",
                "deadline_iso": delivery_before,
                "is_new_task": is_new_task
            }
            
            if delivery_before and delivery_before in parsed_times:
                delivery_tw_end, grace_used = parsed_times[delivery_before]
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
        
        else:  # PAIRED
            if pickup_loc is None or len(pickup_loc) != 2:
                raise ValueError(f"Task {tid} missing valid restaurant_location")
            if delivery_loc is None or len(delivery_loc) != 2:
                raise ValueError(f"Task {tid} missing valid delivery_location")
            
            pickup_index = location_indices[tuple(pickup_loc)]
            delivery_index = location_indices[tuple(delivery_loc)]
            
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
            
            # Process pickup time window
            if pickup_before and pickup_before in parsed_times:
                pickup_tw_end, grace_used = parsed_times[pickup_before]
                grace_penalties[pickup_index] = grace_used
                if isinstance(pickup_tw_end, tuple):
                    pickup_tw = pickup_tw_end
                    dt = datetime.fromisoformat(pickup_before.replace("Z", "+00:00")).astimezone(timezone.utc)
                    now = datetime.now(timezone.utc)
                    deadline_seconds = int((dt - now).total_seconds())
                    late_flags[pickup_index] = max(deadline_seconds, 0)
                else:
                    pickup_tw = (0, pickup_tw_end if pickup_tw_end is not None else MAX_TIME)
                    late_flags[pickup_index] = pickup_tw_end if pickup_tw_end is not None else MAX_TIME
            else:
                pickup_tw = (0, MAX_TIME)
                late_flags[pickup_index] = MAX_TIME
                grace_penalties[pickup_index] = 0
            
            # Process delivery time window
            if delivery_before and delivery_before in parsed_times:
                delivery_tw_end, grace_used = parsed_times[delivery_before]
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
            
            time_windows[pickup_index] = pickup_tw
            time_windows[delivery_index] = delivery_tw
            task_id_to_indices[tid] = (pickup_index, delivery_index)
    
    # Build pickups_deliveries list
    for tid, (pidx, didx) in task_id_to_indices.items():
        if pidx is not None and didx is not None:
            pickups_deliveries.append((pidx, didx))
    
    # Get current coordinates and build OSRM matrix with caching
    current_coordinates = _data_builder.get_current_coordinates()
    time_matrix_raw = build_osrm_time_matrix_cached(current_coordinates)
    time_matrix = [[int(round(cell)) for cell in row] for row in time_matrix_raw]
    
    # Build final data structure
    data.update({
        'agent_starts': agent_starts,
        'pickups_deliveries': pickups_deliveries,
        'time_windows': time_windows,
        'time_matrix': time_matrix,
        'num_agents': len(agent_starts),
        'num_locations': len(current_coordinates),
        'late_flags': late_flags,
        'node_metadata': node_metadata,
        'grace_penalties': grace_penalties,
        'coordinates': current_coordinates  # For reference
    })
    
    # Agent info mapping
    agents_info = {}
    for i, agent in enumerate(agents):
        driver_id = agent.get("driver_id", f"Agent{i}")
        name = agent.get("name", f"Agent {i}")
        agents_info[i] = {"driver_id": driver_id, "name": name}
    data['agents_info'] = agents_info
    
    # Assigned tasks per agent
    assigned_tasks_per_agent = {i: [] for i in range(data['num_agents'])}
    
    for task in current_tasks:
        assigned_driver = task.get("assigned_driver")
        if assigned_driver is None:
            continue
        agent_idx = agent_id_to_index.get(assigned_driver)
        if agent_idx is None:
            continue
        tid = task.get("id")
        if tid not in task_id_to_indices:
            continue
        pidx, didx = task_id_to_indices[tid]
        if pidx is not None and didx is not None:
            assigned_tasks_per_agent[agent_idx].append((pidx, didx))
        elif didx is not None:
            assigned_tasks_per_agent[agent_idx].append((None, didx))
    
    # New task indices
    new_tid = new_task.get("id")
    new_task_indices = task_id_to_indices.get(new_tid)
    
    return data, assigned_tasks_per_agent, new_task_indices

def solve_single_agent_routing(data, agent_id, assigned_tasks, new_task):
    """
    Build and solve routing model for a single agent.
    """
    # Combine assigned tasks and new task
    tasks = assigned_tasks.copy()
    if new_task is not None:
        tasks.append(new_task)

    # Build list of unique nodes
    nodes = set()
    for p, d in tasks:
        if p is not None:
            nodes.add(p)
        if d is not None:
            nodes.add(d)
    
    nodes = list(nodes)
    
    # If no tasks, return trivial solution
    if not nodes:
        return {
            'route_time': 0,
            'route_sequence': [agent_id],
            'lateness': 0,
            'feasible': True,
            'detailed_route': []
        }

    # Create list of all locations for this agent's problem: [agent_start] + task_nodes
    locations = [agent_id] + nodes
    num_locations = len(locations)

    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(num_locations, 1, 0)

    # Create routing model
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a time callback
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        from_location = locations[from_node]
        to_location = locations[to_node]
        return data['time_matrix'][from_location][to_location]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add time window constraints
    time = 'Time'
    max_time = 10000  # Large enough horizon
    routing.AddDimension(
        transit_callback_index,
        max_time,  # allow waiting time
        max_time,  # maximum time per vehicle
        False,     # Don't force start cumul to zero
        time)
    time_dimension = routing.GetDimensionOrDie(time)

    # Set time windows for nodes
    for i, location in enumerate(locations):
        if location in data['time_windows']:
            start_time, end_time = data['time_windows'][location]
            index = manager.NodeToIndex(i)
            time_dimension.CumulVar(index).SetRange(start_time, end_time)

    # Add pickup and delivery constraints
    for pickup_node, delivery_node in data['pickups_deliveries']:
        if pickup_node in locations and delivery_node in locations:
            pickup_index = locations.index(pickup_node)
            delivery_index = locations.index(delivery_node)
            pickup_route_index = manager.NodeToIndex(pickup_index)
            delivery_route_index = manager.NodeToIndex(delivery_index)
            routing.AddPickupAndDelivery(pickup_route_index, delivery_route_index)
            routing.solver().Add(
                routing.VehicleVar(pickup_route_index) == routing.VehicleVar(delivery_route_index))
            routing.solver().Add(
                time_dimension.CumulVar(pickup_route_index) <= time_dimension.CumulVar(delivery_route_index))

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)

    # Match original OR-Tools prototype settings exactly
    search_parameters.time_limit.FromSeconds(2)  # Same 2-second timeout as original
    search_parameters.use_full_propagation = True  # Same as original
    
    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        # Extract route
        route_sequence = []
        total_route_time = solution.ObjectiveValue()
        lateness = 0
        detailed_route = []

        index = routing.Start(0)
        while not routing.IsEnd(index):
            location_index = locations[manager.IndexToNode(index)]
            route_sequence.append(location_index)
            
            # Add detailed route info
            if location_index in data['node_metadata']:
                metadata = data['node_metadata'][location_index]
                time_var = time_dimension.CumulVar(index)
                route_time_at_node = solution.Value(time_var)
                detailed_route.append({
                    "task_id": metadata["task_id"],
                    "type": metadata["type"],
                    "location": data['coordinates'][location_index],
                    "deadline_iso": metadata["deadline_iso"],
                    "route_time_seconds": route_time_at_node,
                    "is_new_task": metadata["is_new_task"]
                })

            index = solution.Value(routing.NextVar(index))

        return {
            'route_time': total_route_time,
            'route_sequence': route_sequence,
            'lateness': lateness,
            'feasible': True,
            'detailed_route': detailed_route
        }
    else:
        return {
            'route_time': float('inf'),
            'route_sequence': [],
            'lateness': float('inf'),
            'feasible': False,
            'detailed_route': []
        }

def calculate_agent_score(grace_penalty_seconds: float, additional_time_minutes: float, 
                         current_task_count: int, already_late_stops: int,
                         total_route_time: int, max_grace_period: int) -> int:
    """
    Calculate agent score with debug information.
    """
    score = 100.0
    
    print(f"  - Grace penalty: {grace_penalty_seconds:.5f}s")
    print(f"  - Additional time: {additional_time_minutes} min")
    print(f"  - Current tasks: {current_task_count}")
    print(f"  - Already late stops: {already_late_stops}")
    print(f"  - Total route time: {total_route_time}s")
    
    # Grace period penalty (0-25 points)
    if grace_penalty_seconds > 0:
        grace_penalty_ratio = min(grace_penalty_seconds / max_grace_period, 1.0)
        score -= grace_penalty_ratio * 25
    
    # Additional time penalty (0-30 points)
    max_reasonable_additional_time = 60.0  # 60 minutes
    additional_time_penalty = min(additional_time_minutes / max_reasonable_additional_time, 1.0)
    score -= additional_time_penalty * 30
    
    # Current workload penalty (0-20 points)
    max_reasonable_tasks = 8
    workload_penalty = min(current_task_count / max_reasonable_tasks, 1.0)
    score -= workload_penalty * 20
    
    # Route efficiency penalty (0-15 points)
    max_reasonable_route_time = 28800  # 8 hours in seconds
    route_efficiency_penalty = min(total_route_time / max_reasonable_route_time, 1.0)
    score -= route_efficiency_penalty * 15
    
    # Late stops penalty (0-10 points)
    max_reasonable_late_stops = 3
    late_stops_penalty = min(already_late_stops / max_reasonable_late_stops, 1.0)
    score -= late_stops_penalty * 10
    
    final_score = max(0, round(score))
    print(f"  - Final score: {final_score}")
    
    return final_score

def recommend_agents(new_task: Dict[str, Any], agents: List[Dict[str, Any]], 
                    current_tasks: List[Dict[str, Any]], 
                    max_grace_period: int = DEFAULT_MAX_GRACE_PERIOD,
                    use_proximity: bool = True,
                    area_type: str = "urban",
                    enable_debug: bool = False,
                    max_distance_km: Optional[float] = None):
    """
    Main recommendation function with light optimizations and proximity filtering.
    
    Args:
        use_proximity: Enable proximity-based agent filtering (default: True)
        area_type: "urban" or "rural" - affects proximity settings
        enable_debug: Enable debug output
        max_distance_km: Maximum distance in kilometers for agent selection (default: None, allows up to 50km)
    """
    start_time = time.time()
    grace_period = INITIAL_GRACE_PERIOD
    best_recommendations = []
    agents_evaluated = 0

    # Proximity-based agent filtering BEFORE data model building
    if use_proximity and len(agents) > 1:  # Changed from > 5 to > 1 to allow proximity filtering with smaller fleets
        proximity_config = get_proximity_config(len(agents), area_type)
        candidate_agents_data = get_proximate_agents(
            new_task, agents, 
            max_candidates=proximity_config["max_candidates"],
            initial_radius_km=proximity_config["initial_radius_km"],
            enable_debug=enable_debug,
            max_distance_km=max_distance_km
        )
        agents_to_evaluate = [agent for _, agent, _ in candidate_agents_data]
        agent_distance_map = {agent['driver_id']: distance for _, agent, distance in candidate_agents_data}
    else:
        agents_to_evaluate = agents
        candidate_agents_data = [(i, agent, 0) for i, agent in enumerate(agents)]
        agent_distance_map = {agent['driver_id']: 0 for agent in agents}

    while grace_period <= max_grace_period:
        if enable_debug:
            print(f"=== DEBUG INFO ===")
            print(f"Agents: {[a.get('driver_id') for a in agents_to_evaluate]} (from {len(agents)} total)")
            print(f"New Task: {new_task.get('id')} {new_task.get('job_type')}")
        
        # Use incremental data model building with filtered agents
        data, assigned_tasks_per_agent, new_task_indices = build_data_model_incremental(
            new_task, agents_to_evaluate, current_tasks, grace_period)

        if new_task_indices is None:
            if enable_debug:
                print(f"Warning: new task {new_task.get('id')} not found in data model")
            break

        if enable_debug:
            print(f"Task indices: {new_task_indices}")
            print(f"Assigned tasks per agent: {assigned_tasks_per_agent}")
            print(f"Time windows: {data['time_windows']}")
            
            # Print time matrix for debugging
            print(f"Time matrix:")
            for row in data['time_matrix']:
                print(row)

        # Calculate base route times for all agents (without new task)
        base_route_times = {}
        for agent in range(data['num_agents']):
            assigned = assigned_tasks_per_agent.get(agent, [])
            base_result = solve_single_agent_routing(data, agent, assigned, None)
            base_route_times[agent] = base_result['route_time'] if base_result['feasible'] else float('inf')

        # Evaluate each agent for the new task
        recommendations = []
        feasible_found = False
        high_score_threshold = 75

        for agent in range(data['num_agents']):
            agents_evaluated += 1
            assigned = assigned_tasks_per_agent.get(agent, [])
            result = solve_single_agent_routing(data, agent, assigned, new_task_indices)
            
            if not result['feasible']:
                score = 0
                additional_time_minutes = 0
                grace_penalty_seconds = 0
                already_late_stops = 0
                route = []
            else:
                feasible_found = True
                additional_time = result['route_time'] - base_route_times[agent]
                additional_time_minutes = round(additional_time / 60.0, 1)
                
                # Convert route to old API format
                route = convert_route_to_old_format(result['detailed_route'], agent, data)

                # Calculate penalties
                grace_penalty_seconds = 0
                already_late_stops = 0
                agent_tasks = assigned + [new_task_indices]
                
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

                score = calculate_agent_score(
                    grace_penalty_seconds=grace_penalty_seconds,
                    additional_time_minutes=additional_time_minutes,
                    current_task_count=len(assigned),
                    already_late_stops=already_late_stops,
                    total_route_time=result['route_time'],
                    max_grace_period=max_grace_period
                )
                
                # Add proximity bonus
                agent_info = data['agents_info'][agent]
                distance_km = agent_distance_map.get(agent_info['driver_id'], 0)
                proximity_bonus = max(0, 5 * (1 - distance_km / 20))
                score = int(score + proximity_bonus)
                
                if enable_debug:
                    print(f"Agent {agent_info['driver_id']} Score Breakdown:")
                    print(f"  Base Score: {score - proximity_bonus:.1f}")
                    print(f"  Distance: {distance_km:.1f}km")
                    print(f"  Proximity Bonus: {proximity_bonus:.1f}")
                    print(f"  Final Score: {score}")
                    print("---")

            agent_info = data['agents_info'][agent]
            recommendations.append({
                "driver_id": agent_info["driver_id"],
                "name": agent_info["name"],
                "score": score,
                "additional_time_minutes": additional_time_minutes,
                "grace_penalty_seconds": grace_penalty_seconds,
                "already_late_stops": already_late_stops,
                "distance_km": agent_distance_map.get(agent_info['driver_id'], 0),
                "proximity_bonus": max(0, 5 * (1 - agent_distance_map.get(agent_info['driver_id'], 0) / 20)),
                "route": route
            })

        if feasible_found:
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            best_recommendations = recommendations
            
            # Early termination if we have high-quality candidates
            if len(best_recommendations) >= 3 and best_recommendations[0]['score'] >= high_score_threshold:
                if enable_debug:
                    print(f"Early termination: found high-quality candidates (score >= {high_score_threshold})")
                break

        # Increase grace period for next iteration
        grace_period = min(grace_period * 2, max_grace_period)

    execution_time = time.time() - start_time
    task_id = new_task.get("id", "unknown")
    
    if best_recommendations:
        result = {
            "task_id": task_id,
            "recommendations": best_recommendations[:3],  # Top 3
            "performance": {
                "execution_time_seconds": round(execution_time, 3),
                "agents_evaluated": len(agents_to_evaluate),
                "total_agents": len(agents),
                "proximity_filtering": use_proximity,
                "or_tools_optimizations": agents_evaluated
            }
        }
        return json.dumps(result, indent=2)
    else:
        result = {
            "task_id": task_id,
            "recommendations": [],
            "error": "No feasible assignment found",
            "performance": {
                "execution_time_seconds": round(execution_time, 3),
                "agents_evaluated": len(agents_to_evaluate),
                "total_agents": len(agents),
                "proximity_filtering": use_proximity,
                "or_tools_optimizations": agents_evaluated
            }
        }
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

def haversine_distance_km(lat1_lng1: Tuple[float, float], lat2_lng2: Tuple[float, float]) -> float:
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    lat1, lng1 = lat1_lng1
    lat2, lng2 = lat2_lng2
    
    # Convert decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])

    # Haversine formula
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    return c * r

def get_proximate_agents(new_task: Dict[str, Any], agents: List[Dict[str, Any]], 
                        max_candidates: int = 8, initial_radius_km: float = 5.0,
                        enable_debug: bool = False, max_distance_km: Optional[float] = None) -> List[Tuple[int, Dict[str, Any], float]]:
    """
    Get agents sorted by proximity to new task, expanding radius as needed.
    Returns: List of (original_agent_index, agent_data, distance_km)
    """
    # Use restaurant location as primary proximity point
    task_location = tuple(new_task['restaurant_location'])
    
    # Calculate distances to all agents
    agent_distances = []
    for i, agent in enumerate(agents):
        agent_loc = tuple(agent['current_location'])
        distance_km = haversine_distance_km(task_location, agent_loc)
        agent_distances.append((i, agent, distance_km))
    
    # Sort by distance
    agent_distances.sort(key=lambda x: x[2])
    
    # Start with closest agents, expand radius if needed
    radius_km = initial_radius_km
    selected_agents = []
    min_candidates = min(3, len(agents))  # Ensure we have at least 3 candidates
    
    # Determine maximum allowed radius (either user-specified or default 50km)
    max_radius_km = max_distance_km if max_distance_km is not None else 50
    
    while len(selected_agents) < max_candidates and radius_km <= max_radius_km:
        for agent_idx, agent, distance in agent_distances:
            if distance <= radius_km and len(selected_agents) < max_candidates:
                # Check if agent not already selected
                if not any(a[0] == agent_idx for a in selected_agents):
                    selected_agents.append((agent_idx, agent, distance))
        
        # If we have minimum candidates and some are reasonably close, we can stop
        if len(selected_agents) >= min_candidates and selected_agents[-1][2] <= 15:
            break
            
        # If we have max candidates, stop
        if len(selected_agents) >= max_candidates:
            break
            
        radius_km *= 1.5  # Expand by 50%
    
    # If still no candidates and max_distance_km is not set, take the closest ones regardless of distance
    if len(selected_agents) < min_candidates and max_distance_km is None:
        selected_agents = agent_distances[:min_candidates]
    elif len(selected_agents) < min_candidates and max_distance_km is not None:
        # With max_distance_km set, only take agents within the limit
        selected_agents = [(idx, agent, dist) for idx, agent, dist in agent_distances 
                          if dist <= max_distance_km][:min_candidates]
    
    if enable_debug:
        distances = [dist for _, _, dist in selected_agents]
        print(f"Proximity filtering: {len(agents)} -> {len(selected_agents)} agents")
        if distances:
            print(f"Distance range: {min(distances):.1f}km - {max(distances):.1f}km")
    
    return selected_agents

def get_proximity_config(num_agents: int, area_type: str = "urban") -> dict:
    """Get optimal proximity settings based on fleet size and area."""
    if area_type == "urban":
        return {
            "initial_radius_km": 3.0,
            "max_candidates": min(10, max(4, num_agents // 10))  # Smaller for OR-Tools
        }
    else:  # rural
        return {
            "initial_radius_km": 15.0, 
            "max_candidates": min(12, max(6, num_agents // 8))
        } 