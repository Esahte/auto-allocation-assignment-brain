INITIAL_GRACE_PERIOD = 600  # 10 minutes
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import json
from typing import Dict, Any, List
from datetime import datetime, timezone, timedelta

def parse_iso_to_seconds_from_now(iso_time: str, grace_period=INITIAL_GRACE_PERIOD):
    """Convert ISO time string to seconds from now, and flag if already late.
    If late, return (0, grace_period) as the time window and the grace period used.
    Returns: (time_window, grace_period_used)
    """
    if iso_time is None:
        return None, 0
    dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00")).astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    diff = (dt - now).total_seconds()
    print(f"[DEBUG] Parsed time for {iso_time}: {diff} seconds from now")
    is_late = diff < 0
    if is_late:
        grace_used = abs(diff)  # How much past deadline
        return (0, grace_period), grace_used
    return max(0, int(diff)), 0

def build_data_model(new_task: Dict[str, Any], agents: List[Dict[str, Any]], current_tasks: List[Dict[str, Any]], grace_period=INITIAL_GRACE_PERIOD):
    """
    Build data model dictionary compatible with solve_single_agent_routing() from structured inputs.
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

    # Tasks: Each task has pickup and delivery locations and time windows
    # We will create unique locations list for pickups and deliveries
    # Also build pickups_deliveries tuples referencing indices in coordinates list

    # Start indices for tasks will be after agent start locations
    location_index = len(coordinates)
    location_map = {}  # (type, task_id) -> index in coordinates

    pickups_deliveries = []
    time_windows = {}
    
    # NEW: Track node metadata for detailed route output
    node_metadata = {}  # node_index -> {task_id, type, deadline_iso, is_new_task}

    # Helper to add location if not already added
    def add_location(loc_tuple):
        nonlocal location_index
        if loc_tuple not in location_map:
            location_map[loc_tuple] = location_index
            coordinates.append(loc_tuple)
            location_index += 1
        return location_map[loc_tuple]

    # Process all current tasks plus the new task (for time windows, we need all tasks)
    all_tasks = current_tasks.copy()
    # Only add new_task if not already in current_tasks by id
    if all(t.get("id") != new_task.get("id") for t in current_tasks):
        all_tasks.append(new_task)

    # We will map id to (pickup_index, delivery_index) or (None, delivery_index) for DELIVERY_ONLY
    task_id_to_indices = {}

    # For time windows, we will convert ISO time strings to seconds from now, with 0 as earliest start
    # We will set a large max window (e.g. 5000) for upper bound if no limit or far future

    MAX_TIME = 5000
    late_flags = {}
    grace_penalties = {}  # Track grace period usage per task/node

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
            # Only delivery location and window
            if delivery_loc is None or len(delivery_loc) != 2:
                raise ValueError(f"Task {tid} missing valid delivery_location")
            delivery_index = add_location(tuple(delivery_loc))
            
            # Store metadata
            node_metadata[delivery_index] = {
                "task_id": tid,
                "type": "delivery",
                "deadline_iso": delivery_before,
                "is_new_task": is_new_task
            }
            
            # Time window only for delivery
            if delivery_before:
                delivery_tw_end, grace_used = parse_iso_to_seconds_from_now(delivery_before, grace_period)
                grace_penalties[delivery_index] = grace_used
                if isinstance(delivery_tw_end, tuple):  # For late: (0, grace_period)
                    delivery_tw = delivery_tw_end
                    # Compute and store the actual deadline value, even if negative, clamped to 0
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
            # PAIRED or default: both pickup and delivery
            if pickup_loc is None or len(pickup_loc) != 2:
                raise ValueError(f"Task {tid} missing valid restaurant_location")
            if delivery_loc is None or len(delivery_loc) != 2:
                raise ValueError(f"Task {tid} missing valid delivery_location")
            pickup_index = add_location(tuple(pickup_loc))
            delivery_index = add_location(tuple(delivery_loc))

            # Store metadata
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

            if pickup_before:
                pickup_tw_end, grace_used = parse_iso_to_seconds_from_now(pickup_before, grace_period)
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
            if delivery_before:
                delivery_tw_end, grace_used = parse_iso_to_seconds_from_now(delivery_before, grace_period)
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

    # Build pickups_deliveries list for all tasks (only those with pickup and delivery)
    for tid, (pidx, didx) in task_id_to_indices.items():
        if pidx is not None and didx is not None:
            pickups_deliveries.append((pidx, didx))

    # Build time matrix using OSRM for all coordinates
    from osrm_tables_test import build_osrm_time_matrix
    time_matrix_raw = build_osrm_time_matrix(coordinates)
    time_matrix = [[int(round(cell)) for cell in row] for row in time_matrix_raw]

    # Compose data dict
    data['agent_starts'] = agent_starts
    data['pickups_deliveries'] = pickups_deliveries
    data['time_windows'] = time_windows
    data['time_matrix'] = time_matrix
    data['num_agents'] = len(agent_starts)
    data['num_locations'] = len(coordinates)
    data['late_flags'] = late_flags
    data['node_metadata'] = node_metadata  # NEW
    data['grace_penalties'] = grace_penalties  # NEW: Track grace period usage

    # Map agent indices to agents info for output
    agents_info = {}
    for i, agent in enumerate(agents):
        driver_id = agent.get("driver_id", f"Agent{i}")
        name = agent.get("name", f"Agent {i}")
        agents_info[i] = {"driver_id": driver_id, "name": name}
    data['agents_info'] = agents_info

    # Prepare assigned tasks per agent from current_tasks
    # Map from agent index to list of (pickup_index, delivery_index) tuples
    assigned_tasks_per_agent = {i: [] for i in range(data['num_agents'])}
    # We assume each task has an assigned_driver that matches driver_id in agents
    driver_id_to_agent_index = {agent.get("driver_id", str(i)): i for i, agent in enumerate(agents)}

    for task in current_tasks:
        assigned_driver = task.get("assigned_driver")
        if assigned_driver is None:
            continue
        agent_idx = driver_id_to_agent_index.get(assigned_driver)
        if agent_idx is None:
            continue
        tid = task.get("id")
        if tid not in task_id_to_indices:
            continue
        pidx, didx = task_id_to_indices[tid]
        if pidx is not None and didx is not None:
            assigned_tasks_per_agent[agent_idx].append((pidx, didx))
        elif didx is not None:
            assigned_tasks_per_agent[agent_idx].append((None, didx))  # Handle DELIVERY_ONLY properly

    # Prepare new_task indices tuple for recommendation call
    new_tid = new_task.get("id")
    new_task_indices = task_id_to_indices.get(new_tid)

    # Print debug info for time windows before returning
    print(f"[DEBUG] Final time windows: {time_windows}")
    # Return all needed data structures
    return data, assigned_tasks_per_agent, new_task_indices

def solve_single_agent_routing(data, agent_id, assigned_tasks, new_task):
    """
    Build and solve routing model for a single agent with assigned tasks plus the new task.
    assigned_tasks: list of (pickup, delivery) tuples currently assigned to the agent
    new_task: (pickup, delivery) tuple to consider adding, pickup or delivery may be None for DELIVERY_ONLY
    Returns: dict with keys 'route_time', 'route_sequence', 'lateness', 'feasible', 'detailed_route'
    """
    # Combine assigned tasks and new task
    tasks = assigned_tasks.copy()
    if new_task is not None:
        tasks.append(new_task)

    # Nodes involved: agent start + all pickups/deliveries in tasks
    # Build list of unique nodes
    nodes = set()
    for p, d in tasks:
        if p is not None:
            nodes.add(p)
        if d is not None:
            nodes.add(d)
    nodes = list(nodes)

    # Map original node index to local index in routing model
    node_to_local = {node: idx+1 for idx, node in enumerate(nodes)}  # index 0 reserved for start
    local_to_node = {idx+1: node for idx, node in enumerate(nodes)}

    # Build time matrix for this subset including start node
    size = len(nodes) + 1
    time_matrix = [[0]*size for _ in range(size)]

    start_node = data['agent_starts'][agent_id]

    # Fill time matrix:
    # Row 0 and col 0 correspond to start_node
    for i in range(size):
        for j in range(size):
            if i == 0 and j == 0:
                time_matrix[i][j] = 0
            elif i == 0:
                # from start_node to node j
                to_node = local_to_node[j]
                time_matrix[i][j] = data['time_matrix'][start_node][to_node]
            elif j == 0:
                # from node i to start_node (not used but set to 0)
                time_matrix[i][j] = 0
            else:
                from_node = local_to_node[i]
                to_node = local_to_node[j]
                time_matrix[i][j] = data['time_matrix'][from_node][to_node]

    # Create routing index manager and model for single vehicle
    manager = pywrapcp.RoutingIndexManager(size, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add time dimension
    routing.AddDimension(
        transit_callback_index,
        600,  # allow waiting
        10000,  # increased max time per route
        False,
        'Time'
    )
    time_dimension = routing.GetDimensionOrDie('Time')

    # Add pickup and delivery pairs
    for p, d in tasks:
        # Skip pairs where either pickup or delivery is None (e.g. DELIVERY_ONLY pickup is None)
        if p is None or d is None:
            continue
        pickup_index = manager.NodeToIndex(node_to_local[p])
        delivery_index = manager.NodeToIndex(node_to_local[d])
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index))
        routing.solver().Add(time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(delivery_index))

    # Apply time windows for pickups and deliveries
    for node in nodes:
        local_idx = node_to_local[node]
        idx = manager.NodeToIndex(local_idx)
        if node in data['time_windows']:
            start, end = data['time_windows'][node]
            time_dimension.CumulVar(idx).SetRange(start, end)
        else:
            # If no window, allow full range
            time_dimension.CumulVar(idx).SetRange(0, 5000)

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(2)
    search_parameters.use_full_propagation = True

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        route_time = solution.Value(time_dimension.CumulVar(routing.End(0)))
        route_sequence = []
        detailed_route = []
        current_time = datetime.now(timezone.utc)
        
        index = routing.Start(0)
        
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            arrival_seconds = solution.Value(time_dimension.CumulVar(index))
            
            if node_idx == 0:
                route_sequence.append(f"Start({start_node})")
                detailed_route.append({
                    "type": "start",
                    "index": start_node
                })
            else:
                orig_node = local_to_node[node_idx]
                route_sequence.append(f"Node {orig_node}")
                
                # Get metadata for this node
                metadata = data['node_metadata'].get(orig_node, {})
                task_id = metadata.get("task_id")
                node_type = metadata.get("type")
                deadline_iso = metadata.get("deadline_iso")
                is_new_task = metadata.get("is_new_task", False)
                
                # Calculate arrival time
                arrival_time = current_time + timedelta(seconds=arrival_seconds)
                
                # No lateness calculation needed - grace periods handle all deadline issues
                lateness_seconds = 0  # Always 0 since grace periods handle everything
                
                # For display purposes, still show original deadline
                if deadline_iso:
                    deadline_dt = datetime.fromisoformat(deadline_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
                
                # Determine route entry type
                if is_new_task and node_type == "pickup":
                    entry_type = "new_task_pickup"
                elif is_new_task and node_type == "delivery":
                    entry_type = "new_task_delivery"
                elif not is_new_task and node_type == "pickup":
                    entry_type = "existing_task_pickup"
                elif not is_new_task and node_type == "delivery":
                    entry_type = "existing_task_delivery"
                else:
                    entry_type = "unknown"
                
                # Format timestamps in YYYY-MM-DDThh:mm:ssZ format
                arrival_time_str = arrival_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                
                route_entry = {
                    "type": entry_type,
                    "task_id": task_id,
                    f"{node_type}_index": orig_node,
                    "arrival_time": arrival_time_str,
                    "lateness": lateness_seconds
                }
                
                if deadline_dt:
                    deadline_str = deadline_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    route_entry["deadline"] = deadline_str
                
                detailed_route.append(route_entry)
                
            index = solution.Value(routing.NextVar(index))
            
        # Add end node
        node_idx = manager.IndexToNode(index)
        route_sequence.append(f"End({start_node})")
        detailed_route.append({
            "type": "end", 
            "index": start_node
        })

        return {
            'route_time': route_time,
            'route_sequence': route_sequence,
            'lateness': 0,  # Always 0 now
            'feasible': True,
            'already_late_count': 0,  # Not used anymore
            'detailed_route': detailed_route
        }
    else:
        # No feasible solution
        return {
            'route_time': None,
            'route_sequence': [],
            'lateness': None,
            'feasible': False,
            'already_late_count': 0,
            'detailed_route': []
        }

def recommend_agents(new_task: Dict[str, Any], agents: List[Dict[str, Any]], current_tasks: List[Dict[str, Any]]):
    """
    Recommend top 3 agents for the new task based on routing optimization.
    new_task: dict with task info
    agents: list of agent dicts
    current_tasks: list of task dicts
    Returns JSON string with task and top agents info.
    """
    grace_period = INITIAL_GRACE_PERIOD
    max_grace = 1800  # 30 minutes
    while grace_period <= max_grace:
        data, assigned_tasks_per_agent, new_task_indices = build_data_model(new_task, agents, current_tasks, grace_period)

        # === DEBUG BLOCK START ===
        print("=== DEBUG INFO ===")
        print("Agents:", [a["driver_id"] for a in agents])
        print("New Task:", new_task["id"], new_task["job_type"])
        print("Task indices:", new_task_indices)
        print("Assigned tasks per agent:", assigned_tasks_per_agent)
        print("Time windows:", data["time_windows"])
        print("Time matrix:")
        for row in data["time_matrix"]:
            print(row)
        # === DEBUG BLOCK END ===

        base_route_times = {}
        # Calculate base route times for each agent with current assigned tasks only
        for agent in range(data['num_agents']):
            res = solve_single_agent_routing(data, agent, assigned_tasks_per_agent.get(agent, []), None)
            base_route_times[agent] = res['route_time'] if res['feasible'] else float('inf')

        recommendations = []
        feasible_found = False

        for agent in range(data['num_agents']):
            assigned = assigned_tasks_per_agent.get(agent, [])
            res = solve_single_agent_routing(data, agent, assigned, new_task_indices)
            if not res['feasible']:
                score = 0
                additional_time_minutes = 0
                lateness_penalty_seconds = 0
                grace_penalty_seconds = 0
                already_late_stops = 0
                route = []
            else:
                feasible_found = True
                additional_time = res['route_time'] - base_route_times[agent]
                additional_time_minutes = round(additional_time / 60.0, 1)
                route = res['detailed_route']

                # Calculate grace period penalties and statistics for all tasks assigned to this agent
                grace_penalty_seconds = 0
                already_late_stops = 0  # Count of tasks that needed grace periods
                agent_tasks = assigned + ([new_task_indices] if new_task_indices else [])
                
                for task_tuple in agent_tasks:
                    pickup_idx, delivery_idx = task_tuple if isinstance(task_tuple, tuple) else (task_tuple, None)
                    
                    # Check pickup
                    if pickup_idx and pickup_idx in data['grace_penalties']:
                        grace_used = data['grace_penalties'][pickup_idx]
                        if grace_used > 0:
                            grace_penalty_seconds += grace_used
                            already_late_stops += 1
                    
                    # Check delivery  
                    if delivery_idx and delivery_idx in data['grace_penalties']:
                        grace_used = data['grace_penalties'][delivery_idx]
                        if grace_used > 0:
                            grace_penalty_seconds += grace_used
                            already_late_stops += 1

                # lateness_penalty_seconds is now the total grace period time used
                lateness_penalty_seconds = grace_penalty_seconds

                # Simplified scoring: Only based on grace period usage
                if grace_penalty_seconds == 0:
                    score = 100  # Perfect score when no grace periods used
                else:
                    # Normalize grace period penalty (max 30 minutes worth)
                    MAX_GRACE_PENALTY = 1800  # 30 minutes worth of grace period
                    norm_grace = min(grace_penalty_seconds / MAX_GRACE_PENALTY, 1.0)
                    score = round((1.0 - norm_grace) * 100)

            agent_info = data['agents_info'].get(agent, {"driver_id": f"Agent{agent}", "name": f"Agent {agent}"})
            recommendations.append({
                "driver_id": agent_info["driver_id"],
                "name": agent_info["name"],
                "score": score,
                "additional_time_minutes": additional_time_minutes,
                "lateness_penalty_seconds": lateness_penalty_seconds,
                "grace_penalty_seconds": grace_penalty_seconds,
                "already_late_stops": already_late_stops,
                "route": route
            })

        if feasible_found:
            # Sort by score descending and pick top 3
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            top_recommendations = recommendations[:3]
            output = {
                "task_id": new_task.get("id", "unknown"),
                "recommendations": top_recommendations
            }
            return json.dumps(output, indent=2)
        else:
            grace_period += 600  # Increase by 10 minutes
            # Also extend the new task's delivery time window
            if new_task_indices and new_task_indices[1] is not None:
                delivery_idx = new_task_indices[1]
                current_window = data["time_windows"].get(delivery_idx, (0, INITIAL_GRACE_PERIOD))
                new_window = (current_window[0], current_window[1] + 600)
                data["time_windows"][delivery_idx] = new_window

    # If no feasible solution found after increasing grace period
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    top_recommendations = recommendations[:3]
    output = {
        "task_id": new_task.get("id", "unknown"),
        "recommendations": top_recommendations
    }
    return json.dumps(output, indent=2)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print("Usage: python OR_tool_prototype.py new_task.json agents.json current_tasks.json")
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

    recommendations_json = recommend_agents(new_task, agents, current_tasks)
    print(recommendations_json)