"""
Ultra-fast recommendation system using heuristic algorithms instead of full OR-Tools optimization.
Provides good recommendations in under 1 second.
"""

import json
import time
import hashlib
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone
from functools import lru_cache
import math

INITIAL_GRACE_PERIOD = 600
DEFAULT_MAX_GRACE_PERIOD = 3600

# Cache for OSRM results
_osrm_cache = {}
_cache_timeout = 300

def get_cache_key(locations: List[Tuple[float, float]]) -> str:
    """Generate a cache key for locations list."""
    locations_str = json.dumps(sorted(locations), sort_keys=True)
    return hashlib.md5(locations_str.encode()).hexdigest()

def is_cache_valid(timestamp: float) -> bool:
    """Check if cache entry is still valid."""
    return time.time() - timestamp < _cache_timeout

def build_osrm_time_matrix_cached(locations: List[Tuple[float, float]]) -> List[List[int]]:
    """Cached OSRM matrix building."""
    cache_key = get_cache_key(locations)
    
    if cache_key in _osrm_cache:
        cached_data, timestamp = _osrm_cache[cache_key]
        if is_cache_valid(timestamp):
            return cached_data
        else:
            del _osrm_cache[cache_key]
    
    from osrm_tables_test import build_osrm_time_matrix
    matrix = build_osrm_time_matrix(locations)
    
    _osrm_cache[cache_key] = (matrix, time.time())
    
    # Cleanup old entries
    if len(_osrm_cache) > 100:
        expired_keys = [k for k, (_, ts) in _osrm_cache.items() if not is_cache_valid(ts)]
        for k in expired_keys:
            del _osrm_cache[k]
    
    return matrix

@lru_cache(maxsize=1000)
def parse_iso_to_seconds_cached(iso_time: str) -> int:
    """Fast cached datetime parsing."""
    if iso_time is None:
        return 999999
    dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00")).astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    return int((dt - now).total_seconds())

def calculate_distance_heuristic(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
    """Fast distance calculation using haversine formula."""
    lat1, lon1 = loc1
    lat2, lon2 = loc2
    
    # Approximate distance in meters
    R = 6371000  # Earth radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2) * math.sin(dlat/2) + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2) * math.sin(dlon/2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    # Convert to approximate travel time (assuming 30 km/h average speed)
    return distance / (30000/3600)  # seconds

def calculate_agent_score_ultra_fast(agent_id: str, agent_data: Dict[str, Any], 
                                   new_task: Dict[str, Any], current_tasks: List[Dict[str, Any]],
                                   time_matrix: Optional[List[List[int]]] = None,
                                   agent_index: int = 0) -> Dict[str, Any]:
    """
    Ultra-fast scoring using heuristics instead of full optimization.
    """
    agent_location = tuple(agent_data.get("current_location", [0, 0]))
    
    # Get tasks assigned to this agent
    agent_tasks = [task for task in current_tasks if agent_id in task.get("assigned_to", [])]
    
    # Basic scoring factors
    score = 100.0
    additional_time = 0.0
    grace_penalty = 0.0
    late_stops = 0
    
    # Factor 1: Distance to new task (30% weight)
    new_task_pickup = tuple(new_task.get("restaurant_location", [0, 0]))
    if time_matrix and agent_index < len(time_matrix):
        # Use OSRM data if available
        pickup_locations = [tuple(task.get("restaurant_location", [0, 0])) for task in [new_task]]
        try:
            # Find new task pickup index in matrix
            # This is a simplified approach
            distance_to_pickup = time_matrix[agent_index][agent_index + len(agent_tasks) + 1]
        except (IndexError, TypeError):
            distance_to_pickup = calculate_distance_heuristic(agent_location, new_task_pickup)
    else:
        distance_to_pickup = calculate_distance_heuristic(agent_location, new_task_pickup)
    
    distance_penalty = min(distance_to_pickup / 1800, 1.0)  # Normalize to 30 minutes max
    score -= distance_penalty * 30
    additional_time = distance_to_pickup / 60.0  # Convert to minutes
    
    # Factor 2: Current workload (25% weight)
    workload_penalty = min(len(agent_tasks) / 5.0, 1.0)  # Normalize to 5 tasks max
    score -= workload_penalty * 25
    
    # Factor 3: Time constraints (25% weight)
    new_task_pickup_deadline = parse_iso_to_seconds_cached(new_task.get("pickup_before"))
    new_task_delivery_deadline = parse_iso_to_seconds_cached(new_task.get("delivery_before"))
    
    # Check if deadlines are tight
    current_time = 0  # Start from now
    estimated_pickup_time = current_time + distance_to_pickup
    
    if new_task_pickup_deadline < estimated_pickup_time:
        grace_penalty += abs(new_task_pickup_deadline - estimated_pickup_time)
        late_stops += 1
    
    if new_task_delivery_deadline < estimated_pickup_time + 1800:  # 30 min delivery estimate
        grace_penalty += abs(new_task_delivery_deadline - (estimated_pickup_time + 1800))
        late_stops += 1
    
    time_penalty = min(grace_penalty / 3600, 1.0)  # Normalize to 1 hour
    score -= time_penalty * 25
    
    # Factor 4: Agent-specific factors (20% weight)
    # Check if agent has no-cash tag but task requires cash
    if agent_data.get("has_tag_no_cash", False) and new_task.get("payment_method") == "cash":
        score -= 15
    
    # Agent status
    if agent_data.get("status") != "online":
        score -= 20
    
    # Same team bonus (small)
    new_task_area = new_task.get("delivery_area", "")
    agent_team = agent_data.get("team_name", "")
    if new_task_area and agent_team and new_task_area.lower() in agent_team.lower():
        score += 5
    
    return {
        "driver_id": agent_id,
        "name": agent_data.get("name", "Unknown"),
        "score": max(0, round(score)),
        "additional_time_minutes": round(additional_time, 1),
        "grace_penalty_seconds": round(grace_penalty),
        "already_late_stops": late_stops,
        "current_task_count": len(agent_tasks),
        "total_route_time_seconds": round(distance_to_pickup + sum(
            calculate_distance_heuristic(
                tuple(task.get("restaurant_location", [0, 0])),
                tuple(task.get("delivery_location", [0, 0]))
            ) for task in agent_tasks
        )),
        "route": []  # Skip detailed route for speed
    }

def recommend_agents_ultra_fast(new_task: Dict[str, Any], agents: List[Dict[str, Any]], 
                               current_tasks: List[Dict[str, Any]], 
                               max_grace_period: int = DEFAULT_MAX_GRACE_PERIOD,
                               use_osrm: bool = True) -> str:
    """
    Ultra-fast recommendation using heuristic algorithms.
    Provides good recommendations in under 1 second.
    """
    start_time = time.time()
    
    recommendations = []
    time_matrix = None
    
    if use_osrm and len(agents) <= 20:  # Only use OSRM for small datasets
        try:
            # Build minimal coordinate list
            coordinates = [tuple(agent.get("current_location", [0, 0])) for agent in agents]
            coordinates.append(tuple(new_task.get("restaurant_location", [0, 0])))
            coordinates.append(tuple(new_task.get("delivery_location", [0, 0])))
            
            time_matrix = build_osrm_time_matrix_cached(coordinates)
        except Exception:
            # Fall back to heuristic distance calculation
            time_matrix = None
    
    # Score each agent using fast heuristics
    for i, agent in enumerate(agents):
        agent_id = agent.get("driver_id", f"Agent{i}")
        
        score_data = calculate_agent_score_ultra_fast(
            agent_id, agent, new_task, current_tasks, time_matrix, i
        )
        recommendations.append(score_data)
    
    # Sort by score and return top 3
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    top_recommendations = recommendations[:3]
    
    execution_time = time.time() - start_time
    
    output = {
        "task_id": new_task.get("id", "unknown"),
        "recommendations": top_recommendations,
        "execution_time_seconds": round(execution_time, 3),
        "cache_hits": len(_osrm_cache),
        "algorithm": "ultra_fast_heuristic"
    }
    
    return json.dumps(output, indent=2)

# Legacy compatibility
def recommend_agents(new_task: Dict[str, Any], agents: List[Dict[str, Any]], 
                    current_tasks: List[Dict[str, Any]], 
                    max_grace_period: int = DEFAULT_MAX_GRACE_PERIOD):
    """Legacy wrapper for compatibility."""
    return recommend_agents_ultra_fast(new_task, agents, current_tasks, max_grace_period)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print("Usage: python OR_tool_prototype_ultra_fast.py new_task.json agents.json current_tasks.json")
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

    print("Running ultra-fast recommendation algorithm...")
    recommendations_json = recommend_agents_ultra_fast(new_task, agents, current_tasks)
    print(recommendations_json) 