"""
Batch-optimized OR-Tools prototype for agent recommendation without task reassignment.

This version evaluates ALL agents simultaneously to recommend the best ones for a NEW task,
while keeping all existing task assignments fixed (no reassignment between agents).

Key Features:
- Batch processing of all agents at once
- No task reassignment - agents keep their existing tasks  
- Only recommends where to INSERT the new task into each agent's route
- Returns top recommendations with scores and route details
"""

import numpy as np
import json
import time
import requests
import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from functools import lru_cache
import hashlib

# Constants
INITIAL_GRACE_PERIOD = 600  # 10 minutes
DEFAULT_MAX_GRACE_PERIOD = 3600  # 60 minutes

# OSRM cache for API calls
_osrm_cache = {}
_cache_timeout = 300  # 5 minutes

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
                        max_candidates: int = 10, initial_radius_km: float = 5.0,
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
            "max_candidates": min(12, max(5, num_agents // 8))
        }
    else:  # rural
        return {
            "initial_radius_km": 15.0, 
            "max_candidates": min(15, max(8, num_agents // 6))
        }

def get_cache_key(locations: List[Tuple[float, float]]) -> str:
    """Generate cache key for location set."""
    sorted_locations = sorted(locations)
    location_str = str(sorted_locations)
    return hashlib.md5(location_str.encode()).hexdigest()

def build_osrm_time_matrix_cached(locations: List[Tuple[float, float]], use_cache: bool = True) -> List[List[int]]:
    """
    Build time matrix using OSRM API with caching.
    Returns travel times in seconds between all location pairs.
    """
    if len(locations) <= 1:
        return [[0]]
    
    cache_key = get_cache_key(locations)
    current_time = time.time()
    
    # Check cache
    if use_cache and cache_key in _osrm_cache:
        cached_data, cache_time = _osrm_cache[cache_key]
        if current_time - cache_time < _cache_timeout:
            return cached_data
    
    try:
        # Format coordinates for OSRM API
        coordinates = ";".join([f"{lng},{lat}" for lat, lng in locations])
        url = f"http://router.project-osrm.org/table/v1/driving/{coordinates}?annotations=duration"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('code') != 'Ok':
            raise Exception(f"OSRM API error: {data.get('message', 'Unknown error')}")
        
        # Convert to integer seconds
        durations = data['durations']
        time_matrix = [[int(duration) for duration in row] for row in durations]
        
        # Cache the result
        if use_cache:
            _osrm_cache[cache_key] = (time_matrix, current_time)
        
        return time_matrix
        
    except Exception as e:
        print(f"OSRM API call failed: {e}")
        # Fallback to approximate distances
        return build_approximate_time_matrix(locations)

def build_approximate_time_matrix(locations: List[Tuple[float, float]]) -> List[List[int]]:
    """Fallback approximate time matrix based on Haversine distance."""
    def haversine_distance(lat1, lng1, lat2, lng2):
        R = 6371  # Earth's radius in kilometers
        dlat = np.radians(lat2 - lat1)
        dlng = np.radians(lng2 - lng1)
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    n = len(locations)
    matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                lat1, lng1 = locations[i]
                lat2, lng2 = locations[j]
                distance_km = haversine_distance(lat1, lng1, lat2, lng2)
                # Assume average speed of 30 km/h in urban areas
                time_seconds = int((distance_km / 30) * 3600)
                matrix[i][j] = time_seconds
    
    return matrix

def parse_time(time_str: str) -> datetime:
    """Parse ISO 8601 timestamp."""
    if time_str.endswith('Z'):
        time_str = time_str[:-1] + '+00:00'
    return datetime.fromisoformat(time_str)

def evaluate_agent_for_new_task(
    agent: Dict[str, Any],
    new_task: Dict[str, Any], 
    current_tasks: List[Dict[str, Any]],
    time_matrix: List[List[int]],
    location_to_index: Dict[Tuple[float, float], int],
    max_grace_period: int
) -> Dict[str, Any]:
    """
    Evaluate how adding the new task affects a specific agent's route.
    Does NOT reassign existing tasks - only finds best insertion points.
    """
    agent_id = agent['driver_id']
    agent_name = agent.get('name', agent_id)
    
    # Find tasks assigned to this agent
    agent_tasks = [task for task in current_tasks if task.get('assigned_driver') == agent_id]
    
    # Build agent's current route
    current_route = build_agent_route(agent, agent_tasks, location_to_index)
    
    if not current_route:
        # Agent has no existing tasks - simple case
        return evaluate_empty_agent(agent, new_task, time_matrix, location_to_index, max_grace_period)
    
    # Find best insertion points for new task
    best_insertion = find_best_insertion_point(
        agent, new_task, current_route, time_matrix, location_to_index, max_grace_period
    )
    
    return best_insertion

def build_agent_route(
    agent: Dict[str, Any], 
    agent_tasks: List[Dict[str, Any]], 
    location_to_index: Dict[Tuple[float, float], int]
) -> List[Dict[str, Any]]:
    """Build the current route for an agent with their existing tasks."""
    if not agent_tasks:
        return []
    
    agent_location = tuple(agent['current_location'])
    agent_index = location_to_index[agent_location]
    
    route_points = []
    
    # Add start point
    route_points.append({
        'type': 'start',
        'index': agent_index,
        'location': agent_location
    })
    
    # Add existing tasks (in chronological order by pickup time)
    sorted_tasks = sorted(agent_tasks, key=lambda t: parse_time(t['pickup_before']))
    
    for task in sorted_tasks:
        restaurant_loc = tuple(task['restaurant_location'])
        delivery_loc = tuple(task['delivery_location'])
        
        # Add pickup
        route_points.append({
            'type': 'existing_task_pickup',
            'task_id': task['id'],
            'index': location_to_index[restaurant_loc],
            'location': restaurant_loc,
            'deadline': parse_time(task['pickup_before'])
        })
        
        # Add delivery
        route_points.append({
            'type': 'existing_task_delivery', 
            'task_id': task['id'],
            'index': location_to_index[delivery_loc],
            'location': delivery_loc,
            'deadline': parse_time(task['delivery_before'])
        })
    
    # Add end point (same as start)
    route_points.append({
        'type': 'end',
        'index': agent_index,
        'location': agent_location
    })
    
    return route_points

def evaluate_empty_agent(
    agent: Dict[str, Any],
    new_task: Dict[str, Any],
    time_matrix: List[List[int]],
    location_to_index: Dict[Tuple[float, float], int],
    max_grace_period: int
) -> Dict[str, Any]:
    """Evaluate agent with no existing tasks."""
    
    agent_location = tuple(agent['current_location'])
    restaurant_location = tuple(new_task['restaurant_location'])
    delivery_location = tuple(new_task['delivery_location'])
    
    agent_index = location_to_index[agent_location]
    restaurant_index = location_to_index[restaurant_location]
    delivery_index = location_to_index[delivery_location]
    
    # Calculate times
    current_time = datetime.now(timezone.utc)
    
    # Time to reach restaurant
    travel_to_restaurant = time_matrix[agent_index][restaurant_index]
    pickup_arrival = current_time + timedelta(seconds=travel_to_restaurant)
    
    # Time from restaurant to delivery
    restaurant_to_delivery = time_matrix[restaurant_index][delivery_index]
    delivery_arrival = pickup_arrival + timedelta(seconds=restaurant_to_delivery)
    
    # Parse deadlines
    pickup_deadline = parse_time(new_task['pickup_before'])
    delivery_deadline = parse_time(new_task['delivery_before'])
    
    # Calculate lateness and penalties
    pickup_lateness = max(0, (pickup_arrival - pickup_deadline).total_seconds())
    delivery_lateness = max(0, (delivery_arrival - delivery_deadline).total_seconds())
    
    total_lateness = pickup_lateness + delivery_lateness
    grace_penalty = min(total_lateness, max_grace_period)
    already_late_stops = (1 if pickup_lateness > 0 else 0) + (1 if delivery_lateness > 0 else 0)
    
    # Calculate score (0-100)
    total_time = travel_to_restaurant + restaurant_to_delivery
    time_penalty = min(total_time / 3600, 1.0) * 30  # Max 30 point penalty for long routes
    lateness_penalty = min(grace_penalty / max_grace_period, 1.0) * 70  # Max 70 point penalty for lateness
    
    score = max(0, 100 - time_penalty - lateness_penalty)
    
    # Build route
    route = [
        {
            "type": "start",
            "index": agent_index
        },
        {
            "type": "new_task_pickup",
            "task_id": new_task['id'],
            "pickup_index": restaurant_index,
            "arrival_time": pickup_arrival.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "deadline": new_task['pickup_before'],
            "lateness": int(pickup_lateness)
        },
        {
            "type": "new_task_delivery", 
            "task_id": new_task['id'],
            "delivery_index": delivery_index,
            "arrival_time": delivery_arrival.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "deadline": new_task['delivery_before'],
            "lateness": int(delivery_lateness)
        },
        {
            "type": "end",
            "index": agent_index
        }
    ]
    
    return {
        "driver_id": agent['driver_id'],
        "name": agent.get('name', agent['driver_id']),
        "score": int(score),
        "additional_time_minutes": total_time / 60.0,
        "grace_penalty_seconds": int(grace_penalty),
        "already_late_stops": already_late_stops,
        "route": route
    }

def find_best_insertion_point(
    agent: Dict[str, Any],
    new_task: Dict[str, Any], 
    current_route: List[Dict[str, Any]],
    time_matrix: List[List[int]],
    location_to_index: Dict[Tuple[float, float], int],
    max_grace_period: int
) -> Dict[str, Any]:
    """
    Find the best insertion points for new task pickup and delivery in agent's existing route.
    """
    
    restaurant_location = tuple(new_task['restaurant_location'])
    delivery_location = tuple(new_task['delivery_location'])
    restaurant_index = location_to_index[restaurant_location]
    delivery_index = location_to_index[delivery_location]
    
    pickup_deadline = parse_time(new_task['pickup_before'])
    delivery_deadline = parse_time(new_task['delivery_before'])
    
    best_score = -1
    best_result = None
    
    # Try all possible insertion positions
    route_length = len(current_route)
    
    # Pickup can be inserted at positions 1 to route_length-1 (after start, before end)
    for pickup_pos in range(1, route_length):
        # Delivery must be after pickup
        for delivery_pos in range(pickup_pos + 1, route_length):
            
            # Create new route with insertions
            new_route = current_route[:pickup_pos] + [
                {
                    'type': 'new_task_pickup',
                    'task_id': new_task['id'],
                    'index': restaurant_index,
                    'location': restaurant_location,
                    'deadline': pickup_deadline
                }
            ] + current_route[pickup_pos:delivery_pos] + [
                {
                    'type': 'new_task_delivery',
                    'task_id': new_task['id'], 
                    'index': delivery_index,
                    'location': delivery_location,
                    'deadline': delivery_deadline
                }
            ] + current_route[delivery_pos:]
            
            # Calculate timing for this route
            result = calculate_route_timing(new_route, time_matrix, max_grace_period)
            
            if result and (best_score < 0 or result['score'] > best_score):
                best_score = result['score']
                best_result = result
    
    # Format for API response
    if best_result:
        best_result['driver_id'] = agent['driver_id']
        best_result['name'] = agent.get('name', agent['driver_id'])
    
    return best_result

def calculate_route_timing(
    route: List[Dict[str, Any]], 
    time_matrix: List[List[int]], 
    max_grace_period: int
) -> Optional[Dict[str, Any]]:
    """Calculate timing, lateness, and score for a route."""
    
    if len(route) < 2:
        return None
    
    current_time = datetime.now(timezone.utc)
    arrival_time = current_time
    
    total_lateness = 0
    late_stops = 0
    total_travel_time = 0
    route_details = []
    
    for i in range(len(route)):
        point = route[i]
        
        if point['type'] == 'start':
            route_details.append({
                "type": "start",
                "index": point['index']
            })
            
        elif point['type'] == 'end':
            route_details.append({
                "type": "end", 
                "index": point['index']
            })
            
        else:
            # Travel to this point
            prev_index = route[i-1]['index']
            curr_index = point['index']
            travel_time = time_matrix[prev_index][curr_index]
            
            arrival_time += timedelta(seconds=travel_time)
            total_travel_time += travel_time
            
            # Check lateness
            deadline = point['deadline']
            lateness_seconds = max(0, (arrival_time - deadline).total_seconds())
            
            if lateness_seconds > 0:
                total_lateness += lateness_seconds
                late_stops += 1
            
            # Add to route details
            if point['type'] in ['new_task_pickup', 'existing_task_pickup']:
                route_details.append({
                    "type": point['type'],
                    "task_id": point['task_id'],
                    "pickup_index": point['index'],
                    "arrival_time": arrival_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "deadline": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "lateness": int(lateness_seconds)
                })
            else:  # delivery
                route_details.append({
                    "type": point['type'],
                    "task_id": point['task_id'],
                    "delivery_index": point['index'], 
                    "arrival_time": arrival_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "deadline": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "lateness": int(lateness_seconds)
                })
    
    # Calculate score
    grace_penalty = min(total_lateness, max_grace_period)
    
    # Score components
    time_penalty = min(total_travel_time / 3600, 1.0) * 30  # Max 30 points for travel time
    lateness_penalty = min(grace_penalty / max_grace_period, 1.0) * 70  # Max 70 points for lateness
    
    score = max(0, 100 - time_penalty - lateness_penalty)
    
    return {
        "score": int(score),
        "additional_time_minutes": total_travel_time / 60.0,
        "grace_penalty_seconds": int(grace_penalty),
        "already_late_stops": late_stops,
        "route": route_details
    }

def recommend_agents_batch_optimized(
    new_task: Dict[str, Any],
    agents: List[Dict[str, Any]], 
    current_tasks: List[Dict[str, Any]],
    max_grace_period: int = DEFAULT_MAX_GRACE_PERIOD,
    enable_debug: bool = False,
    use_proximity: bool = True,
    area_type: str = "urban",
    max_distance_km: Optional[float] = None
) -> Dict[str, Any]:
    """
    Batch-optimized agent recommendation with proximity-based filtering.
    
    Args:
        use_proximity: Enable proximity-based agent filtering (default: True)
        area_type: "urban" or "rural" - affects proximity settings
        max_distance_km: Maximum distance in kilometers for agent selection (default: None, allows up to 50km)
    
    Returns recommendations sorted by score (best first).
    """
    
    start_time = time.time()
    
    try:
        # Proximity-based agent filtering
        if use_proximity and len(agents) > 1:  # Changed from > 5 to > 1 to allow proximity filtering with smaller fleets
            proximity_config = get_proximity_config(len(agents), area_type)
            candidate_agents_data = get_proximate_agents(
                new_task, agents, 
                max_candidates=proximity_config["max_candidates"],
                initial_radius_km=proximity_config["initial_radius_km"],
                enable_debug=enable_debug,
                max_distance_km=max_distance_km
            )
            candidate_agents = [agent for _, agent, _ in candidate_agents_data]
        else:
            candidate_agents = agents
            candidate_agents_data = [(i, agent, 0) for i, agent in enumerate(agents)]
        
        # Collect all unique locations (now with filtered agents)
        locations = set()
        
        # Agent locations (only for selected candidates)
        for _, agent, _ in candidate_agents_data:
            locations.add(tuple(agent['current_location']))
        
        # New task locations  
        locations.add(tuple(new_task['restaurant_location']))
        locations.add(tuple(new_task['delivery_location']))
        
        # Current task locations
        for task in current_tasks:
            locations.add(tuple(task['restaurant_location']))
            locations.add(tuple(task['delivery_location']))
        
        # Convert to list and create mapping
        location_list = list(locations)
        location_to_index = {loc: i for i, loc in enumerate(location_list)}
        
        if enable_debug:
            print(f"Batch optimization: {len(candidate_agents)}/{len(agents)} agents, {len(current_tasks)} current tasks, {len(location_list)} unique locations")
        
        # Build time matrix using OSRM (smaller matrix due to proximity filtering)
        time_matrix = build_osrm_time_matrix_cached(location_list)
        
        # Evaluate each candidate agent for the new task
        recommendations = []
        high_score_threshold = 75  # Stop early if we find high-quality candidates
        
        for agent_idx, agent, distance_km in candidate_agents_data:
            try:
                result = evaluate_agent_for_new_task(
                    agent, new_task, current_tasks, time_matrix, location_to_index, max_grace_period
                )
                if result:
                    # Add proximity bonus to score (up to 5 points)
                    proximity_bonus = max(0, 5 * (1 - distance_km / 20))
                    result['score'] = int(result['score'] + proximity_bonus)
                    result['distance_km'] = round(distance_km, 1)
                    result['proximity_bonus'] = round(proximity_bonus, 1)
                    recommendations.append(result)
                    
            except Exception as e:
                if enable_debug:
                    print(f"Error evaluating agent {agent.get('driver_id', 'unknown')}: {e}")
                continue
        
        # Sort by score (highest first) and return top 3
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        top_recommendations = recommendations[:3]
        
        # Early termination if we have high-quality candidates
        if len(top_recommendations) >= 3 and top_recommendations[0]['score'] >= high_score_threshold:
            early_termination = True
        else:
            early_termination = False
        
        execution_time = time.time() - start_time
        
        if enable_debug:
            print(f"Batch optimization completed in {execution_time:.3f}s")
            print(f"Generated {len(top_recommendations)} recommendations from {len(candidate_agents)} candidates")
            if early_termination:
                print(f"Early termination: found high-quality candidates (score >= {high_score_threshold})")
        
        return {
            "task_id": new_task.get('id', 'unknown'),
            "recommendations": top_recommendations,
            "performance": {
                "execution_time_seconds": round(execution_time, 3),
                "agents_evaluated": len(candidate_agents),
                "total_agents": len(agents),
                "proximity_filtering": use_proximity,
                "early_termination": early_termination
            }
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Batch optimization failed after {execution_time:.3f}s: {str(e)}"
        if enable_debug:
            print(error_msg)
        
        return {
            "task_id": new_task.get('id', 'unknown'),
            "recommendations": [],
            "error": error_msg,
            "performance": {
                "execution_time_seconds": round(execution_time, 3),
                "agents_evaluated": 0,
                "total_agents": len(agents),
                "proximity_filtering": use_proximity
            }
        }

def clear_cache():
    """Clear the OSRM API cache."""
    global _osrm_cache
    _osrm_cache.clear()
    return {"cache_cleared": True} 