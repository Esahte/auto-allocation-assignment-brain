"""
Fleet Optimizer - Many-to-Many Route Optimization

This module implements a fleet-wide optimization system that:
- Takes ALL unassigned tasks + ALL eligible agents
- Builds optimal routes for EVERY agent
- Returns complete route assignments

Conservative mode: Only assigns unassigned/declined tasks, doesn't reassign existing tasks.
"""

import time
import requests
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math

# =============================================================================
# CONFIGURATION
# =============================================================================

OSRM_SERVER = "https://osrm-caribbean-785077267034.us-central1.run.app"
DEFAULT_WALLET_THRESHOLD = 2500
DEFAULT_SERVICE_TIME_SECONDS = 180  # 3 minutes per stop
MAX_SOLVER_TIME_SECONDS = 10  # Reduced from 30 - sufficient for typical fleet sizes

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Location:
    """Geographic location with lat/lng"""
    lat: float
    lng: float
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.lat, self.lng)
    
    def to_osrm_str(self) -> str:
        """OSRM expects lng,lat format"""
        return f"{self.lng},{self.lat}"


@dataclass
class Task:
    """Represents a delivery task (pickup + delivery pair)"""
    id: str
    restaurant_location: Location
    delivery_location: Location
    pickup_before: datetime
    delivery_before: datetime
    pickup_completed: bool = False
    assigned_driver: Optional[str] = None
    payment_type: str = "CARD"
    tags: List[str] = field(default_factory=list)
    declined_by: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Task':
        """Parse task from API response"""
        # Parse locations
        rest_loc = data.get('restaurant_location', [0, 0])
        del_loc = data.get('delivery_location', [0, 0])
        
        # Parse timestamps
        pickup_before = cls._parse_timestamp(data.get('pickup_before'))
        delivery_before = cls._parse_timestamp(data.get('delivery_before'))
        
        # Get declined_by list (extract just driver IDs)
        declined_by = []
        if 'declined_by' in data:
            declined_by = [d.get('driver_id') for d in data['declined_by'] if d.get('driver_id')]
        
        # Get metadata
        meta = data.get('_meta', {})
        
        return cls(
            id=data.get('id', ''),
            restaurant_location=Location(rest_loc[0], rest_loc[1]),
            delivery_location=Location(del_loc[0], del_loc[1]),
            pickup_before=pickup_before,
            delivery_before=delivery_before,
            pickup_completed=data.get('pickup_completed', False),
            assigned_driver=data.get('assigned_driver'),
            payment_type=meta.get('payment_type', 'CARD'),
            tags=meta.get('tags', []),
            declined_by=declined_by,
            meta=meta
        )
    
    @staticmethod
    def _parse_timestamp(ts_str: Optional[str]) -> datetime:
        """Parse ISO timestamp string to datetime"""
        if not ts_str:
            return datetime.now(timezone.utc)
        try:
            # Handle various ISO formats
            if ts_str.endswith('Z'):
                ts_str = ts_str[:-1] + '+00:00'
            return datetime.fromisoformat(ts_str)
        except:
            return datetime.now(timezone.utc)


@dataclass 
class Agent:
    """Represents a delivery agent/driver"""
    id: str
    name: str
    current_location: Location
    wallet_balance: float = 0.0
    current_tasks: List[Task] = field(default_factory=list)
    max_tasks: int = 2
    available_capacity: int = 2
    tags: List[str] = field(default_factory=list)
    has_no_cash_tag: bool = False
    is_scooter_agent: bool = False
    geofence_regions: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Agent':
        """Parse agent from API response"""
        loc = data.get('current_location', [0, 0])
        meta = data.get('_meta', {})
        
        # Parse current tasks
        current_tasks = []
        for task_data in data.get('current_tasks', []):
            current_tasks.append(Task.from_dict(task_data))
        
        # Handle max_tasks being string or int
        max_tasks = meta.get('max_tasks', 2)
        if isinstance(max_tasks, str):
            max_tasks = int(max_tasks)
        
        return cls(
            id=data.get('driver_id', ''),
            name=data.get('name', ''),
            current_location=Location(loc[0], loc[1]),
            wallet_balance=data.get('wallet_balance', 0.0),
            current_tasks=current_tasks,
            max_tasks=max_tasks,
            available_capacity=meta.get('available_capacity', 2),
            tags=meta.get('tags', []),
            has_no_cash_tag=meta.get('has_no_cash_tag', False),
            is_scooter_agent=meta.get('is_scooter_agent', False),
            geofence_regions=meta.get('geofence_regions', [])
        )


@dataclass
class GeofenceRegion:
    """Represents a geofence polygon"""
    region_id: int
    region_name: str
    polygon: List[Tuple[float, float]]  # List of (lat, lng) vertices
    fleet_ids: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GeofenceRegion':
        """Parse geofence from API response"""
        polygon = [(p[0], p[1]) for p in data.get('polygon', [])]
        return cls(
            region_id=data.get('region_id', 0),
            region_name=data.get('region_name', ''),
            polygon=polygon,
            fleet_ids=data.get('fleet_ids', [])
        )


# =============================================================================
# GEOFENCE UTILITIES
# =============================================================================

def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Ray casting algorithm to check if point is inside polygon.
    
    Args:
        point: (lat, lng) tuple
        polygon: List of (lat, lng) vertices
    
    Returns:
        True if point is inside polygon
    """
    if len(polygon) < 3:
        return False
    
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


# =============================================================================
# COMPATIBILITY MATRIX
# =============================================================================

class CompatibilityChecker:
    """
    Builds and checks agent-task compatibility based on business rules.
    """
    
    def __init__(self, 
                 wallet_threshold: float = DEFAULT_WALLET_THRESHOLD,
                 geofence_regions: List[GeofenceRegion] = None):
        self.wallet_threshold = wallet_threshold
        self.geofence_regions = geofence_regions or []
        # Build lookup for geofence by name
        self.geofence_by_name = {g.region_name: g for g in self.geofence_regions}
    
    def is_compatible(self, agent: Agent, task: Task) -> Tuple[bool, str]:
        """
        Check if an agent can be assigned a specific task.
        
        Returns:
            (is_compatible, reason) tuple
        """
        # Rule 1: Decline History - Agent previously declined this task
        if agent.id in task.declined_by:
            return False, "agent_declined_task"
        
        # Rule 2: NoCash Tag - Agent can't handle cash
        if task.payment_type == "CASH" and agent.has_no_cash_tag:
            return False, "no_cash_tag"
        
        # Rule 3: Wallet Threshold - Agent wallet too high for cash orders
        if task.payment_type == "CASH" and agent.wallet_balance > self.wallet_threshold:
            return False, "wallet_threshold_exceeded"
        
        # Rule 4: Tag Matching - Task tags must match agent tags
        if task.tags:
            if not any(tag in agent.tags for tag in task.tags):
                return False, "tag_mismatch"
        
        # Rule 5: Scooter Geofence - Both locations must be in geofence
        if agent.is_scooter_agent and agent.geofence_regions:
            for region_name in agent.geofence_regions:
                if region_name in self.geofence_by_name:
                    geofence = self.geofence_by_name[region_name]
                    pickup_in = point_in_polygon(
                        task.restaurant_location.to_tuple(), 
                        geofence.polygon
                    )
                    delivery_in = point_in_polygon(
                        task.delivery_location.to_tuple(),
                        geofence.polygon
                    )
                    if not (pickup_in and delivery_in):
                        return False, "outside_scooter_geofence"
        
        return True, "compatible"
    
    def build_compatibility_matrix(self, 
                                   agents: List[Agent], 
                                   tasks: List[Task]) -> Dict[str, Dict[str, Tuple[bool, str]]]:
        """
        Build full compatibility matrix for all agent-task pairs.
        
        Returns:
            {agent_id: {task_id: (is_compatible, reason)}}
        """
        matrix = {}
        for agent in agents:
            matrix[agent.id] = {}
            for task in tasks:
                matrix[agent.id][task.id] = self.is_compatible(agent, task)
        return matrix


# =============================================================================
# OSRM TRAVEL TIME MATRIX
# =============================================================================

def get_travel_time_matrix(locations: List[Location]) -> List[List[int]]:
    """
    Get travel time matrix between all locations using OSRM.
    
    Args:
        locations: List of Location objects
    
    Returns:
        NxN matrix of travel times in seconds
    """
    if len(locations) < 2:
        return [[0]]
    
    # Build coordinates string for OSRM
    coords_str = ";".join([loc.to_osrm_str() for loc in locations])
    url = f"{OSRM_SERVER}/table/v1/driving/{coords_str}?annotations=duration"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get('code') == 'Ok':
            durations = data.get('durations', [])
            # Convert to integers (seconds)
            return [[int(d) if d is not None else 999999 for d in row] for row in durations]
    except Exception as e:
        print(f"[OSRM] Error getting travel times: {e}")
    
    # Fallback to Haversine estimation
    return _estimate_travel_times(locations)


def _estimate_travel_times(locations: List[Location]) -> List[List[int]]:
    """Fallback: estimate travel times using Haversine distance"""
    n = len(locations)
    matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = _haversine_km(
                    locations[i].lat, locations[i].lng,
                    locations[j].lat, locations[j].lng
                )
                # Assume average speed of 30 km/h
                matrix[i][j] = int(dist / 30 * 3600)
    
    return matrix


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Haversine distance between two points in kilometers"""
    R = 6371  # Earth's radius in km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


# =============================================================================
# FLEET OPTIMIZER (VRP SOLVER)
# =============================================================================

class FleetOptimizer:
    """
    Multi-vehicle routing problem solver using OR-Tools.
    
    Builds optimal routes for all agents considering:
    - Agent-task compatibility
    - Time windows
    - Pickup-delivery precedence
    - Agent capacity
    - Existing assigned tasks
    """
    
    def __init__(self,
                 agents: List[Agent],
                 unassigned_tasks: List[Task],
                 compatibility_checker: CompatibilityChecker,
                 current_time: datetime = None):
        self.agents = agents
        self.unassigned_tasks = unassigned_tasks
        self.compatibility_checker = compatibility_checker
        self.current_time = current_time or datetime.now(timezone.utc)
        
        # Build compatibility matrix
        self.compatibility_matrix = compatibility_checker.build_compatibility_matrix(
            agents, unassigned_tasks
        )
        
        # Results
        self.solution = None
        self.routes = {}
        self.unassigned = []
        self.metadata = {}
    
    def _build_location_index(self) -> Tuple[List[Location], Dict]:
        """
        Build unified location index for all stops.
        
        Returns:
            locations: List of all locations
            index_map: Mapping information for nodes
        """
        locations = []
        index_map = {
            'agent_starts': {},      # agent_id -> location_index
            'agent_ends': {},        # agent_id -> location_index
            'pickups': {},           # task_id -> location_index
            'deliveries': {},        # task_id -> location_index
            'task_at_index': {},     # location_index -> (task_id, 'pickup'|'delivery')
            'agent_at_index': {},    # location_index -> agent_id
        }
        
        # Add depot (index 0) - we'll use a central point
        # For now, use first agent's location as reference
        if self.agents:
            locations.append(self.agents[0].current_location)
        else:
            locations.append(Location(17.12, -61.82))  # Antigua center
        
        idx = 1
        
        # Add agent start/end locations
        for agent in self.agents:
            index_map['agent_starts'][agent.id] = idx
            index_map['agent_at_index'][idx] = agent.id
            locations.append(agent.current_location)
            idx += 1
            
            # End location (same as start for now - they return to their position)
            index_map['agent_ends'][agent.id] = idx
            locations.append(agent.current_location)
            idx += 1
        
        # Add existing task locations (pickups and deliveries for current tasks)
        for agent in self.agents:
            for task in agent.current_tasks:
                if not task.pickup_completed:
                    # Add pickup location
                    index_map['pickups'][task.id] = idx
                    index_map['task_at_index'][idx] = (task.id, 'pickup', agent.id)
                    locations.append(task.restaurant_location)
                    idx += 1
                
                # Add delivery location
                index_map['deliveries'][task.id] = idx
                index_map['task_at_index'][idx] = (task.id, 'delivery', agent.id)
                locations.append(task.delivery_location)
                idx += 1
        
        # Add unassigned task locations
        for task in self.unassigned_tasks:
            # Pickup
            index_map['pickups'][task.id] = idx
            index_map['task_at_index'][idx] = (task.id, 'pickup', None)
            locations.append(task.restaurant_location)
            idx += 1
            
            # Delivery
            index_map['deliveries'][task.id] = idx
            index_map['task_at_index'][idx] = (task.id, 'delivery', None)
            locations.append(task.delivery_location)
            idx += 1
        
        return locations, index_map
    
    def _time_to_seconds(self, dt: datetime) -> int:
        """Convert datetime to seconds from current time"""
        delta = (dt - self.current_time).total_seconds()
        return max(0, int(delta))
    
    def optimize(self) -> Dict:
        """
        Run the fleet optimization.
        
        Returns:
            Optimization results with routes for each agent
        """
        start_time = time.time()
        
        # Handle edge cases
        if not self.agents:
            return self._empty_result("no_agents")
        
        if not self.unassigned_tasks and not any(a.current_tasks for a in self.agents):
            return self._empty_result("no_tasks")
        
        # Build location index
        locations, index_map = self._build_location_index()
        num_locations = len(locations)
        num_vehicles = len(self.agents)
        
        print(f"[FleetOptimizer] Building model: {num_locations} locations, {num_vehicles} vehicles")
        print(f"[FleetOptimizer] Unassigned tasks: {len(self.unassigned_tasks)}")
        
        # Get travel time matrix
        print("[FleetOptimizer] Fetching travel times from OSRM...")
        time_matrix = get_travel_time_matrix(locations)
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(
            num_locations,
            num_vehicles,
            [index_map['agent_starts'][a.id] for a in self.agents],  # starts
            [index_map['agent_ends'][a.id] for a in self.agents]     # ends
        )
        routing = pywrapcp.RoutingModel(manager)
        
        # Travel time callback
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return time_matrix[from_node][to_node] + DEFAULT_SERVICE_TIME_SECONDS
        
        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add time dimension
        routing.AddDimension(
            transit_callback_index,
            3600 * 4,  # 4 hour max slack (waiting time)
            3600 * 8,  # 8 hour max total time
            False,     # Don't force start cumul to zero
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # Add capacity dimension to enforce max_tasks per agent
        # Track NEW tasks only - count pickups of unassigned tasks
        # Build set of unassigned task IDs for quick lookup
        unassigned_task_ids = {t.id for t in self.unassigned_tasks}
        
        def demand_callback(from_index):
            """Returns demand: +1 for new task pickup, 0 otherwise"""
            node = manager.IndexToNode(from_index)
            task_info = index_map['task_at_index'].get(node)
            if task_info:
                task_id, stop_type, original_agent = task_info
                # Only count NEW task pickups (not existing tasks, not deliveries)
                if stop_type == 'pickup' and task_id in unassigned_task_ids:
                    return 1  # New task being assigned
            return 0  # Existing tasks, deliveries, start/end nodes
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        
        # Get available capacity for each agent (how many MORE tasks they can take)
        vehicle_capacities = []
        for agent in self.agents:
            cap = agent.available_capacity
            # Ensure at least 0 capacity (shouldn't be negative)
            vehicle_capacities.append(max(0, cap))
        
        print(f"[FleetOptimizer] Agent capacities: {[(a.name, a.available_capacity) for a in self.agents]}")
        
        # Only add capacity constraint if there are agents with limited capacity
        if any(cap < 100 for cap in vehicle_capacities):  # Skip if all have high capacity
            routing.AddDimensionWithVehicleCapacity(
                demand_callback_index,
                0,  # No slack
                vehicle_capacities,  # Max NEW tasks per vehicle
                True,  # Start cumul at zero
                'Capacity'
            )
        
        # Add pickup and delivery constraints
        pickup_delivery_pairs = []
        
        # Existing tasks (already assigned to specific agents)
        for agent_idx, agent in enumerate(self.agents):
            for task in agent.current_tasks:
                pickup_idx = index_map['pickups'].get(task.id)
                delivery_idx = index_map['deliveries'].get(task.id)
                
                if pickup_idx is not None and delivery_idx is not None:
                    # Both pickup and delivery needed
                    pickup_delivery_pairs.append((pickup_idx, delivery_idx))
                    
                    # Force this task to this agent
                    pickup_index = manager.NodeToIndex(pickup_idx)
                    delivery_index = manager.NodeToIndex(delivery_idx)
                    
                    routing.SetAllowedVehiclesForIndex([agent_idx], pickup_index)
                    routing.SetAllowedVehiclesForIndex([agent_idx], delivery_index)
                    
                elif delivery_idx is not None:
                    # Delivery only (pickup completed)
                    delivery_index = manager.NodeToIndex(delivery_idx)
                    routing.SetAllowedVehiclesForIndex([agent_idx], delivery_index)
                    
                    # Add time window for delivery - this is the real deadline
                    deadline_seconds = self._time_to_seconds(task.delivery_before)
                    time_dimension.CumulVar(delivery_index).SetMax(deadline_seconds + 1800)  # 30min grace
        
        # Unassigned tasks
        for task in self.unassigned_tasks:
            pickup_idx = index_map['pickups'].get(task.id)
            delivery_idx = index_map['deliveries'].get(task.id)
            
            if pickup_idx is None or delivery_idx is None:
                continue
            
            pickup_delivery_pairs.append((pickup_idx, delivery_idx))
            
            pickup_index = manager.NodeToIndex(pickup_idx)
            delivery_index = manager.NodeToIndex(delivery_idx)
            
            # Set allowed vehicles based on compatibility
            allowed_vehicles = []
            for agent_idx, agent in enumerate(self.agents):
                is_compatible, _ = self.compatibility_matrix[agent.id][task.id]
                if is_compatible:
                    allowed_vehicles.append(agent_idx)
            
            # Log compatibility
            if allowed_vehicles:
                print(f"[FleetOptimizer] Task {task.id[:20]}... compatible with {len(allowed_vehicles)} agents: {[self.agents[i].name for i in allowed_vehicles[:3]]}")
                routing.SetAllowedVehiclesForIndex(allowed_vehicles, pickup_index)
                routing.SetAllowedVehiclesForIndex(allowed_vehicles, delivery_index)
            else:
                print(f"[FleetOptimizer] Task {task.id[:20]}... NO compatible agents!")
            
            # Allow dropping this task as a pair (pickup + delivery together)
            # Penalty is high but allows partial solutions when capacity is exceeded
            penalty = 50000 if allowed_vehicles else 1000
            routing.AddDisjunction([pickup_index, delivery_index], penalty, 2)
            
            # Add time windows
            # pickup_before = when food is READY (earliest pickup time)
            # delivery_before = delivery DEADLINE (must arrive before this)
            pickup_ready_time = self._time_to_seconds(task.pickup_before)
            delivery_deadline = self._time_to_seconds(task.delivery_before)
            
            # Pickup: Don't arrive before food is ready (or you wait)
            # Set minimum arrival time to when food is ready
            time_dimension.CumulVar(pickup_index).SetMin(pickup_ready_time)
            # Allow late pickup with grace period (food can wait a bit)
            time_dimension.CumulVar(pickup_index).SetMax(pickup_ready_time + 3600)  # 1hr grace for pickup
            
            # Delivery: This is the real deadline - prioritize this!
            # Allow some grace but less than pickup
            time_dimension.CumulVar(delivery_index).SetMax(delivery_deadline + 1800)  # 30min grace for delivery
        
        # Add pickup-delivery constraints
        for pickup_idx, delivery_idx in pickup_delivery_pairs:
            pickup_index = manager.NodeToIndex(pickup_idx)
            delivery_index = manager.NodeToIndex(delivery_idx)
            
            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
            )
            routing.solver().Add(
                time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(delivery_index)
            )
        
        # Set solver parameters - optimized for speed
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        
        # Dynamic time limit based on problem size
        num_tasks = len(self.unassigned_tasks) + sum(len(a.current_tasks) for a in self.agents)
        if num_tasks <= 5:
            time_limit = 3  # Small problems: 3 seconds
        elif num_tasks <= 15:
            time_limit = 5  # Medium problems: 5 seconds
        elif num_tasks <= 30:
            time_limit = 10  # Larger problems: 10 seconds
        else:
            time_limit = MAX_SOLVER_TIME_SECONDS  # Very large: use max
        
        search_parameters.time_limit.FromSeconds(time_limit)
        
        # Stop early if we find a solution with no improvement for a while
        search_parameters.solution_limit = 100  # Stop after finding 100 solutions
        
        print(f"[FleetOptimizer] Solver time limit: {time_limit}s for {num_tasks} tasks")
        
        # Solve
        print("[FleetOptimizer] Solving VRP...")
        solution = routing.SolveWithParameters(search_parameters)
        
        solve_time = time.time() - start_time
        print(f"[FleetOptimizer] Solved in {solve_time:.2f}s")
        
        if solution:
            return self._extract_solution(routing, manager, solution, index_map, time_dimension, solve_time)
        else:
            print("[FleetOptimizer] No solution found")
            return self._empty_result("no_solution", solve_time)
    
    def _extract_solution(self, routing, manager, solution, index_map, time_dimension, solve_time) -> Dict:
        """Extract solution into structured format"""
        
        agent_routes = []
        assigned_task_ids = set()
        total_lateness = 0
        
        for vehicle_idx, agent in enumerate(self.agents):
            route_stops = []
            assigned_new_tasks = []
            
            index = routing.Start(vehicle_idx)
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                time_var = time_dimension.CumulVar(index)
                arrival_seconds = solution.Value(time_var)
                arrival_time = self.current_time.timestamp() + arrival_seconds
                arrival_dt = datetime.fromtimestamp(arrival_time, tz=timezone.utc)
                
                # Check what this node represents
                task_info = index_map['task_at_index'].get(node)
                
                if task_info:
                    task_id, stop_type, original_agent = task_info
                    
                    # Find the task
                    task = None
                    if original_agent:
                        # Existing task
                        for a in self.agents:
                            for t in a.current_tasks:
                                if t.id == task_id:
                                    task = t
                                    break
                    else:
                        # Unassigned task
                        for t in self.unassigned_tasks:
                            if t.id == task_id:
                                task = t
                                if stop_type == 'pickup':
                                    assigned_new_tasks.append(task_id)
                                    assigned_task_ids.add(task_id)
                                break
                    
                    if task:
                        # Get readable names from task metadata
                        restaurant_name = task.meta.get('restaurant_name', 'Unknown Restaurant')
                        customer_name = task.meta.get('customer_name', 'Unknown Customer')
                        
                        if stop_type == 'pickup':
                            # Pickup: pickup_before = food ready time
                            # If we arrive early, we wait. If we arrive late, food waits.
                            food_ready_time = task.pickup_before
                            wait_seconds = max(0, (food_ready_time - arrival_dt).total_seconds())
                            late_seconds = max(0, (arrival_dt - food_ready_time).total_seconds())
                            
                            stop = {
                                'type': f"{'new' if not original_agent else 'existing'}_task_pickup",
                                'task_id': task_id,
                                'location': task.restaurant_location.to_tuple(),
                                'arrival_time': arrival_dt.isoformat(),
                                'food_ready_time': food_ready_time.isoformat(),
                                'wait_seconds': int(wait_seconds),  # Time waiting for food
                                'late_seconds': int(late_seconds),  # How late after food ready
                                'restaurant_name': restaurant_name,
                                'customer_name': customer_name,
                                'display_name': restaurant_name
                            }
                        else:
                            # Delivery: delivery_before = deadline (must arrive before!)
                            deadline = task.delivery_before
                            lateness = max(0, (arrival_dt - deadline).total_seconds())
                            total_lateness += lateness  # Only count delivery lateness
                            early_seconds = max(0, (deadline - arrival_dt).total_seconds())
                            
                            stop = {
                                'type': f"{'new' if not original_agent else 'existing'}_task_delivery",
                                'task_id': task_id,
                                'location': task.delivery_location.to_tuple(),
                                'arrival_time': arrival_dt.isoformat(),
                                'deadline': deadline.isoformat(),
                                'lateness_seconds': int(lateness),  # How late to customer
                                'early_seconds': int(early_seconds),  # How early (good!)
                                'restaurant_name': restaurant_name,
                                'customer_name': customer_name,
                                'display_name': customer_name
                            }
                        
                        route_stops.append(stop)
                
                index = solution.Value(routing.NextVar(index))
            
            # Build route info
            route_info = {
                'agent_id': agent.id,
                'agent_name': agent.name,
                'assigned_new_tasks': assigned_new_tasks,
                'total_stops': len(route_stops),
                'route': route_stops,
                'total_lateness_seconds': sum(s.get('lateness_seconds', 0) for s in route_stops)
            }
            
            # Generate Google Maps URL
            if route_stops:
                waypoints = [f"{s['location'][0]},{s['location'][1]}" for s in route_stops]
                origin = f"{agent.current_location.lat},{agent.current_location.lng}"
                destination = waypoints[-1] if waypoints else origin
                waypoints_str = "|".join(waypoints[:-1]) if len(waypoints) > 1 else ""
                
                maps_url = f"https://www.google.com/maps/dir/?api=1&origin={origin}&destination={destination}"
                if waypoints_str:
                    maps_url += f"&waypoints={waypoints_str}"
                route_info['maps_url'] = maps_url
            
            agent_routes.append(route_info)
        
        # Find unassigned tasks
        unassigned_tasks = []
        for task in self.unassigned_tasks:
            if task.id not in assigned_task_ids:
                # Find reason
                reasons = []
                for agent in self.agents:
                    compatible, reason = self.compatibility_matrix[agent.id][task.id]
                    if not compatible:
                        reasons.append(reason)
                
                primary_reason = max(set(reasons), key=reasons.count) if reasons else "no_feasible_route"
                
                unassigned_tasks.append({
                    'task_id': task.id,
                    'reason': primary_reason,
                    'restaurant_name': task.meta.get('restaurant_name', ''),
                    'customer_name': task.meta.get('customer_name', '')
                })
        
        return {
            'success': True,
            'metadata': {
                'total_agents': len(self.agents),
                'total_unassigned_tasks': len(self.unassigned_tasks),
                'tasks_assigned': len(assigned_task_ids),
                'tasks_unassigned': len(unassigned_tasks),
                'total_lateness_seconds': int(total_lateness),
                'optimization_time_seconds': round(solve_time, 3),
                'solver': 'or_tools_vrp'
            },
            'agent_routes': agent_routes,
            'unassigned_tasks': unassigned_tasks
        }
    
    def _empty_result(self, reason: str, solve_time: float = 0) -> Dict:
        """Return empty result with reason"""
        return {
            'success': False,
            'reason': reason,
            'metadata': {
                'total_agents': len(self.agents),
                'total_unassigned_tasks': len(self.unassigned_tasks),
                'tasks_assigned': 0,
                'tasks_unassigned': len(self.unassigned_tasks),
                'optimization_time_seconds': round(solve_time, 3)
            },
            'agent_routes': [],
            'unassigned_tasks': [{'task_id': t.id, 'reason': reason} for t in self.unassigned_tasks]
        }


# =============================================================================
# MAIN OPTIMIZATION FUNCTION
# =============================================================================

def optimize_fleet(agents_data: Dict, tasks_data: Dict) -> Dict:
    """
    Main entry point for fleet optimization.
    
    Args:
        agents_data: Response from /api/test/or-tools/agents endpoint
        tasks_data: Response from /api/test/or-tools/unassigned-tasks endpoint
    
    Returns:
        Optimization results with routes for each agent
    """
    start_time = time.time()
    
    # Parse agents
    agents = [Agent.from_dict(a) for a in agents_data.get('agents', [])]
    print(f"[optimize_fleet] Parsed {len(agents)} agents")
    
    # Parse unassigned tasks
    tasks = [Task.from_dict(t) for t in tasks_data.get('tasks', [])]
    print(f"[optimize_fleet] Parsed {len(tasks)} unassigned tasks")
    
    # Parse geofence regions
    geofences = [GeofenceRegion.from_dict(g) for g in agents_data.get('geofence_data', [])]
    print(f"[optimize_fleet] Parsed {len(geofences)} geofence regions")
    
    # Get settings
    settings = agents_data.get('settings_used', {})
    wallet_threshold = settings.get('walletNoCashThreshold', DEFAULT_WALLET_THRESHOLD)
    
    # Build compatibility checker
    compatibility_checker = CompatibilityChecker(
        wallet_threshold=wallet_threshold,
        geofence_regions=geofences
    )
    
    # Run optimizer
    optimizer = FleetOptimizer(
        agents=agents,
        unassigned_tasks=tasks,
        compatibility_checker=compatibility_checker
    )
    
    result = optimizer.optimize()
    
    # Add total execution time
    result['total_execution_time_seconds'] = round(time.time() - start_time, 3)
    
    return result


# =============================================================================
# STANDALONE TESTING
# =============================================================================

if __name__ == "__main__":
    import json
    
    # Test with fetching from local dashboard
    print("Testing Fleet Optimizer...")
    print("=" * 60)
    
    try:
        # Fetch data from dashboard endpoints
        print("Fetching agents from http://localhost:8000/api/test/or-tools/agents...")
        agents_response = requests.get("http://localhost:8000/api/test/or-tools/agents", timeout=30)
        agents_data = agents_response.json()
        print(f"  Got {agents_data.get('summary', {}).get('eligible_agents', 0)} eligible agents")
        
        print("Fetching tasks from http://localhost:8000/api/test/or-tools/unassigned-tasks...")
        tasks_response = requests.get("http://localhost:8000/api/test/or-tools/unassigned-tasks", timeout=30)
        tasks_data = tasks_response.json()
        print(f"  Got {tasks_data.get('summary', {}).get('total', 0)} unassigned tasks")
        
        print("\nRunning optimization...")
        print("=" * 60)
        
        result = optimize_fleet(agents_data, tasks_data)
        
        print("\n" + "=" * 60)
        print("RESULTS:")
        print("=" * 60)
        print(json.dumps(result, indent=2, default=str))
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to dashboard at localhost:8000")
        print("Make sure the dashboard is running.")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

