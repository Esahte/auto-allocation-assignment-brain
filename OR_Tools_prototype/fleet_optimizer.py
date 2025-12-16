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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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

# =============================================================================
# HTTP SESSION WITH RETRY LOGIC
# =============================================================================

def create_retry_session(retries=3, backoff_factor=0.5, status_forcelist=(500, 502, 503, 504)):
    """Create a requests session with retry logic for OSRM calls."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Global session for OSRM calls
_osrm_session = None

def get_osrm_session():
    """Get or create the OSRM HTTP session with retry logic."""
    global _osrm_session
    if _osrm_session is None:
        _osrm_session = create_retry_session(retries=3, backoff_factor=1.0)
    return _osrm_session
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
        """OSRM expects lng,lat format. Truncate to 6 decimal places (0.1m precision)"""
        return f"{self.lng:.6f},{self.lat:.6f}"


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

def get_osrm_distances_matrix(origins: List[Location], destinations: List[Location]) -> List[List[float]]:
    """
    Get road distances from multiple origins to multiple destinations using OSRM.
    Returns distances in kilometers.
    """
    if not origins or not destinations:
        return []
    
    try:
        # Build coordinates: origins first, then destinations
        all_locations = origins + destinations
        coords_str = ";".join([loc.to_osrm_str() for loc in all_locations])
        num_origins = len(origins)
        
        # Request FULL matrix without sources/destinations filter
        # (the sources/destinations filter causes "InvalidQuery" on some OSRM servers)
        url = f"{OSRM_SERVER}/table/v1/driving/{coords_str}?annotations=distance"
        
        session = get_osrm_session()
        response = session.get(url, timeout=30)
        
        if response.status_code != 200:
            # Get more details about the error
            try:
                error_data = response.json()
                error_msg = error_data.get('message', 'Unknown error')
                error_code = error_data.get('code', 'Unknown')
                print(f"[OSRM] Distance matrix failed: {error_code} - {error_msg}")
                
                # Log problematic coordinates
                print(f"[OSRM] Origins ({len(origins)}):")
                for i, loc in enumerate(origins):
                    print(f"[OSRM]   {i}: ({loc.lat:.6f}, {loc.lng:.6f})")
                print(f"[OSRM] Destinations ({len(destinations)}):")
                for i, loc in enumerate(destinations):
                    print(f"[OSRM]   {i}: ({loc.lat:.6f}, {loc.lng:.6f})")
            except:
                print(f"[OSRM] Distance matrix failed with status {response.status_code}")
        
        response.raise_for_status()
        data = response.json()
        
        if data.get('code') == 'Ok':
            # Extract only the origin→destination subset from the full matrix
            # Full matrix is NxN where N = num_origins + num_destinations
            # We want rows 0..num_origins-1, columns num_origins..N-1
            full_distances = data.get('distances', [])
            result = []
            for i in range(num_origins):
                row = []
                for j in range(num_origins, len(all_locations)):
                    d = full_distances[i][j]
                    row.append(d / 1000.0 if d is not None else float('inf'))
                result.append(row)
            return result
        else:
            print(f"[OSRM] Unexpected response code: {data.get('code')} - {data.get('message', '')}")
    except Exception as e:
        print(f"[OSRM] Distance matrix error: {e}, falling back to Haversine")
    
    # Fallback to Haversine
    result = []
    for origin in origins:
        row = []
        for dest in destinations:
            dist = _haversine_km(origin.lat, origin.lng, dest.lat, dest.lng)
            row.append(dist)
        result.append(row)
    return result


class CompatibilityChecker:
    """
    Builds and checks agent-task compatibility based on business rules.
    """
    
    def __init__(self, 
                 wallet_threshold: float = DEFAULT_WALLET_THRESHOLD,
                 geofence_regions: List[GeofenceRegion] = None,
                 max_distance_km: float = None):
        self.wallet_threshold = wallet_threshold
        self.geofence_regions = geofence_regions or []
        self.max_distance_km = max_distance_km  # Max distance for assignment
        self.distance_cache = {}  # Cache for agent-task distances
        # Build lookup for geofence by name
        self.geofence_by_name = {g.region_name: g for g in self.geofence_regions}
    
    def _get_agent_projected_location(self, agent: Agent) -> Location:
        """
        Get the agent's projected location for distance checks.
        
        If agent has existing tasks, returns their LAST DELIVERY location
        (where they'll be after completing current tasks).
        Otherwise, returns their current location.
        
        This enables smarter chaining - an agent far from a restaurant NOW
        might be close AFTER their current deliveries.
        """
        if agent.current_tasks:
            # Find the last delivery location in their current task chain
            last_task = agent.current_tasks[-1]
            return last_task.delivery_location
        else:
            return agent.current_location
    
    def precompute_distances(self, agents: List[Agent], tasks: List[Task]):
        """
        Precompute OSRM road distances from all agents to all task pickup locations.
        
        IMPORTANT: For agents with existing tasks, we use their PROJECTED location
        (last delivery) instead of current location. This enables smart chaining
        where an agent finishing nearby can pick up new tasks in the area.
        """
        if not agents or not tasks:
            return
        
        # Use projected locations (last delivery) for agents with existing tasks
        agent_projected_locations = [self._get_agent_projected_location(agent) for agent in agents]
        task_locations = [task.restaurant_location for task in tasks]
        
        # Log which agents are using projected locations
        for agent in agents:
            if agent.current_tasks:
                proj_loc = self._get_agent_projected_location(agent)
                print(f"[CompatibilityChecker] {agent.name}: Using projected location "
                      f"(last delivery at {proj_loc.lat:.4f}, {proj_loc.lng:.4f}) instead of current location")
        
        print(f"[CompatibilityChecker] Getting OSRM distances for {len(agents)} agents x {len(tasks)} tasks...")
        
        distances = get_osrm_distances_matrix(agent_projected_locations, task_locations)
        
        # Store in cache with projected flag
        for i, agent in enumerate(agents):
            for j, task in enumerate(tasks):
                self.distance_cache[(agent.id, task.id)] = distances[i][j] if distances else float('inf')
        
        print(f"[CompatibilityChecker] Cached {len(self.distance_cache)} distances (using projected locations)")
    
    def is_compatible(self, agent: Agent, task: Task) -> Tuple[bool, str]:
        """
        Check if an agent can be assigned a specific task.
        
        Returns:
            (is_compatible, reason) tuple
        """
        # Rule 1: Decline History - Agent previously declined this task
        if task.declined_by and agent.id in task.declined_by:
            return False, "agent_declined_task"
        
        # Rule 2: NoCash Tag - Agent can't handle cash
        if task.payment_type == "CASH" and agent.has_no_cash_tag:
            return False, "no_cash_tag"
        
        # Rule 3: Wallet Threshold - Agent wallet too high for cash orders
        if task.payment_type == "CASH" and agent.wallet_balance > self.wallet_threshold:
            return False, "wallet_threshold_exceeded"
        
        # Rule 4: Tag Matching - Task tags must match agent tags
        # task.tags must be a non-empty list for tag filtering to apply
        if task.tags and len(task.tags) > 0:
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
        
        # Rule 6: Max Distance - Agent must be within maxDistanceKm of pickup location
        # Uses OSRM road distance (precomputed) for accuracy
        # IMPORTANT: Uses PROJECTED location (last delivery) for agents with existing tasks
        if self.max_distance_km is not None:
            # Check cache first (OSRM road distance from projected location)
            cache_key = (agent.id, task.id)
            if cache_key in self.distance_cache:
                distance = self.distance_cache[cache_key]
            else:
                # Fallback to Haversine if not in cache - use projected location
                projected_loc = self._get_agent_projected_location(agent)
                distance = _haversine_km(
                    projected_loc.lat, projected_loc.lng,
                    task.restaurant_location.lat, task.restaurant_location.lng
                )
            
            if distance > self.max_distance_km:
                # Include whether this is from projected location in the reason
                loc_type = "projected" if agent.current_tasks else "current"
                return False, f"distance_exceeded_{distance:.1f}km_from_{loc_type}"
        
        return True, "compatible"
    
    def build_compatibility_matrix(self, 
                                   agents: List[Agent], 
                                   tasks: List[Task]) -> Dict[str, Dict[str, Tuple[bool, str]]]:
        """
        Build full compatibility matrix for all agent-task pairs.
        
        Returns:
            {agent_id: {task_id: (is_compatible, reason)}}
        """
        # Precompute OSRM distances for maxDistanceKm check
        if self.max_distance_km is not None and tasks:
            self.precompute_distances(agents, tasks)
        
        matrix = {}
        stats = {
            'total_pairs': 0,
            'compatible': 0,
            'filtered_by_distance': 0,
            'filtered_by_cash': 0,
            'filtered_by_declined': 0,
            'filtered_by_other': 0
        }
        
        for agent in agents:
            matrix[agent.id] = {}
            for task in tasks:
                stats['total_pairs'] += 1
                is_compatible, reason = self.is_compatible(agent, task)
                matrix[agent.id][task.id] = (is_compatible, reason)
                
                if is_compatible:
                    stats['compatible'] += 1
                elif 'distance_exceeded' in reason:
                    stats['filtered_by_distance'] += 1
                elif reason in ['no_cash_tag', 'wallet_threshold_exceeded']:
                    stats['filtered_by_cash'] += 1
                elif reason == 'agent_declined_task':
                    stats['filtered_by_declined'] += 1
                else:
                    stats['filtered_by_other'] += 1
        
        print(f"[CompatibilityMatrix] {stats}")
        self.compatibility_stats = stats  # Store for reporting
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
        session = get_osrm_session()
        response = session.get(url, timeout=30)
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
        
        # Store max distance for chain-aware filtering
        self.max_distance_km = getattr(compatibility_checker, 'max_distance_km', None)
        
        # Build compatibility matrix
        self.compatibility_matrix = compatibility_checker.build_compatibility_matrix(
            agents, unassigned_tasks
        )
        
        # Results
        self.solution = None
        self.routes = {}
        self.unassigned = []
        self.metadata = {}
    
    def _build_location_index(self, routable_tasks: List[Task] = None) -> Tuple[List[Location], Dict]:
        """
        Build unified location index for all stops.
        
        Args:
            routable_tasks: List of unassigned tasks that have compatible agents.
                           If None, uses self.unassigned_tasks (legacy behavior).
        
        Returns:
            locations: List of all locations
            index_map: Mapping information for nodes
        """
        # Default to all unassigned tasks if not specified (legacy behavior)
        if routable_tasks is None:
            routable_tasks = self.unassigned_tasks
        
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
            print(f"[FleetOptimizer] Agent {agent.name} has {len(agent.current_tasks)} current tasks")
            for task in agent.current_tasks:
                print(f"[FleetOptimizer]   - Task {task.id[:20]}... pickup_completed={task.pickup_completed}")
                if not task.pickup_completed:
                    # Add pickup location
                    index_map['pickups'][task.id] = idx
                    index_map['task_at_index'][idx] = (task.id, 'pickup', agent.id)
                    locations.append(task.restaurant_location)
                    print(f"[FleetOptimizer]     Added PICKUP at index {idx}")
                    idx += 1
                else:
                    print(f"[FleetOptimizer]     SKIPPED pickup (already completed)")
                
                # Add delivery location
                index_map['deliveries'][task.id] = idx
                index_map['task_at_index'][idx] = (task.id, 'delivery', agent.id)
                locations.append(task.delivery_location)
                print(f"[FleetOptimizer]     Added DELIVERY at index {idx}")
                idx += 1
        
        # Add routable unassigned task locations (pre-filtered tasks excluded)
        for task in routable_tasks:
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
    
    def _build_distance_matrix(self, locations: List[Location]) -> List[List[float]]:
        """
        Build a distance matrix (in km) for all location pairs.
        Used for chain-aware distance enforcement.
        
        Returns:
            2D list where matrix[i][j] = distance in km from location i to j
        """
        n = len(locations)
        
        # Try OSRM first
        try:
            # Build coordinate string for OSRM
            coords_str = ";".join([loc.to_osrm_str() for loc in locations])
            url = f"{OSRM_SERVER}/table/v1/driving/{coords_str}?annotations=distance"
            
            session = get_osrm_session()
            response = session.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 'Ok' and 'distances' in data:
                    # OSRM returns distances in meters, convert to km
                    osrm_distances = data['distances']
                    return [[d / 1000.0 for d in row] for row in osrm_distances]
        except Exception as e:
            print(f"[FleetOptimizer] OSRM distance matrix error: {e}, using Haversine fallback")
        
        # Fallback to Haversine distances
        matrix = []
        for i, loc_i in enumerate(locations):
            row = []
            for j, loc_j in enumerate(locations):
                if i == j:
                    row.append(0.0)
                else:
                    dist = _haversine_km(loc_i.lat, loc_i.lng, loc_j.lat, loc_j.lng)
                    row.append(dist)
            matrix.append(row)
        
        return matrix
    
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
        
        # PRE-FILTER: Check compatibility BEFORE building location index
        # This prevents pre-filtered tasks from getting indices in the routing model
        self.pre_filtered_tasks = []
        self.routable_tasks = []  # Tasks that have at least one compatible agent
        
        for task in self.unassigned_tasks:
            # Check if ANY agent is compatible with this task
            compatible_agents = []
            incompatibility_reasons = []
            
            for agent in self.agents:
                is_compatible, reason = self.compatibility_matrix[agent.id][task.id]
                if is_compatible:
                    compatible_agents.append(agent.name)
                else:
                    incompatibility_reasons.append(reason)
            
            if compatible_agents:
                self.routable_tasks.append(task)
            else:
                # No compatible agents - pre-filter this task
                reason = max(set(incompatibility_reasons), key=incompatibility_reasons.count) if incompatibility_reasons else "no_compatible_agents"
                print(f"[FleetOptimizer] PRE-FILTER: Task {task.id[:20]}... no compatible agents (reason: {reason})")
                self.pre_filtered_tasks.append({
                    'task': task,
                    'reason': reason
                })
        
        print(f"[FleetOptimizer] Pre-filtered {len(self.pre_filtered_tasks)} tasks, {len(self.routable_tasks)} tasks routable")
        
        # Build location index with ONLY routable tasks
        locations, index_map = self._build_location_index(self.routable_tasks)
        num_locations = len(locations)
        num_vehicles = len(self.agents)
        
        print(f"[FleetOptimizer] Building model: {num_locations} locations, {num_vehicles} vehicles")
        print(f"[FleetOptimizer] Routable tasks: {len(self.routable_tasks)}")
        
        # Get travel time matrix
        print("[FleetOptimizer] Fetching travel times from OSRM...")
        time_matrix = get_travel_time_matrix(locations)
        
        # Build distance matrix for chain-aware enforcement
        # This ensures every hop in the route is within maxDistanceKm
        max_distance_km = self.max_distance_km
        distance_matrix = None
        if max_distance_km is not None:
            print(f"[FleetOptimizer] Building distance matrix for chain-aware filtering (max {max_distance_km}km per hop)...")
            distance_matrix = self._build_distance_matrix(locations)
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(
            num_locations,
            num_vehicles,
            [index_map['agent_starts'][a.id] for a in self.agents],  # starts
            [index_map['agent_ends'][a.id] for a in self.agents]     # ends
        )
        routing = pywrapcp.RoutingModel(manager)
        
        # Travel time callback with chain-aware distance enforcement
        # Progressive penalty: the further over maxDistanceKm, the heavier the penalty
        # Uses quadratic scaling so small overages are tolerable but large ones are punishing
        BASE_PENALTY_PER_KM = 600  # 10 min base penalty per km over
        
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            base_time = time_matrix[from_node][to_node] + DEFAULT_SERVICE_TIME_SECONDS
            
            # Chain-aware enforcement: progressive penalty for exceeding maxDistanceKm
            if distance_matrix is not None and max_distance_km is not None:
                hop_distance_km = distance_matrix[from_node][to_node]
                if hop_distance_km > max_distance_km:
                    # Progressive (quadratic) penalty:
                    # - 1km over: 1^2 * 600 = 10 min penalty
                    # - 2km over: 2^2 * 600 = 40 min penalty  
                    # - 3km over: 3^2 * 600 = 1.5 hour penalty
                    # - 5km over: 5^2 * 600 = 4+ hour penalty (strongly discouraged)
                    excess_km = hop_distance_km - max_distance_km
                    penalty = int((excess_km ** 2) * BASE_PENALTY_PER_KM)
                    return base_time + penalty
            
            return base_time
        
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
        # Track NEW tasks only - count pickups of routable unassigned tasks
        # Build set of routable task IDs for quick lookup
        routable_task_ids = {t.id for t in self.routable_tasks}
        
        def demand_callback(from_index):
            """Returns demand: +1 for new task pickup, 0 otherwise"""
            node = manager.IndexToNode(from_index)
            task_info = index_map['task_at_index'].get(node)
            if task_info:
                task_id, stop_type, original_agent = task_info
                # Only count NEW task pickups (not existing tasks, not deliveries)
                if stop_type == 'pickup' and task_id in routable_task_ids:
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
        
        # Existing tasks (already assigned to specific agents) - MANDATORY, no disjunction
        print(f"[FleetOptimizer] Setting up existing task constraints...")
        for agent_idx, agent in enumerate(self.agents):
            for task in agent.current_tasks:
                pickup_idx = index_map['pickups'].get(task.id)
                delivery_idx = index_map['deliveries'].get(task.id)
                
                print(f"[FleetOptimizer] Existing task {task.id[:20]}... for {agent.name}: pickup_idx={pickup_idx}, delivery_idx={delivery_idx}")
                
                if pickup_idx is not None and delivery_idx is not None:
                    # Both pickup and delivery needed
                    pickup_delivery_pairs.append((pickup_idx, delivery_idx))
                    
                    # Force this task to this agent (MANDATORY - no disjunction)
                    pickup_index = manager.NodeToIndex(pickup_idx)
                    delivery_index = manager.NodeToIndex(delivery_idx)
                    
                    routing.SetAllowedVehiclesForIndex([agent_idx], pickup_index)
                    routing.SetAllowedVehiclesForIndex([agent_idx], delivery_index)
                    print(f"[FleetOptimizer]   Locked to agent {agent_idx} ({agent.name}): pickup->delivery")
                    
                elif delivery_idx is not None:
                    # Delivery only (pickup completed)
                    delivery_index = manager.NodeToIndex(delivery_idx)
                    routing.SetAllowedVehiclesForIndex([agent_idx], delivery_index)
                    print(f"[FleetOptimizer]   Locked to agent {agent_idx} ({agent.name}): delivery only (pickup done)")
                    
                    # Add time window for delivery - this is the real deadline
                    deadline_seconds = self._time_to_seconds(task.delivery_before)
                    time_dimension.CumulVar(delivery_index).SetMax(deadline_seconds + 1800)  # 30min grace
                else:
                    print(f"[FleetOptimizer]   WARNING: No indices found for existing task!")
        
        # Routable tasks (pre-filtering already done before building location index)
        for task in self.routable_tasks:
            pickup_idx = index_map['pickups'].get(task.id)
            delivery_idx = index_map['deliveries'].get(task.id)
            
            if pickup_idx is None or delivery_idx is None:
                continue
            
            # Get compatible vehicles (we know at least one exists because of pre-filtering)
            allowed_vehicles = []
            for agent_idx, agent in enumerate(self.agents):
                is_compatible, _ = self.compatibility_matrix[agent.id][task.id]
                if is_compatible:
                    allowed_vehicles.append(agent_idx)
            
            print(f"[FleetOptimizer] Task {task.id[:20]}... compatible with {len(allowed_vehicles)} agents: {[self.agents[i].name for i in allowed_vehicles[:3]]} (vehicle_indices: {allowed_vehicles})")
            
            pickup_delivery_pairs.append((pickup_idx, delivery_idx))
            
            pickup_index = manager.NodeToIndex(pickup_idx)
            delivery_index = manager.NodeToIndex(delivery_idx)
            
            # Set allowed vehicles - only agents who haven't declined and pass other checks
            print(f"[FleetOptimizer] Setting allowed vehicles for pickup_index={pickup_index}, delivery_index={delivery_index} -> vehicles={allowed_vehicles}")
            routing.SetAllowedVehiclesForIndex(allowed_vehicles, pickup_index)
            routing.SetAllowedVehiclesForIndex(allowed_vehicles, delivery_index)
            
            # Allow dropping this task as a pair (pickup + delivery together)
            # High penalty ensures we try to assign before dropping
            routing.AddDisjunction([pickup_index, delivery_index], 50000, 2)
            
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
                        # Routable unassigned task
                        for t in self.routable_tasks:
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
        
        # First, add pre-filtered tasks (those with NO compatible agents)
        for pre_filtered in getattr(self, 'pre_filtered_tasks', []):
            task = pre_filtered['task']
            
            # Build detailed agent-level breakdown
            agent_details = []
            for agent in self.agents:
                if task.id in self.compatibility_matrix.get(agent.id, {}):
                    compatible, reason = self.compatibility_matrix[agent.id][task.id]
                    agent_details.append({
                        'agent_id': agent.id,
                        'agent_name': agent.name,
                        'compatible': compatible,
                        'reason': reason
                    })
            
            unassigned_tasks.append({
                'task_id': task.id,
                'reason': pre_filtered['reason'],
                'restaurant_name': task.meta.get('restaurant_name', ''),
                'customer_name': task.meta.get('customer_name', ''),
                'agents_considered': agent_details
            })
        
        # Then add routable tasks that were in the routing model but not assigned
        for task in getattr(self, 'routable_tasks', []):
            if task.id not in assigned_task_ids:
                # Build detailed agent-level breakdown
                agent_details = []
                compatible_agents = []
                
                for agent in self.agents:
                    if task.id in self.compatibility_matrix.get(agent.id, {}):
                        compatible, reason = self.compatibility_matrix[agent.id][task.id]
                        agent_details.append({
                            'agent_id': agent.id,
                            'agent_name': agent.name,
                            'compatible': compatible,
                            'reason': reason
                        })
                        if compatible:
                            compatible_agents.append(agent.name)
                
                # Determine primary reason
                if compatible_agents:
                    # Had compatible agents but solver couldn't fit it
                    primary_reason = "no_feasible_route"
                    reason_detail = f"Compatible with {', '.join(compatible_agents)} but no feasible time slot"
                else:
                    incompatible_reasons = [a['reason'] for a in agent_details if not a['compatible']]
                    primary_reason = max(set(incompatible_reasons), key=incompatible_reasons.count) if incompatible_reasons else "unknown"
                    reason_detail = primary_reason
                
                unassigned_tasks.append({
                    'task_id': task.id,
                    'reason': primary_reason,
                    'reason_detail': reason_detail,
                    'restaurant_name': task.meta.get('restaurant_name', ''),
                    'customer_name': task.meta.get('customer_name', ''),
                    'agents_considered': agent_details
                })
        
        # Get compatibility stats
        compat_stats = getattr(self.compatibility_checker, 'compatibility_stats', {})
        
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
            'compatibility': {
                'total_agent_task_pairs': compat_stats.get('total_pairs', 0),
                'compatible_pairs': compat_stats.get('compatible', 0),
                'filtered_by_distance': compat_stats.get('filtered_by_distance', 0),
                'filtered_by_cash_rules': compat_stats.get('filtered_by_cash', 0),
                'filtered_by_declined': compat_stats.get('filtered_by_declined', 0),
                'filtered_by_other': compat_stats.get('filtered_by_other', 0)
            },
            'agent_routes': agent_routes,
            'unassigned_tasks': unassigned_tasks
        }
    
    def _empty_result(self, reason: str, solve_time: float = 0) -> Dict:
        """Return empty result with reason"""
        # Build agent details for each unassigned task
        unassigned_with_details = []
        for t in self.unassigned_tasks:
            agent_details = []
            for agent in self.agents:
                if hasattr(self, 'compatibility_matrix') and t.id in self.compatibility_matrix.get(agent.id, {}):
                    compatible, agent_reason = self.compatibility_matrix[agent.id][t.id]
                    agent_details.append({
                        'agent_id': agent.id,
                        'agent_name': agent.name,
                        'compatible': compatible,
                        'reason': agent_reason
                    })
                else:
                    agent_details.append({
                        'agent_id': agent.id,
                        'agent_name': agent.name,
                        'compatible': False,
                        'reason': reason
                    })
            
            unassigned_with_details.append({
                'task_id': t.id, 
                'reason': reason,
                'restaurant_name': t.meta.get('restaurant_name', ''),
                'customer_name': t.meta.get('customer_name', ''),
                'agents_considered': agent_details
            })
        
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
            'unassigned_tasks': unassigned_with_details
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
    max_distance_km = settings.get('maxDistanceKm', None)  # Max distance for assignment
    
    print(f"[optimize_fleet] Settings: wallet_threshold={wallet_threshold}, max_distance_km={max_distance_km}")
    
    # Build compatibility checker
    compatibility_checker = CompatibilityChecker(
        wallet_threshold=wallet_threshold,
        geofence_regions=geofences,
        max_distance_km=max_distance_km
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

