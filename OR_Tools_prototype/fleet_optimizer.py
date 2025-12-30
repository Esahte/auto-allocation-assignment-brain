"""
Fleet Optimizer - Many-to-Many Route Optimization

This module implements a fleet-wide optimization system that:
- Takes ALL unassigned tasks + ALL eligible agents
- Builds optimal routes for EVERY agent
- Returns complete route assignments

Conservative mode: Only assigns unassigned/declined tasks, doesn't reassign existing tasks.
"""

import time
import logging
import requests

# Set up logger for fleet optimizer
logger = logging.getLogger('fleet_optimizer')
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math

# =============================================================================
# CONFIGURATION
# =============================================================================

OSRM_SERVER = "https://osrm-caribbean-785077267034.us-central1.run.app"

# Maximum allowed lateness for deliveries (in minutes)
# If a new task assignment would make ANY delivery (new or existing) later than this, drop the new task
DEFAULT_MAX_LATENESS_MINUTES = 45

# Maximum allowed delay for pickups (in minutes)
# How long after food is ready can the agent arrive before the task is considered infeasible
DEFAULT_MAX_PICKUP_DELAY_MINUTES = 60

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
    tips: float = 0.0
    delivery_fee: float = 0.0
    
    @property
    def is_premium_task(self) -> bool:
        """Check if this is a premium task (tips >= $5 OR delivery_fee >= $18)"""
        return self.tips >= 5.0 or self.delivery_fee >= 18.0
    
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
        # CRITICAL: Ensure all IDs are strings for consistent comparison
        declined_by = []
        if 'declined_by' in data and data['declined_by']:
            raw_declined = data['declined_by']
            # Handle both formats: [{"driver_id": "123"}, ...] or ["123", ...]
            for d in raw_declined:
                if isinstance(d, dict):
                    driver_id = d.get('driver_id')
                    if driver_id:
                        declined_by.append(str(driver_id))
                else:
                    # Direct ID (string or int)
                    declined_by.append(str(d))
            if declined_by:
                logger.info(f"[FleetOptimizer] 📥 Task {data.get('id', '')[:20]}... has {len(declined_by)} declines: {declined_by}")
        
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
            meta=meta,
            tips=float(data.get('tips') or meta.get('tips') or 0.0),
            delivery_fee=float(data.get('delivery_fee') or meta.get('delivery_fee') or 0.0)
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
    priority: Optional[int] = None  # Priority level: 1 = highest (premium tasks only)
    
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
            geofence_regions=meta.get('geofence_regions', []),
            priority=meta.get('priority')  # None if not a priority agent
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
    
    Supports two modes controlled by prefilter_distance:
    - True (default): Pre-filter by distance for FAST proximity-based optimization
    - False: THOROUGH event-based optimization (still enforces max_distance_km as hard constraint)
    
    NOTE: max_distance_km is ALWAYS enforced as a hard constraint in both modes.
    The prefilter_distance flag only affects whether we pre-filter candidates for the solver.
    """
    
    def __init__(self, 
                 wallet_threshold: float = DEFAULT_WALLET_THRESHOLD,
                 geofence_regions: List[GeofenceRegion] = None,
                 max_distance_km: float = None,
                 prefilter_distance: bool = True,
                 max_lateness_minutes: int = DEFAULT_MAX_LATENESS_MINUTES,
                 max_pickup_delay_minutes: int = DEFAULT_MAX_PICKUP_DELAY_MINUTES):
        self.wallet_threshold = wallet_threshold
        self.geofence_regions = geofence_regions or []
        self.max_distance_km = max_distance_km  # Max distance for assignment
        self.prefilter_distance = prefilter_distance  # Whether to pre-filter by distance
        self.max_lateness_minutes = max_lateness_minutes  # Max allowed delivery lateness
        self.max_pickup_delay_minutes = max_pickup_delay_minutes  # Max delay after food ready
        self.distance_cache = {}  # Cache for agent-task distances
        # Build lookup for geofence by name
        self.geofence_by_name = {g.region_name: g for g in self.geofence_regions}
        
        if not prefilter_distance:
            print(f"[CompatibilityChecker] THOROUGH mode - max_distance_km={max_distance_km}km still enforced as hard constraint")
    
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
        Precompute Haversine (straight-line) distances from all agents to all task pickup locations.
        
        Uses Haversine instead of OSRM for:
        - Faster computation (no API calls)
        - More reliable (no network dependency)
        - Sufficient accuracy for proximity checks
        
        For agents with existing tasks, we compute BOTH:
        1. Distance from CURRENT location (for opportunistic pickups)
        2. Distance from PROJECTED location (for efficient chaining)
        
        If EITHER distance is within max_distance_km, the agent is compatible.
        This allows agents to pick up nearby tasks while on their way to deliveries.
        """
        if not agents or not tasks:
            return
        
        # Log which agents have both locations
        busy_agents = [a for a in agents if a.current_tasks]
        if busy_agents:
            print(f"[CompatibilityChecker] {len(busy_agents)} busy agents - checking BOTH current and projected locations")
            for agent in busy_agents:
                proj_loc = self._get_agent_projected_location(agent)
                print(f"[CompatibilityChecker] {agent.name}: current=({agent.current_location.lat:.4f}, {agent.current_location.lng:.4f}), "
                      f"projected=({proj_loc.lat:.4f}, {proj_loc.lng:.4f})")
        
        print(f"[CompatibilityChecker] Computing Haversine distances for {len(agents)} agents x {len(tasks)} tasks...")
        
        # Compute distances using Haversine (fast, no API calls)
        for agent in agents:
            proj_loc = self._get_agent_projected_location(agent)
            curr_loc = agent.current_location
            
            for task in tasks:
                task_loc = task.restaurant_location
                
                # Calculate both distances using Haversine
                proj_dist = _haversine_km(proj_loc.lat, proj_loc.lng, task_loc.lat, task_loc.lng)
                curr_dist = _haversine_km(curr_loc.lat, curr_loc.lng, task_loc.lat, task_loc.lng)
                
                # Use the MINIMUM distance - if agent is near NOW or will be near LATER
                min_dist = min(proj_dist, curr_dist)
                self.distance_cache[(agent.id, task.id)] = min_dist
                
                # Also store both for logging
                self.projected_distance_cache = getattr(self, 'projected_distance_cache', {})
                self.current_distance_cache = getattr(self, 'current_distance_cache', {})
                self.projected_distance_cache[(agent.id, task.id)] = proj_dist
                self.current_distance_cache[(agent.id, task.id)] = curr_dist
        
        print(f"[CompatibilityChecker] Cached {len(self.distance_cache)} Haversine distances (using MIN of current and projected)")
    
    def is_compatible(self, agent: Agent, task: Task) -> Tuple[bool, str]:
        """
        Check if an agent can be assigned a specific task.
        
        Returns:
            (is_compatible, reason) tuple
        """
        # =================================================================
        # PRIORITY AGENT RULES (Check first)
        # =================================================================
        # Priority 1 agents ONLY get premium tasks (tips >= $5 OR delivery_fee >= $18)
        # They bypass distance constraints for qualifying tasks
        if agent.priority == 1:
            if not task.is_premium_task:
                return False, "priority1_non_premium_task"
            # Priority 1 agents bypass distance for premium tasks (handled below)
        
        # =================================================================
        # STANDARD RULES
        # =================================================================
        
        # Rule 1: Decline History - Agent previously declined this task
        # CRITICAL: Normalize both agent ID and declined_by list to strings for comparison
        agent_id_str = str(agent.id)
        declined_by_str = [str(d) for d in task.declined_by] if task.declined_by else []
        
        if declined_by_str and agent_id_str in declined_by_str:
            logger.info(f"[FleetOptimizer] ⛔ BLOCKED: Agent {agent.name} ({agent_id_str}) previously declined task {task.id[:20]}...")
            return False, "agent_declined_task"
        
        # Rule 2: Cash Handling - All agents can handle cash BY DEFAULT
        # Only agents with NoCash tag are blocked from cash orders
        # Checks if 'nocash' appears ANYWHERE in the tag (handles AbsolutelyNoCash, etc.)
        if task.payment_type == "CASH":
            has_no_cash = False
            for tag in agent.tags:
                # Normalize: lowercase, remove dashes/spaces/underscores
                normalized = tag.lower().replace('-', '').replace('_', '').replace(' ', '')
                if 'nocash' in normalized:
                    has_no_cash = True
                    break
            
            if has_no_cash:
                return False, "no_cash_tag"
        
        # Rule 3: Wallet Threshold - Agent wallet too high for cash orders
        if task.payment_type == "CASH" and agent.wallet_balance > self.wallet_threshold:
            return False, "wallet_threshold_exceeded"
        
        # Rule 4: Tag Matching
        # Normalize tags for comparison (lowercase, replace - and space with _)
        agent_tags_lower = [t.lower().replace('-', '_').replace(' ', '_') for t in agent.tags]
        task_tags_lower = [t.lower().replace('-', '_').replace(' ', '_') for t in task.tags] if task.tags else []
        
        # Check if agent has TEST tag (INTERNAL is just a regular tag with no special rules)
        agent_is_test = any(t == 'test' for t in agent_tags_lower)
        # Check if task has TEST tag
        task_is_test = any(t == 'test' for t in task_tags_lower)
        
        # Rule 4a: TEST agents can ONLY get TEST tasks
        if agent_is_test and not task_is_test:
            return False, "test_agent_non_test_task"
        
        # Rule 4b: TEST tasks can ONLY go to TEST agents
        if task_is_test and not agent_is_test:
            return False, "test_task_non_test_agent"
        
        # Rule 4c: If task has tags (excluding TEST), agent must have at least one matching tag
        non_test_task_tags = [t for t in task_tags_lower if t != 'test']
        if non_test_task_tags:
            non_test_agent_tags = [t for t in agent_tags_lower if t != 'test']
            if not any(tag in non_test_agent_tags for tag in non_test_task_tags):
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
        # Uses MINIMUM of current and projected distance (opportunistic pickup + chaining)
        # ALWAYS enforce max_distance_km as a HARD constraint (even in THOROUGH mode)
        # SKIP for Priority 1 agents on premium tasks (they bypass distance)
        bypass_distance = (agent.priority == 1 and task.is_premium_task)
        
        if self.max_distance_km is not None and not bypass_distance:
            # Check cache first (uses MIN of current and projected)
            cache_key = (agent.id, task.id)
            if cache_key in self.distance_cache:
                distance = self.distance_cache[cache_key]
            else:
                # Fallback to Haversine - check BOTH current and projected
                curr_distance = _haversine_km(
                    agent.current_location.lat, agent.current_location.lng,
                    task.restaurant_location.lat, task.restaurant_location.lng
                )
                projected_loc = self._get_agent_projected_location(agent)
                proj_distance = _haversine_km(
                    projected_loc.lat, projected_loc.lng,
                    task.restaurant_location.lat, task.restaurant_location.lng
                )
                distance = min(curr_distance, proj_distance)
            
            if distance > self.max_distance_km:
                # Get both distances for detailed logging
                proj_dist = getattr(self, 'projected_distance_cache', {}).get(cache_key, float('inf'))
                curr_dist = getattr(self, 'current_distance_cache', {}).get(cache_key, float('inf'))
                
                if agent.current_tasks:
                    return False, f"distance_exceeded_curr={curr_dist:.1f}km_proj={proj_dist:.1f}km"
                else:
                    return False, f"distance_exceeded_{distance:.1f}km"
        
        # Rule 7: Delivery Direction Coherence - New task's delivery should be in same direction as existing tasks
        # This prevents inefficient zig-zag routes where agent has to go north then south
        if agent.current_tasks and len(agent.current_tasks) > 0:
            direction_check = self._check_delivery_direction_coherence(agent, task)
            if not direction_check[0]:
                return direction_check
        
        # Priority 1 agents get a special compatible reason for premium tasks
        if agent.priority == 1 and task.is_premium_task:
            return True, "compatible_priority1_premium"
        
        return True, "compatible"
    
    def _check_delivery_direction_coherence(self, agent: Agent, new_task: Task) -> Tuple[bool, str]:
        """
        Check if the new task's delivery is in a coherent direction with existing task deliveries.
        
        This prevents assigning tasks where:
        - Agent is at Restaurant A
        - Existing task delivery goes NORTH
        - New task delivery goes SOUTH (opposite direction = inefficient)
        
        Returns:
            (is_coherent, reason) tuple
        """
        if not agent.current_tasks:
            return True, "no_existing_tasks"
        
        # Get the reference point - use the new task's pickup location as the center
        # (since agent will be at pickup to collect both orders)
        center_lat = new_task.restaurant_location.lat
        center_lng = new_task.restaurant_location.lng
        
        # Calculate direction vector for new task's delivery (from pickup to delivery)
        new_delivery_lat = new_task.delivery_location.lat
        new_delivery_lng = new_task.delivery_location.lng
        new_delta_lat = new_delivery_lat - center_lat
        new_delta_lng = new_delivery_lng - center_lng
        
        # Check against each existing task's delivery direction
        for existing_task in agent.current_tasks:
            # Only check tasks from the SAME or NEARBY restaurant (within 0.5km)
            # Tasks from different restaurants don't need direction coherence
            pickup_distance = _haversine_km(
                center_lat, center_lng,
                existing_task.restaurant_location.lat, existing_task.restaurant_location.lng
            )
            if pickup_distance > 0.5:  # Different restaurant, skip
                continue
            
            # Calculate direction vector for existing task's delivery
            existing_delivery_lat = existing_task.delivery_location.lat
            existing_delivery_lng = existing_task.delivery_location.lng
            existing_delta_lat = existing_delivery_lat - center_lat
            existing_delta_lng = existing_delivery_lng - center_lng
            
            # Calculate dot product to determine if directions are aligned
            # dot > 0 means same direction, dot < 0 means opposite
            dot_product = (new_delta_lat * existing_delta_lat) + (new_delta_lng * existing_delta_lng)
            
            # Calculate magnitudes for angle calculation
            new_magnitude = (new_delta_lat**2 + new_delta_lng**2) ** 0.5
            existing_magnitude = (existing_delta_lat**2 + existing_delta_lng**2) ** 0.5
            
            if new_magnitude < 0.001 or existing_magnitude < 0.001:
                continue  # Skip if delivery is at same location as pickup
            
            # Calculate cosine of angle between directions
            cos_angle = dot_product / (new_magnitude * existing_magnitude)
            
            # If cos_angle < -0.3 (angle > ~107 degrees), deliveries are in opposite directions
            # This threshold allows some flexibility but rejects clearly opposite routes
            if cos_angle < -0.3:
                # Calculate actual distances for logging
                new_dist = _haversine_km(center_lat, center_lng, new_delivery_lat, new_delivery_lng)
                existing_dist = _haversine_km(center_lat, center_lng, existing_delivery_lat, existing_delivery_lng)
                
                logger.info(f"[FleetOptimizer] ⛔ DIRECTION BLOCK: {agent.name} has delivery going "
                          f"{'N' if existing_delta_lat > 0 else 'S'} ({existing_dist:.1f}km), "
                          f"new task goes {'N' if new_delta_lat > 0 else 'S'} ({new_dist:.1f}km) - OPPOSITE DIRECTION")
                return False, f"opposite_delivery_direction"
        
        return True, "direction_coherent"
    
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
            'filtered_by_direction': 0,
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
                elif reason == 'opposite_delivery_direction':
                    stats['filtered_by_direction'] += 1
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
        
        # For large requests (>50 locations), use chunking to avoid URL length limits
        if len(locations) > 50:
            print(f"[OSRM] Large request ({len(locations)} locations) - using chunked approach")
            return _get_travel_times_chunked(locations, session)
        
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
    print(f"[OSRM] Using Haversine fallback for {len(locations)} locations")
    return _estimate_travel_times(locations)


def _get_travel_times_chunked(locations: List[Location], session) -> List[List[int]]:
    """
    Get travel times in chunks to avoid OSRM URL length limits.
    Splits large requests into smaller batches and combines results.
    """
    n = len(locations)
    matrix = [[999999] * n for _ in range(n)]  # Initialize with high values
    
    # Set diagonal to 0
    for i in range(n):
        matrix[i][i] = 0
    
    CHUNK_SIZE = 40  # Safe size for OSRM GET requests
    
    # Process in chunks
    for i_start in range(0, n, CHUNK_SIZE):
        i_end = min(i_start + CHUNK_SIZE, n)
        
        for j_start in range(0, n, CHUNK_SIZE):
            j_end = min(j_start + CHUNK_SIZE, n)
            
            # Get subset of locations
            sources = list(range(i_start, i_end))
            destinations = list(range(j_start, j_end))
            
            # Build URL with sources and destinations parameters
            coords_str = ";".join([locations[idx].to_osrm_str() for idx in sources + destinations])
            
            # Create source and destination indices for the combined array
            source_indices = ";".join(str(i) for i in range(len(sources)))
            dest_indices = ";".join(str(i) for i in range(len(sources), len(sources) + len(destinations)))
            
            url = f"{OSRM_SERVER}/table/v1/driving/{coords_str}?annotations=duration&sources={source_indices}&destinations={dest_indices}"
            
            try:
                response = session.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data.get('code') == 'Ok':
                    durations = data.get('durations', [])
                    
                    # Copy results into the main matrix
                    for i_local, i_global in enumerate(sources):
                        for j_local, j_global in enumerate(destinations):
                            if i_local < len(durations) and j_local < len(durations[i_local]):
                                val = durations[i_local][j_local]
                                matrix[i_global][j_global] = int(val) if val is not None else 999999
            except Exception as e:
                print(f"[OSRM] Chunk error ({i_start}-{i_end} x {j_start}-{j_end}): {e}")
                # Fill chunk with Haversine estimates
                for i_global in sources:
                    for j_global in destinations:
                        if i_global != j_global:
                            dist = _haversine_km(
                                locations[i_global].lat, locations[i_global].lng,
                                locations[j_global].lat, locations[j_global].lng
                            )
                            matrix[i_global][j_global] = int(dist / 30 * 3600)
    
    print(f"[OSRM] Chunked request complete: {n}x{n} matrix built")
    return matrix


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
                 current_time: datetime = None,
                 prefilter_distance: bool = True):
        self.agents = agents
        self.unassigned_tasks = unassigned_tasks
        self.compatibility_checker = compatibility_checker
        self.current_time = current_time or datetime.now(timezone.utc)
        self.prefilter_distance = prefilter_distance
        
        # Store max distance for chain-aware filtering
        # ALWAYS enforce max_distance_km as a hard constraint in ALL modes
        # This prevents assigning far-away agents even in THOROUGH mode
        self.max_distance_km = getattr(compatibility_checker, 'max_distance_km', None)
        
        # Store max lateness for post-solution validation
        self.max_lateness_minutes = getattr(compatibility_checker, 'max_lateness_minutes', DEFAULT_MAX_LATENESS_MINUTES)
        
        # Store max pickup delay for time window constraints
        self.max_pickup_delay_minutes = getattr(compatibility_checker, 'max_pickup_delay_minutes', DEFAULT_MAX_PICKUP_DELAY_MINUTES)
        
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
        
        print(f"[FleetOptimizer] Checking compatibility for {len(self.unassigned_tasks)} unassigned task(s)...")
        
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
        
        # DEBUG: Show Priority agents and their compatibility
        p1_agents = [a for a in self.agents if a.priority == 1]
        if p1_agents:
            print(f"[FleetOptimizer] 🔍 Priority 1 agents detected: {[a.name for a in p1_agents]}")
            for p1 in p1_agents:
                print(f"[FleetOptimizer] 🔍 {p1.name}: priority={p1.priority} (type={type(p1.priority).__name__}), tags={p1.tags}")
        else:
            # Check if any agent has priority set at all
            agents_with_priority = [(a.name, a.priority, type(a.priority).__name__) for a in self.agents if a.priority is not None]
            if agents_with_priority:
                print(f"[FleetOptimizer] ⚠️ Agents with priority (but not == 1): {agents_with_priority}")
            else:
                print(f"[FleetOptimizer] ⚠️ No agents have priority field set")
        
        # Show why there's nothing to assign
        if len(self.routable_tasks) == 0 and len(self.unassigned_tasks) > 0:
            print(f"[FleetOptimizer] ⚠️ ALL {len(self.unassigned_tasks)} tasks were pre-filtered!")
            for pf in self.pre_filtered_tasks[:3]:
                task = pf['task']
                reason = pf['reason']
                print(f"  → {task.meta.get('restaurant_name', task.id[:15])}: {reason}")
        
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
        # max_distance_km is ALWAYS enforced in compatibility checks (both FAST and THOROUGH modes)
        max_distance_km = self.max_distance_km
        distance_matrix = None
        if max_distance_km is not None:
            print(f"[FleetOptimizer] Building distance matrix for chain-aware filtering (max {max_distance_km}km per hop)...")
            distance_matrix = self._build_distance_matrix(locations)
        else:
            print(f"[FleetOptimizer] No max_distance_km configured - distance not constrained")
        
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
        
        # DEBUG: Log travel times for same-restaurant chains (pickup → multiple deliveries)
        logger.info(f"[FleetOptimizer] 🕐 DEBUG: Checking travel times for chained deliveries:")
        
        # Find tasks from the same restaurant
        pickup_to_tasks = {}
        for task in self.routable_tasks:
            pickup_key = (round(task.restaurant_location.lat, 4), round(task.restaurant_location.lng, 4))
            if pickup_key not in pickup_to_tasks:
                pickup_to_tasks[pickup_key] = []
            pickup_to_tasks[pickup_key].append(task)
        
        # Log travel times for same-pickup chains
        for pickup_loc, tasks in pickup_to_tasks.items():
            if len(tasks) > 1:
                logger.info(f"[FleetOptimizer]   Same-restaurant chain at {pickup_loc} ({len(tasks)} tasks):")
                for task in tasks:
                    pickup_idx = index_map['pickups'].get(task.id)
                    delivery_idx = index_map['deliveries'].get(task.id)
                    if pickup_idx is not None and delivery_idx is not None:
                        travel_sec = time_matrix[pickup_idx][delivery_idx]
                        travel_min = travel_sec / 60
                        customer = task.meta.get('customer_name', task.id[:15])
                        logger.info(f"[FleetOptimizer]     → {customer}: pickup[{pickup_idx}] to delivery[{delivery_idx}] = {travel_sec}s ({travel_min:.1f}min)")
                        
                        # Also log delivery-to-delivery times for ordering
                        for other_task in tasks:
                            if other_task.id != task.id:
                                other_delivery_idx = index_map['deliveries'].get(other_task.id)
                                if other_delivery_idx:
                                    d2d_time = time_matrix[delivery_idx][other_delivery_idx]
                                    other_customer = other_task.meta.get('customer_name', other_task.id[:15])
                                    logger.info(f"[FleetOptimizer]       {customer} delivery → {other_customer} delivery = {d2d_time}s ({d2d_time/60:.1f}min)")
        
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
                    
                    # Add time windows for existing task - handle already-late gracefully
                    raw_pickup_delta = (task.pickup_before - self.current_time).total_seconds()
                    raw_delivery_delta = (task.delivery_before - self.current_time).total_seconds()
                    
                    if raw_pickup_delta < 0:
                        # Pickup already late - allow pickup anytime in next 2 hours
                        pickup_seconds = 0
                        pickup_max = 7200  # 2 hours
                        late_by = abs(raw_pickup_delta) / 60
                        print(f"[FleetOptimizer]     ⚠️ Pickup already {late_by:.0f}min late - relaxing to 2hr window")
                    else:
                        pickup_seconds = int(raw_pickup_delta)
                        pickup_max = pickup_seconds + self.max_pickup_delay_minutes * 60
                    
                    if raw_delivery_delta < 0:
                        # Delivery already late - allow delivery anytime in next 2 hours
                        delivery_max = 7200  # 2 hours
                        late_by = abs(raw_delivery_delta) / 60
                        print(f"[FleetOptimizer]     ⚠️ Delivery already {late_by:.0f}min late - relaxing to 2hr window")
                    else:
                        delivery_max = int(raw_delivery_delta) + self.max_lateness_minutes * 60
                    
                    time_dimension.CumulVar(pickup_index).SetMax(pickup_max)
                    time_dimension.CumulVar(delivery_index).SetMax(delivery_max)
                    
                elif delivery_idx is not None:
                    # Delivery only (pickup completed)
                    delivery_index = manager.NodeToIndex(delivery_idx)
                    routing.SetAllowedVehiclesForIndex([agent_idx], delivery_index)
                    print(f"[FleetOptimizer]   Locked to agent {agent_idx} ({agent.name}): delivery only (pickup done)")
                    
                    # Add time window for delivery - handle already-late tasks gracefully
                    # If deadline is in the past, give generous window to avoid ROUTING_FAIL
                    raw_deadline_delta = (task.delivery_before - self.current_time).total_seconds()
                    if raw_deadline_delta < 0:
                        # Task is ALREADY LATE - give 2 hours from now to complete
                        # This prevents late tasks from making entire solver fail
                        late_by_minutes = abs(raw_deadline_delta) / 60
                        deadline_seconds = 7200  # 2 hours from now
                        print(f"[FleetOptimizer]     ⚠️ Task already {late_by_minutes:.0f}min late - relaxing constraint to 2hr window")
                    else:
                        deadline_seconds = int(raw_deadline_delta)
                    time_dimension.CumulVar(delivery_index).SetMax(deadline_seconds + 1800)  # +30min grace
                else:
                    print(f"[FleetOptimizer]   WARNING: No indices found for existing task!")
        
        # Routable tasks (pre-filtering already done before building location index)
        for task in self.routable_tasks:
            pickup_idx = index_map['pickups'].get(task.id)
            delivery_idx = index_map['deliveries'].get(task.id)
            
            if pickup_idx is None or delivery_idx is None:
                continue
            
            # Get compatible vehicles (we know at least one exists because of pre-filtering)
            # For PREMIUM tasks: Priority 1 agents get FIRST DIBS (exclusive if any are available)
            all_compatible_vehicles = []
            priority1_compatible_vehicles = []
            
            for agent_idx, agent in enumerate(self.agents):
                is_compatible, reason = self.compatibility_matrix[agent.id][task.id]
                if is_compatible:
                    all_compatible_vehicles.append(agent_idx)
                    # Track Priority 1 agents separately
                    if agent.priority == 1:
                        priority1_compatible_vehicles.append(agent_idx)
                elif agent.priority == 1 and task.is_premium_task:
                    # Debug: Why is P1 agent not compatible with a premium task?
                    logger.warning(f"[FleetOptimizer] ⚠️ P1 agent {agent.name} NOT compatible with premium task {task.id[:20]}... Reason: {reason}")
            
            # PRIORITY ASSIGNMENT LOGIC:
            # If task is premium AND there are eligible Priority 1 agents -> they get exclusive access
            # Otherwise -> all compatible agents can take it
            if task.is_premium_task and priority1_compatible_vehicles:
                allowed_vehicles = priority1_compatible_vehicles
                priority_note = f" ⭐ PRIORITY (P1 agents: {[self.agents[i].name for i in priority1_compatible_vehicles]})"
                # Log this important event to persistent logs
                logger.info(f"[FleetOptimizer] ⭐ PREMIUM TASK {task.id[:20]}... → Priority 1 agents get exclusive: {[self.agents[i].name for i in priority1_compatible_vehicles]}")
            else:
                allowed_vehicles = all_compatible_vehicles
                priority_note = ""
                if task.is_premium_task and not priority1_compatible_vehicles:
                    priority_note = " (no P1 agents available - using regular agents)"
                    logger.info(f"[FleetOptimizer] ⭐ PREMIUM TASK {task.id[:20]}... → No P1 agents available, using regular agents")
            
            print(f"[FleetOptimizer] Task {task.id[:20]}... compatible with {len(allowed_vehicles)} agents: {[self.agents[i].name for i in allowed_vehicles[:3]]} (vehicle_indices: {allowed_vehicles}){priority_note}")
            
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
            # Allow late pickup with configurable grace period (food can wait)
            max_pickup_delay = self.max_pickup_delay_minutes * 60  # Convert to seconds
            time_dimension.CumulVar(pickup_index).SetMax(pickup_ready_time + max_pickup_delay)
            
            # Delivery: Use configurable max lateness as hard constraint
            # This is the real deadline - prioritize this!
            max_lateness_grace = self.max_lateness_minutes * 60  # Convert to seconds
            time_dimension.CumulVar(delivery_index).SetMax(delivery_deadline + max_lateness_grace)
        
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
        
        # ================================================================
        # DELIVERY ORDER OPTIMIZATION
        # ================================================================
        # For chained tasks from the same pickup, compute optimal delivery order
        # based on distance/travel time, not just time windows
        
        # Find same-pickup chains and sort by delivery distance from pickup
        pickup_chains = {}
        for task in self.routable_tasks:
            pickup_idx = index_map['pickups'].get(task.id)
            delivery_idx = index_map['deliveries'].get(task.id)
            if pickup_idx is not None and delivery_idx is not None:
                # Use pickup location as key (rounded to avoid floating point issues)
                pickup_key = (round(task.restaurant_location.lat, 4), round(task.restaurant_location.lng, 4))
                if pickup_key not in pickup_chains:
                    pickup_chains[pickup_key] = []
                # Store travel time from pickup to delivery
                travel_time = time_matrix[pickup_idx][delivery_idx]
                pickup_chains[pickup_key].append({
                    'task': task,
                    'delivery_idx': delivery_idx,
                    'pickup_idx': pickup_idx,
                    'travel_time': travel_time
                })
        
        # Log optimal delivery orders for chains
        for pickup_key, chain in pickup_chains.items():
            if len(chain) > 1:
                # Sort by travel time (closest delivery first)
                chain_sorted = sorted(chain, key=lambda x: x['travel_time'])
                logger.info(f"[FleetOptimizer] 🔗 CHAIN OPTIMIZATION: {len(chain)} deliveries from same pickup:")
                for i, item in enumerate(chain_sorted):
                    customer = item['task'].meta.get('customer_name', 'Unknown')[:20]
                    travel_min = item['travel_time'] / 60
                    logger.info(f"[FleetOptimizer]   {i+1}. {customer} ({travel_min:.1f}min from pickup)")
        
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
            return self._extract_solution(routing, manager, solution, index_map, time_dimension, solve_time, time_matrix)
        else:
            # Diagnose WHY no solution was found
            status = routing.status()
            status_names = {
                0: "ROUTING_NOT_SOLVED",
                1: "ROUTING_SUCCESS", 
                2: "ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED",
                3: "ROUTING_FAIL",
                4: "ROUTING_FAIL_TIMEOUT",
                5: "ROUTING_INVALID"
            }
            status_name = status_names.get(status, f"UNKNOWN_{status}")
            
            # Check capacity issues
            agents_at_capacity = sum(1 for a in self.agents if a.available_capacity <= 0)
            agents_with_space = len(self.agents) - agents_at_capacity
            
            # Build detailed reason
            if agents_with_space == 0:
                reason = "all_agents_at_capacity"
                print(f"[FleetOptimizer] No solution: ALL {len(self.agents)} agents at max capacity")
            elif status == 4:
                reason = "solver_timeout"
                print(f"[FleetOptimizer] No solution: Solver timed out ({solve_time:.1f}s)")
            elif status == 5:
                reason = "invalid_model"
                print(f"[FleetOptimizer] No solution: Model invalid (check constraints)")
            else:
                reason = f"infeasible_{status_name}"
                print(f"[FleetOptimizer] No solution: {status_name} - {agents_with_space}/{len(self.agents)} agents have capacity")
                print(f"[FleetOptimizer] Hint: Check time windows, distances, or constraint conflicts")
            
            return self._empty_result(reason, solve_time)
    
    def _extract_solution(self, routing, manager, solution, index_map, time_dimension, solve_time, time_matrix) -> Dict:
        """Extract solution into structured format"""
        
        agent_routes = []
        assigned_task_ids = set()
        total_lateness = 0
        
        # DEBUG: Check which tasks were dropped
        dropped_tasks = []
        for task in self.routable_tasks:
            task_id = task.id
            is_assigned = False
            for vehicle_idx in range(len(self.agents)):
                index = routing.Start(vehicle_idx)
                while not routing.IsEnd(index):
                    node = manager.IndexToNode(index)
                    task_info = index_map['task_at_index'].get(node)
                    if task_info and task_info[0] == task_id:
                        is_assigned = True
                        break
                    index = solution.Value(routing.NextVar(index))
                if is_assigned:
                    break
            if not is_assigned:
                dropped_tasks.append(task)
        
        if dropped_tasks:
            print(f"[FleetOptimizer] ⚠️ Solver DROPPED {len(dropped_tasks)} task(s):")
            for task in dropped_tasks:
                time_to_pickup = (task.pickup_before - self.current_time).total_seconds() / 60
                time_to_delivery = (task.delivery_before - self.current_time).total_seconds() / 60
                print(f"  → {task.meta.get('restaurant_name', task.id[:15])}: pickup in {time_to_pickup:.1f}min, deliver in {time_to_delivery:.1f}min")
        
        for vehicle_idx, agent in enumerate(self.agents):
            route_stops = []
            assigned_new_tasks = []
            
            index = routing.Start(vehicle_idx)
            
            prev_arrival_seconds = 0
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                time_var = time_dimension.CumulVar(index)
                arrival_seconds = solution.Value(time_var)
                arrival_time = self.current_time.timestamp() + arrival_seconds
                arrival_dt = datetime.fromtimestamp(arrival_time, tz=timezone.utc)
                
                # DEBUG: Log travel time between stops
                delta_seconds = arrival_seconds - prev_arrival_seconds
                logger.info(f"[FleetOptimizer] 🚗 Route extraction: node={node}, arrival_seconds={arrival_seconds} (+{delta_seconds}s / {delta_seconds/60:.1f}min from prev), arrival_time={arrival_dt.strftime('%H:%M:%S')} UTC")
                prev_arrival_seconds = arrival_seconds
                
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
            
            # DEBUG: Log the complete route with times and verify travel times
            if route_stops and assigned_new_tasks:
                logger.info(f"[FleetOptimizer] 📋 Route for {agent.name} ({len(route_stops)} stops):")
                prev_arrival_dt = None
                prev_location = None
                prev_node_idx = None
                
                for i, stop in enumerate(route_stops):
                    stop_type = stop.get('type', 'unknown')
                    display = stop.get('display_name', stop.get('task_id', 'unknown')[:15])
                    arrival = stop.get('arrival_time', 'N/A')
                    location = stop.get('location')
                    
                    # Get the index for this stop
                    task_id = stop.get('task_id')
                    if 'pickup' in stop_type:
                        node_idx = index_map['pickups'].get(task_id)
                    elif 'delivery' in stop_type:
                        node_idx = index_map['deliveries'].get(task_id)
                    else:
                        node_idx = None
                    
                    # Calculate actual vs expected travel time
                    if prev_arrival_dt and arrival and prev_node_idx is not None and node_idx is not None:
                        try:
                            current_arrival = datetime.fromisoformat(arrival.replace('Z', '+00:00'))
                            actual_delta = (current_arrival - prev_arrival_dt).total_seconds()
                            expected_travel = time_matrix[prev_node_idx][node_idx]
                            expected_with_service = expected_travel + DEFAULT_SERVICE_TIME_SECONDS
                            
                            if abs(actual_delta - expected_with_service) > 300:  # More than 5 min difference
                                logger.warning(f"[FleetOptimizer] ⚠️ TRAVEL TIME MISMATCH: node {prev_node_idx}→{node_idx}")
                                logger.warning(f"[FleetOptimizer]   Actual delta: {actual_delta}s ({actual_delta/60:.1f}min)")
                                logger.warning(f"[FleetOptimizer]   Expected (OSRM + service): {expected_with_service}s ({expected_with_service/60:.1f}min)")
                        except Exception as e:
                            pass
                    
                    if 'late_seconds' in stop:
                        logger.info(f"[FleetOptimizer]   {i+1}. {stop_type}: {display} @ {arrival} (wait: {stop.get('wait_seconds', 0)}s)")
                    elif 'lateness_seconds' in stop:
                        logger.info(f"[FleetOptimizer]   {i+1}. {stop_type}: {display} @ {arrival} (early: {stop.get('early_seconds', 0)}s)")
                    else:
                        logger.info(f"[FleetOptimizer]   {i+1}. {stop_type}: {display} @ {arrival}")
                    
                    # Update previous values
                    if arrival:
                        try:
                            prev_arrival_dt = datetime.fromisoformat(arrival.replace('Z', '+00:00'))
                        except:
                            pass
                    prev_location = location
                    prev_node_idx = node_idx
            
            # ================================================================
            # RECALCULATE ACTUAL ARRIVAL TIMES
            # ================================================================
            # The solver's cumulative times may include slack (waiting).
            # Recalculate based on actual travel sequence and OSRM times.
            
            if route_stops:
                recalc_time = self.current_time
                prev_node_idx = index_map['agent_starts'].get(agent.id)
                
                for stop in route_stops:
                    task_id = stop.get('task_id')
                    stop_type = stop.get('type', '')
                    
                    # Get node index
                    if 'pickup' in stop_type:
                        node_idx = index_map['pickups'].get(task_id)
                    elif 'delivery' in stop_type:
                        node_idx = index_map['deliveries'].get(task_id)
                    else:
                        continue
                    
                    if node_idx is None or prev_node_idx is None:
                        continue
                    
                    # Get travel time from OSRM
                    travel_seconds = time_matrix[prev_node_idx][node_idx]
                    service_seconds = DEFAULT_SERVICE_TIME_SECONDS
                    
                    # Calculate new arrival time
                    recalc_time = recalc_time + timedelta(seconds=travel_seconds + service_seconds)
                    
                    # For pickups, we might need to wait for food to be ready
                    if 'pickup' in stop_type:
                        task = None
                        for t in self.routable_tasks:
                            if t.id == task_id:
                                task = t
                                break
                        if task and task.pickup_before > recalc_time:
                            # Need to wait for food
                            wait_seconds = (task.pickup_before - recalc_time).total_seconds()
                            stop['wait_for_food_seconds'] = int(wait_seconds)
                            recalc_time = task.pickup_before
                    
                    # Store recalculated time
                    original_arrival = stop.get('arrival_time', '')
                    stop['recalculated_arrival_time'] = recalc_time.isoformat()
                    
                    # Update the main arrival_time with the recalculated value
                    if original_arrival:
                        try:
                            orig_dt = datetime.fromisoformat(original_arrival.replace('Z', '+00:00'))
                            diff = abs((recalc_time - orig_dt).total_seconds())
                            if diff > 120:  # More than 2 min difference
                                logger.warning(f"[FleetOptimizer] ⏱️ ARRIVAL CORRECTION: {stop.get('display_name', task_id[:15])}")
                                logger.warning(f"[FleetOptimizer]   Solver said: {orig_dt.strftime('%H:%M:%S')} UTC")
                                logger.warning(f"[FleetOptimizer]   Actual (recalc): {recalc_time.strftime('%H:%M:%S')} UTC (diff: {diff/60:.1f}min)")
                                # Use recalculated time
                                stop['arrival_time'] = recalc_time.isoformat()
                        except Exception:
                            pass
                    
                    prev_node_idx = node_idx
            
            # ================================================================
            # POST-SOLUTION LATENESS VALIDATION
            # ================================================================
            # Check if adding new tasks causes ANY delivery to exceed max lateness
            # If so, reject the new task assignments to protect existing deliveries
            
            max_lateness_seconds = self.max_lateness_minutes * 60
            rejected_new_tasks = []
            
            if assigned_new_tasks:
                # Check lateness for all delivery stops
                max_delivery_lateness = 0
                max_existing_delivery_lateness = 0
                late_existing_tasks = []
                
                for stop in route_stops:
                    if 'lateness_seconds' in stop:
                        lateness = stop.get('lateness_seconds', 0)
                        max_delivery_lateness = max(max_delivery_lateness, lateness)
                        
                        # Track lateness specifically for EXISTING tasks
                        if stop['type'].startswith('existing_task'):
                            max_existing_delivery_lateness = max(max_existing_delivery_lateness, lateness)
                            if lateness > max_lateness_seconds:
                                late_existing_tasks.append({
                                    'task_id': stop['task_id'],
                                    'restaurant_name': stop.get('restaurant_name', ''),
                                    'lateness_minutes': lateness // 60
                                })
                
                # RULE: If new tasks cause existing tasks to exceed max lateness, reject the new tasks
                if late_existing_tasks and assigned_new_tasks:
                    print(f"[FleetOptimizer] ⚠️ LATENESS VIOLATION for {agent.name}:")
                    print(f"  → New tasks would make {len(late_existing_tasks)} existing delivery(ies) >{self.max_lateness_minutes}min late")
                    for lt in late_existing_tasks:
                        print(f"    - {lt['restaurant_name']}: {lt['lateness_minutes']}min late")
                    print(f"  → Rejecting {len(assigned_new_tasks)} new task(s) to protect existing deliveries")
                    
                    # Move new tasks to rejected (they'll be added to unassigned)
                    rejected_new_tasks = assigned_new_tasks.copy()
                    assigned_new_tasks.clear()
                    
                    # Also remove new task stops from the route
                    route_stops = [s for s in route_stops if not s['type'].startswith('new_task')]
                    
                    # Remove from global assigned set
                    for task_id in rejected_new_tasks:
                        assigned_task_ids.discard(task_id)
                
                # RULE: Also reject new tasks if THEY themselves would be too late
                elif max_delivery_lateness > max_lateness_seconds and assigned_new_tasks:
                    # Find which new tasks are too late
                    too_late_new_tasks = []
                    for stop in route_stops:
                        if stop['type'] == 'new_task_delivery':
                            lateness = stop.get('lateness_seconds', 0)
                            if lateness > max_lateness_seconds:
                                too_late_new_tasks.append(stop['task_id'])
                    
                    if too_late_new_tasks:
                        print(f"[FleetOptimizer] ⚠️ NEW TASK TOO LATE for {agent.name}:")
                        print(f"  → {len(too_late_new_tasks)} new task(s) would be >{self.max_lateness_minutes}min late - rejecting")
                        
                        rejected_new_tasks = too_late_new_tasks
                        assigned_new_tasks = [t for t in assigned_new_tasks if t not in too_late_new_tasks]
                        
                        # Remove late new task stops from the route
                        route_stops = [s for s in route_stops if s.get('task_id') not in too_late_new_tasks]
                        
                        # Remove from global assigned set
                        for task_id in too_late_new_tasks:
                            assigned_task_ids.discard(task_id)
            
            # Build route info
            route_info = {
                'agent_id': agent.id,
                'agent_name': agent.name,
                'assigned_new_tasks': assigned_new_tasks,
                'rejected_due_to_lateness': rejected_new_tasks,
                'total_stops': len(route_stops),
                'route': route_stops,
                'total_lateness_seconds': sum(s.get('lateness_seconds', 0) for s in route_stops),
                'max_lateness_minutes_setting': self.max_lateness_minutes
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
            
            # Build detailed agent-level breakdown with capacity info
            agent_details = []
            for agent in self.agents:
                if task.id in self.compatibility_matrix.get(agent.id, {}):
                    compatible, reason = self.compatibility_matrix[agent.id][task.id]
                    has_capacity = agent.available_capacity > 0
                    agent_details.append({
                        'agent_id': agent.id,
                        'agent_name': agent.name,
                        'compatible': compatible,
                        'reason': reason,
                        'available_capacity': agent.available_capacity,
                        'can_be_assigned': compatible and has_capacity
                    })
            
            unassigned_tasks.append({
                'task_id': task.id,
                'reason': pre_filtered['reason'],
                'reason_detail': f"No agents pass compatibility rules ({pre_filtered['reason']})",
                'restaurant_name': task.meta.get('restaurant_name', ''),
                'customer_name': task.meta.get('customer_name', ''),
                'agents_considered': agent_details,
                'compatible_agents': 0,
                'compatible_with_capacity': 0
            })
        
        # Collect tasks rejected due to lateness
        lateness_rejected_task_ids = set()
        for route in agent_routes:
            for task_id in route.get('rejected_due_to_lateness', []):
                lateness_rejected_task_ids.add(task_id)
        
        # Then add routable tasks that were in the routing model but not assigned
        for task in getattr(self, 'routable_tasks', []):
            if task.id not in assigned_task_ids:
                # Build detailed agent-level breakdown
                # Include capacity info so dashboard can show WHY compatible agents can't be assigned
                agent_details = []
                compatible_agents = []
                
                for agent in self.agents:
                    if task.id in self.compatibility_matrix.get(agent.id, {}):
                        compatible, reason = self.compatibility_matrix[agent.id][task.id]
                        has_capacity = agent.available_capacity > 0
                        
                        # Provide accurate reason for compatible agents
                        if compatible and not has_capacity:
                            display_reason = "at_max_capacity"
                        else:
                            display_reason = reason
                        
                        agent_details.append({
                            'agent_id': agent.id,
                            'agent_name': agent.name,
                            'compatible': compatible,
                            'reason': display_reason,
                            'available_capacity': agent.available_capacity,
                            'can_be_assigned': compatible and has_capacity
                        })
                        if compatible:
                            compatible_agents.append(agent.name)
                
                # Determine primary reason
                if task.id in lateness_rejected_task_ids:
                    # Task was rejected due to lateness constraint
                    primary_reason = "would_cause_excessive_lateness"
                    reason_detail = f"Assignment would cause delivery to be >{self.max_lateness_minutes}min late"
                elif compatible_agents:
                    # Had compatible agents but solver couldn't fit it
                    # Check if ALL compatible agents are at capacity
                    compatible_at_capacity = []
                    compatible_with_space = []
                    for a in self.agents:
                        if a.name in compatible_agents:
                            if a.available_capacity <= 0:
                                compatible_at_capacity.append(a.name)
                            else:
                                compatible_with_space.append(a.name)
                    
                    if len(compatible_at_capacity) == len(compatible_agents):
                        # ALL compatible agents are at max capacity
                        primary_reason = "compatible_agents_at_max_capacity"
                        reason_detail = f"Compatible with {len(compatible_agents)} agent(s) but ALL are at max_tasks limit"
                    elif compatible_at_capacity and not compatible_with_space:
                        # Same case, phrased differently
                        primary_reason = "compatible_agents_at_max_capacity"
                        reason_detail = f"Compatible agents ({', '.join(compatible_at_capacity)}) all at max_tasks limit"
                    else:
                        # Some compatible agents have space, but solver still couldn't fit
                        primary_reason = "no_feasible_route"
                        if compatible_with_space:
                            reason_detail = f"Agents with space ({', '.join(compatible_with_space)}) but no feasible time slot"
                        else:
                            reason_detail = f"Compatible with {', '.join(compatible_agents)} but no feasible time slot"
                else:
                    incompatible_reasons = [a['reason'] for a in agent_details if not a['compatible']]
                    primary_reason = max(set(incompatible_reasons), key=incompatible_reasons.count) if incompatible_reasons else "unknown"
                    reason_detail = primary_reason
                
                # Count compatible agents with/without capacity for dashboard
                compatible_count = len(compatible_agents)
                compatible_with_capacity = len([a for a in agent_details if a.get('can_be_assigned', False)])
                
                unassigned_tasks.append({
                    'task_id': task.id,
                    'reason': primary_reason,
                    'reason_detail': reason_detail,
                    'restaurant_name': task.meta.get('restaurant_name', ''),
                    'customer_name': task.meta.get('customer_name', ''),
                    'agents_considered': agent_details,
                    'compatible_agents': compatible_count,
                    'compatible_with_capacity': compatible_with_capacity
                })
        
        # Get compatibility stats
        compat_stats = getattr(self.compatibility_checker, 'compatibility_stats', {})
        
        # Count tasks rejected due to lateness
        tasks_rejected_for_lateness = len(lateness_rejected_task_ids)
        
        return {
            'success': True,
            'metadata': {
                'total_agents': len(self.agents),
                'total_unassigned_tasks': len(self.unassigned_tasks),
                'tasks_assigned': len(assigned_task_ids),
                'tasks_unassigned': len(unassigned_tasks),
                'tasks_rejected_for_lateness': tasks_rejected_for_lateness,
                'max_lateness_minutes_setting': self.max_lateness_minutes,
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
                'filtered_by_direction': compat_stats.get('filtered_by_direction', 0),
                'filtered_by_other': compat_stats.get('filtered_by_other', 0)
            },
            'agent_routes': agent_routes,
            'unassigned_tasks': unassigned_tasks
        }
    
    def _empty_result(self, reason: str, solve_time: float = 0) -> Dict:
        """Return empty result with reason"""
        # Build agent details for each unassigned task
        # IMPORTANT: Add capacity info so dashboard knows WHY compatible agents can't be assigned
        unassigned_with_details = []
        for t in self.unassigned_tasks:
            agent_details = []
            compatible_count = 0
            compatible_with_capacity_count = 0
            
            for agent in self.agents:
                if hasattr(self, 'compatibility_matrix') and t.id in self.compatibility_matrix.get(agent.id, {}):
                    compatible, agent_reason = self.compatibility_matrix[agent.id][t.id]
                    
                    # Add capacity information for compatible agents
                    has_capacity = agent.available_capacity > 0
                    
                    if compatible:
                        compatible_count += 1
                        if has_capacity:
                            compatible_with_capacity_count += 1
                            # Compatible AND has capacity - the ROUTING_FAIL is due to time constraints
                            final_reason = "time_window_conflict"
                        else:
                            # Compatible but NO capacity - this is the real reason
                            final_reason = "at_max_capacity"
                    else:
                        final_reason = agent_reason
                    
                    agent_details.append({
                        'agent_id': agent.id,
                        'agent_name': agent.name,
                        'compatible': compatible,
                        'reason': final_reason,
                        'available_capacity': agent.available_capacity,
                        'can_be_assigned': compatible and has_capacity
                    })
                else:
                    agent_details.append({
                        'agent_id': agent.id,
                        'agent_name': agent.name,
                        'compatible': False,
                        'reason': reason,
                        'available_capacity': agent.available_capacity,
                        'can_be_assigned': False
                    })
            
            # Determine the real reason for unassignment
            if compatible_count == 0:
                task_reason = "no_compatible_agents"
                reason_detail = "No agents pass compatibility rules"
            elif compatible_with_capacity_count == 0:
                task_reason = "compatible_agents_at_max_capacity"
                reason_detail = f"All {compatible_count} compatible agent(s) at max_tasks limit"
            else:
                task_reason = f"infeasible_{reason}"
                reason_detail = f"{compatible_with_capacity_count} compatible agent(s) with capacity but time windows cannot be met"
            
            unassigned_with_details.append({
                'task_id': t.id, 
                'reason': task_reason,
                'reason_detail': reason_detail,
                'restaurant_name': t.meta.get('restaurant_name', ''),
                'customer_name': t.meta.get('customer_name', ''),
                'agents_considered': agent_details,
                'compatible_agents': compatible_count,
                'compatible_with_capacity': compatible_with_capacity_count
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

def optimize_fleet(agents_data: Dict, tasks_data: Dict, prefilter_distance: bool = True) -> Dict:
    """
    Main entry point for fleet optimization.
    
    Args:
        agents_data: Response from /api/test/or-tools/agents endpoint
        tasks_data: Response from /api/test/or-tools/unassigned-tasks endpoint
        prefilter_distance: Whether to pre-filter by distance (True for fast proximity,
                           False for thorough event-based where solver handles distance)
    
    Returns:
        Optimization results with routes for each agent
    """
    start_time = time.time()
    
    mode = "PROXIMITY (fast pre-filter)" if prefilter_distance else "EVENT-BASED (max_distance enforced)"
    print(f"[optimize_fleet] Mode: {mode}")
    
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
    max_lateness_minutes = settings.get('maxLatenessMinutes', DEFAULT_MAX_LATENESS_MINUTES)  # Max allowed delivery lateness
    max_pickup_delay_minutes = settings.get('maxPickupDelayMinutes', DEFAULT_MAX_PICKUP_DELAY_MINUTES)  # Max delay after food ready
    
    print(f"[optimize_fleet] Settings: wallet_threshold={wallet_threshold}, max_distance_km={max_distance_km}, "
          f"max_lateness={max_lateness_minutes}min, max_pickup_delay={max_pickup_delay_minutes}min")
    
    # Build compatibility checker
    compatibility_checker = CompatibilityChecker(
        wallet_threshold=wallet_threshold,
        geofence_regions=geofences,
        max_distance_km=max_distance_km,
        prefilter_distance=prefilter_distance,
        max_lateness_minutes=max_lateness_minutes,
        max_pickup_delay_minutes=max_pickup_delay_minutes
    )
    
    # Run optimizer
    optimizer = FleetOptimizer(
        agents=agents,
        unassigned_tasks=tasks,
        compatibility_checker=compatibility_checker,
        prefilter_distance=prefilter_distance
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

