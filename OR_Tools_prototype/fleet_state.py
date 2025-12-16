"""
Fleet State - Abstract Map for Real-Time Fleet Management

This module maintains an in-memory representation of the entire fleet:
- All agents with their real-time locations and projected locations
- All tasks (unassigned, in-progress, completed)
- Proximity monitoring for smart optimization triggers
- Eligibility caching for efficient decision making

The abstract map enables:
1. Radius-based assignment triggers (only optimize when agent is near task)
2. Proactive chaining (assign next task before agent finishes current)
3. Incremental optimization (optimize for one agent, not entire fleet)
4. Smart eligibility checks (pre-filter before expensive optimization)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timezone
from enum import Enum
import threading
import math
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"      # Has tasks but available for more
    AT_CAPACITY = "at_capacity"  # Max tasks reached


class TaskStatus(Enum):
    UNASSIGNED = "unassigned"
    ASSIGNED = "assigned"
    PICKUP_IN_PROGRESS = "pickup_in_progress"
    DELIVERY_IN_PROGRESS = "delivery_in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class Location:
    lat: float
    lng: float
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.lat, self.lng)
    
    def distance_to(self, other: 'Location') -> float:
        """Calculate Haversine distance in km"""
        R = 6371  # Earth's radius in km
        lat1, lon1 = math.radians(self.lat), math.radians(self.lng)
        lat2, lon2 = math.radians(other.lat), math.radians(other.lng)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def __str__(self):
        return f"({self.lat:.6f}, {self.lng:.6f})"


@dataclass
class TaskState:
    id: str
    job_type: str  # "PAIRED", "PICKUP", "DELIVERY"
    restaurant_location: Location
    delivery_location: Location
    pickup_before: datetime
    delivery_before: datetime
    status: TaskStatus = TaskStatus.UNASSIGNED
    assigned_agent_id: Optional[str] = None
    pickup_completed: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def restaurant_name(self) -> str:
        return self.meta.get('restaurant_name', 'Unknown Restaurant')
    
    @property
    def customer_name(self) -> str:
        return self.meta.get('customer_name', 'Unknown Customer')
    
    @property
    def is_urgent(self) -> bool:
        """Check if task pickup time is within 15 minutes"""
        now = datetime.now(timezone.utc)
        time_until_pickup = (self.pickup_before - now).total_seconds()
        return time_until_pickup <= 900  # 15 minutes
    
    @property
    def is_overdue(self) -> bool:
        """Check if task pickup time has passed"""
        now = datetime.now(timezone.utc)
        return now > self.pickup_before
    
    def urgency_score(self) -> float:
        """Higher score = more urgent (0-100)"""
        now = datetime.now(timezone.utc)
        time_until_pickup = (self.pickup_before - now).total_seconds()
        
        if time_until_pickup <= 0:
            return 100  # Overdue
        elif time_until_pickup <= 300:  # 5 min
            return 90
        elif time_until_pickup <= 600:  # 10 min
            return 70
        elif time_until_pickup <= 900:  # 15 min
            return 50
        elif time_until_pickup <= 1800:  # 30 min
            return 30
        else:
            return 10


@dataclass
class CurrentTask:
    """Represents a task currently assigned to an agent"""
    id: str
    job_type: str
    restaurant_location: Location
    delivery_location: Location
    pickup_before: datetime
    delivery_before: datetime
    pickup_completed: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    id: str
    name: str
    current_location: Location
    status: AgentStatus = AgentStatus.ONLINE
    current_tasks: List[CurrentTask] = field(default_factory=list)
    max_capacity: int = 2  # Max tasks an agent can handle
    declined_task_ids: Set[str] = field(default_factory=set)
    cash_tags: List[str] = field(default_factory=list)
    last_location_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def is_online(self) -> bool:
        return self.status != AgentStatus.OFFLINE
    
    @property
    def is_idle(self) -> bool:
        return self.is_online and len(self.current_tasks) == 0
    
    @property
    def has_capacity(self) -> bool:
        return self.is_online and len(self.current_tasks) < self.max_capacity
    
    @property
    def available_capacity(self) -> int:
        return max(0, self.max_capacity - len(self.current_tasks))
    
    @property
    def projected_location(self) -> Location:
        """
        Where will this agent be after completing current tasks?
        - If no tasks: current location
        - If has tasks: last delivery location
        """
        if not self.current_tasks:
            return self.current_location
        
        # Find the last task in the chain
        last_task = self.current_tasks[-1]
        return last_task.delivery_location
    
    @property
    def next_destination(self) -> Optional[Location]:
        """
        Where is the agent heading next?
        - If no tasks: None
        - If has task with pickup not done: restaurant location
        - If pickup done: delivery location
        """
        if not self.current_tasks:
            return None
        
        first_task = self.current_tasks[0]
        if first_task.pickup_completed:
            return first_task.delivery_location
        else:
            return first_task.restaurant_location
    
    def has_declined(self, task_id: str) -> bool:
        return task_id in self.declined_task_ids
    
    def estimated_time_to_idle(self) -> Optional[float]:
        """
        Estimate seconds until agent is idle (finishes all tasks).
        Returns None if can't estimate.
        """
        if not self.current_tasks:
            return 0
        
        # Simple estimate: sum of delivery deadlines minus now
        # This is a rough approximation
        now = datetime.now(timezone.utc)
        last_delivery = self.current_tasks[-1].delivery_before
        return max(0, (last_delivery - now).total_seconds())


class ProximityTrigger:
    """Represents a condition where an agent is close enough to a task for assignment"""
    def __init__(
        self,
        agent: AgentState,
        task: TaskState,
        distance_km: float,
        trigger_type: str,  # "current_location", "projected_location"
        eligibility_reason: Optional[str] = None
    ):
        self.agent = agent
        self.task = task
        self.distance_km = distance_km
        self.trigger_type = trigger_type
        self.eligibility_reason = eligibility_reason
        self.is_eligible = eligibility_reason is None
        self.timestamp = datetime.now(timezone.utc)
    
    def __repr__(self):
        status = "✓ ELIGIBLE" if self.is_eligible else f"✗ {self.eligibility_reason}"
        return f"ProximityTrigger({self.agent.name} → {self.task.restaurant_name}, {self.distance_km:.2f}km, {status})"


class FleetState:
    """
    The Abstract Map - maintains real-time state of the entire fleet.
    Thread-safe for concurrent WebSocket updates.
    """
    
    def __init__(
        self,
        assignment_radius_km: float = 3.0,
        chain_lookahead_radius_km: float = 5.0,
        max_distance_km: float = 3.0,
        optimization_cooldown_seconds: float = 30.0
    ):
        """
        Initialize the fleet state.
        
        Args:
            assignment_radius_km: Distance threshold for triggering assignment optimization
            chain_lookahead_radius_km: Distance to look for chain opportunities
            max_distance_km: Maximum distance for agent-task compatibility
            optimization_cooldown_seconds: Minimum time between optimizations per agent
        """
        self.assignment_radius_km = assignment_radius_km
        self.chain_lookahead_radius_km = chain_lookahead_radius_km
        self.max_distance_km = max_distance_km
        self.optimization_cooldown_seconds = optimization_cooldown_seconds
        
        # State storage
        self._agents: Dict[str, AgentState] = {}
        self._tasks: Dict[str, TaskState] = {}  # All tasks (unassigned, in-progress)
        
        # Optimization tracking
        self._last_optimization_time: Dict[str, float] = {}  # agent_id -> timestamp
        self._pending_triggers: List[ProximityTrigger] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'location_updates': 0,
            'task_updates': 0,
            'proximity_checks': 0,
            'triggers_generated': 0,
            'optimizations_triggered': 0,
            'optimizations_skipped_cooldown': 0
        }
        
        logger.info(f"[FleetState] Initialized with radius={assignment_radius_km}km, "
                   f"max_distance={max_distance_km}km, cooldown={optimization_cooldown_seconds}s")
    
    # =========================================================================
    # AGENT STATE MANAGEMENT
    # =========================================================================
    
    def update_agent_location(
        self,
        agent_id: str,
        lat: float,
        lng: float,
        name: Optional[str] = None
    ) -> List[ProximityTrigger]:
        """
        Update agent's current location and check for proximity triggers.
        
        Returns list of triggers if agent is now within assignment radius of any task.
        """
        with self._lock:
            self._stats['location_updates'] += 1
            
            location = Location(lat=lat, lng=lng)
            now = datetime.now(timezone.utc)
            
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                agent.current_location = location
                agent.last_location_update = now
                agent.last_updated = now
                if name:
                    agent.name = name
            else:
                # New agent - create state
                agent = AgentState(
                    id=agent_id,
                    name=name or f"Agent {agent_id}",
                    current_location=location,
                    status=AgentStatus.ONLINE,
                    last_location_update=now
                )
                self._agents[agent_id] = agent
                logger.info(f"[FleetState] New agent added: {agent.name} ({agent_id})")
            
            # Check proximity triggers
            triggers = self._check_proximity_triggers(agent)
            
            if triggers:
                logger.info(f"[FleetState] {len(triggers)} proximity triggers for {agent.name}")
                for t in triggers:
                    logger.debug(f"  → {t}")
            
            return triggers
    
    def set_agent_online(
        self,
        agent_id: str,
        name: Optional[str] = None,
        location: Optional[Tuple[float, float]] = None
    ) -> Optional[AgentState]:
        """Mark agent as online"""
        with self._lock:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                agent.status = AgentStatus.ONLINE
                if name:
                    agent.name = name
                if location:
                    agent.current_location = Location(lat=location[0], lng=location[1])
                agent.last_updated = datetime.now(timezone.utc)
                logger.info(f"[FleetState] Agent online: {agent.name}")
                return agent
            elif location:
                # Create new agent
                agent = AgentState(
                    id=agent_id,
                    name=name or f"Agent {agent_id}",
                    current_location=Location(lat=location[0], lng=location[1]),
                    status=AgentStatus.ONLINE
                )
                self._agents[agent_id] = agent
                logger.info(f"[FleetState] New agent online: {agent.name}")
                return agent
            return None
    
    def set_agent_offline(self, agent_id: str) -> Optional[AgentState]:
        """Mark agent as offline"""
        with self._lock:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                agent.status = AgentStatus.OFFLINE
                agent.last_updated = datetime.now(timezone.utc)
                logger.info(f"[FleetState] Agent offline: {agent.name}")
                return agent
            return None
    
    def update_agent_tasks(
        self,
        agent_id: str,
        tasks: List[Dict[str, Any]]
    ) -> Optional[AgentState]:
        """Update agent's current task list"""
        with self._lock:
            if agent_id not in self._agents:
                return None
            
            agent = self._agents[agent_id]
            agent.current_tasks = []
            
            for t in tasks:
                # Parse locations
                restaurant_loc = t.get('restaurant_location', [0, 0])
                delivery_loc = t.get('delivery_location', [0, 0])
                
                # Parse times
                pickup_before = t.get('pickup_before')
                delivery_before = t.get('delivery_before')
                
                if isinstance(pickup_before, str):
                    pickup_before = datetime.fromisoformat(pickup_before.replace('Z', '+00:00'))
                if isinstance(delivery_before, str):
                    delivery_before = datetime.fromisoformat(delivery_before.replace('Z', '+00:00'))
                
                current_task = CurrentTask(
                    id=t.get('id', ''),
                    job_type=t.get('job_type', 'PAIRED'),
                    restaurant_location=Location(lat=restaurant_loc[0], lng=restaurant_loc[1]),
                    delivery_location=Location(lat=delivery_loc[0], lng=delivery_loc[1]),
                    pickup_before=pickup_before or datetime.now(timezone.utc),
                    delivery_before=delivery_before or datetime.now(timezone.utc),
                    pickup_completed=t.get('pickup_completed', False),
                    meta=t.get('_meta', {})
                )
                agent.current_tasks.append(current_task)
            
            # Update status based on capacity
            if len(agent.current_tasks) >= agent.max_capacity:
                agent.status = AgentStatus.AT_CAPACITY
            elif len(agent.current_tasks) > 0:
                agent.status = AgentStatus.BUSY
            elif agent.status != AgentStatus.OFFLINE:
                agent.status = AgentStatus.ONLINE
            
            agent.last_updated = datetime.now(timezone.utc)
            self._stats['task_updates'] += 1
            
            return agent
    
    def add_declined_task(self, agent_id: str, task_id: str):
        """Record that an agent declined a task"""
        with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].declined_task_ids.add(task_id)
                logger.info(f"[FleetState] Agent {self._agents[agent_id].name} declined task {task_id[:20]}...")
    
    def get_agent(self, agent_id: str) -> Optional[AgentState]:
        """Get agent state by ID"""
        with self._lock:
            return self._agents.get(agent_id)
    
    def get_all_agents(self) -> List[AgentState]:
        """Get all agents"""
        with self._lock:
            return list(self._agents.values())
    
    def get_online_agents(self) -> List[AgentState]:
        """Get all online agents"""
        with self._lock:
            return [a for a in self._agents.values() if a.is_online]
    
    def get_available_agents(self) -> List[AgentState]:
        """Get agents with capacity for more tasks"""
        with self._lock:
            return [a for a in self._agents.values() if a.has_capacity]
    
    # =========================================================================
    # TASK STATE MANAGEMENT
    # =========================================================================
    
    def add_task(self, task_data: Dict[str, Any]) -> TaskState:
        """Add or update a task"""
        with self._lock:
            task_id = task_data.get('id', '')
            
            # Parse locations
            restaurant_loc = task_data.get('restaurant_location', [0, 0])
            delivery_loc = task_data.get('delivery_location', [0, 0])
            
            # Parse times
            pickup_before = task_data.get('pickup_before')
            delivery_before = task_data.get('delivery_before')
            
            if isinstance(pickup_before, str):
                pickup_before = datetime.fromisoformat(pickup_before.replace('Z', '+00:00'))
            if isinstance(delivery_before, str):
                delivery_before = datetime.fromisoformat(delivery_before.replace('Z', '+00:00'))
            
            # Determine status
            assigned_agent = task_data.get('assigned_driver') or task_data.get('assigned_agent')
            status = TaskStatus.ASSIGNED if assigned_agent else TaskStatus.UNASSIGNED
            
            if task_id in self._tasks:
                # Update existing
                task = self._tasks[task_id]
                task.restaurant_location = Location(lat=restaurant_loc[0], lng=restaurant_loc[1])
                task.delivery_location = Location(lat=delivery_loc[0], lng=delivery_loc[1])
                task.pickup_before = pickup_before or task.pickup_before
                task.delivery_before = delivery_before or task.delivery_before
                task.assigned_agent_id = assigned_agent
                task.status = status
                task.pickup_completed = task_data.get('pickup_completed', False)
                task.meta = task_data.get('_meta', task.meta)
                task.last_updated = datetime.now(timezone.utc)
            else:
                # Create new
                task = TaskState(
                    id=task_id,
                    job_type=task_data.get('job_type', 'PAIRED'),
                    restaurant_location=Location(lat=restaurant_loc[0], lng=restaurant_loc[1]),
                    delivery_location=Location(lat=delivery_loc[0], lng=delivery_loc[1]),
                    pickup_before=pickup_before or datetime.now(timezone.utc),
                    delivery_before=delivery_before or datetime.now(timezone.utc),
                    status=status,
                    assigned_agent_id=assigned_agent,
                    pickup_completed=task_data.get('pickup_completed', False),
                    meta=task_data.get('_meta', {})
                )
                self._tasks[task_id] = task
                logger.info(f"[FleetState] New task added: {task.restaurant_name} → {task.customer_name}")
            
            self._stats['task_updates'] += 1
            return task
    
    def update_task_status(self, task_id: str, status: TaskStatus) -> Optional[TaskState]:
        """Update task status"""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = status
                task.last_updated = datetime.now(timezone.utc)
                return task
            return None
    
    def complete_task(self, task_id: str) -> Optional[TaskState]:
        """Mark task as completed and remove from active tasks"""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus.COMPLETED
                task.last_updated = datetime.now(timezone.utc)
                
                # Remove from agent's current tasks
                if task.assigned_agent_id and task.assigned_agent_id in self._agents:
                    agent = self._agents[task.assigned_agent_id]
                    agent.current_tasks = [t for t in agent.current_tasks if t.id != task_id]
                    
                    # Update agent status
                    if len(agent.current_tasks) == 0 and agent.status != AgentStatus.OFFLINE:
                        agent.status = AgentStatus.ONLINE
                    elif len(agent.current_tasks) < agent.max_capacity:
                        agent.status = AgentStatus.BUSY
                
                logger.info(f"[FleetState] Task completed: {task_id[:20]}...")
                return task
            return None
    
    def cancel_task(self, task_id: str) -> Optional[TaskState]:
        """Mark task as cancelled"""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus.CANCELLED
                task.last_updated = datetime.now(timezone.utc)
                
                # Remove from agent's current tasks if assigned
                if task.assigned_agent_id and task.assigned_agent_id in self._agents:
                    agent = self._agents[task.assigned_agent_id]
                    agent.current_tasks = [t for t in agent.current_tasks if t.id != task_id]
                
                logger.info(f"[FleetState] Task cancelled: {task_id[:20]}...")
                return task
            return None
    
    def get_task(self, task_id: str) -> Optional[TaskState]:
        """Get task by ID"""
        with self._lock:
            return self._tasks.get(task_id)
    
    def get_unassigned_tasks(self) -> List[TaskState]:
        """Get all unassigned tasks"""
        with self._lock:
            return [t for t in self._tasks.values() if t.status == TaskStatus.UNASSIGNED]
    
    def get_active_tasks(self) -> List[TaskState]:
        """Get all active (not completed/cancelled) tasks"""
        with self._lock:
            return [t for t in self._tasks.values() 
                    if t.status not in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)]
    
    # =========================================================================
    # PROXIMITY AND ELIGIBILITY CHECKING
    # =========================================================================
    
    def _check_proximity_triggers(self, agent: AgentState) -> List[ProximityTrigger]:
        """
        Check if agent is within assignment radius of any unassigned task.
        Returns list of proximity triggers.
        """
        triggers = []
        
        if not agent.has_capacity:
            return triggers
        
        unassigned_tasks = self.get_unassigned_tasks()
        self._stats['proximity_checks'] += len(unassigned_tasks)
        
        for task in unassigned_tasks:
            # Check current location
            current_distance = agent.current_location.distance_to(task.restaurant_location)
            
            if current_distance <= self.assignment_radius_km:
                eligibility = self._check_eligibility(agent, task)
                trigger = ProximityTrigger(
                    agent=agent,
                    task=task,
                    distance_km=current_distance,
                    trigger_type="current_location",
                    eligibility_reason=eligibility
                )
                triggers.append(trigger)
                self._stats['triggers_generated'] += 1
            
            # Check projected location (for proactive chaining)
            if agent.current_tasks:  # Only if agent is busy
                projected_distance = agent.projected_location.distance_to(task.restaurant_location)
                
                if projected_distance <= self.chain_lookahead_radius_km:
                    eligibility = self._check_eligibility(agent, task)
                    trigger = ProximityTrigger(
                        agent=agent,
                        task=task,
                        distance_km=projected_distance,
                        trigger_type="projected_location",
                        eligibility_reason=eligibility
                    )
                    triggers.append(trigger)
                    self._stats['triggers_generated'] += 1
        
        return triggers
    
    def _check_eligibility(self, agent: AgentState, task: TaskState) -> Optional[str]:
        """
        Check if agent is eligible to take task.
        Returns None if eligible, or reason string if not.
        """
        # Check declined
        if agent.has_declined(task.id):
            return "declined"
        
        # Check capacity
        if not agent.has_capacity:
            return "at_capacity"
        
        # Check online
        if not agent.is_online:
            return "offline"
        
        # Check max distance (using current or projected location)
        if agent.current_tasks:
            # Use projected location for busy agents
            distance = agent.projected_location.distance_to(task.restaurant_location)
        else:
            distance = agent.current_location.distance_to(task.restaurant_location)
        
        if distance > self.max_distance_km:
            return f"too_far ({distance:.1f}km > {self.max_distance_km}km)"
        
        # TODO: Add cash tag checking
        # TODO: Add geofence checking
        
        return None  # Eligible!
    
    def find_eligible_agents_for_task(self, task_id: str) -> List[Tuple[AgentState, float, Optional[str]]]:
        """
        Find all agents that could potentially take this task.
        Returns list of (agent, distance_km, eligibility_reason) tuples.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return []
            
            results = []
            for agent in self._agents.values():
                if not agent.is_online:
                    continue
                
                # Calculate distance from effective location
                if agent.current_tasks:
                    distance = agent.projected_location.distance_to(task.restaurant_location)
                else:
                    distance = agent.current_location.distance_to(task.restaurant_location)
                
                eligibility = self._check_eligibility(agent, task)
                results.append((agent, distance, eligibility))
            
            # Sort by distance (closest first)
            results.sort(key=lambda x: x[1])
            return results
    
    def find_tasks_near_agent(
        self,
        agent_id: str,
        radius_km: Optional[float] = None,
        use_projected: bool = False
    ) -> List[Tuple[TaskState, float]]:
        """
        Find unassigned tasks near an agent.
        Returns list of (task, distance_km) tuples.
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return []
            
            radius = radius_km or self.assignment_radius_km
            location = agent.projected_location if use_projected else agent.current_location
            
            results = []
            for task in self.get_unassigned_tasks():
                distance = location.distance_to(task.restaurant_location)
                if distance <= radius:
                    results.append((task, distance))
            
            # Sort by distance
            results.sort(key=lambda x: x[1])
            return results
    
    # =========================================================================
    # OPTIMIZATION CONTROL
    # =========================================================================
    
    def should_trigger_optimization(self, agent_id: str) -> bool:
        """
        Check if we should trigger optimization for this agent.
        Implements cooldown to prevent over-optimization.
        """
        with self._lock:
            now = time.time()
            last_opt = self._last_optimization_time.get(agent_id, 0)
            
            if now - last_opt < self.optimization_cooldown_seconds:
                self._stats['optimizations_skipped_cooldown'] += 1
                return False
            
            return True
    
    def record_optimization(self, agent_id: str):
        """Record that optimization was triggered for this agent"""
        with self._lock:
            self._last_optimization_time[agent_id] = time.time()
            self._stats['optimizations_triggered'] += 1
    
    def get_optimization_candidates(self) -> List[ProximityTrigger]:
        """
        Get list of triggers that should result in optimization.
        Filters by eligibility and cooldown.
        """
        with self._lock:
            candidates = []
            
            for agent in self.get_available_agents():
                if not self.should_trigger_optimization(agent.id):
                    continue
                
                triggers = self._check_proximity_triggers(agent)
                eligible_triggers = [t for t in triggers if t.is_eligible]
                
                if eligible_triggers:
                    # Take the best trigger (closest task)
                    best = min(eligible_triggers, key=lambda t: t.distance_km)
                    candidates.append(best)
            
            return candidates
    
    # =========================================================================
    # BULK OPERATIONS (for syncing with dashboard)
    # =========================================================================
    
    def sync_agents(self, agents_data: List[Dict[str, Any]]):
        """Sync all agents from dashboard data"""
        with self._lock:
            for agent_data in agents_data:
                agent_id = str(agent_data.get('id', ''))
                location = agent_data.get('location', [0, 0])
                
                if agent_id in self._agents:
                    agent = self._agents[agent_id]
                    agent.current_location = Location(lat=location[0], lng=location[1])
                    agent.name = agent_data.get('name', agent.name)
                else:
                    agent = AgentState(
                        id=agent_id,
                        name=agent_data.get('name', f'Agent {agent_id}'),
                        current_location=Location(lat=location[0], lng=location[1]),
                        status=AgentStatus.ONLINE
                    )
                    self._agents[agent_id] = agent
                
                # Update tasks
                if 'current_tasks' in agent_data:
                    self.update_agent_tasks(agent_id, agent_data['current_tasks'])
                
                # Update declined
                if 'declined_task_ids' in agent_data:
                    agent.declined_task_ids = set(agent_data['declined_task_ids'])
            
            logger.info(f"[FleetState] Synced {len(agents_data)} agents")
    
    def sync_tasks(self, tasks_data: List[Dict[str, Any]]):
        """Sync all tasks from dashboard data"""
        with self._lock:
            for task_data in tasks_data:
                self.add_task(task_data)
            
            logger.info(f"[FleetState] Synced {len(tasks_data)} tasks")
    
    # =========================================================================
    # STATISTICS AND DEBUGGING
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fleet state statistics"""
        with self._lock:
            return {
                **self._stats,
                'total_agents': len(self._agents),
                'online_agents': len(self.get_online_agents()),
                'available_agents': len(self.get_available_agents()),
                'total_tasks': len(self._tasks),
                'unassigned_tasks': len(self.get_unassigned_tasks()),
                'active_tasks': len(self.get_active_tasks())
            }
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current fleet state"""
        with self._lock:
            agents_summary = []
            for agent in self._agents.values():
                agents_summary.append({
                    'id': agent.id,
                    'name': agent.name,
                    'status': agent.status.value,
                    'location': agent.current_location.to_tuple(),
                    'projected_location': agent.projected_location.to_tuple(),
                    'current_tasks': len(agent.current_tasks),
                    'capacity': f"{len(agent.current_tasks)}/{agent.max_capacity}",
                    'declined_count': len(agent.declined_task_ids)
                })
            
            tasks_summary = []
            for task in self.get_unassigned_tasks():
                tasks_summary.append({
                    'id': task.id[:20] + '...',
                    'restaurant': task.restaurant_name,
                    'customer': task.customer_name,
                    'urgency': task.urgency_score(),
                    'is_urgent': task.is_urgent,
                    'is_overdue': task.is_overdue
                })
            
            return {
                'agents': agents_summary,
                'unassigned_tasks': tasks_summary,
                'stats': self.get_stats()
            }
    
    def clear(self):
        """Clear all state (for testing)"""
        with self._lock:
            self._agents.clear()
            self._tasks.clear()
            self._last_optimization_time.clear()
            logger.info("[FleetState] State cleared")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Create a global fleet state instance
fleet_state = FleetState(
    assignment_radius_km=3.0,
    chain_lookahead_radius_km=5.0,
    max_distance_km=3.0,
    optimization_cooldown_seconds=30.0
)

