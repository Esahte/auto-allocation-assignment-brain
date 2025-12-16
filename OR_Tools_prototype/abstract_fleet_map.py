"""
Abstract Fleet Map - Real-time in-memory representation of fleet state.

This module maintains a live view of all agents and tasks, enabling:
- Proximity-based optimization triggers (when agent enters task radius)
- Eligibility pre-checking before optimization
- Targeted/incremental optimization for specific agents
- Smart decision-making about when to optimize

Author: Auto-generated for fleet optimization
"""

import math
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable
from datetime import datetime, timezone
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Location:
    """Geographic location with lat/lng coordinates."""
    lat: float
    lng: float
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.lat, self.lng)
    
    def __hash__(self):
        return hash((round(self.lat, 6), round(self.lng, 6)))


@dataclass
class AgentTask:
    """A task currently assigned to an agent."""
    task_id: str
    pickup_location: Location
    delivery_location: Location
    pickup_completed: bool = False
    pickup_before: Optional[datetime] = None
    delivery_before: Optional[datetime] = None
    meta: Dict = field(default_factory=dict)


@dataclass
class AgentState:
    """Real-time state of an agent in the fleet."""
    agent_id: str
    name: str
    location: Location
    status: str = "online"  # online, offline, busy
    current_tasks: List[AgentTask] = field(default_factory=list)
    max_capacity: int = 2
    tags: List[str] = field(default_factory=list)  # cash, geofence tags
    declined_tasks: Set[str] = field(default_factory=set)
    last_location_update: float = field(default_factory=time.time)
    
    @property
    def capacity(self) -> int:
        """Available capacity for new tasks."""
        return max(0, self.max_capacity - len(self.current_tasks))
    
    @property
    def is_available(self) -> bool:
        """Whether agent can accept new tasks."""
        return self.status == "online" and self.capacity > 0
    
    @property
    def projected_location(self) -> Location:
        """
        Where the agent will be after completing current tasks.
        If busy, return last delivery location. Otherwise, current location.
        """
        if self.current_tasks:
            return self.current_tasks[-1].delivery_location
        return self.location


@dataclass
class UnassignedTask:
    """An unassigned task waiting for assignment."""
    task_id: str
    pickup_location: Location
    delivery_location: Location
    pickup_before: Optional[datetime] = None
    delivery_before: Optional[datetime] = None
    required_tags: List[str] = field(default_factory=list)  # cash, geofence
    geofence: Optional[str] = None
    declined_by: Set[str] = field(default_factory=set)  # agent IDs who declined
    meta: Dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    # Pre-computed eligibility (updated when agents move)
    eligible_agents: Set[str] = field(default_factory=set)
    agents_in_radius: Set[str] = field(default_factory=set)


def haversine_distance_km(loc1: Location, loc2: Location) -> float:
    """Calculate the great-circle distance between two points in kilometers."""
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1 = math.radians(loc1.lat), math.radians(loc1.lng)
    lat2, lon2 = math.radians(loc2.lat), math.radians(loc2.lng)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


class AbstractFleetMap:
    """
    Real-time in-memory representation of the fleet state.
    
    Maintains live positions of all agents and tasks, enabling:
    - Proximity-based optimization triggers
    - Eligibility pre-checking
    - Targeted/incremental optimization
    
    Events emitted via callbacks:
    - on_agent_enters_radius(agent_id, task_id, distance_km)
    - on_optimization_needed(trigger_type, agent_ids, task_ids)
    """
    
    def __init__(
        self,
        assignment_radius_km: float = 3.0,
        max_distance_km: float = 5.0,
        optimization_cooldown_seconds: float = 10.0,
        emit_callback: Optional[Callable] = None
    ):
        """
        Initialize the Abstract Fleet Map.
        
        Args:
            assignment_radius_km: Distance at which agent is considered "in range" of a task
            max_distance_km: Maximum distance for agent-task compatibility
            optimization_cooldown_seconds: Minimum time between optimizations
            emit_callback: Function to emit WebSocket events
        """
        self.assignment_radius_km = assignment_radius_km
        self.max_distance_km = max_distance_km
        self.optimization_cooldown_seconds = optimization_cooldown_seconds
        self.emit_callback = emit_callback
        
        # State storage
        self.agents: Dict[str, AgentState] = {}
        self.unassigned_tasks: Dict[str, UnassignedTask] = {}
        self.in_progress_tasks: Dict[str, AgentTask] = {}
        
        # Optimization tracking
        self.last_optimization_time: float = 0
        self.pending_optimizations: List[Dict] = []
        self.optimization_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "location_updates": 0,
            "proximity_triggers": 0,
            "optimizations_triggered": 0,
            "optimizations_skipped_cooldown": 0,
            "eligibility_checks": 0,
        }
        
        logger.info(f"[FleetMap] Initialized with radius={assignment_radius_km}km, max_distance={max_distance_km}km")
    
    # =========================================================================
    # AGENT MANAGEMENT
    # =========================================================================
    
    def update_agent_location(self, agent_id: str, lat: float, lng: float, name: str = None) -> Dict:
        """
        Update an agent's location and check for proximity triggers.
        
        Returns dict with any triggered events.
        """
        self.stats["location_updates"] += 1
        new_location = Location(lat, lng)
        
        # Create or update agent
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentState(
                agent_id=agent_id,
                name=name or f"Agent {agent_id}",
                location=new_location
            )
            logger.info(f"[FleetMap] New agent added: {name or agent_id}")
        else:
            agent = self.agents[agent_id]
            old_location = agent.location
            agent.location = new_location
            agent.last_location_update = time.time()
            if name:
                agent.name = name
        
        # Check proximity to unassigned tasks
        triggers = self._check_proximity_triggers(agent_id)
        
        return {
            "agent_id": agent_id,
            "location": {"lat": lat, "lng": lng},
            "triggers": triggers
        }
    
    def set_agent_status(self, agent_id: str, status: str, name: str = None):
        """Set agent online/offline status."""
        if agent_id in self.agents:
            self.agents[agent_id].status = status
            logger.info(f"[FleetMap] Agent {self.agents[agent_id].name} is now {status}")
        elif status == "online":
            # Create new agent if coming online
            self.agents[agent_id] = AgentState(
                agent_id=agent_id,
                name=name or f"Agent {agent_id}",
                location=Location(0, 0),  # Will be updated with location event
                status=status
            )
            logger.info(f"[FleetMap] New agent online: {name or agent_id}")
    
    def update_agent_tasks(self, agent_id: str, tasks: List[Dict]):
        """Update an agent's current tasks."""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        agent.current_tasks = []
        
        for task_data in tasks:
            pickup_loc = task_data.get('restaurant_location', task_data.get('pickup_location', [0, 0]))
            delivery_loc = task_data.get('delivery_location', [0, 0])
            
            agent_task = AgentTask(
                task_id=task_data.get('id', task_data.get('task_id', '')),
                pickup_location=Location(pickup_loc[0], pickup_loc[1]),
                delivery_location=Location(delivery_loc[0], delivery_loc[1]),
                pickup_completed=task_data.get('pickup_completed', False),
                meta=task_data.get('_meta', {})
            )
            agent.current_tasks.append(agent_task)
            
            # Also track in in_progress_tasks
            self.in_progress_tasks[agent_task.task_id] = agent_task
        
        logger.debug(f"[FleetMap] Agent {agent.name} has {len(agent.current_tasks)} tasks")
    
    def add_agent_declined_task(self, agent_id: str, task_id: str):
        """Record that an agent declined a task."""
        if agent_id in self.agents:
            self.agents[agent_id].declined_tasks.add(task_id)
            logger.info(f"[FleetMap] Agent {self.agents[agent_id].name} declined task {task_id[:20]}...")
        
        # Also update task's declined_by set
        if task_id in self.unassigned_tasks:
            self.unassigned_tasks[task_id].declined_by.add(agent_id)
            # Recalculate eligible agents
            self._update_task_eligibility(task_id)
    
    # =========================================================================
    # TASK MANAGEMENT
    # =========================================================================
    
    def add_unassigned_task(self, task_data: Dict) -> Dict:
        """
        Add a new unassigned task and check for immediate assignment opportunities.
        
        Returns dict with eligible agents and any triggered optimizations.
        """
        task_id = task_data.get('id', task_data.get('task_id', ''))
        pickup_loc = task_data.get('restaurant_location', task_data.get('pickup_location', [0, 0]))
        delivery_loc = task_data.get('delivery_location', [0, 0])
        
        task = UnassignedTask(
            task_id=task_id,
            pickup_location=Location(pickup_loc[0], pickup_loc[1]),
            delivery_location=Location(delivery_loc[0], delivery_loc[1]),
            required_tags=task_data.get('required_tags', []),
            geofence=task_data.get('geofence'),
            meta=task_data.get('_meta', task_data.get('meta', {}))
        )
        
        # Parse time constraints
        if 'pickup_before' in task_data:
            try:
                task.pickup_before = datetime.fromisoformat(task_data['pickup_before'].replace('Z', '+00:00'))
            except:
                pass
        if 'delivery_before' in task_data:
            try:
                task.delivery_before = datetime.fromisoformat(task_data['delivery_before'].replace('Z', '+00:00'))
            except:
                pass
        
        self.unassigned_tasks[task_id] = task
        
        # Calculate initial eligibility
        eligible = self._calculate_eligible_agents(task)
        task.eligible_agents = eligible['eligible']
        task.agents_in_radius = eligible['in_radius']
        
        logger.info(f"[FleetMap] New task {task_id[:20]}... - {len(task.eligible_agents)} eligible, {len(task.agents_in_radius)} in radius")
        
        result = {
            "task_id": task_id,
            "eligible_agents": list(task.eligible_agents),
            "agents_in_radius": list(task.agents_in_radius),
            "should_optimize": len(task.agents_in_radius) > 0
        }
        
        # If agents are already in radius, trigger optimization
        if task.agents_in_radius:
            self._queue_optimization("task_added_agents_in_radius", list(task.agents_in_radius), [task_id])
        
        return result
    
    def remove_task(self, task_id: str, reason: str = "completed"):
        """Remove a task (completed, cancelled, or assigned)."""
        if task_id in self.unassigned_tasks:
            del self.unassigned_tasks[task_id]
            logger.info(f"[FleetMap] Task {task_id[:20]}... removed ({reason})")
        
        if task_id in self.in_progress_tasks:
            del self.in_progress_tasks[task_id]
    
    def task_assigned(self, task_id: str, agent_id: str):
        """Mark a task as assigned to an agent."""
        if task_id in self.unassigned_tasks:
            task = self.unassigned_tasks[task_id]
            # Move to in_progress
            agent_task = AgentTask(
                task_id=task_id,
                pickup_location=task.pickup_location,
                delivery_location=task.delivery_location,
                pickup_before=task.pickup_before,
                delivery_before=task.delivery_before,
                meta=task.meta
            )
            self.in_progress_tasks[task_id] = agent_task
            del self.unassigned_tasks[task_id]
            
            # Add to agent's tasks
            if agent_id in self.agents:
                self.agents[agent_id].current_tasks.append(agent_task)
            
            logger.info(f"[FleetMap] Task {task_id[:20]}... assigned to {self.agents.get(agent_id, {}).name if agent_id in self.agents else agent_id}")
    
    # =========================================================================
    # PROXIMITY & ELIGIBILITY CHECKING
    # =========================================================================
    
    def _check_proximity_triggers(self, agent_id: str) -> List[Dict]:
        """
        Check if an agent has entered the radius of any unassigned tasks.
        Returns list of triggered task assignments to consider.
        """
        if agent_id not in self.agents:
            return []
        
        agent = self.agents[agent_id]
        if not agent.is_available:
            return []
        
        triggers = []
        tasks_to_optimize = []
        
        for task_id, task in self.unassigned_tasks.items():
            # Skip if agent declined this task
            if agent_id in task.declined_by or task_id in agent.declined_tasks:
                continue
            
            # Calculate distance from agent to task pickup
            # Use projected location if agent has tasks
            agent_location = agent.projected_location
            distance = haversine_distance_km(agent_location, task.pickup_location)
            
            was_in_radius = agent_id in task.agents_in_radius
            is_in_radius = distance <= self.assignment_radius_km
            is_eligible = distance <= self.max_distance_km
            
            # Update task's tracking sets
            if is_eligible and agent_id not in task.eligible_agents:
                task.eligible_agents.add(agent_id)
            elif not is_eligible and agent_id in task.eligible_agents:
                task.eligible_agents.discard(agent_id)
            
            if is_in_radius and agent_id not in task.agents_in_radius:
                task.agents_in_radius.add(agent_id)
            elif not is_in_radius and agent_id in task.agents_in_radius:
                task.agents_in_radius.discard(agent_id)
            
            # Trigger if agent just entered radius
            if is_in_radius and not was_in_radius and is_eligible:
                self.stats["proximity_triggers"] += 1
                trigger = {
                    "type": "agent_entered_radius",
                    "agent_id": agent_id,
                    "agent_name": agent.name,
                    "task_id": task_id,
                    "distance_km": round(distance, 2),
                    "restaurant": task.meta.get('restaurant_name', 'Unknown')
                }
                triggers.append(trigger)
                tasks_to_optimize.append(task_id)
                
                logger.info(f"[FleetMap] 🎯 PROXIMITY TRIGGER: {agent.name} entered radius of task {task_id[:15]}... ({distance:.2f}km)")
        
        # Queue optimization if we have triggers
        if tasks_to_optimize:
            self._queue_optimization("proximity_trigger", [agent_id], tasks_to_optimize)
        
        return triggers
    
    def _calculate_eligible_agents(self, task: UnassignedTask) -> Dict:
        """
        Calculate which agents are eligible for a task and which are in radius.
        """
        self.stats["eligibility_checks"] += 1
        
        eligible = set()
        in_radius = set()
        
        for agent_id, agent in self.agents.items():
            if not agent.is_available:
                continue
            
            # Skip if declined
            if agent_id in task.declined_by or task.task_id in agent.declined_tasks:
                continue
            
            # Check tag requirements (cash, geofence)
            if task.required_tags:
                if not all(tag in agent.tags for tag in task.required_tags):
                    continue
            
            # Calculate distance
            agent_location = agent.projected_location
            distance = haversine_distance_km(agent_location, task.pickup_location)
            
            # Check max distance
            if distance <= self.max_distance_km:
                eligible.add(agent_id)
                
                # Check assignment radius
                if distance <= self.assignment_radius_km:
                    in_radius.add(agent_id)
        
        return {
            "eligible": eligible,
            "in_radius": in_radius
        }
    
    def _update_task_eligibility(self, task_id: str):
        """Recalculate eligibility for a specific task."""
        if task_id not in self.unassigned_tasks:
            return
        
        task = self.unassigned_tasks[task_id]
        result = self._calculate_eligible_agents(task)
        task.eligible_agents = result['eligible']
        task.agents_in_radius = result['in_radius']
    
    # =========================================================================
    # OPTIMIZATION TRIGGERING
    # =========================================================================
    
    def _queue_optimization(self, trigger_type: str, agent_ids: List[str], task_ids: List[str]):
        """
        Queue an optimization request with cooldown protection.
        """
        current_time = time.time()
        time_since_last = current_time - self.last_optimization_time
        
        with self.optimization_lock:
            if time_since_last < self.optimization_cooldown_seconds:
                # Add to pending instead of triggering immediately
                self.stats["optimizations_skipped_cooldown"] += 1
                self.pending_optimizations.append({
                    "trigger_type": trigger_type,
                    "agent_ids": agent_ids,
                    "task_ids": task_ids,
                    "queued_at": current_time
                })
                logger.info(f"[FleetMap] Optimization queued (cooldown: {self.optimization_cooldown_seconds - time_since_last:.1f}s remaining)")
                return
            
            self.stats["optimizations_triggered"] += 1
            self.last_optimization_time = current_time
        
        # Emit optimization event
        self._emit_optimization_needed(trigger_type, agent_ids, task_ids)
    
    def process_pending_optimizations(self) -> bool:
        """
        Process any pending optimizations that were queued during cooldown.
        Returns True if an optimization was triggered.
        """
        current_time = time.time()
        
        with self.optimization_lock:
            if not self.pending_optimizations:
                return False
            
            time_since_last = current_time - self.last_optimization_time
            if time_since_last < self.optimization_cooldown_seconds:
                return False
            
            # Merge all pending optimizations
            all_agent_ids = set()
            all_task_ids = set()
            
            for pending in self.pending_optimizations:
                all_agent_ids.update(pending['agent_ids'])
                all_task_ids.update(pending['task_ids'])
            
            self.pending_optimizations = []
            self.last_optimization_time = current_time
            self.stats["optimizations_triggered"] += 1
        
        # Emit merged optimization
        self._emit_optimization_needed(
            "merged_pending",
            list(all_agent_ids),
            list(all_task_ids)
        )
        return True
    
    def _emit_optimization_needed(self, trigger_type: str, agent_ids: List[str], task_ids: List[str]):
        """Emit an optimization event via callback."""
        event_data = {
            "trigger_type": trigger_type,
            "agent_ids": agent_ids,
            "task_ids": task_ids,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agents": [
                {
                    "id": aid,
                    "name": self.agents[aid].name if aid in self.agents else "Unknown",
                    "location": self.agents[aid].location.to_tuple() if aid in self.agents else None,
                    "capacity": self.agents[aid].capacity if aid in self.agents else 0
                }
                for aid in agent_ids
            ],
            "tasks": [
                {
                    "id": tid,
                    "pickup_location": self.unassigned_tasks[tid].pickup_location.to_tuple() if tid in self.unassigned_tasks else None,
                    "restaurant": self.unassigned_tasks[tid].meta.get('restaurant_name', 'Unknown') if tid in self.unassigned_tasks else 'Unknown'
                }
                for tid in task_ids
            ]
        }
        
        logger.info(f"[FleetMap] 🚀 OPTIMIZATION NEEDED: {trigger_type} - {len(agent_ids)} agents, {len(task_ids)} tasks")
        
        if self.emit_callback:
            self.emit_callback('fleet:optimization_needed', event_data)
    
    # =========================================================================
    # STATE QUERIES
    # =========================================================================
    
    def get_state_snapshot(self) -> Dict:
        """Get a complete snapshot of the current fleet state."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "assignment_radius_km": self.assignment_radius_km,
                "max_distance_km": self.max_distance_km,
                "optimization_cooldown_seconds": self.optimization_cooldown_seconds
            },
            "agents": {
                aid: {
                    "id": agent.agent_id,
                    "name": agent.name,
                    "status": agent.status,
                    "location": {"lat": agent.location.lat, "lng": agent.location.lng},
                    "projected_location": {"lat": agent.projected_location.lat, "lng": agent.projected_location.lng},
                    "capacity": agent.capacity,
                    "current_tasks": len(agent.current_tasks),
                    "declined_tasks": len(agent.declined_tasks),
                    "last_update_seconds_ago": round(time.time() - agent.last_location_update, 1)
                }
                for aid, agent in self.agents.items()
            },
            "unassigned_tasks": {
                tid: {
                    "id": task.task_id,
                    "pickup_location": {"lat": task.pickup_location.lat, "lng": task.pickup_location.lng},
                    "delivery_location": {"lat": task.delivery_location.lat, "lng": task.delivery_location.lng},
                    "restaurant": task.meta.get('restaurant_name', 'Unknown'),
                    "customer": task.meta.get('customer_name', 'Unknown'),
                    "eligible_agents": list(task.eligible_agents),
                    "agents_in_radius": list(task.agents_in_radius),
                    "declined_by": list(task.declined_by),
                    "age_seconds": round(time.time() - task.created_at, 1)
                }
                for tid, task in self.unassigned_tasks.items()
            },
            "in_progress_tasks": len(self.in_progress_tasks),
            "stats": self.stats.copy(),
            "pending_optimizations": len(self.pending_optimizations)
        }
    
    def get_agents_near_task(self, task_id: str) -> List[Dict]:
        """Get all agents near a specific task with distances."""
        if task_id not in self.unassigned_tasks:
            return []
        
        task = self.unassigned_tasks[task_id]
        result = []
        
        for agent_id, agent in self.agents.items():
            if not agent.is_available:
                continue
            
            distance = haversine_distance_km(agent.projected_location, task.pickup_location)
            
            result.append({
                "agent_id": agent_id,
                "name": agent.name,
                "distance_km": round(distance, 2),
                "in_radius": distance <= self.assignment_radius_km,
                "eligible": distance <= self.max_distance_km and agent_id not in task.declined_by,
                "declined": agent_id in task.declined_by,
                "capacity": agent.capacity
            })
        
        # Sort by distance
        result.sort(key=lambda x: x['distance_km'])
        return result
    
    def get_tasks_near_agent(self, agent_id: str) -> List[Dict]:
        """Get all unassigned tasks near a specific agent with distances."""
        if agent_id not in self.agents:
            return []
        
        agent = self.agents[agent_id]
        result = []
        
        for task_id, task in self.unassigned_tasks.items():
            distance = haversine_distance_km(agent.projected_location, task.pickup_location)
            
            result.append({
                "task_id": task_id,
                "restaurant": task.meta.get('restaurant_name', 'Unknown'),
                "distance_km": round(distance, 2),
                "in_radius": distance <= self.assignment_radius_km,
                "eligible": distance <= self.max_distance_km and agent_id not in task.declined_by,
                "declined": agent_id in task.declined_by
            })
        
        # Sort by distance
        result.sort(key=lambda x: x['distance_km'])
        return result
    
    # =========================================================================
    # BULK OPERATIONS
    # =========================================================================
    
    def sync_from_dashboard(self, agents_data: List[Dict], tasks_data: List[Dict]):
        """
        Sync the fleet map state from dashboard data.
        Useful for initial load or periodic sync.
        """
        logger.info(f"[FleetMap] Syncing from dashboard: {len(agents_data)} agents, {len(tasks_data)} tasks")
        
        # Update agents
        for agent_data in agents_data:
            agent_id = str(agent_data.get('id', agent_data.get('agent_id', '')))
            location = agent_data.get('location', agent_data.get('current_location', [0, 0]))
            
            if agent_id not in self.agents:
                self.agents[agent_id] = AgentState(
                    agent_id=agent_id,
                    name=agent_data.get('name', f'Agent {agent_id}'),
                    location=Location(location[0], location[1]),
                    status="online"
                )
            else:
                self.agents[agent_id].location = Location(location[0], location[1])
                self.agents[agent_id].name = agent_data.get('name', self.agents[agent_id].name)
            
            # Update current tasks
            if 'current_tasks' in agent_data:
                self.update_agent_tasks(agent_id, agent_data['current_tasks'])
            
            # Update tags
            self.agents[agent_id].tags = agent_data.get('tags', [])
            
            # Update declined history
            declined = agent_data.get('declined_task_ids', [])
            self.agents[agent_id].declined_tasks = set(declined)
        
        # Update tasks
        current_task_ids = set()
        for task_data in tasks_data:
            task_id = task_data.get('id', task_data.get('task_id', ''))
            current_task_ids.add(task_id)
            
            if task_id not in self.unassigned_tasks:
                self.add_unassigned_task(task_data)
            else:
                # Update existing task's declined_by from task data if available
                if 'declined_by' in task_data:
                    self.unassigned_tasks[task_id].declined_by = set(task_data['declined_by'])
        
        # Remove tasks no longer in the list
        for task_id in list(self.unassigned_tasks.keys()):
            if task_id not in current_task_ids:
                self.remove_task(task_id, reason="no longer in dashboard")
        
        logger.info(f"[FleetMap] Sync complete: {len(self.agents)} agents, {len(self.unassigned_tasks)} unassigned tasks")
    
    def clear(self):
        """Clear all state."""
        self.agents.clear()
        self.unassigned_tasks.clear()
        self.in_progress_tasks.clear()
        self.pending_optimizations.clear()
        logger.info("[FleetMap] State cleared")


# Singleton instance for use across the application
_fleet_map_instance: Optional[AbstractFleetMap] = None


def get_fleet_map(
    assignment_radius_km: float = 3.0,
    max_distance_km: float = 5.0,
    emit_callback: Optional[Callable] = None
) -> AbstractFleetMap:
    """Get or create the singleton fleet map instance."""
    global _fleet_map_instance
    
    if _fleet_map_instance is None:
        _fleet_map_instance = AbstractFleetMap(
            assignment_radius_km=assignment_radius_km,
            max_distance_km=max_distance_km,
            emit_callback=emit_callback
        )
    elif emit_callback and _fleet_map_instance.emit_callback is None:
        _fleet_map_instance.emit_callback = emit_callback
    
    return _fleet_map_instance


def reset_fleet_map():
    """Reset the singleton instance (useful for testing)."""
    global _fleet_map_instance
    if _fleet_map_instance:
        _fleet_map_instance.clear()
    _fleet_map_instance = None

