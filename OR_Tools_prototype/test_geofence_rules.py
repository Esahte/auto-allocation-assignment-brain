"""
Tests for geofence region assignment rules.

Covers:
1. PAIRED task in one region (region agents online -> restricted, distance bypassed)
2. PAIRED task with pickup in region A and delivery in region B (must use pickup region)
3. DELIVERY_ONLY task in a region (uses delivery location for region lookup)
4. No online region agents (fallback to global rules, distance applies)
5. Task in no region (fallback to global rules)
6. Agent assigned to multiple regions (eligible in any of their regions)
7. Scooter region unchanged (both pickup+delivery must be in geofence)
8. Backward compatibility: no geofences synced -> all agents eligible per normal rules
9. Defensive sync: empty geofences list does NOT clear existing geofences
10. Optimizer CompatibilityChecker non-scooter geofence rules
"""

import sys
import os
import unittest
from datetime import datetime, timezone, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fleet_state import (
    FleetState, AgentState, TaskState, Location, AgentStatus, TaskStatus, CurrentTask
)
from fleet_optimizer import (
    GeofenceRegion, CompatibilityChecker, Agent, Task, Location as OptimizerLocation, point_in_polygon
)


# ---------------------------------------------------------------------------
# Helpers – reusable locations and polygon
# ---------------------------------------------------------------------------

# A rectangular region roughly in Kingston, Jamaica
# (18.00, -76.80) -> (18.05, -76.80) -> (18.05, -76.75) -> (18.00, -76.75)
KINGSTON_POLYGON = [
    [18.00, -76.80],
    [18.05, -76.80],
    [18.05, -76.75],
    [18.00, -76.75],
]

# A second region polygon – Montego Bay area
MOBAY_POLYGON = [
    [18.45, -77.95],
    [18.50, -77.95],
    [18.50, -77.90],
    [18.45, -77.90],
]

# Scooter region polygon – small zone inside Kingston
SCOOTER_POLYGON = [
    [18.01, -76.79],
    [18.03, -76.79],
    [18.03, -76.77],
    [18.01, -76.77],
]

# Locations INSIDE Kingston region
LOC_IN_KINGSTON = Location(18.02, -76.78)
LOC_IN_KINGSTON_2 = Location(18.03, -76.77)

# Location INSIDE Montego Bay region
LOC_IN_MOBAY = Location(18.47, -77.92)

# Location INSIDE scooter region
LOC_IN_SCOOTER = Location(18.02, -76.78)

# Location OUTSIDE all regions (far away)
LOC_OUTSIDE = Location(17.90, -76.50)

# A future timestamp for pickup/delivery
FUTURE = datetime.now(timezone.utc) + timedelta(hours=2)


def _test_road_distance_provider(origins, destinations):
    """Deterministic stand-in for OSRM in eligibility unit tests."""
    return [
        [
            Location(origin.lat, origin.lng).distance_to(
                Location(destination.lat, destination.lng)
            )
            for destination in destinations
        ]
        for origin in origins
    ], "osrm"


def _make_fleet_state() -> FleetState:
    """Create a fresh FleetState with generous defaults."""
    fs = FleetState(
        assignment_radius_km=10.0,
        max_distance_km=10.0,
        road_distance_provider=_test_road_distance_provider,
    )
    return fs


def _make_agent(agent_id: str, name: str, location: Location, online: bool = True,
                priority: int = None, tags: list = None) -> AgentState:
    """Create an AgentState."""
    return AgentState(
        id=agent_id,
        name=name,
        current_location=location,
        status=AgentStatus.IDLE if online else AgentStatus.OFFLINE,
        tags=tags or [],
        priority=priority,
    )


def _make_task(task_id: str, pickup: Location, delivery: Location,
               job_type: str = "PAIRED", payment: str = "card",
               tips: float = 0.0, delivery_fee: float = 8.0) -> TaskState:
    """Create a TaskState."""
    return TaskState(
        id=task_id,
        job_type=job_type,
        restaurant_location=pickup,
        delivery_location=delivery,
        pickup_before=FUTURE,
        delivery_before=FUTURE + timedelta(hours=1),
        payment_method=payment,
        tips=tips,
        delivery_fee=delivery_fee,
    )


# ===========================================================================
# FleetState tests
# ===========================================================================

class TestGetTaskRegion(unittest.TestCase):
    """Test FleetState._get_task_region()"""

    def setUp(self):
        self.fs = _make_fleet_state()
        self.fs._geofences = {
            'region_kgn': {
                'id': 'region_kgn',
                'name': 'Kingston Downtown',
                'polygon': KINGSTON_POLYGON,
                'agent_ids': {'agent_1', 'agent_2'},
            },
            'region_mobay': {
                'id': 'region_mobay',
                'name': 'Montego Bay',
                'polygon': MOBAY_POLYGON,
                'agent_ids': {'agent_3'},
            },
            'region_scooter': {
                'id': 'region_scooter',
                'name': 'Scooter Zone Kingston',
                'polygon': SCOOTER_POLYGON,
                'agent_ids': {'agent_4'},
            },
        }

    def test_pickup_in_region(self):
        """Pickup is in Kingston, delivery outside -> returns Kingston region."""
        task = _make_task('t1', LOC_IN_KINGSTON, LOC_OUTSIDE)
        result = self.fs._get_task_region(task)
        self.assertIsNotNone(result)
        self.assertEqual(result['id'], 'region_kgn')
        self.assertFalse(result['is_scooter'])

    def test_delivery_in_region(self):
        """Pickup outside, delivery in Kingston -> returns Kingston region."""
        task = _make_task('t2', LOC_OUTSIDE, LOC_IN_KINGSTON)
        result = self.fs._get_task_region(task)
        self.assertIsNotNone(result)
        self.assertEqual(result['id'], 'region_kgn')

    def test_both_in_different_regions_pickup_priority(self):
        """Pickup in Kingston, delivery in MoBay -> pickup region (Kingston) wins."""
        task = _make_task('t3', LOC_IN_KINGSTON, LOC_IN_MOBAY)
        result = self.fs._get_task_region(task)
        self.assertIsNotNone(result)
        self.assertEqual(result['id'], 'region_kgn')

    def test_neither_in_region(self):
        """Both locations outside all regions -> returns None."""
        task = _make_task('t4', LOC_OUTSIDE, LOC_OUTSIDE)
        result = self.fs._get_task_region(task)
        self.assertIsNone(result)

    def test_scooter_region_excluded(self):
        """Location only in scooter region -> returns None (scooter excluded from non-scooter logic)."""
        task = _make_task('t5', LOC_IN_SCOOTER, LOC_OUTSIDE)
        # LOC_IN_SCOOTER is inside both Kingston and Scooter regions
        # Kingston is non-scooter, so it should match Kingston first
        result = self.fs._get_task_region(task)
        # Since LOC_IN_SCOOTER (18.02, -76.78) is inside Kingston polygon too, 
        # it should return the non-scooter Kingston region
        self.assertIsNotNone(result)
        self.assertFalse(result['is_scooter'])

    def test_only_scooter_region(self):
        """Location ONLY in scooter region (and no other) -> returns None."""
        # Clear non-scooter regions so only scooter region remains
        self.fs._geofences = {
            'region_scooter': {
                'id': 'region_scooter',
                'name': 'Scooter Zone Kingston',
                'polygon': SCOOTER_POLYGON,
                'agent_ids': {'agent_4'},
            },
        }
        task = _make_task('t6', LOC_IN_SCOOTER, LOC_OUTSIDE)
        result = self.fs._get_task_region(task)
        self.assertIsNone(result)

    def test_no_geofences(self):
        """No geofences loaded -> returns None."""
        self.fs._geofences = {}
        task = _make_task('t7', LOC_IN_KINGSTON, LOC_IN_KINGSTON)
        result = self.fs._get_task_region(task)
        self.assertIsNone(result)


class TestGetOnlineRegionAgents(unittest.TestCase):
    """Test FleetState._get_online_region_agents()"""

    def setUp(self):
        self.fs = _make_fleet_state()
        self.fs._geofences = {
            'region_kgn': {
                'id': 'region_kgn',
                'name': 'Kingston Downtown',
                'polygon': KINGSTON_POLYGON,
                'agent_ids': {'agent_1', 'agent_2', 'agent_3'},
            },
        }
        # Add agents – 1 & 2 online, 3 offline
        self.fs._agents['agent_1'] = _make_agent('agent_1', 'Alice', LOC_IN_KINGSTON, online=True)
        self.fs._agents['agent_2'] = _make_agent('agent_2', 'Bob', LOC_IN_KINGSTON, online=True)
        self.fs._agents['agent_3'] = _make_agent('agent_3', 'Charlie', LOC_OUTSIDE, online=False)

    def test_online_agents_returned(self):
        online = self.fs._get_online_region_agents('region_kgn')
        self.assertIn('agent_1', online)
        self.assertIn('agent_2', online)
        self.assertNotIn('agent_3', online)

    def test_no_online_agents(self):
        """All region agents offline -> empty list."""
        self.fs._agents['agent_1'].status = AgentStatus.OFFLINE
        self.fs._agents['agent_2'].status = AgentStatus.OFFLINE
        online = self.fs._get_online_region_agents('region_kgn')
        self.assertEqual(online, [])

    def test_unknown_region(self):
        online = self.fs._get_online_region_agents('nonexistent')
        self.assertEqual(online, [])


class TestCheckEligibilityGeofence(unittest.TestCase):
    """Test _check_eligibility with geofence region restrictions."""

    def setUp(self):
        self.fs = _make_fleet_state()
        self.fs._geofences = {
            'region_kgn': {
                'id': 'region_kgn',
                'name': 'Kingston Downtown',
                'polygon': KINGSTON_POLYGON,
                'agent_ids': {'agent_1', 'agent_2'},
            },
        }
        # Online region agent
        self.agent_in_region = _make_agent('agent_1', 'Alice', LOC_IN_KINGSTON, online=True)
        # Online agent NOT in region
        self.agent_outside_region = _make_agent('agent_99', 'Zara', LOC_IN_KINGSTON, online=True)
        
        self.fs._agents['agent_1'] = self.agent_in_region
        self.fs._agents['agent_2'] = _make_agent('agent_2', 'Bob', LOC_IN_KINGSTON, online=True)
        self.fs._agents['agent_99'] = self.agent_outside_region

    def test_region_agent_eligible_and_distance_bypassed(self):
        """Task in Kingston region, agent in region -> eligible, even if far away."""
        # Task pickup in Kingston, agent far from pickup but in region
        far_agent = _make_agent('agent_1', 'Alice', Location(18.50, -77.90), online=True)  # Far location
        self.fs._agents['agent_1'] = far_agent
        
        task = _make_task('t1', LOC_IN_KINGSTON, LOC_OUTSIDE)
        result = self.fs._check_eligibility(far_agent, task)
        self.assertIsNone(result, f"Expected eligible, got: {result}")

    def test_non_region_agent_blocked(self):
        """Task in Kingston region, agent NOT in region -> blocked."""
        task = _make_task('t1', LOC_IN_KINGSTON, LOC_OUTSIDE)
        result = self.fs._check_eligibility(self.agent_outside_region, task)
        self.assertIsNotNone(result)
        self.assertIn('not_in_task_region', result)

    def test_no_online_region_agents_fallback(self):
        """Task in region but all region agents offline -> fallback, distance applies."""
        self.fs._agents['agent_1'].status = AgentStatus.OFFLINE
        self.fs._agents['agent_2'].status = AgentStatus.OFFLINE
        
        task = _make_task('t1', LOC_IN_KINGSTON, LOC_OUTSIDE)
        # Agent not in region, but since no region agents are online -> fallback (no region restriction)
        # Distance check applies normally
        result = self.fs._check_eligibility(self.agent_outside_region, task)
        # Should NOT be blocked by region restriction, may be blocked by distance or pass
        self.assertTrue(result is None or 'not_in_task_region' not in result)

    def test_task_in_no_region(self):
        """Task not in any region -> normal rules apply."""
        task = _make_task('t1', LOC_OUTSIDE, LOC_OUTSIDE)
        result = self.fs._check_eligibility(self.agent_outside_region, task, override_max_distance_km=100)
        # Should not be blocked by region (task isn't in one)
        self.assertTrue(result is None or 'not_in_task_region' not in result)

    def test_backward_compat_no_geofences(self):
        """No geofences synced -> all agents eligible per normal rules."""
        self.fs._geofences = {}
        task = _make_task('t1', LOC_IN_KINGSTON, LOC_OUTSIDE)
        result = self.fs._check_eligibility(self.agent_outside_region, task, override_max_distance_km=100)
        self.assertTrue(result is None or 'not_in_task_region' not in result)


class TestSyncGeofencesDefensive(unittest.TestCase):
    """Test that empty geofences list does NOT clear existing geofences."""

    def setUp(self):
        self.fs = _make_fleet_state()
        self.fs._geofences = {
            'region_kgn': {
                'id': 'region_kgn',
                'name': 'Kingston Downtown',
                'polygon': KINGSTON_POLYGON,
                'agent_ids': {'agent_1'},
            },
        }

    def test_empty_list_preserves_geofences(self):
        """sync_geofences([]) should not clear existing geofences."""
        self.fs.sync_geofences([])
        self.assertEqual(len(self.fs._geofences), 1)

    def test_none_preserves_geofences(self):
        """sync_geofences(None-ish) should not clear existing geofences."""
        self.fs.sync_geofences(None)
        self.assertEqual(len(self.fs._geofences), 1)

    def test_nonempty_list_replaces_geofences(self):
        """sync_geofences with real data replaces geofences."""
        self.fs.sync_geofences([{
            'id': 'new_region',
            'name': 'New Region',
            'polygon': MOBAY_POLYGON,
            'agent_ids': ['agent_5'],
        }])
        self.assertEqual(len(self.fs._geofences), 1)
        self.assertIn('new_region', self.fs._geofences)
        self.assertNotIn('region_kgn', self.fs._geofences)


class TestDeliveryOnlyRegion(unittest.TestCase):
    """DELIVERY_ONLY task uses delivery location for region lookup."""

    def setUp(self):
        self.fs = _make_fleet_state()
        self.fs._geofences = {
            'region_kgn': {
                'id': 'region_kgn',
                'name': 'Kingston Downtown',
                'polygon': KINGSTON_POLYGON,
                'agent_ids': {'agent_1'},
            },
        }
        self.fs._agents['agent_1'] = _make_agent('agent_1', 'Alice', LOC_IN_KINGSTON, online=True)

    def test_delivery_only_in_region(self):
        """DELIVERY_ONLY task with delivery in Kingston -> region restricted."""
        task = _make_task('t1', LOC_OUTSIDE, LOC_IN_KINGSTON, job_type='DELIVERY_ONLY')
        result = self.fs._get_task_region(task)
        self.assertIsNotNone(result)
        self.assertEqual(result['id'], 'region_kgn')


class TestMultipleRegionAgent(unittest.TestCase):
    """Agent assigned to multiple regions is eligible in any of them."""

    def setUp(self):
        self.fs = _make_fleet_state()
        self.fs._geofences = {
            'region_kgn': {
                'id': 'region_kgn',
                'name': 'Kingston Downtown',
                'polygon': KINGSTON_POLYGON,
                'agent_ids': {'agent_multi'},
            },
            'region_mobay': {
                'id': 'region_mobay',
                'name': 'Montego Bay',
                'polygon': MOBAY_POLYGON,
                'agent_ids': {'agent_multi'},
            },
        }
        self.fs._agents['agent_multi'] = _make_agent(
            'agent_multi', 'MultiAgent', LOC_IN_KINGSTON, online=True
        )

    def test_agent_eligible_in_kingston(self):
        task = _make_task('t1', LOC_IN_KINGSTON, LOC_OUTSIDE)
        result = self.fs._check_eligibility(
            self.fs._agents['agent_multi'], task, override_max_distance_km=100
        )
        self.assertIsNone(result, f"Expected eligible, got: {result}")

    def test_agent_eligible_in_mobay(self):
        task = _make_task('t2', LOC_IN_MOBAY, LOC_OUTSIDE)
        result = self.fs._check_eligibility(
            self.fs._agents['agent_multi'], task, override_max_distance_km=100
        )
        self.assertIsNone(result, f"Expected eligible, got: {result}")


# ===========================================================================
# Optimizer (fleet_optimizer) tests
# ===========================================================================

class TestGeofenceRegionFromDict(unittest.TestCase):
    """Test GeofenceRegion.from_dict() handles both old and new formats."""

    def test_old_format(self):
        data = {
            'region_id': 42,
            'region_name': 'Kingston Zone',
            'polygon': [[18.0, -76.8], [18.05, -76.8], [18.05, -76.75], [18.0, -76.75]],
            'fleet_ids': ['a1', 'a2'],
        }
        gf = GeofenceRegion.from_dict(data)
        self.assertEqual(gf.region_id, 42)
        self.assertEqual(gf.region_name, 'Kingston Zone')
        self.assertEqual(len(gf.polygon), 4)
        self.assertEqual(gf.fleet_ids, ['a1', 'a2'])
        self.assertFalse(gf.is_scooter)

    def test_new_format(self):
        data = {
            'id': 99,
            'name': 'Montego Bay',
            'polygon': [[18.45, -77.95], [18.50, -77.95], [18.50, -77.90], [18.45, -77.90]],
            'agent_ids': [101, 102],
        }
        gf = GeofenceRegion.from_dict(data)
        self.assertEqual(gf.region_id, 99)
        self.assertEqual(gf.region_name, 'Montego Bay')
        self.assertEqual(gf.fleet_ids, ['101', '102'])
        self.assertFalse(gf.is_scooter)

    def test_scooter_detection(self):
        data = {
            'id': 5,
            'name': 'Scooter Zone Kingston',
            'polygon': SCOOTER_POLYGON,
            'agent_ids': ['a4'],
        }
        gf = GeofenceRegion.from_dict(data)
        self.assertTrue(gf.is_scooter)


class TestOptimizerNonScooterGeofence(unittest.TestCase):
    """Test CompatibilityChecker non-scooter geofence rules."""

    def _make_optimizer_task(self, task_id, pickup, delivery, tips=0.0, delivery_fee=8.0):
        return Task(
            id=task_id,
            restaurant_location=pickup,
            delivery_location=delivery,
            pickup_before=FUTURE,
            delivery_before=FUTURE + timedelta(hours=1),
            payment_type='CARD',
            tags=[],
            declined_by=[],
            tips=tips,
            delivery_fee=delivery_fee,
        )

    def _make_optimizer_agent(self, agent_id, name, location, tags=None, priority=None, fleet_ids=None):
        return Agent(
            id=agent_id,
            name=name,
            current_location=location,
            tags=tags or [],
            priority=priority,
        )

    def _make_checker(self, **kwargs):
        return CompatibilityChecker(
            road_distance_provider=_test_road_distance_provider,
            **kwargs
        )

    def setUp(self):
        self.kgn_geofence = GeofenceRegion(
            region_id='region_kgn',
            region_name='Kingston Downtown',
            polygon=[(p[0], p[1]) for p in KINGSTON_POLYGON],
            fleet_ids=['agent_1', 'agent_2'],
        )
        self.mobay_geofence = GeofenceRegion(
            region_id='region_mobay',
            region_name='Montego Bay',
            polygon=[(p[0], p[1]) for p in MOBAY_POLYGON],
            fleet_ids=['agent_3'],
        )
        self.scooter_geofence = GeofenceRegion(
            region_id='region_scooter',
            region_name='Scooter Zone Kingston',
            polygon=[(p[0], p[1]) for p in SCOOTER_POLYGON],
            fleet_ids=['agent_4'],
        )
        
        self.agent_in_kgn = self._make_optimizer_agent('agent_1', 'Alice', OptimizerLocation(18.02, -76.78))
        self.agent_in_mobay = self._make_optimizer_agent('agent_3', 'Charlie', OptimizerLocation(18.47, -77.92))
        self.agent_outside = self._make_optimizer_agent('agent_99', 'Zara', OptimizerLocation(18.02, -76.78))
        
        self.all_agents = [self.agent_in_kgn, self.agent_in_mobay, self.agent_outside]

    def test_region_agent_compatible_distance_bypassed(self):
        """Agent in Kingston region + task in Kingston -> compatible, even far away."""
        checker = self._make_checker(
            geofence_regions=[self.kgn_geofence, self.mobay_geofence],
            max_distance_km=1.0,  # Very short distance
        )
        checker._all_agents = self.all_agents
        
        task = self._make_optimizer_task(
            't1',
            OptimizerLocation(18.03, -76.77),  # Inside Kingston
            OptimizerLocation(17.90, -76.50),   # Outside
        )
        is_compat, reason = checker.is_compatible(self.agent_in_kgn, task)
        self.assertTrue(is_compat, f"Expected compatible, got: {reason}")

    def test_non_region_agent_blocked(self):
        """Agent NOT in Kingston region + task in Kingston (region staffed) -> blocked."""
        checker = self._make_checker(
            geofence_regions=[self.kgn_geofence, self.mobay_geofence],
            max_distance_km=50.0,
        )
        checker._all_agents = self.all_agents
        
        task = self._make_optimizer_task(
            't1',
            OptimizerLocation(18.03, -76.77),  # Inside Kingston
            OptimizerLocation(17.90, -76.50),   # Outside
        )
        is_compat, reason = checker.is_compatible(self.agent_outside, task)
        self.assertFalse(is_compat)
        self.assertIn('not_in_task_region', reason)

    def test_no_region_agents_in_solver_fallback(self):
        """Task in region but no region agents in solver -> fallback to global rules."""
        checker = self._make_checker(
            geofence_regions=[self.kgn_geofence],
            max_distance_km=50.0,
        )
        # Only non-region agent in solver
        checker._all_agents = [self.agent_outside]
        
        task = self._make_optimizer_task(
            't1',
            OptimizerLocation(18.03, -76.77),  # Inside Kingston
            OptimizerLocation(17.90, -76.50),   # Outside
        )
        is_compat, reason = checker.is_compatible(self.agent_outside, task)
        # Should not be blocked by region (no region agents in solver)
        self.assertTrue('not_in_task_region' not in reason)

    def test_task_in_no_region(self):
        """Task outside all regions -> normal rules, no region restriction."""
        checker = self._make_checker(
            geofence_regions=[self.kgn_geofence],
            max_distance_km=50.0,
        )
        checker._all_agents = self.all_agents
        
        task = self._make_optimizer_task(
            't1',
            OptimizerLocation(17.90, -76.50),  # Outside all regions
            OptimizerLocation(17.85, -76.45),   # Outside
        )
        is_compat, reason = checker.is_compatible(self.agent_outside, task)
        self.assertTrue(is_compat, f"Expected compatible, got: {reason}")

    def test_scooter_region_blocks_trip_leaving_zone(self):
        """Agent in a scooter region's fleet_ids -> both locations must be inside."""
        scooter_agent = self._make_optimizer_agent(
            'agent_4', 'ScooterGuy', OptimizerLocation(18.02, -76.78)
        )
        checker = self._make_checker(
            geofence_regions=[self.scooter_geofence],
            max_distance_km=50.0,
        )
        checker._all_agents = [scooter_agent]
        
        # Pickup inside scooter zone, delivery outside -> should fail scooter rule
        task = self._make_optimizer_task(
            't1',
            OptimizerLocation(18.02, -76.78),  # Inside scooter zone
            OptimizerLocation(17.90, -76.50),   # Outside
        )
        is_compat, reason = checker.is_compatible(scooter_agent, task)
        self.assertFalse(is_compat)
        self.assertIn('outside_scooter_geofence', reason)

    def test_scooter_region_allows_trip_inside_zone(self):
        """Agent in a scooter region -> trip staying fully inside is compatible."""
        scooter_agent = self._make_optimizer_agent(
            'agent_4', 'ScooterGuy', OptimizerLocation(18.02, -76.78)
        )
        checker = self._make_checker(
            geofence_regions=[self.scooter_geofence],
            max_distance_km=50.0,
        )
        checker._all_agents = [scooter_agent]

        task = self._make_optimizer_task(
            't1',
            OptimizerLocation(18.02, -76.78),    # Inside scooter zone
            OptimizerLocation(18.025, -76.775),  # Also inside
        )
        is_compat, reason = checker.is_compatible(scooter_agent, task)
        self.assertTrue(is_compat, f"Expected compatible, got: {reason}")

    def test_scooter_rule_ignores_agent_tags(self):
        """A 'scooter'-tagged agent NOT in the region is not scooter-restricted."""
        tagged_agent = self._make_optimizer_agent(
            'agent_99', 'Zara', OptimizerLocation(18.02, -76.78), tags=['Scooter']
        )
        checker = self._make_checker(
            geofence_regions=[self.scooter_geofence],
            max_distance_km=50.0,
        )
        checker._all_agents = [tagged_agent]

        task = self._make_optimizer_task(
            't1',
            OptimizerLocation(18.02, -76.78),  # Inside scooter zone
            OptimizerLocation(17.90, -76.50),   # Outside
        )
        is_compat, reason = checker.is_compatible(tagged_agent, task)
        self.assertTrue(is_compat, f"Expected compatible, got: {reason}")

    def test_backward_compat_no_geofences(self):
        """No geofences in checker -> no region restriction."""
        checker = self._make_checker(
            geofence_regions=[],
            max_distance_km=50.0,
        )
        checker._all_agents = [self.agent_outside]
        
        task = self._make_optimizer_task(
            't1',
            OptimizerLocation(18.03, -76.77),
            OptimizerLocation(17.90, -76.50),
        )
        is_compat, reason = checker.is_compatible(self.agent_outside, task)
        self.assertTrue(is_compat, f"Expected compatible, got: {reason}")

    def test_pickup_region_priority_over_delivery(self):
        """Pickup in Kingston, delivery in MoBay -> Kingston region used."""
        checker = self._make_checker(
            geofence_regions=[self.kgn_geofence, self.mobay_geofence],
            max_distance_km=50.0,
        )
        checker._all_agents = self.all_agents
        
        task = self._make_optimizer_task(
            't1',
            OptimizerLocation(18.03, -76.77),  # Inside Kingston
            OptimizerLocation(18.47, -77.92),   # Inside MoBay
        )
        # Kingston agent should be compatible (pickup region = Kingston)
        is_compat_kgn, reason_kgn = checker.is_compatible(self.agent_in_kgn, task)
        self.assertTrue(is_compat_kgn, f"Expected Kingston agent compatible, got: {reason_kgn}")
        
        # MoBay agent should be blocked (not in Kingston region)
        is_compat_mobay, reason_mobay = checker.is_compatible(self.agent_in_mobay, task)
        self.assertFalse(is_compat_mobay)
        self.assertIn('not_in_task_region', reason_mobay)


# ===========================================================================
# Export helper tests
# ===========================================================================

class TestExportGeofenceData(unittest.TestCase):
    """Test the _export_geofence_data helper in app.py logic."""

    def test_export_format(self):
        """Verify export translates fleet_state format to optimizer format."""
        fs = _make_fleet_state()
        fs.sync_geofences([{
            'id': '42',
            'name': 'Kingston',
            'polygon': KINGSTON_POLYGON,
            'agent_ids': ['a1', 'a2'],
        }])
        
        # Simulate what _export_geofence_data does
        exported = []
        for gf in fs._geofences.values():
            exported.append({
                'region_id': gf['id'],
                'region_name': gf['name'],
                'polygon': gf['polygon'],
                'fleet_ids': list(gf['agent_ids']),
            })
        
        self.assertEqual(len(exported), 1)
        self.assertEqual(exported[0]['region_id'], '42')
        self.assertEqual(exported[0]['region_name'], 'Kingston')
        
        # Verify GeofenceRegion.from_dict can parse the exported format
        gf = GeofenceRegion.from_dict(exported[0])
        self.assertEqual(str(gf.region_id), '42')
        self.assertEqual(gf.region_name, 'Kingston')


if __name__ == '__main__':
    unittest.main()
