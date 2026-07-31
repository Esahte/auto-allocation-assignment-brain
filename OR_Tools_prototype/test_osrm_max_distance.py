import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from fleet_optimizer import (
    Agent,
    CompatibilityChecker,
    Location as OptimizerLocation,
    Task,
    get_osrm_distances_matrix_with_source,
)
from fleet_state import FleetState


class _FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "code": "Ok",
            "distances": [
                [0, 4500],
                [4500, 0],
            ],
        }


class _FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def get(self, *_args, **_kwargs):
        if self.error:
            raise self.error
        return self.response


def _task_payload():
    now = datetime.now(timezone.utc)
    return {
        "id": "task-1",
        "job_type": "PAIRED",
        "restaurant_location": [17.13, -61.84],
        "delivery_location": [17.14, -61.85],
        "pickup_before": (now + timedelta(minutes=10)).isoformat(),
        "delivery_before": (now + timedelta(minutes=40)).isoformat(),
        "_meta": {"restaurant_name": "Test Merchant"},
    }


class OsrmDistanceMatrixTests(unittest.TestCase):
    def test_osrm_distance_is_returned_in_kilometers(self):
        with patch(
            "fleet_optimizer.get_osrm_session",
            return_value=_FakeSession(response=_FakeResponse()),
        ):
            matrix, source = get_osrm_distances_matrix_with_source(
                [OptimizerLocation(17.12, -61.84)],
                [OptimizerLocation(17.13, -61.84)],
            )

        self.assertEqual(source, "osrm")
        self.assertEqual(matrix, [[4.5]])

    def test_haversine_is_used_when_osrm_fails(self):
        with patch(
            "fleet_optimizer.get_osrm_session",
            return_value=_FakeSession(error=TimeoutError("OSRM timeout")),
        ):
            matrix, source = get_osrm_distances_matrix_with_source(
                [OptimizerLocation(17.12, -61.84)],
                [OptimizerLocation(17.13, -61.84)],
            )

        self.assertEqual(source, "haversine_fallback")
        self.assertGreater(matrix[0][0], 1.0)
        self.assertLess(matrix[0][0], 1.2)


class MaxDistanceEligibilityTests(unittest.TestCase):
    def test_fleet_eligibility_uses_cached_road_distance(self):
        calls = []

        def road_provider(origins, destinations):
            calls.append((origins, destinations))
            return [[8.0] for _ in origins], "osrm"

        fleet = FleetState(
            max_distance_km=5,
            road_distance_provider=road_provider,
        )
        fleet.sync_agents([{
            "id": "agent-1",
            "name": "Agent One",
            "location": [17.12, -61.84],
            "status": "online",
        }])
        task = fleet.add_task(_task_payload())
        agent = fleet.get_agent("agent-1")

        first = fleet._check_eligibility(agent, task)
        second = fleet._check_eligibility(agent, task)

        self.assertIn("8.0km road > 5km", first)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)

    def test_optimizer_compatibility_uses_road_distance(self):
        checker = CompatibilityChecker(
            max_distance_km=5,
            road_distance_provider=lambda origins, destinations: (
                [[8.0] for _ in origins],
                "osrm",
            ),
        )
        agent = Agent(
            id="agent-1",
            name="Agent One",
            current_location=OptimizerLocation(17.12, -61.84),
        )
        now = datetime.now(timezone.utc)
        task = Task(
            id="task-1",
            restaurant_location=OptimizerLocation(17.13, -61.84),
            delivery_location=OptimizerLocation(17.14, -61.85),
            pickup_before=now + timedelta(minutes=10),
            delivery_before=now + timedelta(minutes=40),
        )

        compatible, reason = checker.is_compatible(agent, task)

        self.assertFalse(compatible)
        self.assertEqual(reason, "distance_exceeded_8.0km")


if __name__ == "__main__":
    unittest.main()
