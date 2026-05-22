import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app as app_module


class ProximityCapacityReservationTests(unittest.TestCase):
    def setUp(self):
        app_module.fleet_state.clear()
        with app_module._proximity_lock:
            app_module._agent_pending_broadcasts.clear()
            app_module._agent_pending_directions.clear()
            app_module._agent_pending_routes.clear()
            app_module._task_agent_broadcast_counts.clear()

        app_module.fleet_state.sync_agents([
            {
                "id": "agent-1",
                "name": "Agent One",
                "location": [17.12, -61.84],
                "status": "online",
                "max_capacity": 2,
            }
        ])

    def tearDown(self):
        app_module.fleet_state.clear()
        with app_module._proximity_lock:
            app_module._agent_pending_broadcasts.clear()
            app_module._agent_pending_directions.clear()
            app_module._agent_pending_routes.clear()
            app_module._task_agent_broadcast_counts.clear()

    def _add_assigned_task(self, task_id="current-task"):
        now = datetime.now(timezone.utc)
        app_module.fleet_state.add_task({
            "id": task_id,
            "job_type": "PAIRED",
            "restaurant_location": [17.12, -61.84],
            "delivery_location": [17.13, -61.85],
            "pickup_before": (now + timedelta(minutes=10)).isoformat(),
            "delivery_before": (now + timedelta(minutes=40)).isoformat(),
            "assigned_agent_id": "agent-1",
            "_meta": {
                "restaurant_name": "Current Restaurant",
                "customer_name": "Current Customer",
            },
        })

    def test_reservation_consumes_only_remaining_capacity(self):
        self._add_assigned_task()

        reserved, dropped = app_module.reserve_pending_broadcasts([
            {"agent_id": "agent-1", "task_id": "offer-1", "bearing": 90, "has_route": True},
            {"agent_id": "agent-1", "task_id": "offer-2", "bearing": 90, "has_route": True},
        ])

        self.assertEqual([r["task_id"] for r in reserved], ["offer-1"])
        self.assertEqual([(d["task_id"], d["reason"]) for d in dropped], [("offer-2", "at_capacity")])
        self.assertEqual(app_module.get_agent_pending_tasks("agent-1"), {"offer-1"})
        self.assertEqual(app_module.get_agent_broadcast_capacity("agent-1"), 0)

    def test_duplicate_reservation_does_not_consume_capacity_twice(self):
        app_module.reserve_pending_broadcasts([
            {"agent_id": "agent-1", "task_id": "offer-1", "bearing": 90, "has_route": True}
        ])

        reserved, dropped = app_module.reserve_pending_broadcasts([
            {"agent_id": "agent-1", "task_id": "offer-1", "bearing": 90, "has_route": True},
            {"agent_id": "agent-1", "task_id": "offer-2", "bearing": 90, "has_route": True},
        ])

        self.assertEqual([r["task_id"] for r in reserved], ["offer-2"])
        self.assertEqual([(d["task_id"], d["reason"]) for d in dropped], [("offer-1", "already_pending")])
        self.assertEqual(app_module.get_agent_pending_tasks("agent-1"), {"offer-1", "offer-2"})


if __name__ == "__main__":
    unittest.main()
