import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app as app_module
from reliable_event_outbox import ReliableEventOutbox


class ReliableLifecycleEventTests(unittest.TestCase):
    def setUp(self):
        app_module.fleet_state.clear()
        with app_module._lifecycle_event_lock:
            app_module._lifecycle_event_acks.clear()
            app_module._terminal_lifecycle_tasks.clear()

        app_module.fleet_state.sync_agents([
            {
                "id": "agent-1",
                "name": "Agent One",
                "location": [17.12, -61.84],
                "status": "online",
                "max_capacity": 2,
            }
        ])
        self.client = app_module.socketio.test_client(app_module.app)

    def tearDown(self):
        self.client.disconnect()
        app_module.fleet_state.clear()
        with app_module._lifecycle_event_lock:
            app_module._lifecycle_event_acks.clear()
            app_module._terminal_lifecycle_tasks.clear()

    def _add_assigned_task(self, task_id="task-1"):
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
                "restaurant_name": "Test Restaurant",
                "customer_name": "Test Customer",
            },
        })

    def test_task_completed_ack_removes_assigned_task(self):
        self._add_assigned_task()

        ack = self.client.emit(
            "task:completed",
            {
                "id": "task-1",
                "agent_id": "agent-1",
                "agent_name": "Agent One",
                "job_type": 1,
                "event_id": "evt-complete-1",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            callback=True,
        )

        self.assertTrue(ack["success"])
        self.assertTrue(ack["task_removed"])
        self.assertFalse(ack["duplicate"])
        agent = app_module.fleet_state.get_agent("agent-1")
        self.assertEqual(len(agent.current_tasks), 0)

    def test_duplicate_completion_event_is_idempotent(self):
        self._add_assigned_task()
        payload = {
            "id": "task-1",
            "agent_id": "agent-1",
            "agent_name": "Agent One",
            "job_type": 1,
            "event_id": "evt-complete-dup",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        first_ack = self.client.emit("task:completed", payload, callback=True)
        second_ack = self.client.emit("task:completed", payload, callback=True)

        self.assertTrue(first_ack["success"])
        self.assertTrue(second_ack["success"])
        self.assertTrue(second_ack["duplicate"])
        agent = app_module.fleet_state.get_agent("agent-1")
        self.assertEqual(len(agent.current_tasks), 0)

    def test_unknown_delivery_completion_cleans_placeholder_and_blocks_late_create(self):
        # Simulates task:assigned arriving before task:created, then delivery completion.
        app_module.fleet_state.assign_task("ghost-task", "agent-1", "Agent One")
        self.assertEqual(len(app_module.fleet_state.get_agent("agent-1").current_tasks), 1)

        ack = self.client.emit(
            "task:completed",
            {
                "id": "ghost-task",
                "agent_id": "agent-1",
                "agent_name": "Agent One",
                "job_type": 1,
                "event_id": "evt-ghost-complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            callback=True,
        )

        self.assertTrue(ack["success"])
        self.assertTrue(ack["unknown_task"])
        self.assertEqual(len(app_module.fleet_state.get_agent("agent-1").current_tasks), 0)

        now = datetime.now(timezone.utc)
        created_ack = self.client.emit(
            "task:created",
            {
                "event_id": "evt-ghost-created",
                "task": {
                    "id": "ghost-task",
                    "job_type": "PAIRED",
                    "restaurant_location": [17.12, -61.84],
                    "delivery_location": [17.13, -61.85],
                    "pickup_before": (now + timedelta(minutes=10)).isoformat(),
                    "delivery_before": (now + timedelta(minutes=40)).isoformat(),
                    "_meta": {
                        "restaurant_name": "Late Create",
                        "customer_name": "Already Done",
                    },
                },
            },
            callback=True,
        )

        self.assertTrue(created_ack["success"])
        self.assertTrue(created_ack["ignored_terminal"])
        self.assertIsNone(app_module.fleet_state.get_task("ghost-task"))

    def test_unknown_pickup_completion_is_stale_noop_success(self):
        payload = {
            "id": "ghost-pickup",
            "agent_id": "agent-1",
            "agent_name": "Agent One",
            "event_id": "evt-ghost-pickup",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        first_ack = self.client.emit("pickup:completed", payload, callback=True)
        second_ack = self.client.emit("pickup:completed", payload, callback=True)

        self.assertTrue(first_ack["success"])
        self.assertTrue(first_ack["stale_noop"])
        self.assertTrue(first_ack["unknown_task"])
        self.assertEqual(first_ack["reason"], "pickup_completion_for_unknown_or_terminal_task")
        self.assertTrue(second_ack["success"])
        self.assertTrue(second_ack["duplicate"])

    def test_unknown_task_updated_is_stale_noop_success(self):
        self.client.emit(
            "task:updated",
            {
                "id": "ghost-update",
                "event_id": "evt-ghost-update",
                "status": "Unassigned",
            },
        )

        received = self.client.get_received()
        ack_packets = [packet for packet in received if packet["name"] == "task:updated_ack"]
        self.assertEqual(len(ack_packets), 1)
        ack = ack_packets[0]["args"][0]
        self.assertTrue(ack["success"])
        self.assertTrue(ack["stale_noop"])
        self.assertTrue(ack["unknown_task"])
        self.assertEqual(ack["reason"], "task_not_found")

    def test_missing_pickup_completion_task_id_still_fails(self):
        ack = self.client.emit(
            "pickup:completed",
            {
                "agent_id": "agent-1",
                "event_id": "evt-missing-task-id",
            },
            callback=True,
        )

        self.assertFalse(ack["success"])
        self.assertEqual(ack["error"], "Missing task id")


class FakeSocketClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def call(self, event_type, payload, timeout):
        self.calls.append((event_type, payload, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ReliableEventOutboxTests(unittest.TestCase):
    def test_outbox_retries_until_matching_success_ack(self):
        payload = {
            "id": "task-1",
            "agent_id": "agent-1",
            "job_type": 1,
            "completed_at": "2026-04-27T14:31:28Z",
        }
        event_id = "task:completed:task-1::1:agent-1:2026-04-27T14:31:28Z"
        sio = FakeSocketClient([
            {"success": True, "event_id": "wrong-event"},
            {"success": True, "event_id": event_id},
        ])

        with tempfile.TemporaryDirectory() as tmp:
            outbox = ReliableEventOutbox(
                f"{tmp}/outbox.sqlite",
                sio,
                ack_timeout_seconds=0.01,
                base_backoff_seconds=0,
                max_attempts=3,
            )

            first = outbox.emit("task:completed", payload)
            second = outbox.emit("task:completed", payload)

        self.assertFalse(first["success"])
        self.assertTrue(first["retry_scheduled"])
        self.assertTrue(second["success"])
        self.assertEqual(len(sio.calls), 2)


if __name__ == "__main__":
    unittest.main()
