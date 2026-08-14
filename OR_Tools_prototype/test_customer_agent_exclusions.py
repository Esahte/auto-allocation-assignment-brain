import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import app as app_module
from fleet_optimizer import (
    Agent,
    CompatibilityChecker,
    Location as OptimizerLocation,
    Task,
    optimize_fleet,
)
from fleet_state import FleetState


def _test_road_distance_provider(origins, destinations):
    return [
        [
            ((origin.lat - destination.lat) ** 2
             + (origin.lng - destination.lng) ** 2) ** 0.5 * 111
            for destination in destinations
        ]
        for origin in origins
    ], "osrm"


def _task_payload(task_id="task-1", customer="JohnDoe"):
    now = datetime.now(timezone.utc)
    return {
        "id": task_id,
        "job_type": "PAIRED",
        "restaurant_location": [17.12, -61.84],
        "delivery_location": [17.13, -61.85],
        "pickup_before": (now + timedelta(minutes=10)).isoformat(),
        "delivery_before": (now + timedelta(minutes=40)).isoformat(),
        "_meta": {"restaurant_name": "KFC", "customer_name": customer},
    }


class CustomerExemptionEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.fleet = FleetState(
            max_distance_km=10,
            road_distance_provider=_test_road_distance_provider
        )
        self.fleet.sync_agents([
            {
                "id": "123",
                "name": "Agent 123",
                "location": [17.12, -61.84],
                "status": "online",
                "blocked_customers": ["JohnDoe"],
            },
            {
                "id": "999",
                "name": "Agent 999",
                "location": [17.12, -61.84],
                "status": "online",
                "blocked_customers": [],
            },
        ])
        self.task = self.fleet.add_task(_task_payload(customer="JohnDoe"))

    def test_agent_field_blocks_exact_username(self):
        reason = self.fleet._check_eligibility(
            self.fleet.get_agent("123"), self.task
        )
        self.assertEqual(reason, "customer_exemption_A")

    def test_agent_field_does_not_block_different_case(self):
        agent = self.fleet.get_agent("123")
        other = self.fleet.add_task(_task_payload("task-case", "johndoe"))

        self.assertIsNone(self.fleet._check_eligibility(agent, other))

    def test_agent_field_does_not_block_extra_space(self):
        agent = self.fleet.get_agent("123")
        other = self.fleet.add_task(_task_payload("task-space", "John Doe"))

        self.assertIsNone(self.fleet._check_eligibility(agent, other))

    def test_agent_field_does_not_block_similar_special_characters(self):
        agent = self.fleet.get_agent("123")
        self.fleet.update_agent("123", blocked_customers=["John.Doe"])
        other = self.fleet.add_task(_task_payload("task-special", "John_Doe"))

        self.assertIsNone(self.fleet._check_eligibility(agent, other))

    def test_config_map_blocks_even_when_agent_list_is_empty(self):
        task = self.fleet.add_task(_task_payload("task-2", "Jane Doe!"))
        self.fleet.set_customer_agent_exclusions({"Jane Doe!": ["999"]})

        reason = self.fleet._check_eligibility(
            self.fleet.get_agent("999"), task
        )

        self.assertEqual(reason, "customer_exemption_C")

    def test_config_map_preserves_exact_username_key(self):
        self.fleet.set_customer_agent_exclusions({"  JohnDoe  ": ["999"]})

        self.assertEqual(
            list(self.fleet.customer_agent_exclusions.keys()),
            ["  JohnDoe  "],
        )
        padded = self.fleet.add_task(_task_payload("task-pad", "  JohnDoe  "))
        unpadded = self.fleet.add_task(_task_payload("task-unpad", "JohnDoe"))

        self.assertEqual(
            self.fleet._check_eligibility(self.fleet.get_agent("999"), padded),
            "customer_exemption_C",
        )
        self.assertIsNone(
            self.fleet._check_eligibility(self.fleet.get_agent("999"), unpadded)
        )

    def test_config_map_does_not_partial_match(self):
        village = self.fleet.add_task(_task_payload("task-partial", "JohnDoeJr"))
        self.fleet.set_customer_agent_exclusions({"JohnDoe": ["999"]})

        self.assertIsNone(
            self.fleet._check_eligibility(self.fleet.get_agent("999"), village)
        )

    def test_clearing_both_restores_eligibility(self):
        agent = self.fleet.get_agent("123")
        self.fleet.update_agent("123", blocked_customers=[])
        self.fleet.set_customer_agent_exclusions({})

        self.assertIsNone(self.fleet._check_eligibility(agent, self.task))

    def test_optimizer_uses_union_of_agent_and_config_sources(self):
        agent = Agent(
            id="123",
            name="Agent 123",
            current_location=OptimizerLocation(17.12, -61.84),
            blocked_customers=["JohnDoe"],
        )
        task = Task.from_dict(_task_payload(customer="JohnDoe"))
        checker = CompatibilityChecker(
            max_distance_km=10,
            customer_agent_exclusions={"JohnDoe": ["123"]},
        )

        compatible, reason = checker.is_compatible(agent, task)

        self.assertFalse(compatible)
        self.assertEqual(reason, "customer_exemption_A+C")

    def test_optimizer_does_not_block_case_or_partial_matches(self):
        agent = Agent(
            id="123",
            name="Agent 123",
            current_location=OptimizerLocation(17.12, -61.84),
            blocked_customers=["john"],
        )
        task = Task.from_dict(_task_payload(customer="JohnDoe"))
        checker = CompatibilityChecker(
            max_distance_km=None,
            customer_agent_exclusions={"JOHNDOE": ["123"]},
        )

        compatible, reason = checker.is_compatible(agent, task)

        self.assertTrue(compatible)
        self.assertFalse(reason.startswith("customer_exemption"))

    def test_optimize_flow_receives_exported_config_exclusions(self):
        fleet = FleetState(
            max_distance_km=10,
            road_distance_provider=_test_road_distance_provider
        )
        fleet.sync_agents([{
            "id": "999",
            "name": "Agent 999",
            "location": [17.12, -61.84],
            "status": "online",
            "blocked_customers": [],
        }])
        fleet.set_customer_agent_exclusions({"JohnDoe": ["999"]})
        fleet.add_task(_task_payload())

        with patch(
            "fleet_optimizer.CompatibilityChecker",
        ) as checker_class, patch(
            "fleet_optimizer.FleetOptimizer.optimize",
            return_value={"success": True},
        ):
            result = optimize_fleet(
                fleet.export_agents_for_optimizer(),
                fleet.export_tasks_for_optimizer(),
                prefilter_distance=False,
            )

        self.assertTrue(result["success"])
        self.assertEqual(
            checker_class.call_args.kwargs["customer_agent_exclusions"],
            {"JohnDoe": ["999"]},
        )


class CustomerExemptionEventTests(unittest.TestCase):
    def setUp(self):
        app_module.fleet_state.clear()
        self.client = app_module.socketio.test_client(app_module.app)

    def tearDown(self):
        self.client.disconnect()
        app_module.fleet_state.clear()

    def test_sync_and_updates_replace_and_clear_exemptions(self):
        self.client.emit("fleet:sync", {
            "agents": [{
                "id": "123",
                "name": "Agent 123",
                "location": [17.12, -61.84],
                "status": "online",
                "blocked_customers": ["JohnDoe"],
            }],
            "unassigned_tasks": [],
            "in_progress_tasks": [],
            "config": {
                "customer_agent_exclusions": {"Jane Doe!": ["123"]}
            },
        })

        agent = app_module.fleet_state.get_agent("123")
        self.assertEqual(agent.blocked_customers, ["JohnDoe"])
        self.assertEqual(
            app_module.fleet_state.customer_agent_exclusions,
            {"Jane Doe!": ["123"]},
        )

        self.client.emit("agent:update", {
            "agent_id": "123",
            "blocked_customers": [],
        })
        self.client.emit("config:update", {
            "config": {"customer_agent_exclusions": {}},
        })

        self.assertEqual(agent.blocked_customers, [])
        self.assertEqual(
            app_module.fleet_state.customer_agent_exclusions, {}
        )

    def test_agent_online_persists_blocked_customers(self):
        self.client.emit("agent:online", {
            "agent_id": "456",
            "name": "Agent 456",
            "location": [17.12, -61.84],
            "blocked_customers": ["User_Name"],
        })

        self.assertEqual(
            app_module.fleet_state.get_agent("456").blocked_customers,
            ["User_Name"],
        )


if __name__ == "__main__":
    unittest.main()
