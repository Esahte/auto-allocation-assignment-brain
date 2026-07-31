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


def _task_payload(task_id="task-1", merchant="KFC St. John's"):
    now = datetime.now(timezone.utc)
    return {
        "id": task_id,
        "job_type": "PAIRED",
        "restaurant_location": [17.12, -61.84],
        "delivery_location": [17.13, -61.85],
        "pickup_before": (now + timedelta(minutes=10)).isoformat(),
        "delivery_before": (now + timedelta(minutes=40)).isoformat(),
        "_meta": {"restaurant_name": merchant},
    }


class MerchantExemptionEligibilityTests(unittest.TestCase):
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
                "blocked_merchants": ["  KFC ST. JOHN'S  "],
            },
            {
                "id": "999",
                "name": "Agent 999",
                "location": [17.12, -61.84],
                "status": "online",
                "blocked_merchants": [],
            },
        ])
        self.task = self.fleet.add_task(
            _task_payload(merchant="kfc st. john's")
        )

    def test_agent_field_blocks_case_insensitively_after_trim(self):
        reason = self.fleet._check_eligibility(
            self.fleet.get_agent("123"), self.task
        )
        self.assertEqual(reason, "merchant_exemption_A")

    def test_agent_field_blocks_case_insensitive_partial_match(self):
        agent = self.fleet.get_agent("123")
        self.fleet.update_agent("123", blocked_merchants=["KFC"])
        airport_task = self.fleet.add_task(
            _task_payload("task-airport", "KfC Airport")
        )

        reason = self.fleet._check_eligibility(agent, airport_task)

        self.assertEqual(reason, "merchant_exemption_A")

    def test_config_map_blocks_even_when_agent_list_is_empty(self):
        subway = self.fleet.add_task(_task_payload("task-2", " subway "))
        self.fleet.set_merchant_agent_exclusions({"  SUBWAY  ": ["999"]})

        reason = self.fleet._check_eligibility(
            self.fleet.get_agent("999"), subway
        )

        self.assertEqual(reason, "merchant_exemption_C")

    def test_config_map_blocks_case_insensitive_partial_match(self):
        village_task = self.fleet.add_task(
            _task_payload("task-village", "Village KFC")
        )
        self.fleet.set_merchant_agent_exclusions({"kFc": ["999"]})

        reason = self.fleet._check_eligibility(
            self.fleet.get_agent("999"), village_task
        )

        self.assertEqual(reason, "merchant_exemption_C")

    def test_clearing_both_restores_eligibility(self):
        agent = self.fleet.get_agent("123")
        self.fleet.update_agent("123", blocked_merchants=[])
        self.fleet.set_merchant_agent_exclusions({})

        self.assertIsNone(self.fleet._check_eligibility(agent, self.task))

    def test_optimizer_uses_union_of_agent_and_config_sources(self):
        agent = Agent(
            id="123",
            name="Agent 123",
            current_location=OptimizerLocation(17.12, -61.84),
            blocked_merchants=["KFC St. John's"],
        )
        task = Task.from_dict(_task_payload(merchant=" kfc st. john's "))
        checker = CompatibilityChecker(
            max_distance_km=10,
            merchant_agent_exclusions={"KFC St. John's": ["123"]},
        )

        compatible, reason = checker.is_compatible(agent, task)

        self.assertFalse(compatible)
        self.assertEqual(reason, "merchant_exemption_A+C")

    def test_optimizer_blocks_case_insensitive_partial_matches(self):
        agent = Agent(
            id="123",
            name="Agent 123",
            current_location=OptimizerLocation(17.12, -61.84),
            blocked_merchants=["kfc"],
        )
        task = Task.from_dict(_task_payload(merchant="KFC Airport"))
        checker = CompatibilityChecker(
            max_distance_km=10,
            merchant_agent_exclusions={"KfC": ["123"]},
        )

        compatible, reason = checker.is_compatible(agent, task)

        self.assertFalse(compatible)
        self.assertEqual(reason, "merchant_exemption_A+C")

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
            "blocked_merchants": [],
        }])
        fleet.set_merchant_agent_exclusions({"KFC St. John's": ["999"]})
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
            checker_class.call_args.kwargs["merchant_agent_exclusions"],
            {"kfc st. john's": ["999"]},
        )


class MerchantExemptionEventTests(unittest.TestCase):
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
                "blocked_merchants": ["KFC"],
            }],
            "unassigned_tasks": [],
            "in_progress_tasks": [],
            "config": {
                "merchant_agent_exclusions": {"Subway": ["123"]}
            },
        })

        agent = app_module.fleet_state.get_agent("123")
        self.assertEqual(agent.blocked_merchants, ["KFC"])
        self.assertEqual(
            app_module.fleet_state.merchant_agent_exclusions,
            {"subway": ["123"]},
        )

        self.client.emit("agent:update", {
            "agent_id": "123",
            "blocked_merchants": [],
        })
        self.client.emit("config:update", {
            "config": {"merchant_agent_exclusions": {}},
        })

        self.assertEqual(agent.blocked_merchants, [])
        self.assertEqual(
            app_module.fleet_state.merchant_agent_exclusions, {}
        )

    def test_agent_online_persists_blocked_merchants(self):
        self.client.emit("agent:online", {
            "agent_id": "456",
            "name": "Agent 456",
            "location": [17.12, -61.84],
            "blocked_merchants": ["Subway"],
        })

        self.assertEqual(
            app_module.fleet_state.get_agent("456").blocked_merchants,
            ["Subway"],
        )


if __name__ == "__main__":
    unittest.main()
