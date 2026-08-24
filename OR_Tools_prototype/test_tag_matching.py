import unittest
from datetime import datetime, timedelta, timezone

from fleet_optimizer import Agent, CompatibilityChecker, Location as OptimizerLocation, Task
from fleet_state import FleetState, _extract_incoming_task_tags


def _task_payload(task_id="task-1", tags=None, custom_fields=None, meta=None):
    now = datetime.now(timezone.utc)
    payload = {
        "id": task_id,
        "job_type": "PAIRED",
        "restaurant_location": [17.12, -61.84],
        "delivery_location": [17.13, -61.85],
        "pickup_before": (now + timedelta(minutes=10)).isoformat(),
        "delivery_before": (now + timedelta(minutes=40)).isoformat(),
        "tags": tags or [],
        "_meta": {"restaurant_name": "AllMart Oasis Water Delivery"},
    }
    if custom_fields is not None:
        payload["custom_fields"] = custom_fields
    if meta is not None:
        payload["_meta"].update(meta)
    return payload


class ExtractIncomingTaskTagsTests(unittest.TestCase):
    def test_includes_task_category(self):
        tags = _extract_incoming_task_tags({
            "tags": [],
            "custom_fields": {"Task_Category": "HEAVY"},
        })
        self.assertEqual(tags, ["HEAVY"])

    def test_merges_tookan_tags_and_category(self):
        tags = _extract_incoming_task_tags({
            "tags": ["TEST"],
            "custom_fields": {"Task_Category": "HEAVY"},
        })
        self.assertEqual(tags, ["TEST", "HEAVY"])

    def test_reads_meta_tags(self):
        tags = _extract_incoming_task_tags(
            {"tags": []},
            {"tags": ["AUA"]},
        )
        self.assertEqual(tags, ["AUA"])


class FleetStateTagEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.fleet = FleetState(max_distance_km=10)
        self.fleet.sync_agents([
            {
                "id": "1975009",
                "name": "Orlan Wilson",
                "location": [17.12, -61.84],
                "status": "online",
                "tags": [],
            },
            {
                "id": "heavy-1",
                "name": "Heavy Agent",
                "location": [17.12, -61.84],
                "status": "online",
                "tags": ["HEAVY"],
            },
        ])

    def test_category_only_task_blocks_untagged_agent(self):
        task = self.fleet.add_task(_task_payload(
            custom_fields={"Task_Category": "HEAVY"}
        ))
        self.assertEqual(task.tags, ["HEAVY"])
        reason = self.fleet._check_eligibility(
            self.fleet.get_agent("1975009"), task, override_max_distance_km=10
        )
        self.assertEqual(reason, "tag_mismatch")

    def test_category_only_task_allows_matching_agent(self):
        task = self.fleet.add_task(_task_payload(
            custom_fields={"Task_Category": "HEAVY"}
        ))
        reason = self.fleet._check_eligibility(
            self.fleet.get_agent("heavy-1"), task, override_max_distance_km=10
        )
        self.assertIsNone(reason)

    def test_untagged_task_allows_untagged_agent(self):
        task = self.fleet.add_task(_task_payload())
        reason = self.fleet._check_eligibility(
            self.fleet.get_agent("1975009"), task, override_max_distance_km=10
        )
        self.assertIsNone(reason)


class OptimizerTagSourceTests(unittest.TestCase):
    def _base_payload(self):
        now = datetime.now(timezone.utc)
        return {
            "id": "task-1",
            "restaurant_location": [17.12, -61.84],
            "delivery_location": [17.13, -61.85],
            "pickup_before": (now + timedelta(minutes=10)).isoformat(),
            "delivery_before": (now + timedelta(minutes=40)).isoformat(),
        }

    def test_reads_top_level_tags(self):
        payload = self._base_payload()
        payload["tags"] = ["HEAVY"]
        payload["_meta"] = {"payment_type": "CASH"}
        task = Task.from_dict(payload)
        self.assertEqual(task.tags, ["HEAVY"])

    def test_reads_meta_tags(self):
        payload = self._base_payload()
        payload["_meta"] = {"tags": ["HEAVY"], "payment_type": "CASH"}
        task = Task.from_dict(payload)
        self.assertEqual(task.tags, ["HEAVY"])

    def test_tag_mismatch_in_optimizer(self):
        now = datetime.now(timezone.utc)
        task = Task.from_dict({
            "id": "task-1",
            "restaurant_location": [17.12, -61.84],
            "delivery_location": [17.13, -61.85],
            "pickup_before": (now + timedelta(minutes=10)).isoformat(),
            "delivery_before": (now + timedelta(minutes=40)).isoformat(),
            "tags": ["HEAVY"],
            "_meta": {"payment_type": "CARD", "tags": ["HEAVY"]},
        })
        agent = Agent(
            id="1975009",
            name="Orlan Wilson",
            current_location=OptimizerLocation(17.12, -61.84),
            tags=[],
        )
        checker = CompatibilityChecker(wallet_threshold=9999, max_distance_km=10)
        is_compat, reason = checker.is_compatible(agent, task)
        self.assertFalse(is_compat)
        self.assertEqual(reason, "tag_mismatch")


if __name__ == "__main__":
    unittest.main()
