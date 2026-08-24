"""Policy matrix tests shared by FleetState and CompatibilityChecker."""

import unittest
from dataclasses import dataclass

from region_policy import evaluate_region_eligibility, matching_regions


@dataclass
class Location:
    lat: float
    lng: float


@dataclass
class Task:
    restaurant_location: Location
    delivery_location: Location
    job_type: str = 'PAIRED'


def square(minimum, maximum):
    return [
        [minimum, minimum],
        [maximum, minimum],
        [maximum, maximum],
        [minimum, maximum],
    ]


def region(region_id, polygon, agents, sharing='region_first', coverage='any_endpoint'):
    return {
        'id': region_id,
        'name': region_id,
        'polygon': polygon,
        'agent_ids': agents,
        'sharing_mode': sharing,
        'route_coverage': coverage,
    }


class TestRegionPolicy(unittest.TestCase):
    def test_entire_route_and_any_endpoint(self):
        task = Task(Location(5, 5), Location(20, 20))
        regions = [
            region('entire', square(0, 10), [], 'open_access', 'entire_route'),
            region('any', square(0, 10), [], 'open_access', 'any_endpoint'),
        ]
        self.assertEqual(
            [item['id'] for item in matching_regions(task, regions)],
            ['any'],
        )

    def test_partial_task_uses_only_active_endpoint(self):
        coverage = [region('zone', square(0, 10), [], 'open_access', 'entire_route')]
        delivery_only = Task(Location(20, 20), Location(5, 5), 'DELIVERY_ONLY')
        pickup_only = Task(Location(5, 5), Location(20, 20), 'PICKUP_ONLY')
        self.assertEqual(len(matching_regions(delivery_only, coverage)), 1)
        self.assertEqual(len(matching_regions(pickup_only, coverage)), 1)

    def test_region_first_beats_open_access_in_overlap(self):
        task = Task(Location(5, 5), Location(6, 6))
        regions = [
            region('open', square(0, 10), ['open-member'], 'open_access'),
            region('first', square(0, 10), ['first-member'], 'region_first'),
        ]
        blocked = evaluate_region_eligibility(
            'global', task, regions, ['global', 'first-member']
        )
        open_member = evaluate_region_eligibility(
            'open-member', task, regions, ['open-member', 'first-member']
        )
        first_member = evaluate_region_eligibility(
            'first-member', task, regions, ['open-member', 'first-member']
        )
        self.assertFalse(blocked['eligible'])
        self.assertTrue(open_member['eligible'])
        self.assertFalse(open_member['distance_bypass'])
        self.assertTrue(first_member['distance_bypass'])

    def test_unstaffed_region_first_falls_back_globally(self):
        task = Task(Location(5, 5), Location(20, 20))
        regions = [region('first', square(0, 10), ['offline-member'])]
        decision = evaluate_region_eligibility('global', task, regions, ['global'])
        self.assertTrue(decision['eligible'])
        self.assertFalse(decision['restrict_to_members'])

    def test_cross_region_endpoints_use_member_union(self):
        task = Task(Location(5, 5), Location(25, 25))
        regions = [
            region('pickup', square(0, 10), ['pickup-agent']),
            region('delivery', square(20, 30), ['delivery-agent']),
        ]
        online = ['pickup-agent', 'delivery-agent']
        self.assertTrue(evaluate_region_eligibility('pickup-agent', task, regions, online)['eligible'])
        self.assertTrue(evaluate_region_eligibility('delivery-agent', task, regions, online)['eligible'])

    def test_multi_region_agent_is_reverse_locked_to_any_subscription(self):
        regions = [
            region('one', square(0, 10), ['multi']),
            region('two', square(20, 30), ['multi']),
        ]
        in_second = Task(Location(25, 25), Location(40, 40))
        outside = Task(Location(40, 40), Location(45, 45))
        self.assertTrue(evaluate_region_eligibility('multi', in_second, regions, ['multi'])['eligible'])
        decision = evaluate_region_eligibility('multi', outside, regions, ['multi'])
        self.assertFalse(decision['eligible'])
        self.assertEqual(decision['reason'], 'agent_restricted_to_region')

    def test_invalid_enum_is_rejected(self):
        invalid = [region('bad', square(0, 10), [], sharing='invalid')]
        with self.assertRaises(ValueError):
            matching_regions(Task(Location(5, 5), Location(6, 6)), invalid)


if __name__ == '__main__':
    unittest.main()
