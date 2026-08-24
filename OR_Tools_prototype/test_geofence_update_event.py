import unittest

import app as app_module


POLYGON = [[17.0, -62.0], [17.2, -62.0], [17.2, -61.7], [17.0, -61.7]]


def payload(request_id='region-update-1', sharing_mode='open_access'):
    return {
        'request_id': request_id,
        'geofences': [{
            'id': 'region-1',
            'name': 'Central',
            'polygon': POLYGON,
            'agent_ids': ['agent-1'],
            'sharing_mode': sharing_mode,
            'route_coverage': 'entire_route',
        }],
    }


class GeofenceUpdateEventTests(unittest.TestCase):
    def setUp(self):
        app_module.fleet_state.clear()
        app_module.fleet_state._geofences = {}
        self.client = app_module.socketio.test_client(app_module.app)

    def tearDown(self):
        self.client.disconnect()
        app_module.fleet_state._geofences = {}

    def _ack(self):
        events = self.client.get_received()
        return next(
            event['args'][0]
            for event in events
            if event['name'] == 'geofences:update_ack'
        )

    def test_live_update_applies_policy_and_acknowledges_request(self):
        self.client.emit('geofences:update', payload())
        ack = self._ack()

        self.assertTrue(ack['success'])
        self.assertEqual(ack['request_id'], 'region-update-1')
        self.assertEqual(ack['synced']['geofences'], 1)
        stored = app_module.fleet_state._geofences['region-1']
        self.assertEqual(stored['sharing_mode'], 'open_access')
        self.assertEqual(stored['route_coverage'], 'entire_route')

    def test_invalid_enum_is_rejected_without_replacing_previous_state(self):
        app_module.fleet_state.sync_geofences(payload()['geofences'])
        self.client.emit('geofences:update', payload('region-update-bad', 'invalid'))
        ack = self._ack()

        self.assertFalse(ack['success'])
        self.assertEqual(ack['request_id'], 'region-update-bad')
        self.assertIn('Invalid sharing_mode', ack['error'])
        self.assertEqual(
            app_module.fleet_state._geofences['region-1']['sharing_mode'],
            'open_access',
        )

    def test_export_keeps_policy_fields_for_solver_snapshot(self):
        app_module.fleet_state.sync_geofences(payload()['geofences'])
        exported = app_module._export_geofence_data()
        self.assertEqual(exported[0]['sharing_mode'], 'open_access')
        self.assertEqual(exported[0]['route_coverage'], 'entire_route')


if __name__ == '__main__':
    unittest.main()
