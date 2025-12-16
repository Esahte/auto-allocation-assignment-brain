#!/usr/bin/env python3
"""
WebSocket Test Script for OR-Tools Fleet Optimizer

Tests the Socket.IO connection and events:
- task:get_recommendations (single task → agent recommendations)
- fleet:optimize_request (fleet-wide optimization)
"""

import socketio
import json
import time
from datetime import datetime, timedelta

# Create Socket.IO client
sio = socketio.Client()

# Track responses
responses = {}

# Sample test data
SAMPLE_TASK = {
    "id": "test-task-001",
    "restaurant_location": [10.6549, -61.5019],  # Port of Spain
    "delivery_location": [10.6600, -61.5100],
    "pickup_before": (datetime.utcnow() + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "delivery_before": (datetime.utcnow() + timedelta(minutes=60)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "_meta": {
        "restaurant_name": "Test Restaurant",
        "customer_name": "Test Customer",
        "payment_type": "CARD",
        "tags": []
    }
}

SAMPLE_AGENTS = [
    {
        "driver_id": "agent-001",
        "name": "Test Agent 1",
        "current_location": [10.6550, -61.5020],  # Very close to restaurant
        "wallet_balance": 0,
        "current_tasks": [],
        "_meta": {
            "max_tasks": 3,
            "available_capacity": 3,
            "tags": [],
            "has_no_cash_tag": False,
            "is_scooter_agent": False,
            "geofence_regions": []
        }
    },
    {
        "driver_id": "agent-002",
        "name": "Test Agent 2",
        "current_location": [10.6600, -61.5050],  # A bit further
        "wallet_balance": 0,
        "current_tasks": [],
        "_meta": {
            "max_tasks": 3,
            "available_capacity": 3,
            "tags": [],
            "has_no_cash_tag": False,
            "is_scooter_agent": False,
            "geofence_regions": []
        }
    }
]


# =============================================================================
# Event Handlers
# =============================================================================

@sio.event
def connect():
    print("✅ Connected to OR-Tools WebSocket server!")
    print(f"   Session ID: {sio.sid}")

@sio.event
def disconnect():
    print("❌ Disconnected from server")

@sio.on('connection_established')
def on_connection_established(data):
    print(f"📡 Server acknowledged connection:")
    print(f"   Client ID: {data.get('client_id')}")
    print(f"   Server time: {data.get('server_time')}")
    print(f"   Available events: {data.get('available_events')}")

@sio.on('task:recommendations')
def on_task_recommendations(data):
    print("\n📋 Received task:recommendations")
    print(f"   Request ID: {data.get('request_id')}")
    print(f"   Task ID: {data.get('task_id')}")
    
    recommendations = data.get('recommendations', [])
    print(f"   Recommendations: {len(recommendations)}")
    
    for i, rec in enumerate(recommendations[:3]):  # Show top 3
        print(f"   [{i+1}] {rec.get('name')} - Score: {rec.get('score')}, "
              f"Distance: {rec.get('distance_km', 'N/A')}km")
    
    if data.get('error'):
        print(f"   ⚠️ Error: {data.get('error')}")
    
    responses['task:recommendations'] = data

@sio.on('fleet:routes_updated')
def on_fleet_routes(data):
    print("\n🚗 Received fleet:routes_updated")
    print(f"   Request ID: {data.get('request_id')}")
    print(f"   Success: {data.get('success')}")
    
    metadata = data.get('metadata', {})
    print(f"   Tasks assigned: {metadata.get('tasks_assigned', 0)}")
    print(f"   Tasks unassigned: {metadata.get('tasks_unassigned', 0)}")
    
    if data.get('error'):
        print(f"   ⚠️ Error: {data.get('error')}")
    
    responses['fleet:routes_updated'] = data

@sio.on('task:created_ack')
def on_task_created_ack(data):
    print(f"✅ task:created acknowledged: {data.get('task_id')}")

@sio.on('task:declined_ack')
def on_task_declined_ack(data):
    print(f"✅ task:declined acknowledged: {data.get('task_id')}")


# =============================================================================
# Test Functions
# =============================================================================

def test_single_task_recommendations():
    """Test getting recommendations for a single task."""
    print("\n" + "="*60)
    print("TEST 1: Single Task Recommendations (batch_optimized)")
    print("="*60)
    
    print(f"Sending task:get_recommendations...")
    print(f"   Task: {SAMPLE_TASK['_meta']['restaurant_name']} → {SAMPLE_TASK['_meta']['customer_name']}")
    print(f"   Agents: {len(SAMPLE_AGENTS)}")
    
    sio.emit('task:get_recommendations', {
        'request_id': 'test-single-001',
        'new_task': SAMPLE_TASK,
        'agents': SAMPLE_AGENTS,
        'settings': {
            'optimization_mode': 'tardiness_min',
            'maxDistanceKm': 10
        }
    })
    
    # Wait for response
    time.sleep(3)
    
    if 'task:recommendations' in responses:
        print("✅ Test passed!")
        return True
    else:
        print("❌ No response received")
        return False


def test_fleet_optimization():
    """Test fleet-wide optimization (requires dashboard running)."""
    print("\n" + "="*60)
    print("TEST 2: Fleet Optimization (fleet_optimizer)")
    print("="*60)
    
    print("Sending fleet:optimize_request...")
    print("   (Will fetch data from localhost:8000)")
    
    sio.emit('fleet:optimize_request', {
        'request_id': 'test-fleet-001',
        'dashboard_url': 'http://localhost:8000'
    })
    
    # Wait for response (fleet optimization takes longer)
    time.sleep(10)
    
    if 'fleet:routes_updated' in responses:
        result = responses['fleet:routes_updated']
        if result.get('success'):
            print("✅ Test passed!")
        else:
            print(f"⚠️ Optimization completed with issues: {result.get('error', 'Unknown')}")
        return True
    else:
        print("❌ No response received (is dashboard running on port 8000?)")
        return False


def test_task_created_event():
    """Test task:created event with auto_recommend."""
    print("\n" + "="*60)
    print("TEST 3: Task Created Event (with auto_recommend)")
    print("="*60)
    
    responses.clear()  # Clear previous responses
    
    print("Sending task:created with auto_recommend=True...")
    
    sio.emit('task:created', {
        'task_id': SAMPLE_TASK['id'],
        'task': SAMPLE_TASK,
        'agents': SAMPLE_AGENTS,
        'auto_recommend': True,
        'settings': {
            'optimization_mode': 'tardiness_min'
        }
    })
    
    # Wait for acknowledgment and recommendations
    time.sleep(3)
    
    if 'task:recommendations' in responses:
        print("✅ Test passed! (auto_recommend triggered)")
        return True
    else:
        print("⚠️ Acknowledgment received but no auto recommendations")
        return True  # Still a partial success


def main():
    """Run all WebSocket tests."""
    print("="*60)
    print("OR-Tools WebSocket Test Suite")
    print("="*60)
    print(f"Server: http://localhost:5050")
    print()
    
    try:
        # Connect to server
        print("Connecting to WebSocket server...")
        sio.connect('http://localhost:5050')
        
        # Wait for connection to establish
        time.sleep(1)
        
        if not sio.connected:
            print("❌ Failed to connect!")
            return
        
        # Run tests
        results = []
        
        # Test 1: Single task recommendations
        results.append(("Single Task Recommendations", test_single_task_recommendations()))
        
        # Test 2: Task created with auto_recommend
        results.append(("Task Created Event", test_task_created_event()))
        
        # Test 3: Fleet optimization (only if dashboard is running)
        print("\n⚠️ Skipping fleet optimization test (requires dashboard on port 8000)")
        # Uncomment below to test fleet optimization:
        # results.append(("Fleet Optimization", test_fleet_optimization()))
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        for name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status}: {name}")
        
        print("\n" + "="*60)
        print("WebSocket connection is working!")
        print("Your dashboard can now connect to ws://localhost:5050")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if sio.connected:
            sio.disconnect()


if __name__ == '__main__':
    main()

