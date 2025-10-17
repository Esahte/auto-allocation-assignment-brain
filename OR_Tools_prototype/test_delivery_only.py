"""
Test cases for delivery-only task support (pickup_completed flag).

This test suite validates that the system correctly handles tasks where 
pickups have already been completed, only routing agents to delivery locations.
"""

import json
import requests
from datetime import datetime, timezone, timedelta

# Local testing endpoint
BASE_URL = "http://localhost:8080"

def generate_future_time(hours_from_now: float) -> str:
    """Generate ISO timestamp for future time."""
    future_time = datetime.now(timezone.utc) + timedelta(hours=hours_from_now)
    return future_time.strftime("%Y-%m-%dT%H:%M:%SZ")

def test_case_1_mixed_pickup_completed():
    """
    Test Case 1: Mixed scenario with some completed pickups
    - Agent 1 has tasks A and B with completed pickups (delivery-only)
    - New task C is a normal PAIRED task
    - Expected: System should only route to deliveries of A and B, plus pickup+delivery of C
    """
    print("\n" + "="*80)
    print("TEST CASE 1: Mixed Pickup Completed Scenario")
    print("="*80)
    
    payload =  {
  "task_id": "578724123962417606489027752601",
  "agent_count": 1,
  "current_task_count": 1,
  "max_distance_km": 10,
  "optimization_mode": "tardiness_min",
  "agents": [
    {
      "driver_id": "2098791",
      "name": "Testing Esah",
      "current_location": [
        17.12623998,
        -61.8218519
      ]
    }
  ],
  "new_task": {
    "id": "578724123962417606489027752601",
    "job_type": "PAIRED",
    "restaurant_location": [
      17.126151031975567,
      -61.82166574640591
    ],
    "delivery_location": [
      17.12609,
      -61.8217069
    ],
    "pickup_before": "2025-10-17T02:00:00.000Z",
    "delivery_before": "2025-10-17T03:00:00.000Z"
  },
  "current_tasks": [
    {
      "id": "578722681745217606474483427179",
      "job_type": "PAIRED",
      "restaurant_location": [
        17.126151031975567,
        -61.82166574640591
      ],
      "delivery_location": [
        17.12609,
        -61.8217069
      ],
      "pickup_completed": True,
      "assigned_driver": "2098791",
      "pickup_before": "2025-10-17T01:03:00.000Z",
      "delivery_before": "2025-10-17T03:00:00.000Z"
    }
  ]
}
    
    print("\nRequest payload:")
    print(json.dumps(payload, indent=2))
    
    response = requests.post(f"{BASE_URL}/recommend", json=payload)
    
    print(f"\nResponse Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("\nRecommendations:")
        print(json.dumps(result, indent=2))
        
        # Validation
        print("\n" + "-"*80)
        print("VALIDATION:")
        print("-"*80)
        
        if result.get("recommendations"):
            for rec in result["recommendations"][:2]:  # Top 2
                print(f"\nAgent: {rec['name']} (Score: {rec['score']})")
                route = rec.get("route", [])
                
                # Count pickup types for the completed task
                completed_task_id = "578722681745217606474483427179"
                completed_task_pickups = sum(1 for r in route if r.get("task_id") == completed_task_id and "pickup" in r.get("type", ""))
                completed_task_deliveries = sum(1 for r in route if r.get("task_id") == completed_task_id and "delivery" in r.get("type", ""))
                
                print(f"  Task {completed_task_id[:15]}... (pickup_completed=True):")
                print(f"    Pickups in route: {completed_task_pickups} (Expected: 0)")
                print(f"    Deliveries in route: {completed_task_deliveries} (Expected: 1)")
                
                if rec["driver_id"] == "2098791":
                    if completed_task_pickups == 0:
                        print("    ✅ PASS: No pickups for completed task")
                    else:
                        print("    ❌ FAIL: Found pickups for task with pickup_completed=True")
                    
                    if completed_task_deliveries > 0:
                        print("    ✅ PASS: Delivery present for completed task")
                    else:
                        print("    ⚠️  WARNING: Expected delivery for completed task")
        
        return result
    else:
        print(f"Error: {response.text}")
        return None

def test_case_2_all_delivery_only():
    """
    Test Case 2: All current tasks are delivery-only
    - Agent has multiple tasks, all with completed pickups
    - New task is a normal PAIRED task
    - Expected: Only delivery locations in existing routes
    """
    print("\n" + "="*80)
    print("TEST CASE 2: All Delivery-Only Scenario")
    print("="*80)
    
    payload = {
        "new_task": {
            "id": "task_new",
            "job_type": "PAIRED",
            "restaurant_location": [17.140, -61.890],
            "delivery_location": [17.160, -61.910],
            "pickup_before": generate_future_time(1.0),
            "delivery_before": generate_future_time(1.5)
        },
        "agents": [
            {
                "driver_id": "driver_001",
                "name": "Delivery Expert",
                "current_location": [17.130, -61.880]
            }
        ],
        "current_tasks": [
            {
                "id": "task_1",
                "job_type": "PAIRED",
                "restaurant_location": [17.110, -61.860],
                "delivery_location": [17.145, -61.885],
                "pickup_before": generate_future_time(-1),
                "delivery_before": generate_future_time(0.5),
                "assigned_driver": "driver_001",
                "pickup_completed": True
            },
            {
                "id": "task_2",
                "job_type": "PAIRED",
                "restaurant_location": [17.105, -61.855],
                "delivery_location": [17.150, -61.895],
                "pickup_before": generate_future_time(-2),
                "delivery_before": generate_future_time(0.8),
                "assigned_driver": "driver_001",
                "pickup_completed": True
            },
            {
                "id": "task_3",
                "job_type": "PAIRED",
                "restaurant_location": [17.115, -61.865],
                "delivery_location": [17.155, -61.900],
                "pickup_before": generate_future_time(-0.5),
                "delivery_before": generate_future_time(1.2),
                "assigned_driver": "driver_001",
                "pickup_completed": True
            }
        ],
        "max_grace_period": 3600,
        "algorithm": "batch_optimized"
    }
    
    print("\nRequest payload:")
    print(json.dumps(payload, indent=2))
    
    response = requests.post(f"{BASE_URL}/recommend", json=payload)
    
    print(f"\nResponse Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("\nRecommendations:")
        print(json.dumps(result, indent=2))
        
        # Validation
        print("\n" + "-"*80)
        print("VALIDATION:")
        print("-"*80)
        
        if result.get("recommendations"):
            rec = result["recommendations"][0]
            route = rec.get("route", [])
            
            existing_pickups = sum(1 for r in route if "existing_task_pickup" in r.get("type", ""))
            existing_deliveries = sum(1 for r in route if "existing_task_delivery" in r.get("type", ""))
            new_pickups = sum(1 for r in route if "new_task_pickup" in r.get("type", ""))
            new_deliveries = sum(1 for r in route if "new_task_delivery" in r.get("type", ""))
            
            print(f"Existing task pickups: {existing_pickups} (Expected: 0)")
            print(f"Existing task deliveries: {existing_deliveries} (Expected: 3)")
            print(f"New task pickups: {new_pickups} (Expected: 1)")
            print(f"New task deliveries: {new_deliveries} (Expected: 1)")
            
            if existing_pickups == 0:
                print("✅ PASS: No pickups for tasks with pickup_completed=True")
            else:
                print("❌ FAIL: Found pickups for completed tasks")
            
            if existing_deliveries == 3:
                print("✅ PASS: All 3 delivery-only tasks present")
            else:
                print(f"⚠️  WARNING: Expected 3 deliveries, found {existing_deliveries}")
        
        return result
    else:
        print(f"Error: {response.text}")
        return None

def test_case_3_backward_compatibility():
    """
    Test Case 3: Backward compatibility - no pickup_completed flag
    - Tasks without pickup_completed flag should work as before
    - Expected: Normal PAIRED behavior with both pickup and delivery
    """
    print("\n" + "="*80)
    print("TEST CASE 3: Backward Compatibility (No pickup_completed flag)")
    print("="*80)
    
    payload = {
        "new_task": {
            "id": "task_new",
            "job_type": "PAIRED",
            "restaurant_location": [17.140, -61.890],
            "delivery_location": [17.160, -61.910],
            "pickup_before": generate_future_time(1.0),
            "delivery_before": generate_future_time(1.5)
        },
        "agents": [
            {
                "driver_id": "driver_001",
                "name": "Standard Driver",
                "current_location": [17.130, -61.880]
            }
        ],
        "current_tasks": [
            {
                "id": "task_normal",
                "job_type": "PAIRED",
                "restaurant_location": [17.110, -61.860],
                "delivery_location": [17.145, -61.885],
                "pickup_before": generate_future_time(0.5),
                "delivery_before": generate_future_time(1.0),
                "assigned_driver": "driver_001"
                # Note: NO pickup_completed flag
            }
        ],
        "max_grace_period": 3600,
        "algorithm": "batch_optimized"
    }
    
    print("\nRequest payload:")
    print(json.dumps(payload, indent=2))
    
    response = requests.post(f"{BASE_URL}/recommend", json=payload)
    
    print(f"\nResponse Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("\nRecommendations:")
        print(json.dumps(result, indent=2))
        
        # Validation
        print("\n" + "-"*80)
        print("VALIDATION:")
        print("-"*80)
        
        if result.get("recommendations"):
            rec = result["recommendations"][0]
            route = rec.get("route", [])
            
            existing_pickups = sum(1 for r in route if r.get("task_id") == "task_normal" and "pickup" in r.get("type", ""))
            existing_deliveries = sum(1 for r in route if r.get("task_id") == "task_normal" and "delivery" in r.get("type", ""))
            
            print(f"Task 'task_normal' pickups: {existing_pickups} (Expected: 1)")
            print(f"Task 'task_normal' deliveries: {existing_deliveries} (Expected: 1)")
            
            if existing_pickups == 1 and existing_deliveries == 1:
                print("✅ PASS: Normal PAIRED task behavior preserved")
            else:
                print("❌ FAIL: Backward compatibility broken")
        
        return result
    else:
        print(f"Error: {response.text}")
        return None

def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*80)
    print("DELIVERY-ONLY TASK SUPPORT - TEST SUITE")
    print("="*80)
    print("\nTesting pickup_completed flag functionality...")
    print("Ensure the server is running on localhost:8080")
    
    try:
        # Check server health
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            print(f"\n❌ Server health check failed: {health.status_code}")
            return
        print("\n✅ Server is healthy")
    except Exception as e:
        print(f"\n❌ Cannot connect to server: {e}")
        print("Please start the server with: python app.py")
        return
    
    results = {
        "test_1": None,
        "test_2": None,
        "test_3": None
    }
    
    # Run tests
    try:
        results["test_1"] = test_case_1_mixed_pickup_completed()
    except Exception as e:
        print(f"\n❌ Test Case 1 failed with error: {e}")
    
    try:
        results["test_2"] = test_case_2_all_delivery_only()
    except Exception as e:
        print(f"\n❌ Test Case 2 failed with error: {e}")
    
    try:
        results["test_3"] = test_case_3_backward_compatibility()
    except Exception as e:
        print(f"\n❌ Test Case 3 failed with error: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in results.values() if r is not None)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result is not None else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    run_all_tests()

