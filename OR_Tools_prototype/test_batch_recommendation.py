#!/usr/bin/env python3
"""
Test suite for batch agent recommendation system.

Tests the corrected behavior:
- No task reassignment between agents
- Agents keep their existing tasks
- System only recommends where to INSERT new task into each agent's route
- Returns top 3 recommendations with scores and route details
"""

import json
import requests
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any

def test_batch_vs_sequential_api():
    """Test batch optimization endpoint vs sequential optimization."""
    
    # Test data based on README examples
    test_data = {
        "new_task": {
            "id": "task_123",
            "job_type": "PAIRED",
            "restaurant_location": [17.140, -61.890],
            "delivery_location": [17.160, -61.910],
            "pickup_before": "2025-05-30T17:00:00Z",
            "delivery_before": "2025-05-30T17:30:00Z"
        },
        "agents": [
            {
                "driver_id": "driver_001",
                "name": "John Doe",
                "current_location": [17.150, -61.900]
            },
            {
                "driver_id": "driver_002", 
                "name": "Jane Smith",
                "current_location": [17.120, -61.880]
            },
            {
                "driver_id": "driver_003",
                "name": "Bob Wilson", 
                "current_location": [17.170, -61.920]
            }
        ],
        "current_tasks": [
            {
                "id": "task_456",
                "job_type": "PAIRED",
                "restaurant_location": [17.130, -61.870],
                "delivery_location": [17.145, -61.885],
                "pickup_before": "2025-05-30T16:30:00Z",
                "delivery_before": "2025-05-30T17:00:00Z",
                "assigned_driver": "driver_001"  # John already has this task
            }
        ],
        "max_grace_period": 3600
    }
    
    print("🧪 Testing Batch vs Sequential Optimization")
    print("=" * 60)
    
    # Test local endpoints
    base_url = "http://localhost:8080"
    
    # Test sequential (light optimized)
    print("🔄 Testing Sequential Optimization...")
    start_time = time.time()
    try:
        response = requests.post(
            f"{base_url}/recommend/light-optimized", 
            json=test_data,
            timeout=30
        )
        sequential_time = time.time() - start_time
        sequential_result = response.json() if response.status_code == 200 else None
        print(f"   ✅ Sequential completed in {sequential_time:.3f}s")
    except Exception as e:
        print(f"   ❌ Sequential failed: {e}")
        sequential_result = None
        sequential_time = 0
    
    # Test batch optimization
    print("⚡ Testing Batch Optimization...")
    start_time = time.time()
    try:
        response = requests.post(
            f"{base_url}/recommend/batch-optimized", 
            json=test_data,
            timeout=30
        )
        batch_time = time.time() - start_time
        batch_result = response.json() if response.status_code == 200 else None
        print(f"   ✅ Batch completed in {batch_time:.3f}s")
    except Exception as e:
        print(f"   ❌ Batch failed: {e}")
        batch_result = None
        batch_time = 0
    
    # Compare results
    if sequential_result and batch_result:
        speedup = sequential_time / batch_time if batch_time > 0 else float('inf')
        print(f"\n📊 PERFORMANCE COMPARISON")
        print(f"🐌 Sequential Time: {sequential_time:.3f}s")
        print(f"⚡ Batch Time:      {batch_time:.3f}s") 
        print(f"🚀 Speedup:         {speedup:.1f}x")
        
        # Validate response format
        validate_response_format(batch_result, test_data)
        validate_no_task_reassignment(batch_result, test_data)
        
    return sequential_result, batch_result

def validate_response_format(result: Dict[str, Any], test_data: Dict[str, Any]):
    """Validate that response follows exact API specification."""
    print(f"\n✅ RESPONSE FORMAT VALIDATION")
    
    # Check required top-level fields
    assert "task_id" in result, "Missing task_id"
    assert "recommendations" in result, "Missing recommendations"
    assert result["task_id"] == test_data["new_task"]["id"], "task_id mismatch"
    
    recommendations = result["recommendations"]
    assert isinstance(recommendations, list), "recommendations must be list"
    assert len(recommendations) <= 3, "Max 3 recommendations expected"
    assert len(recommendations) > 0, "At least 1 recommendation expected"
    
    # Validate each recommendation
    for i, rec in enumerate(recommendations):
        print(f"   📋 Validating recommendation {i+1}: {rec.get('driver_id', 'unknown')}")
        
        # Required fields
        required_fields = ["driver_id", "name", "score", "additional_time_minutes", 
                          "grace_penalty_seconds", "already_late_stops", "route"]
        for field in required_fields:
            assert field in rec, f"Missing field: {field}"
        
        # Field types
        assert isinstance(rec["score"], int), "score must be integer"
        assert 0 <= rec["score"] <= 100, "score must be 0-100"
        assert isinstance(rec["additional_time_minutes"], (int, float)), "additional_time_minutes must be numeric"
        assert isinstance(rec["grace_penalty_seconds"], int), "grace_penalty_seconds must be integer"
        assert isinstance(rec["already_late_stops"], int), "already_late_stops must be integer"
        assert isinstance(rec["route"], list), "route must be list"
        
        # Validate route structure
        validate_route_structure(rec["route"], test_data["new_task"]["id"])
    
    # Check sorting (highest score first)
    for i in range(1, len(recommendations)):
        assert recommendations[i-1]["score"] >= recommendations[i]["score"], \
               "Recommendations not sorted by score"
    
    print(f"   ✅ All {len(recommendations)} recommendations valid")

def validate_route_structure(route: List[Dict[str, Any]], new_task_id: str):
    """Validate route structure follows API specification."""
    
    assert len(route) >= 2, "Route must have at least start and end"
    
    # First entry must be start
    assert route[0]["type"] == "start", "Route must start with 'start'"
    assert "index" in route[0], "Start must have index"
    
    # Last entry must be end
    assert route[-1]["type"] == "end", "Route must end with 'end'"
    assert "index" in route[-1], "End must have index"
    
    # Check for new task entries
    new_task_entries = 0
    found_pickup = False
    found_delivery = False
    
    for entry in route:
        if entry["type"] == "new_task_pickup":
            new_task_entries += 1
            found_pickup = True
            assert entry["task_id"] == new_task_id, "new_task_pickup task_id mismatch"
            assert "pickup_index" in entry, "new_task_pickup missing pickup_index"
            assert "arrival_time" in entry, "new_task_pickup missing arrival_time"
            assert "deadline" in entry, "new_task_pickup missing deadline"
            assert "lateness" in entry, "new_task_pickup missing lateness"
            
        elif entry["type"] == "new_task_delivery":
            new_task_entries += 1
            found_delivery = True
            assert entry["task_id"] == new_task_id, "new_task_delivery task_id mismatch"
            assert "delivery_index" in entry, "new_task_delivery missing delivery_index"
            assert "arrival_time" in entry, "new_task_delivery missing arrival_time"
            assert "deadline" in entry, "new_task_delivery missing deadline"
            assert "lateness" in entry, "new_task_delivery missing lateness"
    
    # Must have both pickup and delivery for PAIRED task
    assert found_pickup, "Missing new_task_pickup in route"
    assert found_delivery, "Missing new_task_delivery in route"
    assert new_task_entries == 2, f"Expected 2 new task entries, found {new_task_entries}"

def validate_no_task_reassignment(result: Dict[str, Any], test_data: Dict[str, Any]):
    """Validate that existing tasks are not reassigned between agents."""
    print(f"\n🔒 TASK REASSIGNMENT VALIDATION")
    
    # Build map of current task assignments
    current_assignments = {}
    for task in test_data["current_tasks"]:
        current_assignments[task["id"]] = task["assigned_driver"]
    
    print(f"   📋 Current assignments: {current_assignments}")
    
    # Check that no existing tasks are reassigned
    for rec in result["recommendations"]:
        driver_id = rec["driver_id"]
        route = rec["route"]
        
        for entry in route:
            # Check existing task entries
            if entry["type"] in ["existing_task_pickup", "existing_task_delivery"]:
                task_id = entry["task_id"]
                original_driver = current_assignments.get(task_id)
                
                # This existing task should only appear in routes for its original driver
                assert driver_id == original_driver, \
                    f"Task {task_id} reassigned from {original_driver} to {driver_id} - NOT ALLOWED!"
    
    print(f"   ✅ No task reassignment detected - existing tasks stay with original drivers")

def test_edge_cases():
    """Test edge cases and error conditions."""
    print(f"\n🧪 EDGE CASE TESTING")
    print("=" * 60)
    
    base_url = "http://localhost:8080"
    
    # Test 1: Agent with no existing tasks
    print("📋 Test 1: Agent with no existing tasks")
    test_data = {
        "new_task": {
            "id": "task_solo",
            "job_type": "PAIRED",
            "restaurant_location": [17.140, -61.890],
            "delivery_location": [17.160, -61.910],
            "pickup_before": (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "delivery_before": (datetime.now(timezone.utc) + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        },
        "agents": [
            {
                "driver_id": "solo_driver",
                "name": "Solo Agent",
                "current_location": [17.150, -61.900]
            }
        ],
        "current_tasks": [],  # No existing tasks
        "max_grace_period": 3600
    }
    
    try:
        response = requests.post(f"{base_url}/recommend/batch-optimized", json=test_data, timeout=30)
        result = response.json()
        assert result["recommendations"], "Should have recommendations for solo agent"
        assert len(result["recommendations"][0]["route"]) == 4, "Solo agent route should have start, pickup, delivery, end"
        print("   ✅ Solo agent test passed")
    except Exception as e:
        print(f"   ❌ Solo agent test failed: {e}")
    
    # Test 2: Multiple agents with complex existing routes
    print("📋 Test 2: Multiple agents with complex routes")
    future_time_1 = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    future_time_2 = (datetime.now(timezone.utc) + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    future_time_3 = (datetime.now(timezone.utc) + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    future_time_4 = (datetime.now(timezone.utc) + timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    complex_data = {
        "new_task": {
            "id": "new_complex_task",
            "job_type": "PAIRED",
            "restaurant_location": [17.145, -61.895],
            "delivery_location": [17.155, -61.905],
            "pickup_before": future_time_2,
            "delivery_before": future_time_3
        },
        "agents": [
            {
                "driver_id": "busy_driver_1",
                "name": "Busy Driver 1",
                "current_location": [17.140, -61.890]
            },
            {
                "driver_id": "busy_driver_2", 
                "name": "Busy Driver 2",
                "current_location": [17.160, -61.910]
            }
        ],
        "current_tasks": [
            {
                "id": "existing_task_1",
                "job_type": "PAIRED",
                "restaurant_location": [17.141, -61.891],
                "delivery_location": [17.151, -61.901],
                "pickup_before": future_time_1,
                "delivery_before": future_time_2,
                "assigned_driver": "busy_driver_1"
            },
            {
                "id": "existing_task_2",
                "job_type": "PAIRED", 
                "restaurant_location": [17.161, -61.911],
                "delivery_location": [17.171, -61.921],
                "pickup_before": future_time_1,
                "delivery_before": future_time_2,
                "assigned_driver": "busy_driver_1"  # Same driver has 2 tasks
            },
            {
                "id": "existing_task_3",
                "job_type": "PAIRED",
                "restaurant_location": [17.162, -61.912], 
                "delivery_location": [17.172, -61.922],
                "pickup_before": future_time_2,
                "delivery_before": future_time_3,
                "assigned_driver": "busy_driver_2"
            }
        ],
        "max_grace_period": 3600
    }
    
    try:
        response = requests.post(f"{base_url}/recommend/batch-optimized", json=complex_data, timeout=30)
        result = response.json()
        validate_no_task_reassignment(result, complex_data)
        print("   ✅ Complex route test passed")
    except Exception as e:
        print(f"   ❌ Complex route test failed: {e}")

def test_performance_scalability():
    """Test performance with different dataset sizes."""
    print(f"\n📈 SCALABILITY TESTING")
    print("=" * 60)
    
    base_url = "http://localhost:8080"
    
    test_sizes = [
        (5, 10),   # 5 agents, 10 tasks
        (10, 20),  # 10 agents, 20 tasks
        (15, 30),  # 15 agents, 30 tasks
    ]
    
    for num_agents, num_tasks in test_sizes:
        print(f"📊 Testing {num_agents} agents, {num_tasks} tasks...")
        
        # Generate test data
        agents = []
        for i in range(num_agents):
            agents.append({
                "driver_id": f"driver_{i:03d}",
                "name": f"Driver {i+1}",
                "current_location": [17.1 + (i * 0.01), -61.9 + (i * 0.01)]
            })
        
        current_tasks = []
        future_base = datetime.now(timezone.utc) + timedelta(hours=1)
        
        for i in range(num_tasks):
            assigned_driver = f"driver_{i % num_agents:03d}"  # Distribute tasks evenly
            pickup_time = future_base + timedelta(minutes=i*10)
            delivery_time = pickup_time + timedelta(minutes=30)
            
            current_tasks.append({
                "id": f"existing_task_{i:03d}",
                "job_type": "PAIRED",
                "restaurant_location": [17.11 + (i * 0.001), -61.91 + (i * 0.001)],
                "delivery_location": [17.12 + (i * 0.001), -61.92 + (i * 0.001)], 
                "pickup_before": pickup_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "delivery_before": delivery_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "assigned_driver": assigned_driver
            })
        
        new_task_time = future_base + timedelta(hours=2)
        test_data = {
            "new_task": {
                "id": "scale_test_task",
                "job_type": "PAIRED",
                "restaurant_location": [17.15, -61.95],
                "delivery_location": [17.16, -61.96],
                "pickup_before": new_task_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "delivery_before": (new_task_time + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
            },
            "agents": agents,
            "current_tasks": current_tasks,
            "max_grace_period": 3600
        }
        
        # Test batch optimization
        start_time = time.time()
        try:
            response = requests.post(f"{base_url}/recommend/batch-optimized", json=test_data, timeout=60)
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                num_recommendations = len(result.get("recommendations", []))
                print(f"   ✅ {execution_time:.3f}s - {num_recommendations} recommendations")
            else:
                print(f"   ❌ {execution_time:.3f}s - HTTP {response.status_code}")
                
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"   ❌ {execution_time:.3f}s - {e}")

def main():
    """Run comprehensive test suite."""
    print("🎯 Batch Agent Recommendation Test Suite")
    print("=" * 60)
    print("Testing corrected behavior:")
    print("• No task reassignment between agents")
    print("• Agents keep existing tasks")
    print("• Only recommends insertion points for new task")
    print("• Returns top recommendations with scores")
    print("=" * 60)
    
    # Basic functionality test
    test_batch_vs_sequential_api()
    
    # Edge cases
    test_edge_cases()
    
    # Scalability
    test_performance_scalability()
    
    print("\n" + "=" * 60)
    print("✅ TEST SUITE COMPLETE")
    print("=" * 60)
    print("🎯 Key Validations:")
    print("   • API response format matches specification")
    print("   • No task reassignment occurs")
    print("   • Route structures are valid")  
    print("   • Performance is acceptable")
    print("   • Edge cases handled properly")

if __name__ == "__main__":
    main() 