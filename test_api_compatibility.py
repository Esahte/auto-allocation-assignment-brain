#!/usr/bin/env python3
"""
Test script to verify API compatibility with original documentation format.
Tests both fixed_optimized and light_optimized algorithms.
"""

import json
import requests
import time
from typing import Dict, Any

# API Configuration
BASE_URL = "https://or-tools-recommender-95621826490.us-central1.run.app"

def create_test_request() -> Dict[str, Any]:
    """Create test request matching old API documentation format exactly."""
    return {
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
                "assigned_driver": "driver_001"
            }
        ],
        "max_grace_period": 3600
    }

def validate_response_format(response_data: Dict[str, Any], algorithm_name: str) -> bool:
    """Validate that response matches old API documentation format exactly."""
    print(f"\n🔍 Validating {algorithm_name} response format...")
    
    # Check top-level structure
    required_top_level = ["task_id", "recommendations"]
    for field in required_top_level:
        if field not in response_data:
            print(f"❌ Missing required top-level field: {field}")
            return False
    
    # Check for unexpected top-level fields (shouldn't have execution_time_seconds, cache_hits, etc.)
    allowed_top_level = ["task_id", "recommendations", "error"]
    for field in response_data:
        if field not in allowed_top_level:
            print(f"❌ Unexpected top-level field found: {field}")
            return False
    
    print(f"✅ Top-level structure correct")
    
    # Validate recommendations structure
    recommendations = response_data.get("recommendations", [])
    if not isinstance(recommendations, list):
        print(f"❌ Recommendations should be a list")
        return False
    
    if not recommendations:
        print(f"⚠️  No recommendations returned")
        return True
    
    # Check first recommendation structure
    rec = recommendations[0]
    required_rec_fields = ["driver_id", "name", "score", "additional_time_minutes", 
                          "grace_penalty_seconds", "already_late_stops", "route"]
    
    for field in required_rec_fields:
        if field not in rec:
            print(f"❌ Missing required recommendation field: {field}")
            return False
    
    # Check for unexpected recommendation fields
    for field in rec:
        if field not in required_rec_fields:
            print(f"❌ Unexpected recommendation field: {field}")
            return False
    
    print(f"✅ Recommendation structure correct")
    
    # Validate route structure
    route = rec.get("route", [])
    if not isinstance(route, list):
        print(f"❌ Route should be a list")
        return False
    
    if route:
        # Check route entry structure
        for i, entry in enumerate(route):
            entry_type = entry.get("type", "")
            
            if entry_type == "start":
                if "index" not in entry:
                    print(f"❌ Start entry missing 'index' field")
                    return False
            elif entry_type == "end":
                if "index" not in entry:
                    print(f"❌ End entry missing 'index' field")
                    return False
            elif "pickup" in entry_type:
                if "pickup_index" not in entry:
                    print(f"❌ Pickup entry missing 'pickup_index' field")
                    return False
                if "arrival_time" not in entry:
                    print(f"❌ Pickup entry missing 'arrival_time' field")
                    return False
            elif "delivery" in entry_type:
                if "delivery_index" not in entry:
                    print(f"❌ Delivery entry missing 'delivery_index' field")
                    return False
                if "arrival_time" not in entry:
                    print(f"❌ Delivery entry missing 'arrival_time' field")
                    return False
            
            # Check for unexpected route fields
            allowed_route_fields = ["type", "task_id", "index", "pickup_index", "delivery_index", 
                                  "arrival_time", "deadline", "lateness"]
            for field in entry:
                if field not in allowed_route_fields:
                    print(f"❌ Unexpected route entry field: {field}")
                    return False
    
    print(f"✅ Route structure correct")
    return True

def test_algorithm(endpoint: str, algorithm_name: str, request_data: Dict[str, Any]) -> bool:
    """Test a specific algorithm endpoint."""
    print(f"\n🧪 Testing {algorithm_name} Algorithm")
    print("=" * 50)
    
    try:
        # Make request
        print(f"📤 Sending request to {endpoint}")
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}{endpoint}",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"⏱️  Response time: {response_time:.3f}s")
        print(f"📥 Status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Expected status 200, got {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        # Parse response
        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON response: {e}")
            return False
        
        # Display response summary
        print(f"📋 Task ID: {response_data.get('task_id', 'N/A')}")
        recommendations = response_data.get('recommendations', [])
        print(f"👥 Recommendations count: {len(recommendations)}")
        
        if recommendations:
            best_rec = recommendations[0]
            print(f"🏆 Best recommendation: {best_rec.get('driver_id')} (score: {best_rec.get('score')})")
            print(f"⏰ Additional time: {best_rec.get('additional_time_minutes')} minutes")
            print(f"🛣️  Route steps: {len(best_rec.get('route', []))}")
        
        # Validate format
        is_valid = validate_response_format(response_data, algorithm_name)
        
        if is_valid:
            print(f"✅ {algorithm_name} passed all validation checks!")
        else:
            print(f"❌ {algorithm_name} failed validation checks!")
        
        return is_valid
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Starting API Compatibility Tests")
    print("=" * 60)
    
    # Check if API is healthy
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ API Health Check: {health_data.get('status', 'unknown')}")
            print(f"📊 Available algorithms: {health_data.get('available_algorithms', [])}")
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False
    
    # Create test request
    request_data = create_test_request()
    print(f"\n📝 Request format validation:")
    print(f"✅ Using old API format with {len(request_data)} top-level fields")
    print(f"✅ New task: {request_data['new_task']['id']} ({request_data['new_task']['job_type']})")
    print(f"✅ Agents: {len(request_data['agents'])} agents")
    print(f"✅ Current tasks: {len(request_data['current_tasks'])} tasks")
    
    # Test algorithms
    test_results = []
    
    # Test auto-selection (should pick one of the optimized algorithms)
    test_results.append(test_algorithm("/recommend", "Auto-Selection", request_data))
    
    # Test fixed optimized algorithm directly
    test_results.append(test_algorithm("/recommend/fixed-optimized", "Fixed Optimized", request_data))
    
    # Test light optimized algorithm directly
    test_results.append(test_algorithm("/recommend/light-optimized", "Light Optimized", request_data))
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 30)
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"✅ Passed: {passed}/{total}")
    if passed == total:
        print(f"🎉 All tests passed! API is fully compatible with old documentation format.")
    else:
        print(f"⚠️  Some tests failed. Please review the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 