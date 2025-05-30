import requests
import json
import random
from datetime import datetime, timedelta, timezone

# Antigua coordinates (roughly)
ANTIGUA_LAT_RANGE = (17.05, 17.20)
ANTIGUA_LON_RANGE = (-61.95, -61.70)

# Helper to generate a random coordinate in Antigua
def random_location():
    lat = round(random.uniform(*ANTIGUA_LAT_RANGE), 6)
    lon = round(random.uniform(*ANTIGUA_LON_RANGE), 6)
    return [lat, lon]

# Helper to generate a random ISO8601 time window
def random_time_window(base_time=None, min_offset=0, max_offset=120, duration=30):
    if base_time is None:
        base_time = datetime.now(timezone.utc)
    pickup_offset = random.randint(min_offset, max_offset)
    delivery_offset = pickup_offset + duration
    pickup_time = base_time + timedelta(minutes=pickup_offset)
    delivery_time = base_time + timedelta(minutes=delivery_offset)
    return pickup_time.isoformat(), delivery_time.isoformat()

def generate_agents(num_agents):
    agents = []
    for i in range(num_agents):
        agents.append({
            "driver_id": f"driver{i+1}",
            "name": f"Agent {i+1}",
            "current_location": random_location()
        })
    return agents

def generate_tasks(num_tasks, job_type="PAIRED", id_prefix="task", base_time=None):
    tasks = []
    for i in range(num_tasks):
        pickup_before, delivery_before = random_time_window(base_time)
        tasks.append({
            "id": f"{id_prefix}{i+1}",
            "job_type": job_type,
            "restaurant_location": random_location(),
            "delivery_location": random_location(),
            "pickup_before": pickup_before,
            "delivery_before": delivery_before
        })
    return tasks

def generate_current_tasks(agents, max_tasks_per_agent=5, base_time=None):
    current_tasks = []
    task_counter = 1
    for agent in agents:
        num_tasks = random.randint(0, max_tasks_per_agent)
        for _ in range(num_tasks):
            pickup_before, delivery_before = random_time_window(base_time, min_offset=-60, max_offset=60)
            current_tasks.append({
                "id": f"assigned{task_counter}",
                "job_type": "PAIRED",
                "restaurant_location": random_location(),
                "delivery_location": random_location(),
                "pickup_before": pickup_before,
                "delivery_before": delivery_before,
                "assigned_driver": agent["driver_id"]
            })
            task_counter += 1
    return current_tasks

def test_recommendation():
    # Randomize scenario
    num_agents = random.randint(5, 12)
    num_new_tasks = random.randint(1, 6)
    agents = generate_agents(num_agents)
    new_tasks = generate_tasks(num_new_tasks, id_prefix="newtask")
    current_tasks = generate_current_tasks(agents, max_tasks_per_agent=5)

    # For compatibility with your API, send only one new_task (the first), but you can loop for more
    for i, new_task in enumerate(new_tasks):
        test_data = {
            "new_task": new_task,
            "agents": agents,
            "current_tasks": current_tasks
        }
        url = "http://localhost:8081/recommend"
        print(f"\n--- Scenario {i+1} ---")
        print("Sending request to:", url)
        print("Request data:", json.dumps(test_data, indent=2))
        try:
            response = requests.post(url, json=test_data)
            print("\nResponse status code:", response.status_code)
            print("Response headers:", dict(response.headers))
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    print("\nSuccess! Response:")
                    print(json.dumps(response_data, indent=2))
                except json.JSONDecodeError as e:
                    print("\nError decoding JSON response:")
                    print("Raw response content:", response.text)
                    print("JSON decode error:", str(e))
            else:
                print(f"\nError: {response.status_code}")
                print("Response content:", response.text)
        except requests.exceptions.RequestException as e:
            print(f"\nError making request: {e}")
            print("Request details:", e.request.url if hasattr(e, 'request') else "No request details available")

def test_grace_period_scenario():
    """Test scenario specifically designed to trigger grace period and lateness penalties"""
    
    # Current time
    now = datetime.now(timezone.utc)
    
    # Create a specific scenario with predictable grace period usage
    agents = [
        {"driver_id": "driver1", "name": "Agent 1", "current_location": [17.15, -61.85]},
        {"driver_id": "driver2", "name": "Agent 2", "current_location": [17.10, -61.80]},
        {"driver_id": "driver3", "name": "Agent 3", "current_location": [17.08, -61.90]}
    ]
    
    # Simplified current tasks - just one late task per busy agent
    current_tasks = [
        {
            "id": "slightly_late_pickup",
            "job_type": "PAIRED", 
            "restaurant_location": [17.13, -61.83],  # Close to agent 1
            "delivery_location": [17.14, -61.84],    # Short delivery
            "pickup_before": (now - timedelta(minutes=2)).isoformat(),  # 2 min late - small grace
            "delivery_before": (now + timedelta(minutes=30)).isoformat(),  # Generous delivery time
            "assigned_driver": "driver1"
        },
        {
            "id": "moderately_late_pickup",
            "job_type": "PAIRED",
            "restaurant_location": [17.11, -61.81],  # Close to agent 2
            "delivery_location": [17.12, -61.82],    # Short delivery 
            "pickup_before": (now - timedelta(minutes=5)).isoformat(),  # 5 min late - larger grace
            "delivery_before": (now + timedelta(minutes=35)).isoformat(),  # Generous delivery time
            "assigned_driver": "driver2"
        }
        # Driver3 remains free for perfect score comparison
    ]
    
    # New task that all agents should be able to handle
    new_task = {
        "id": "flexible_new_task",
        "job_type": "PAIRED",
        "restaurant_location": [17.16, -61.86],
        "delivery_location": [17.09, -61.89],
        "pickup_before": (now + timedelta(minutes=40)).isoformat(),   # Far future deadline
        "delivery_before": (now + timedelta(minutes=70)).isoformat()  # Very generous delivery window
    }
    
    test_data = {
        "new_task": new_task,
        "agents": agents,
        "current_tasks": current_tasks
    }
    
    print("=== GRACE PERIOD PENALTY TEST SCENARIO ===")
    print("This scenario should show:")
    print("- Agent 1: Small grace period penalty (2 min late task)")
    print("- Agent 2: Larger grace period penalty (5 min late task)")  
    print("- Agent 3: Perfect score (no existing tasks)")
    print("- All agents should be able to handle the new task")
    print()
    
    url = "http://localhost:8081/recommend"
    print("Sending request to:", url)
    print("Request data:", json.dumps(test_data, indent=2))
    
    try:
        response = requests.post(url, json=test_data)
        print(f"\nResponse status code: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print("\n=== RESULTS ANALYSIS ===")
            print(json.dumps(response_data, indent=2))
            
            print("\n=== SCORE BREAKDOWN & ANALYSIS ===")
            for rec in response_data.get("recommendations", []):
                driver = rec["driver_id"]
                score = rec["score"]
                grace_penalty = rec.get("grace_penalty_seconds", 0)
                lateness_penalty = rec.get("lateness_penalty_seconds", 0)
                
                print(f"\n{driver} (Score: {score}):")
                print(f"  - Grace period penalty: {grace_penalty} seconds ({grace_penalty/60:.1f} minutes)")
                print(f"  - Lateness penalty: {lateness_penalty} seconds ({lateness_penalty/60:.1f} minutes)")
                
                if score == 100:
                    print(f"  ✅ Perfect score - no penalties used!")
                elif score == 0:
                    print(f"  ❌ No feasible solution found")
                else:
                    total_penalty_minutes = (grace_penalty + lateness_penalty) / 60
                    print(f"  ⚠️  Score reduced due to {total_penalty_minutes:.1f} minutes of penalties")
                    print(f"      Grace penalty weight: 30%, Lateness penalty weight: 70%")
                    
        else:
            print(f"Error: {response.status_code}")
            print("Response:", response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def test_manual_scenario():
    """Manually designed scenario with predictable grace period penalties"""
    
    # Current time
    now = datetime.now(timezone.utc)
    
    # Simple scenario: 3 agents in Antigua
    agents = [
        {"driver_id": "agent1", "name": "Agent 1", "current_location": [17.10, -61.85]},
        {"driver_id": "agent2", "name": "Agent 2", "current_location": [17.12, -61.87]}, 
        {"driver_id": "agent3", "name": "Agent 3", "current_location": [17.15, -61.90]}
    ]
    
    # Current tasks with known late penalties:
    current_tasks = [
        {
            "id": "late_task_1",
            "job_type": "PAIRED",
            "restaurant_location": [17.11, -61.86],  # Close to agent1
            "delivery_location": [17.12, -61.86],    # Very short delivery
            "pickup_before": (now - timedelta(minutes=2)).isoformat(),  # 2 min late = 120 sec grace
            "delivery_before": (now + timedelta(minutes=25)).isoformat(),  # Generous delivery window
            "assigned_driver": "agent1"
        },
        {
            "id": "late_task_2", 
            "job_type": "PAIRED",
            "restaurant_location": [17.13, -61.88],  # Close to agent2
            "delivery_location": [17.14, -61.88],    # Very short delivery
            "pickup_before": (now - timedelta(minutes=5)).isoformat(),  # 5 min late = 300 sec grace
            "delivery_before": (now + timedelta(minutes=30)).isoformat(),  # Generous delivery window
            "assigned_driver": "agent2"
        }
        # Agent3 has no tasks - should get perfect score
    ]
    
    # New task with very generous deadlines - all agents should handle easily
    new_task = {
        "id": "easy_new_task",
        "job_type": "PAIRED", 
        "restaurant_location": [17.14, -61.89],
        "delivery_location": [17.16, -61.91],
        "pickup_before": (now + timedelta(minutes=45)).isoformat(),   # 45 min from now
        "delivery_before": (now + timedelta(minutes=75)).isoformat()  # 75 min from now
    }
    
    test_data = {
        "new_task": new_task,
        "agents": agents,
        "current_tasks": current_tasks
    }
    
    print("=== MANUAL TEST SCENARIO ===")
    print("Expected Results:")
    print("• Agent 1: Grace penalty = 120 seconds (2 min late task)")
    print("  - Score calculation: 120/1800 = 0.067 grace penalty")
    print("  - With 30% weight: 0.067 * 0.3 = 0.02 penalty")
    print("  - Expected score: (1 - 0.02) * 100 = 98")
    print("• Agent 2: Grace penalty = 300 seconds (5 min late task)")
    print("  - Score calculation: 300/1800 = 0.167 grace penalty") 
    print("  - With 30% weight: 0.167 * 0.3 = 0.05 penalty")
    print("  - Expected score: (1 - 0.05) * 100 = 95")
    print("• Agent 3: No penalties = Perfect score 100")
    print()
    
    url = "http://localhost:8081/recommend"
    print("Sending request to:", url)
    print("Request data:", json.dumps(test_data, indent=2))
    
    try:
        response = requests.post(url, json=test_data)
        print(f"\nResponse status code: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print("\n=== ACTUAL RESULTS ===")
            print(json.dumps(response_data, indent=2))
            
            print("\n=== VERIFICATION ===")
            for rec in response_data.get("recommendations", []):
                driver = rec["driver_id"]
                score = rec["score"]
                grace_penalty = rec.get("grace_penalty_seconds", 0)
                
                print(f"\n{driver}:")
                print(f"  Actual Score: {score}")
                print(f"  Grace Penalty: {grace_penalty} seconds ({grace_penalty/60:.1f} min)")
                
                # Expected scores based on our calculation
                if driver == "agent1":
                    expected_score = 98
                    expected_grace = 120
                elif driver == "agent2": 
                    expected_score = 95
                    expected_grace = 300
                elif driver == "agent3":
                    expected_score = 100
                    expected_grace = 0
                else:
                    expected_score = "unknown"
                    expected_grace = "unknown"
                
                print(f"  Expected Score: {expected_score}")
                print(f"  Expected Grace: {expected_grace} seconds")
                
                # Verification
                if score == expected_score and grace_penalty == expected_grace:
                    print(f"  ✅ PASS - Results match expectations!")
                else:
                    print(f"  ❌ FAIL - Results don't match expectations")
                    
        else:
            print(f"Error: {response.status_code}")
            print("Response:", response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_manual_scenario() 