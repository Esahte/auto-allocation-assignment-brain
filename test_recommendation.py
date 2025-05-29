import requests
import json

# Sample data for testing
test_data = {
    "new_task": {
        "id": "task123",
        "job_type": "PAIRED",
        "restaurant_location": [37.7749, -122.4194],  # San Francisco coordinates
        "delivery_location": [37.7833, -122.4167],    # San Francisco coordinates
        "pickup_before": "2024-03-20T15:00:00Z",
        "delivery_before": "2024-03-20T15:30:00Z"
    },
    "agents": [
        {
            "driver_id": "driver1",
            "name": "John Doe",
            "current_location": [37.7749, -122.4194]  # San Francisco coordinates
        },
        {
            "driver_id": "driver2",
            "name": "Jane Smith",
            "current_location": [37.7833, -122.4167]  # San Francisco coordinates
        }
    ],
    "current_tasks": [
        {
            "id": "task456",
            "job_type": "PAIRED",
            "restaurant_location": [37.7749, -122.4194],
            "delivery_location": [37.7833, -122.4167],
            "pickup_before": "2024-03-20T14:00:00Z",
            "delivery_before": "2024-03-20T14:30:00Z",
            "assigned_driver": "driver1"
        }
    ]
}

def test_recommendation():
    # URL of your running application
    url = "http://localhost:8081/recommend"
    
    print("Sending request to:", url)
    print("Request data:", json.dumps(test_data, indent=2))
    
    try:
        # Send POST request
        response = requests.post(url, json=test_data)
        
        print("\nResponse status code:", response.status_code)
        print("Response headers:", dict(response.headers))
        
        # Check if request was successful
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

if __name__ == "__main__":
    test_recommendation() 