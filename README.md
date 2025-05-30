# Auto Allocation Assignment Brain

A route optimization service that uses Google OR-Tools to recommend optimal delivery assignments for agents.

## Features

- Route optimization using Google OR-Tools
- REST API endpoint for getting delivery recommendations
- Support for both paired (pickup + delivery) and delivery-only tasks
- Configurable grace period system for handling late tasks
- Detailed route information in JSON format

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask application:
```bash
python OR_Tools_prototype/app.py
```

The server will start on port 8081 by default.

## API Endpoints

### Health Check
```
GET /health
```
Returns the health status of the service.

### Get Recommendations
```
POST /recommend
```
Request body:
```json
{
    "new_task": {
        "id": "task123",
        "job_type": "PAIRED",
        "restaurant_location": [37.7749, -122.4194],
        "delivery_location": [37.7833, -122.4167],
        "pickup_before": "2024-03-20T15:00:00Z",
        "delivery_before": "2024-03-20T15:30:00Z"
    },
    "agents": [
        {
            "driver_id": "driver1",
            "name": "John Doe",
            "current_location": [37.7749, -122.4194]
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
    ],
    "max_grace_period": 1800  // Optional: Maximum grace period in seconds (default: 1800 = 30 minutes)
}
```

Response format:
```json
{
    "task_id": "task123",
    "recommendations": [
        {
            "driver_id": "driver1",
            "name": "John Doe",
            "score": 85,
            "additional_time_minutes": 5.2,
            "grace_penalty_seconds": 300,
            "already_late_stops": 1,
            "route": [
                {
                    "type": "start",
                    "index": 0
                },
                {
                    "type": "new_task_pickup",
                    "task_id": "task123",
                    "pickup_index": 2,
                    "arrival_time": "2024-03-20T15:05:00Z",
                    "deadline": "2024-03-20T15:00:00Z",
                    "lateness": 300
                },
                {
                    "type": "new_task_delivery",
                    "task_id": "task123",
                    "delivery_index": 3,
                    "arrival_time": "2024-03-20T15:25:00Z",
                    "deadline": "2024-03-20T15:30:00Z",
                    "lateness": 0
                },
                {
                    "type": "end",
                    "index": 0
                }
            ]
        }
    ]
}
```

### Response Fields

- `task_id`: ID of the task being assigned
- `recommendations`: List of top 3 recommended agents, sorted by score
  - `driver_id`: Unique identifier for the driver
  - `name`: Driver's name
  - `score`: Recommendation score (0-100)
    - 100: Perfect score (no grace periods needed)
    - 0: Worst score (maximum grace period usage)
  - `additional_time_minutes`: Additional time needed to complete the new task
  - `grace_penalty_seconds`: Total grace period time used for all tasks
  - `already_late_stops`: Number of stops that required grace periods
  - `route`: Detailed route information including:
    - Start and end points
    - Pickup and delivery times
    - Task types (new/existing)
    - Arrival times and deadlines
    - Lateness at each stop

### Grace Period System

The service uses a configurable grace period system to handle late tasks:
- Initial grace period: 10 minutes (600 seconds)
- Default maximum grace period: 30 minutes (1800 seconds)
- Configurable maximum grace period via the `max_grace_period` parameter
- Grace periods are automatically extended if no feasible solution is found
- Tasks that exceed their deadlines use grace periods instead of being marked as late
- The scoring system penalizes agents based on grace period usage relative to the maximum grace period

## Testing

Run the test script:
```bash
python test_recommendation.py
```

## License

MIT 