# Auto Allocation Assignment Brain

A route optimization service that uses Google OR-Tools to recommend optimal delivery assignments for agents.

## Features

- Route optimization using Google OR-Tools
- REST API endpoint for getting delivery recommendations
- Support for both paired (pickup + delivery) and delivery-only tasks
- Time window constraints and lateness penalties
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
    ]
}
```

## Testing

Run the test script:
```bash
python test_recommendation.py
```

## License

MIT 