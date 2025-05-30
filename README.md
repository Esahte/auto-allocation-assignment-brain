# Auto Allocation Assignment Brain

A route optimization service that uses Google OR-Tools to recommend optimal delivery assignments for agents.

## Features

- Route optimization using Google OR-Tools
- REST API endpoint for getting delivery recommendations
- Support for both paired (pickup + delivery) and delivery-only tasks
- Time window constraints and grace period handling
- Detailed route information and scoring with penalty tracking
- Realistic penalty system based on grace period usage

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

**Example Response:**
```json
{
  "task_id": "easy_new_task",
  "recommendations": [
    {
      "driver_id": "agent3",
      "name": "Agent 3",
      "score": 100,
      "additional_time_minutes": 25.1,
      "lateness_penalty_seconds": 0,
      "grace_penalty_seconds": 0,
      "already_late_stops": 0,
      "route": [
        {
          "type": "start",
          "index": 2
        },
        {
          "type": "new_task_pickup",
          "task_id": "easy_new_task",
          "pickup_index": 7,
          "arrival_time": "2025-05-30T08:02:07Z",
          "deadline": "2025-05-30T08:32:07Z",
          "lateness": 0
        },
        {
          "type": "new_task_delivery",
          "task_id": "easy_new_task",
          "delivery_index": 8,
          "arrival_time": "2025-05-30T08:27:47Z",
          "deadline": "2025-05-30T09:02:07Z",
          "lateness": 0
        },
        {
          "type": "end",
          "index": 2
        }
      ]
    },
    {
      "driver_id": "agent1",
      "name": "Agent 1",
      "score": 93,
      "additional_time_minutes": 25.1,
      "lateness_penalty_seconds": 120,
      "grace_penalty_seconds": 120,
      "already_late_stops": 1,
      "route": [
        {
          "type": "start",
          "index": 0
        },
        {
          "type": "existing_task_pickup",
          "task_id": "late_task_1",
          "pickup_index": 3,
          "arrival_time": "2025-05-30T08:07:07Z",
          "deadline": "2025-05-30T07:43:07Z",
          "lateness": 0
        },
        {
          "type": "existing_task_delivery",
          "task_id": "late_task_1",
          "delivery_index": 4,
          "arrival_time": "2025-05-30T08:10:17Z",
          "deadline": "2025-05-30T08:12:07Z",
          "lateness": 0
        },
        {
          "type": "new_task_pickup",
          "task_id": "easy_new_task",
          "pickup_index": 7,
          "arrival_time": "2025-05-30T08:12:47Z",
          "deadline": "2025-05-30T08:32:07Z",
          "lateness": 0
        },
        {
          "type": "new_task_delivery",
          "task_id": "easy_new_task",
          "delivery_index": 8,
          "arrival_time": "2025-05-30T08:38:27Z",
          "deadline": "2025-05-30T09:02:07Z",
          "lateness": 0
        },
        {
          "type": "end",
          "index": 0
        }
      ]
    },
    {
      "driver_id": "agent2",
      "name": "Agent 2",
      "score": 83,
      "additional_time_minutes": 25.1,
      "lateness_penalty_seconds": 300,
      "grace_penalty_seconds": 300,
      "already_late_stops": 1,
      "route": [
        {
          "type": "start",
          "index": 1
        },
        {
          "type": "existing_task_pickup",
          "task_id": "late_task_2",
          "pickup_index": 5,
          "arrival_time": "2025-05-30T08:01:07Z",
          "deadline": "2025-05-30T07:38:07Z",
          "lateness": 0
        },
        {
          "type": "existing_task_delivery",
          "task_id": "late_task_2",
          "delivery_index": 6,
          "arrival_time": "2025-05-30T08:04:17Z",
          "deadline": "2025-05-30T07:56:07Z",
          "lateness": 0
        },
        {
          "type": "new_task_pickup",
          "task_id": "easy_new_task",
          "pickup_index": 7,
          "arrival_time": "2025-05-30T08:06:47Z",
          "deadline": "2025-05-30T08:32:07Z",
          "lateness": 0
        },
        {
          "type": "new_task_delivery",
          "task_id": "easy_new_task",
          "delivery_index": 8,
          "arrival_time": "2025-05-30T08:32:27Z",
          "deadline": "2025-05-30T09:02:07Z",
          "lateness": 0
        },
        {
          "type": "end",
          "index": 1
        }
      ]
    }
  ]
}
```

## Response Fields Explanation

- **task_id**: ID of the new task being assigned
- **recommendations**: Array of up to 3 best agents, sorted by score (highest first)
  - **driver_id**: Unique identifier for the agent
  - **name**: Human-readable name for the agent
  - **score**: Optimization score (0-100, where 100 = perfect, no penalties)
  - **additional_time_minutes**: Extra time added to agent's route if assigned
  - **lateness_penalty_seconds**: Total grace period time used (equals grace_penalty_seconds)
  - **grace_penalty_seconds**: Seconds of grace period applied to late tasks
  - **already_late_stops**: Number of tasks that required grace periods
  - **route**: Detailed step-by-step route including:
    - **type**: start, end, new_task_pickup, new_task_delivery, existing_task_pickup, existing_task_delivery
    - **arrival_time**: When agent arrives at this location (ISO format)
    - **deadline**: Original deadline for this task (ISO format)
    - **lateness**: Always 0 (grace periods handle all deadline violations)

## Scoring System

- **Score 100**: Perfect assignment with no grace periods needed
- **Score 93-99**: Good assignment with minimal grace period usage
- **Score 50-92**: Moderate assignment with some late tasks requiring grace periods
- **Score 1-49**: Challenging assignment with significant grace period usage
- **Score 0**: No feasible solution found

Grace periods are applied when tasks have deadlines in the past, extending them by up to 30 minutes total. The scoring penalizes based on total grace period usage.

## Docker Deployment

Build and run with Docker Compose:
```bash
docker-compose up --build
```

## Testing

Run the test script:
```bash
python test_recommendation.py
```

## License

MIT 