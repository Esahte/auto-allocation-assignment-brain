# Auto-Allocation Recommendation API Documentation

## Overview

The Auto-Allocation Recommendation API is a route optimization service that helps assign delivery tasks to drivers based on their current location, existing assignments, and various constraints. The service uses Google OR-Tools for optimization and implements a grace period system for handling late deliveries.

## Base URL

```
https://recommendation-app-95621826490.us-central1.run.app
```

## Authentication

No authentication required. All endpoints are publicly accessible.

## Endpoints

### 1. Health Check

**GET** `/health`

Check if the service is running and healthy.

#### Response

```json
{
  "status": "healthy"
}
```

#### Status Codes
- `200` - Service is healthy
- `500` - Service error

---

### 2. Get Task Recommendations

**POST** `/recommend`

Get optimized agent recommendations for a new task based on current assignments and constraints.

#### Request Headers
```
Content-Type: application/json
```

#### Request Body

```json
{
  "new_task": {
    "id": "string",
    "job_type": "PAIRED|DELIVERY_ONLY",
    "restaurant_location": [latitude, longitude],
    "delivery_location": [latitude, longitude],
    "pickup_before": "ISO8601 timestamp",
    "delivery_before": "ISO8601 timestamp"
  },
  "agents": [
    {
      "driver_id": "string",
      "name": "string",
      "current_location": [latitude, longitude]
    }
  ],
  "current_tasks": [
    {
      "id": "string",
      "job_type": "PAIRED|DELIVERY_ONLY",
      "restaurant_location": [latitude, longitude],
      "delivery_location": [latitude, longitude],
      "pickup_before": "ISO8601 timestamp",
      "delivery_before": "ISO8601 timestamp",
      "assigned_driver": "string"
    }
  ],
  "max_grace_period": 1800
}
```

#### Field Descriptions

**new_task**
- `id` (string, required): Unique identifier for the new task
- `job_type` (string, required): Either "PAIRED" (pickup + delivery) or "DELIVERY_ONLY"
- `restaurant_location` (array, required): [latitude, longitude] of pickup location
- `delivery_location` (array, required): [latitude, longitude] of delivery location
- `pickup_before` (string, required for PAIRED): ISO8601 timestamp for pickup deadline
- `delivery_before` (string, required): ISO8601 timestamp for delivery deadline

**agents**
- `driver_id` (string, required): Unique identifier for the agent
- `name` (string, required): Display name of the agent
- `current_location` (array, required): [latitude, longitude] of agent's current position

**current_tasks**
- Same structure as `new_task` with additional field:
- `assigned_driver` (string, required): ID of the agent currently assigned to this task

**max_grace_period** (integer, optional)
- Maximum grace period in seconds (default: 1800 = 30 minutes)
- Used for handling late deliveries and calculating penalties

#### Response

```json
{
  "task_id": "string",
  "recommendations": [
    {
      "driver_id": "string",
      "name": "string",
      "score": 100,
      "additional_time_minutes": 0.0,
      "grace_penalty_seconds": 0,
      "already_late_stops": 0,
      "route": [
        {
          "type": "start",
          "index": 0
        },
        {
          "type": "new_task_pickup",
          "task_id": "string",
          "pickup_index": 1,
          "arrival_time": "2025-05-30T16:00:50Z",
          "deadline": "2025-05-30T17:00:00Z",
          "lateness": 0
        },
        {
          "type": "new_task_delivery",
          "task_id": "string",
          "delivery_index": 2,
          "arrival_time": "2025-05-30T16:00:50Z",
          "deadline": "2025-05-30T17:30:00Z",
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

#### Response Field Descriptions

**recommendations** (array): Up to 3 best agent recommendations, sorted by score (highest first)

**Individual Recommendation Fields:**
- `driver_id` (string): Agent's unique identifier
- `name` (string): Agent's display name
- `score` (integer): Optimization score (0-100, higher is better)
- `additional_time_minutes` (float): Extra time needed to complete the new task
- `grace_penalty_seconds` (integer): Total grace period penalty for late tasks
- `already_late_stops` (integer): Number of stops that require grace periods
- `route` (array): Detailed route plan with timing information

**Route Entry Types:**
- `start`: Agent's starting position
- `end`: Agent's ending position
- `new_task_pickup`: Pickup for the new task being assigned
- `new_task_delivery`: Delivery for the new task being assigned
- `existing_task_pickup`: Pickup for an already assigned task
- `existing_task_delivery`: Delivery for an already assigned task

#### Example Request

```bash
curl -X POST https://recommendation-app-95621826490.us-central1.run.app/recommend \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

#### Example Response

```json
{
  "task_id": "task_123",
  "recommendations": [
    {
      "driver_id": "driver_002",
      "name": "Jane Smith", 
      "score": 100,
      "additional_time_minutes": 15.5,
      "grace_penalty_seconds": 0,
      "already_late_stops": 0,
      "route": [
        {
          "type": "start",
          "index": 1
        },
        {
          "type": "new_task_pickup",
          "task_id": "task_123",
          "pickup_index": 2,
          "arrival_time": "2025-05-30T16:15:30Z",
          "deadline": "2025-05-30T17:00:00Z",
          "lateness": 0
        },
        {
          "type": "new_task_delivery",
          "task_id": "task_123", 
          "delivery_index": 3,
          "arrival_time": "2025-05-30T16:20:45Z",
          "deadline": "2025-05-30T17:30:00Z",
          "lateness": 0
        },
        {
          "type": "end",
          "index": 1
        }
      ]
    },
    {
      "driver_id": "driver_001",
      "name": "John Doe",
      "score": 95,
      "additional_time_minutes": 22.3,
      "grace_penalty_seconds": 180,
      "already_late_stops": 1,
      "route": [
        {
          "type": "start",
          "index": 0
        },
        {
          "type": "existing_task_pickup",
          "task_id": "task_456",
          "pickup_index": 4,
          "arrival_time": "2025-05-30T16:35:00Z",
          "deadline": "2025-05-30T16:30:00Z",
          "lateness": 0
        },
        {
          "type": "existing_task_delivery",
          "task_id": "task_456",
          "delivery_index": 5,
          "arrival_time": "2025-05-30T16:45:00Z",
          "deadline": "2025-05-30T17:00:00Z",
          "lateness": 0
        },
        {
          "type": "new_task_pickup",
          "task_id": "task_123",
          "pickup_index": 2,
          "arrival_time": "2025-05-30T16:55:00Z",
          "deadline": "2025-05-30T17:00:00Z",
          "lateness": 0
        },
        {
          "type": "new_task_delivery",
          "task_id": "task_123",
          "delivery_index": 3,
          "arrival_time": "2025-05-30T17:05:00Z",
          "deadline": "2025-05-30T17:30:00Z",
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

#### Status Codes

- `200` - Success
- `400` - Bad Request (missing required fields, invalid data format)
- `500` - Internal Server Error

#### Error Response Format

```json
{
  "error": "Description of the error"
}
```

## Grace Period System

The API implements a grace period system for handling late deliveries:

### How It Works

1. **Initial Grace Period**: 600 seconds (10 minutes)
2. **Maximum Grace Period**: Configurable via `max_grace_period` parameter (default: 1800 seconds / 30 minutes)
3. **Automatic Extension**: If no feasible solution is found, the grace period is automatically extended by 10-minute increments
4. **Scoring Impact**: Grace period usage reduces the agent's score

### Scoring Algorithm

- **Perfect Score**: 100 (no grace periods used)
- **Penalty Calculation**: `(grace_penalty_seconds / max_grace_period) * 100`
- **Final Score**: `100 - penalty_percentage`

## Coordinate System

All location coordinates use the **WGS84** coordinate system:
- **Format**: `[latitude, longitude]`
- **Example**: `[17.140, -61.890]` (Antigua coordinates)
- **Precision**: Up to 6 decimal places recommended

## Rate Limits

Currently no rate limits are enforced, but recommended usage:
- **Maximum requests per minute**: 100
- **Recommended timeout**: 30 seconds per request

## Troubleshooting

### Common Issues

1. **Invalid Coordinates**: Ensure coordinates are in valid ranges
   - Latitude: -90 to 90
   - Longitude: -180 to 180

2. **Invalid Timestamps**: Use ISO8601 format with timezone
   - Correct: `"2025-05-30T17:00:00Z"`
   - Incorrect: `"2025-05-30 17:00:00"`

3. **Missing Required Fields**: All required fields must be provided
   - Check that `new_task`, `agents`, and `current_tasks` are included

4. **No Feasible Solution**: If all agents return score 0
   - Try increasing `max_grace_period`
   - Check if time constraints are too restrictive

### Support

For technical support or questions about the API, please contact the development team or check the project repository for issues and documentation updates.

## Changelog

### Version 1.0.0
- Initial release with core recommendation functionality
- Grace period system implementation
- Support for PAIRED and DELIVERY_ONLY job types
- Real-time route optimization using OR-Tools 