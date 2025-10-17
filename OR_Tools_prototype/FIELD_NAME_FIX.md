# ⚠️ CRITICAL: Field Name Requirements

## The Problem

Your middleware is sending requests with **incorrect field names**. This causes the system to not recognize which driver the tasks belong to, resulting in incorrect routing.

## Field Name Mapping

### ❌ What You're Sending (WRONG)
```json
{
  "agents": [
    {
      "id": "2098791",          // ❌ WRONG
      "location": [17.126, -61.821]  // ❌ WRONG
    }
  ]
}
```

### ✅ What OR-Tools Expects (CORRECT)
```json
{
  "agents": [
    {
      "driver_id": "2098791",        // ✅ CORRECT
      "current_location": [17.126, -61.821]  // ✅ CORRECT
    }
  ]
}
```

## Complete Corrected Request Example

```json
{
  "agents": [
    {
      "driver_id": "2098791",              // ← Changed from "id"
      "name": "Testing Esah",
      "current_location": [                // ← Changed from "location"
        17.12623998,
        -61.8218519
      ]
    }
  ],
  "new_task": {
    "id": "578724123962417606489027752601",
    "job_type": "PAIRED",
    "restaurant_location": [17.126151031975567, -61.82166574640591],
    "delivery_location": [17.12609, -61.8217069],
    "pickup_before": "2025-10-17T02:00:00.000Z",
    "delivery_before": "2025-10-17T03:00:00.000Z"
  },
  "current_tasks": [
    {
      "id": "578722681745217606474483427179",
      "job_type": "PAIRED",
      "restaurant_location": [17.126151031975567, -61.82166574640591],
      "delivery_location": [17.12609, -61.8217069],
      "pickup_completed": true,          // ← This now works!
      "assigned_driver": "2098791",
      "pickup_before": "2025-10-17T01:03:00.000Z",
      "delivery_before": "2025-10-17T03:00:00.000Z"
    }
  ],
  "algorithm": "batch_optimized",
  "max_distance_km": 10,
  "optimization_mode": "tardiness_min"
}
```

## Expected Result (With Correct Field Names)

```json
{
  "route": [
    {
      "type": "start",
      "index": 0
    },
    {
      "type": "new_task_pickup",
      "task_id": "578724123962417606489027752601",
      "pickup_index": 2
    },
    {
      "type": "existing_task_delivery",     // ← OLD TASK: ONLY DELIVERY
      "task_id": "578722681745217606474483427179",
      "delivery_index": 1
    },
    {
      "type": "new_task_delivery",
      "task_id": "578724123962417606489027752601",
      "delivery_index": 1
    },
    {
      "type": "end",
      "index": 0
    }
  ]
}
```

**Notice:** The old task (with `pickup_completed: true`) has **NO** pickup entry - only delivery! ✅

## Where to Fix This

Update your middleware/transformation code that sends requests to OR-Tools:

### Before (Wrong):
```javascript
const orToolsRequest = {
  agents: agents.map(agent => ({
    id: agent.id,                    // ❌ Wrong field name
    location: agent.currentLocation  // ❌ Wrong field name
  })),
  // ...
}
```

### After (Correct):
```javascript
const orToolsRequest = {
  agents: agents.map(agent => ({
    driver_id: agent.id,                    // ✅ Correct
    current_location: agent.currentLocation // ✅ Correct
  })),
  // ...
}
```

## Test Command

```bash
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "agents": [
      {
        "driver_id": "2098791",
        "name": "Testing Esah",
        "current_location": [17.12623998, -61.8218519]
      }
    ],
    "new_task": {
      "id": "578724123962417606489027752601",
      "job_type": "PAIRED",
      "restaurant_location": [17.126151031975567, -61.82166574640591],
      "delivery_location": [17.12609, -61.8217069],
      "pickup_before": "2025-10-17T02:00:00.000Z",
      "delivery_before": "2025-10-17T03:00:00.000Z"
    },
    "current_tasks": [
      {
        "id": "578722681745217606474483427179",
        "job_type": "PAIRED",
        "restaurant_location": [17.126151031975567, -61.82166574640591],
        "delivery_location": [17.12609, -61.8217069],
        "pickup_completed": true,
        "assigned_driver": "2098791",
        "pickup_before": "2025-10-17T01:03:00.000Z",
        "delivery_before": "2025-10-17T03:00:00.000Z"
      }
    ],
    "algorithm": "batch_optimized"
  }'
```

## Summary

1. ✅ **The OR-Tools code is working correctly**
2. ❌ **Your middleware is sending wrong field names**
3. 🔧 **Fix: Change "id" → "driver_id" and "location" → "current_location"**
4. ✅ **Once fixed, pickup_completed will work perfectly**

