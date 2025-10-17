# Timezone Handling Guide

## Summary

**The system uses UTC for all internal calculations but ACCEPTS timestamps in ANY timezone.**

## How It Works

### 1. Current Time (Server)
```python
current_time = datetime.now(timezone.utc)  # Always UTC
```

### 2. Input Times (Your Timestamps)
The system accepts ISO 8601 format with **any timezone**:

| Format | Interpretation | Example |
|--------|---------------|---------|
| `2025-10-16T18:00:00Z` | UTC | 6:00 PM UTC |
| `2025-10-16T18:00:00+00:00` | UTC | 6:00 PM UTC |
| `2025-10-16T14:00:00-04:00` | GMT-4 (AST/EDT) | 2:00 PM local = 6:00 PM UTC |
| `2025-10-16T15:00:00-03:00` | GMT-3 | 3:00 PM local = 6:00 PM UTC |

### 3. Automatic Conversion
Python's `datetime.fromisoformat()` automatically:
- Parses the timezone from the string
- Converts to the correct UTC equivalent when comparing times
- **You don't need to do any conversion yourself!**

## Practical Examples

### Example 1: Using UTC (Current Approach)
```json
{
  "new_task": {
    "pickup_before": "2025-10-16T18:00:00Z",
    "delivery_before": "2025-10-16T18:30:00Z"
  }
}
```
✅ Works perfectly. Times are in UTC.

### Example 2: Using Local Time (GMT-4)
```json
{
  "new_task": {
    "pickup_before": "2025-10-16T14:00:00-04:00",
    "delivery_before": "2025-10-16T14:30:00-04:00"
  }
}
```
✅ Also works! System knows 2:00 PM GMT-4 = 6:00 PM UTC

### Example 3: Mixed Timezones (Not Recommended)
```json
{
  "new_task": {
    "pickup_before": "2025-10-16T18:00:00Z",
    "delivery_before": "2025-10-16T14:30:00-04:00"
  }
}
```
✅ Still works, but confusing. Don't mix timezones in one request.

## Recommendations

### ✅ Best Practice: Use Consistent Timezone

**Option A: Continue with UTC (Simplest)**
```json
{
  "pickup_before": "2025-10-16T18:00:00Z",
  "delivery_before": "2025-10-16T18:30:00Z"
}
```

**Option B: Use Your Local Timezone (More Intuitive)**
If you're in GMT-4 (Atlantic Standard Time / Eastern Daylight Time):
```json
{
  "pickup_before": "2025-10-16T14:00:00-04:00",
  "delivery_before": "2025-10-16T14:30:00-04:00"
}
```

Both work identically! The system handles the conversion.

## Testing Different Timezones

```bash
# Test with UTC
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "new_task": {
      "id": "test",
      "job_type": "PAIRED",
      "restaurant_location": [17.14, -61.89],
      "delivery_location": [17.16, -61.91],
      "pickup_before": "2025-10-16T18:00:00Z",
      "delivery_before": "2025-10-16T18:30:00Z"
    },
    "agents": [{"driver_id": "d1", "name": "Test", "current_location": [17.13, -61.88]}],
    "current_tasks": []
  }'

# Test with GMT-4 (same time, different format)
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "new_task": {
      "id": "test",
      "job_type": "PAIRED",
      "restaurant_location": [17.14, -61.89],
      "delivery_location": [17.16, -61.91],
      "pickup_before": "2025-10-16T14:00:00-04:00",
      "delivery_before": "2025-10-16T14:30:00-04:00"
    },
    "agents": [{"driver_id": "d1", "name": "Test", "current_location": [17.13, -61.88]}],
    "current_tasks": []
  }'
```

Both requests above are **identical** to the system!

## Common Questions

### Q: What timezone is the server running in?
**A:** The server calculates `current_time` using `datetime.now(timezone.utc)`, so it always uses UTC internally.

### Q: Do I need to convert my local times to UTC?
**A:** No! Just send your times with the correct timezone offset (e.g., `-04:00` for GMT-4), and Python handles the conversion automatically.

### Q: What if I send a timestamp without timezone info?
**A:** This creates a "naive" datetime. The system might interpret it as UTC, but it's ambiguous. **Always include timezone info!**

Bad:  ❌ `"2025-10-16T14:00:00"`  (Ambiguous!)  
Good: ✅ `"2025-10-16T14:00:00-04:00"` (Clear!)

### Q: My timestamps have 'Z' at the end - what does that mean?
**A:** The 'Z' means "Zulu time" (UTC). It's shorthand for `+00:00`.

`2025-10-16T18:00:00Z` = `2025-10-16T18:00:00+00:00`

## Code Reference

From `OR_tool_prototype_batch_optimized.py`:

```python
def parse_time(time_str: str) -> datetime:
    """Parse ISO 8601 timestamp."""
    if time_str.endswith('Z'):
        time_str = time_str[:-1] + '+00:00'
    return datetime.fromisoformat(time_str)

# Current time is always UTC
current_time = datetime.now(timezone.utc)

# Parse deadline (preserves timezone info)
pickup_deadline = parse_time(new_task['pickup_before'])

# Compare times (timezone-aware comparison)
pickup_lateness = max(0, (pickup_arrival - pickup_deadline).total_seconds())
```

## Summary

✅ **System calculates current time in UTC**  
✅ **Accepts your timestamps in ANY timezone**  
✅ **Automatically converts for comparison**  
✅ **You can use UTC or local time - both work!**

**Recommendation:** Use whichever timezone makes sense for your users. If your drivers/dispatchers work in GMT-4, send times in GMT-4!

