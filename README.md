# Auto Allocation Assignment Brain

A high-performance route optimization service that uses Google OR-Tools to recommend optimal delivery assignments for agents. **Now 20x faster with sub-second response times!**

## Features

- **Ultra-fast route optimization** with multiple algorithm options
- **Smart algorithm selection** based on dataset size  
- **OSRM result caching** for 40-80x speed improvement
- **REST API endpoints** with built-in performance monitoring
- Support for both paired (pickup + delivery) and delivery-only tasks
- Configurable grace period system for handling late tasks
- Detailed route information in JSON format
- **Real-time performance statistics** and benchmarking

## Performance Improvements

🚀 **Major Speed Boost**: The system now offers multiple optimization levels:
- **Ultra-Fast**: 0.3s average (20x faster) - ideal for real-time applications
- **Optimized**: 2.5s average (8x faster) - maintains OR-Tools accuracy with caching
- **Original**: 6.5s average - maximum precision for small datasets

See `OR_Tools_prototype/SPEED_IMPROVEMENTS_SUMMARY.md` for detailed performance analysis.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the optimized Flask application:
```bash
python OR_Tools_prototype/app.py
```

The server will start on port 8080 by default with automatic algorithm selection.

## API Endpoints

### Health Check
```
GET /health
```
Returns the health status and performance statistics.

### Get Recommendations (Smart Auto-Selection)
```
POST /recommend
```
Automatically selects the best algorithm based on your dataset size:
- **Small** (≤10 agents, ≤20 tasks): Uses `original` for maximum accuracy
- **Medium** (≤50 agents, ≤100 tasks): Uses `optimized` for best balance  
- **Large**: Uses `ultra_fast` for only practical speed

### Force Specific Algorithms
```
POST /recommend/ultra-fast   # Force ultra-fast heuristic algorithm
POST /recommend/optimized    # Force optimized OR-Tools algorithm
```

### Performance Monitoring
```
GET /stats                   # Get detailed performance statistics
POST /cache/clear           # Clear OSRM cache for testing
POST /benchmark             # Compare algorithms on your dataset
```

### Algorithm Selection Override

Add `"algorithm"` to your request body to force a specific algorithm:
```json
{
    "algorithm": "ultra_fast",  // Options: "auto", "ultra_fast", "optimized", "original"
    "new_task": { ... },
    "agents": [ ... ],
    "current_tasks": [ ... ]
}
```

### Enhanced Response Format
```json
{
    "task_id": "task123",
    "algorithm_used": "ultra_fast",
    "execution_time_seconds": 0.342,
    "cache_hits": 15,
    "recommendations": [ ... ]
}
```

## Performance Monitoring

### Real-time Statistics
```bash
curl http://localhost:8080/stats
```
Returns:
- Average response times
- Algorithm usage percentages  
- Cache hit rates
- Total request counts

### Benchmarking
```bash
curl -X POST http://localhost:8080/benchmark \
  -H "Content-Type: application/json" \
  -d @your_request.json
```
Compares execution times of all available algorithms on your data.

## Documentation

- `OR_Tools_prototype/README_OPTIMIZATIONS.md` - Technical optimization details
- `OR_Tools_prototype/SPEED_IMPROVEMENTS_SUMMARY.md` - User-friendly performance guide
- `API_DOCUMENTATION.md` - Complete API reference

## Testing

Run the enhanced test script with performance comparison:
```bash
python test_recommendation.py
```

Run the dedicated performance test:
```bash
python OR_Tools_prototype/performance_test.py
```

## License

MIT 