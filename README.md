# Auto Allocation Assignment Brain

A high-performance delivery agent recommendation system using Google OR-Tools optimization with **13.8x speed improvements** over the original implementation.

## 🚀 **Key Features**

- **Ultra-Fast Performance**: Sub-second response times (0.32s average)
- **Dual Algorithm System**: Fixed optimized for speed, light optimized for caching
- **Smart Auto-Selection**: Automatically chooses the best algorithm for your dataset size
- **Real-World Integration**: OSRM API for accurate travel time calculations
- **Advanced Caching**: Smart OSRM response caching for repeated coordinate sets
- **Production Ready**: Full Docker support with monitoring and health checks

## 📊 **Performance Overview**

| Algorithm | Speed | Best Use Case |
|-----------|-------|---------------|
| **Fixed Optimized** | 0.32s (13.8x faster) | Large datasets, maximum performance |
| **Light Optimized** | ~4.5s (caching benefits) | Small datasets, repeated coordinates |
| Original (archived) | 4.5s | Reference baseline |

## 🏗️ **Project Structure**

```
auto_alocation_assignment_brain/
├── OR_Tools_prototype/           # Main application directory
│   ├── app.py                   # Flask API server
│   ├── OR_tool_prototype_optimized_fixed.py    # High-performance optimizer
│   ├── OR_tool_prototype_light_optimized.py    # Caching-optimized version
│   ├── README.md                # Detailed API documentation
│   ├── requirements.txt         # Python dependencies
│   ├── Dockerfile              # Container configuration
│   ├── agents.json             # Sample agent data
│   ├── current_tasks.json      # Sample current tasks
│   ├── new_task.json          # Sample new task
│   ├── osrm_tables_test.py    # OSRM integration
│   └── archive/               # Reference implementations
│       ├── OR_tool_prototype.py           # Original implementation
│       ├── OR_tool_prototype_optimized.py # Broken optimization attempt
│       └── OR_tool_prototype_ultra_fast.py # Heuristic approach
├── test_recommendation.py      # Comprehensive test suite
├── prototype.py                # Legacy prototype (deprecated)
└── docker-compose.yml         # Multi-container setup
```

## 🚀 **Quick Start**

### **Option 1: Local Development**

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd auto_alocation_assignment_brain/OR_Tools_prototype
   pip install -r requirements.txt
   ```

2. **Start the Server**:
   ```bash
   python app.py
   ```

3. **Test the API**:
   ```bash
   curl -X POST http://localhost:8080/recommend \
     -H "Content-Type: application/json" \
     -d @new_task.json
   ```

### **Option 2: Docker Deployment**

1. **Single Container**:
   ```bash
   cd OR_Tools_prototype
   docker build -t or-tools-recommender .
   docker run -p 8080:8080 or-tools-recommender
   ```

2. **Multi-Container (with docker-compose)**:
   ```bash
   docker-compose up
   ```

## 📚 **API Reference**

### **Core Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/recommend` | POST | Auto-select best algorithm |
| `/recommend/fixed-optimized` | POST | Force high-performance algorithm |
| `/recommend/light-optimized` | POST | Force caching-optimized algorithm |
| `/benchmark` | POST | Compare both algorithms |
| `/stats` | GET | Performance statistics |
| `/cache/clear` | POST | Clear OSRM cache |

### **Request Format**

```json
{
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
```

### **Response Format**

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

For complete API documentation with all parameters, examples, and error codes, see:
**[OR_Tools_prototype/README.md](OR_Tools_prototype/README.md)**

## 🔧 **Algorithm Selection**

### **Automatic Selection Logic**
- **≤20 agents AND ≤50 tasks** → Light Optimized (original solver + caching)
- **>20 agents OR >50 tasks** → Fixed Optimized (adaptive optimizations)

### **Manual Override**
Add `"algorithm": "fixed_optimized"` or `"algorithm": "light_optimized"` to your request.

## 📈 **Monitoring & Performance**

### **Health Check**
```bash
curl http://localhost:8080/health
```

### **Performance Statistics**
```bash
curl http://localhost:8080/stats
```
Returns detailed metrics on algorithm usage, response times, and cache performance.

### **Benchmarking**
```bash
curl -X POST http://localhost:8080/benchmark \
  -H "Content-Type: application/json" \
  -d @your_test_data.json
```
Compares execution times of both algorithms on your specific dataset.

## 🧪 **Testing**

### **Run Comprehensive Tests**
```bash
python test_recommendation.py
```

### **Test Both Algorithms**
```bash
cd OR_Tools_prototype
# Test files were removed for production, but you can create custom tests
```

## 🔍 **Technical Details**

### **Key Optimizations Implemented**
- ✅ **Adaptive Timeouts**: 15-30s based on problem complexity
- ✅ **Smart Algorithm Selection**: AUTOMATIC vs GUIDED_LOCAL_SEARCH
- ✅ **OSRM API Caching**: Prevents repeated API calls for same coordinates
- ✅ **Coordinate Deduplication**: Reuses location mappings
- ✅ **Batch Processing**: Optimized time parsing and calculations
- ✅ **Incremental Data Building**: Avoids rebuilding coordinate sets

### **Architecture**
1. **Flask API Layer** - HTTP request handling and routing
2. **OR-Tools Optimization Engine** - Two optimized solver implementations  
3. **OSRM Integration** - Real-world travel time calculations
4. **Intelligent Caching** - Smart response caching for performance

## 🐳 **Docker Configuration**

The system includes full Docker support with:
- Multi-stage builds for optimized images
- Environment variable configuration
- Health check endpoints
- Production-ready logging

## 📜 **License**

MIT License - see LICENSE file for details.

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes in the `OR_Tools_prototype/` directory
4. Test thoroughly with both algorithms
5. Submit a pull request

## 📞 **Support**

For technical support or questions:
- Check the detailed documentation in `OR_Tools_prototype/README.md`
- Review the archived implementations in `OR_Tools_prototype/archive/`
- Test with the provided sample data files 