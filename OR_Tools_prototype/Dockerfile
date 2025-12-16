FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Use gunicorn with gevent worker for WebSocket support
# --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker enables WebSocket
# -k gevent uses gevent async workers for better concurrency
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--worker-class", "geventwebsocket.gunicorn.workers.GeventWebSocketWorker", "--workers", "1", "--timeout", "300", "app:app"]
