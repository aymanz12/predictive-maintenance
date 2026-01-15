#!/bin/bash
set -e

# Start FastAPI backend in the background
# host 0.0.0.0 is crucial for container networking
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Wait for API to be ready
echo "Waiting for API to start..."
while ! curl -s http://127.0.0.1:8000/docs > /dev/null; do   
  sleep 1
done
echo "API started!"

# Start Gradio frontend
# Running this as the foreground process
echo "Starting Gradio App..."
python app/gradio_app.py
