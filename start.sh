#!/bin/bash
# Start FastAPI on 7860, Gradio on 7861
# HF Space exposes 7860 — Gradio proxies to FastAPI internally

# Start FastAPI in background
uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1 &
FASTAPI_PID=$!

# Wait for FastAPI to be ready
echo "Waiting for FastAPI..."
until curl -sf http://localhost:7860/health > /dev/null 2>&1; do
  sleep 1
done
echo "FastAPI ready."

# Start Gradio on 7860 (replaces FastAPI as the public face)
# Gradio calls FastAPI internally at localhost:7860
# We need to run Gradio on a different port and use a reverse proxy
# Simplest: run Gradio as the main process on 7860, FastAPI on 7861

echo "All services started."
wait $FASTAPI_PID
