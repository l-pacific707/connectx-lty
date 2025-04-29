#!/bin/bash

mkdir -p models
mkdir -p results
mkdir -p logs

MODE=${1:-train}

if [ "$MODE" = "bash" ]; then
  echo "Launching into bash shell inside container..."
  docker run --rm -it --gpus all \
    --platform=linux/amd64 \
    --shm-size=3g \
    --entrypoint /bin/bash \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/results:/app/results" \
    -v "$(pwd)/logs:/app/logs" \
    ltypriv/connectx_lty:c16
else
  echo "Running train.py inside container..."
  docker run --rm -it --gpus all \
    --platform=linux/amd64 \
    --shm-size=3g \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/results:/app/results" \
    ltypriv/connectx_lty:c16 \
    2>&1 | grep -v -E "No pygame|termcolor|matplotlib"
fi
