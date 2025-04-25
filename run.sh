#!/bin/bash

docker run --rm -it --gpus all \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/results:/app/results" \
  ltypriv/connectx_lty:patched \
  2>&1 | grep -v -E "No pygame|termcolor|matplotlib"