#!/bin/bash

docker run --rm -it \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/results:/app/results" \
  ltypriv/connectx_lty \
  2>&1 | grep -v -E "No pygame|termcolor|matplotlib"
