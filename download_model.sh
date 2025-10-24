#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <model_id>"
  exit 1
fi

MODEL_ID="$1"
LOCAL_DIR="${MODEL_ID#*/}"

echo "Downloading model: $MODEL_ID"
echo "Target directory: $LOCAL_DIR"

# If directory exists, overwrite it
if [ -d "$LOCAL_DIR" ]; then
  rm -rf "$LOCAL_DIR"
fi

# Create the directory
mkdir -p "$LOCAL_DIR"

# Run the download
echo "Running: huggingface-cli download $MODEL_ID --local-dir $LOCAL_DIR"
huggingface-cli download "$MODEL_ID" --local-dir "$LOCAL_DIR"

echo "Model saved to: $LOCAL_DIR"
huggingface-cli cache delete --yes

