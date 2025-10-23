#!/usr/bin/env bash
# General script to download any Hugging Face model locally.
# If the target directory exists, it will be overwritten.

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <model_id> <local_dir>"
  exit 1
fi

MODEL_ID="$1"
LOCAL_DIR="$2"

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

