#!/bin/bash
set -e

# If the model checkpoint doesn't exist, run the training pipeline first
if [ ! -f "out/model.safetensors" ]; then
    echo "=========================================================="
    echo "Checkpoint not found. Running quick training pipeline..."
    echo "=========================================================="
    
    echo "-> Downloading dataset..."
    uv run python scripts/download_dataset.py
    
    echo "-> Preparing dataset..."
    uv run python scripts/prepare_data.py
    
    echo "-> Training model (This might take a few minutes on CPU)..."
    uv run python scripts/train.py
    
    echo "=========================================================="
    echo "Training Complete! Proceeding to launch server..."
    echo "=========================================================="
fi

echo "-> Starting FastAPI Server on port 8000..."
exec uv run python scripts/serve.py
