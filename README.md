# Tiny Shakespeare GPT

Building and training a GPT model from scratch using PyTorch on the tiny shakespeare dataset.

## Setup

This project uses `uv` for modern, reproducible dependency management. To sync dependencies:

```bash
uv sync
```

### Data Preparation

Before training the model, you need to download the Tiny Shakespeare dataset. A script is provided to fetch the text and save it to the `data/` directory.

Run the following command using `uv` to ensure it executes in the project's isolated virtual environment:

```bash
uv run python scripts/download_dataset.py
```

This will download `input.txt` (approximately 1.1MB of Shakespeare text) into the `data/` folder.

Next, tokenize the text into integer sequences and save them as binary files (`train.bin` and `val.bin`) for highly efficient loading during training:

```bash
uv run python scripts/prepare_data.py
```

## Training

Once the data is prepared, you can start the training loop. The training script will initialize the model architecture, setup the optimizer, and begin training on the tiny shakespeare dataset.

Run the following command:

```bash
uv run python scripts/train.py
```

The script will periodically evaluate the model on the training and validation sets and print the losses. By default, it runs for a quick 500 steps, but you can adjust `max_iters`, `block_size`, and `batch_size` in the script for longer training. Upon completion, a checkpoint is saved in the `out/` directory.

## Features

This implementation includes several architectural improvements commonly found in state-of-the-art language models:

- **Rotary Position Embeddings (RoPE)**: Replaces absolute positional embeddings with relative ones for better generalization.
- **Grouped-Query Attention (GQA)**: Reduces the number of key/value heads for faster inference and lower memory consumption.
- **RMSNorm**: A strictly faster and more stable alternative to standard LayerNorm.
- **SwiGLU FeedForward**: Replaces the standard ReLU/GELU MLPs with Swish-Gated Linear Units.
- **Flash Attention**: Uses PyTorch's scaled dot product attention for highly optimized, memory-efficient exact attention.
- **Weight Tying**: Shares weights between the token embedding layer and the final output layer.

For more details on the design, see [ARCHITECTURE.md](ARCHITECTURE.md).
