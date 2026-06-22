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
(To be added)
