# Tiny Shakespeare GPT

Building and training a GPT model from scratch using PyTorch on the tiny shakespeare dataset.

## Setup

This project uses `uv` for reproducible dependency management. To sync dependencies:

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

Next, tokenize the text into integer sequences and save them as binary files (`train.bin` and `val.bin`) for loading during training:

```bash
uv run python scripts/prepare_data.py
```

## Training

Once the data is prepared, you can start the training loop. The training script will initialize the model architecture, setup the optimizer, and begin training on the tiny shakespeare dataset.

Run the following command:

```bash
uv run python scripts/train.py
```

The script will periodically evaluate the model on the training and validation sets and print the losses. By default, it runs for a quick 500 steps, but you can adjust `max_iters`, `block_size`, and `batch_size` in the script for longer training. During training, the model tracks validation loss and saves the best checkpoint in the `out/` directory.

## Generation

After the model has been trained and a checkpoint is saved, you can use it to generate new Shakespeare-like text.

Run the generation script:

```bash
uv run python scripts/generate.py
```

This script loads the latest checkpoint from the `out/` directory, initializes the model and tokenizer, and samples new text. You can edit the script to change parameters such as `max_new_tokens`, `temperature`, and `top_k` to experiment with different output styles.

## Limitations & Model Scope

It is important to understand what this model is and what it is not:

- **It is a Base Foundational Model:** This model is trained purely on the objective of **next-token prediction** using a raw, unstructured dataset (Tiny Shakespeare). 
- **It is NOT Conversational:** The model has not undergone Supervised Fine-Tuning (SFT) on question-and-answer pairs, nor has it been optimized with Reinforcement Learning from Human Feedback (RLHF). Because of this, it does not act like an "assistant" or a chatbot. If you prompt it with a question, it will not necessarily answer it; instead, it will simply attempt to continue the text in the style of a Shakespearean play.
- **It is Small and Undertrained:** Being an educational portfolio project, the model has a very low parameter count and is trained on a tiny corpus (~1MB) for a minimal number of steps. While it effectively learns the syntax, formatting, and vocabulary of Shakespeare, it does not possess deep semantic logic, factual knowledge, or long-term coherence.
## Features

### Architecture Improvements

- **Rotary Position Embeddings (RoPE)**: Replaces absolute positional embeddings with relative ones for better generalization.
- **Grouped-Query Attention (GQA)**: Reduces the number of key/value heads for faster inference and lower memory consumption.
- **RMSNorm**: An alternative to standard LayerNorm.
- **SwiGLU FeedForward**: Replaces the standard ReLU/GELU MLPs with Swish-Gated Linear Units.
- **Flash Attention**: Uses PyTorch's scaled dot product attention for highly optimized, memory-efficient exact attention.
- **Weight Tying**: Shares weights between the token embedding layer and the final output layer.
- **Residual Scaling**: Custom initialization (`1/sqrt(2 * n_layer)`) on residual projections to prevent variance explosion.
- **Vocabulary Padding**: Pads the GPT-2 vocabulary to a multiple of 64.

### Training Optimizations

- **Mixed Precision Training**: Uses PyTorch `autocast` with `bfloat16` (or `float16`) to speed up training and reduce memory usage without losing stability.
- **Cosine Learning Rate Scheduler**: Implements a cosine annealing schedule with a linear warmup phase for stable and effective convergence.
- **Gradient Accumulation**: Decouples effective batch size from VRAM limits.
- **Gradient Clipping**: Prevents exploding gradients during training.
- **Checkpoint Tracking**: Automatically tracks and saves only the model checkpoint with the best validation loss.

For more details on the design, see [ARCHITECTURE.md](ARCHITECTURE.md).
