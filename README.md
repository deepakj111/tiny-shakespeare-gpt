# Tiny Shakespeare GPT

Building and training a GPT model from scratch using PyTorch on the tiny shakespeare dataset.

## Requirements & Setup

- **Python**: `>=3.11`
- **Hardware**: A CUDA-capable GPU is strongly recommended for training, though the code supports CPU for testing.

This project uses `uv` for reproducible and lightning-fast dependency management. To install dependencies:

```bash
uv sync
```

*(Optional)* If you plan to use experiment tracking, log in to Weights & Biases:
```bash
uv run wandb login
```

## Project Structure

A clean, modular architecture separates the core model logic from the execution scripts:

```
tiny-shakespeare-gpt/
├── src/
│   └── tiny_shakespeare_gpt/
│       ├── config.py      # Dataclasses for model and training hyperparameters
│       ├── dataset.py     # Memory-mapped PyTorch dataset
│       ├── model.py       # Core GPT architecture (RoPE, GQA, SwiGLU, etc.)
│       └── tokenizer.py   # BPE tokenizer wrapper
├── scripts/
│   ├── download_dataset.py
│   ├── prepare_data.py
│   ├── train.py           # Multi-GPU (DDP) enabled training loop
│   ├── generate.py        # Inference and sampling script
│   └── serve.py           # FastAPI streaming server
├── tests/
│   └── test_model.py      # Unit tests
└── ARCHITECTURE.md        # Deep dive into engineering decisions
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

Thanks to `pyproject.toml` project scripts, you can simply run:

```bash
uv run train
```

The script will periodically evaluate the model on the training and validation sets, print the losses, and generate a sample text to qualitatively track the model's progress. 

### Configuration & Resumability

All hyperparameters and settings are managed in `src/tiny_shakespeare_gpt/config.py` via the `TrainConfig` dataclass. You can adjust `max_iters`, `block_size`, `batch_size`, and other training parameters there.

- **Experiment Tracking**: Set `wandb_log = True` in `config.py` to automatically stream your metrics and generated text samples to your [Weights & Biases](https://wandb.ai/) dashboard.
- **Resuming Training**: If your training run crashes or is preempted, simply set `resume = True` in `config.py`. The script will automatically load `out/ckpt.pt`, restore the optimizer state, and precisely re-seed the RNG state so it picks up exactly where it left off.

## Multi-GPU Distributed Training (DDP)

The training script natively supports PyTorch Distributed Data Parallel (DDP) for scaling across multiple GPUs and compute nodes. 

To launch a distributed training run across all available GPUs on a single node, use `torchrun`:

```bash
uv run torchrun --standalone --nproc_per_node=gpu scripts/train.py
```

**Testing DDP Locally:**
If you want to test the multi-process logic on a machine with a single GPU (or no GPU), you can simulate it by forcing the number of processes. The code is smart enough to detect if you are requesting more processes than available GPUs and will automatically fall back to the CPU (`gloo` backend) to prevent crashes:

```bash
uv run torchrun --standalone --nproc_per_node=2 scripts/train.py
```

## Generation

After the model has been trained and a checkpoint is saved, you can use it to generate new Shakespeare-like text.

Run the generation script via its CLI alias:

```bash
uv run generate --prompt "ROMEO:"
```

This script loads the latest checkpoint from the `out/` directory, initializes the model and tokenizer, and samples new text. You can pass arguments like `--max_new_tokens`, `--temperature`, and `--top_k` to experiment with different output styles.

## API Serving

Once the model is trained, you can expose it as a REST API using FastAPI. The server leverages a global lifespan context for optimized model loading and features an endpoint that supports token-by-token generation via Server-Sent Events (SSE).

Run the server via its CLI alias:
```bash
uv run serve
```

Test the streaming generation endpoint:
```bash
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "ROMEO:", "stream": true, "max_new_tokens": 100}'
```

You can also visit `http://localhost:8000/` in your browser to interact with the API via the built-in Swagger UI.

## Containerization (Docker)

If you want to deploy the project or test it without setting up a local Python environment, a `Dockerfile` is provided. The Docker image uses `uv` for lightning-fast dependency installation and robust caching layers. 

It is configured with a smart entrypoint: if no model checkpoint is found, the container will automatically download the dataset, train a quick model on CPU, and then launch the FastAPI server.

Build the image:
```bash
docker build -t tiny-shakespeare-gpt .
```

Run the container:
```bash
docker run -p 8000:8000 tiny-shakespeare-gpt
```

## Code Quality Enforcement

To ensure production-grade quality, the codebase enforces strict linting, formatting, and static type-checking using `ruff` and `mypy`. 

Run the automated checks:
```bash
uv run ruff check .
uv run ruff format .
uv run mypy src/ scripts/
```

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
- **Checkpoint Tracking & Resumability**: Automatically tracks and saves the model, optimizer, and RNG states. You can perfectly resume training from the exact step it crashed.
- **Distributed Data Parallel (DDP)**: Scales seamlessly across multi-GPU clusters using `torchrun`, with intelligent fallbacks for local CPU testing.
- **Qualitative Evaluation**: Intermittently generates sample text during training evaluation loops to visually confirm grammar and learning progress.
- **Experiment Tracking**: Integrated with Weights & Biases (`wandb`) for real-time visualization of loss curves and text samples.

For more details on the design, see [ARCHITECTURE.md](ARCHITECTURE.md).
