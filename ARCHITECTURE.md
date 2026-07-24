# Architecture Details

> **Design Philosophy**: The `tiny_shakespeare_gpt` architecture is designed to bridge the gap between educational simplicity and modern production realities. While classic tutorials implement the standard GPT-2 architecture, this project replaces legacy components with state-of-the-art techniques used in models like **LLaMA 3** and **Mixtral**. The goal is to provide a pedagogical but highly optimized model that scales efficiently and trains rapidly.

This document outlines the design decisions, mathematical underpinnings, and performance optimizations built into the model.

---

## Core Components

### 1. Tokenization and Vocabulary
The project uses Byte-Pair Encoding (BPE) provided by OpenAI's `tiktoken` library. By default, it employs the state-of-the-art `o200k_base` vocabulary (introduced with GPT-4o). The vocabulary size is padded to a multiple of 64. 
- **Why**: Padding the vocabulary to a multiple of 64 ensures hardware utilization on NVIDIA GPUs (Tensor Cores), leading to faster training times without impacting model quality.

### 2. Normalization (RMSNorm)
Instead of standard Layer Normalization (`nn.LayerNorm`), this architecture uses Root Mean Square Normalization (**RMSNorm**).
- **Why**: RMSNorm drops the mean-centering operation, relying only on variance scaling. This makes it faster while maintaining identical training stability and convergence properties.
- **Biases**: Biases are completely removed from the normalization layers.

### 3. Positional Embeddings (RoPE)
The model does away with absolute positional embeddings in favor of **Rotary Position Embeddings (RoPE)**.
- **How it works**: RoPE encodes positional information directly into the query and key vectors of the self-attention mechanism via complex rotations.
- **Why**: This provides relative positional awareness, improving generalization to longer sequence lengths and offering a strong inductive bias for sequence modeling.

### 4. Attention Mechanism (GQA & Flash Attention)
The attention block incorporates two major optimizations:
- **Grouped-Query Attention (GQA)**: Instead of Multi-Head Attention (where every query head has a dedicated key/value head), GQA groups multiple query heads to share a single key/value head. This drastically reduces the memory bandwidth required for the KV cache during inference, allowing for faster decoding.
- **Flash Attention**: The scaled dot-product attention is computed using PyTorch's `F.scaled_dot_product_attention`, which dispatches to memory-efficient CUDA kernels (like FlashAttention) under the hood. This eliminates the need to materialize the huge `(T, T)` attention matrix in VRAM.

### 5. Sparse Mixture of Experts (MoE) & SwiGLU
The standard dense MLP block is replaced with a **Sparse Mixture of Experts (MoE)** architecture using **SwiGLU** (Swish-Gated Linear Unit) networks for the individual experts.
- **Router**: A gating network dynamically routes each token to the top-K (typically 2) experts out of N available experts.
- **Why MoE**: This allows the model to scale its parameter count massively while keeping the active parameter count (and computational cost) per token low, a paradigm heavily utilized by GPT-4 and modern state-of-the-art models.
- **SwiGLU Experts**: Instead of `ReLU(x * W1) * W2`, each expert uses a gating mechanism: `(x * W1 * Swish(x * W1)) * W3`. We use a hidden dimension expansion factor of `8/3` (rounded to a multiple of 256), rather than the traditional factor of `4`.

### 6. Weight Initialization and Tying
The model employs a custom scaled initialization strategy to maintain stability.
- **Residual Scaling**: The residual projections (the final linear layers in the attention and feed-forward blocks) are initialized with a standard deviation scaled by `1/sqrt(2 * n_layer)`. This prevents variance explosion deep in the network.
- **Weight Tying**: The input token embedding layer (`tok_emb`) and the final output linear projection (`lm_head`) share the same underlying weight matrix. This saves a massive number of parameters (e.g., `vocab_size * n_embd`), acting as a regularizer and making the model more memory-efficient without sacrificing quality.

## Training Optimizations

### Optimizer Settings
The optimizer of choice is **AdamW**, specifically utilizing the fused implementation (`fused=True`) when running on CUDA. The weight decay is properly isolated: it is applied only to multi-dimensional weight matrices (like Linear layers and Embeddings) and explicitly disabled for 1D tensors (like RMSNorm weights).

### Mixed Precision & Bfloat16
The training loop utilizes PyTorch's `autocast` context manager. If supported by the GPU hardware, it defaults to `bfloat16` (Brain Floating Point), which provides the same dynamic range as `float32` while halving the memory requirement and significantly accelerating matrix multiplications on Tensor Cores.

### Learning Rate Scheduler
A **Cosine Annealing** learning rate scheduler is implemented, complete with a linear warmup phase. This prevents early divergence by slowly ramping up the learning rate, then smoothly decays it to a minimum threshold, optimizing convergence in the later stages of training.

### Gradient Accumulation & Clipping
To allow for training with large effective batch sizes on limited VRAM hardware, **Gradient Accumulation** is used. Instead of updating weights every micro-batch, gradients are accumulated and stepped after several micro-batches. Additionally, **Gradient Clipping** (`clip_grad_norm_`) is applied just before the optimizer step to maintain stability and prevent exploding gradients.

### Distributed Data Parallel (DDP)
The training architecture seamlessly scales across multiple GPUs using PyTorch's `DistributedDataParallel`.
- **Dataset Sharding**: `DistributedSampler` is used to split the dataset, ensuring each GPU processes a unique shard of data without overlap.
- **Gradient Synchronization**: The model utilizes the `no_sync()` context manager during gradient accumulation to disable gradient broadcasting until the final micro-step, significantly improving multi-GPU throughput.
- **Local Simulation**: If launched via `torchrun` with more processes than available GPUs, it intelligently falls back to the CPU `gloo` backend to allow local testing and debugging of the distributed logic.

### Checkpoint Resumability & Safetensors
To guard against crashes or preemptions, the model saves a complete snapshot of its state at each validation improvement. The project utilizes the **Safetensors** format for secure and zero-copy loading of model weights.
- **State Preservation**: The model weights are serialized to `model.safetensors`, avoiding the inherent security and performance issues of Python's `pickle`. The `optimizer.state_dict()`, metadata, and the training iteration are saved alongside it in a separate PyTorch metadata file.
- **RNG Synchronization**: Explicitly saves and restores the random number generator states (`torch.get_rng_state()` and `cuda.get_rng_state()`) to guarantee that resuming a run is mathematically identical to an uninterrupted run.

### Logging & Experiment Tracking
- **Weights & Biases (wandb)**: Integrated for real-time tracking of training and validation loss curves, learning rates, and metrics.
- **Qualitative Evaluation**: During evaluation phases, the model runs a live inference pass to generate a small block of text. This is logged to the console and `wandb`, providing a visual indicator of the model's grasp of syntax and vocabulary over time.

### Dataset Loading
To avoid loading the entire dataset into RAM, the project uses NumPy's `memmap` to read the tokenized integer data directly from the binary files (`train.bin`, `val.bin`). This results in instantaneous startup times and zero memory overhead.
