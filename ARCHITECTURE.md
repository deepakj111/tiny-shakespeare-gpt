# Architecture Details

This document outlines the design decisions and architectural components used in the `tiny_shakespeare_gpt` model. The model is built from scratch in PyTorch and incorporates several proven techniques from recent large language models to maximize efficiency and stability.

## Core Components

### 1. Tokenization and Vocabulary
The project uses Byte-Pair Encoding (BPE) provided by OpenAI's `tiktoken` library. By default, it employs the `gpt2` vocabulary. However, the vocabulary size is padded from 50257 to 50304 (a multiple of 64). 
- **Why**: Padding the vocabulary to a multiple of 64 ensures optimal hardware utilization on NVIDIA GPUs (Tensor Cores), leading to faster training times without impacting model quality.

### 2. Normalization (RMSNorm)
Instead of standard Layer Normalization (`nn.LayerNorm`), this architecture uses Root Mean Square Normalization (**RMSNorm**).
- **Why**: RMSNorm drops the mean-centering operation, relying only on variance scaling. This makes it strictly faster while maintaining identical training stability and convergence properties.
- **Biases**: Biases are completely removed from the normalization layers.

### 3. Positional Embeddings (RoPE)
The model does away with absolute positional embeddings in favor of **Rotary Position Embeddings (RoPE)**.
- **How it works**: RoPE encodes positional information directly into the query and key vectors of the self-attention mechanism via complex rotations.
- **Why**: This provides relative positional awareness, improving generalization to longer sequence lengths and offering a strong inductive bias for sequence modeling.

### 4. Attention Mechanism (GQA & Flash Attention)
The attention block incorporates two major optimizations:
- **Grouped-Query Attention (GQA)**: Instead of Multi-Head Attention (where every query head has a dedicated key/value head), GQA groups multiple query heads to share a single key/value head. This drastically reduces the memory bandwidth required for the KV cache during inference, allowing for faster decoding.
- **Flash Attention**: The scaled dot-product attention is computed using PyTorch's `F.scaled_dot_product_attention`, which dispatches to memory-efficient CUDA kernels (like FlashAttention) under the hood. This eliminates the need to materialize the huge `(T, T)` attention matrix in VRAM.

### 5. FeedForward Network (SwiGLU)
The standard MLP block is replaced with a **SwiGLU** (Swish-Gated Linear Unit) network.
- **Why**: Instead of `ReLU(x * W1) * W2`, SwiGLU uses a gating mechanism: `(x * W1 * Swish(x * W1)) * W3`. This has been shown to yield better performance per parameter.
- **Sizing**: We use a hidden dimension expansion factor of `8/3` (rounded to a multiple of 256 for optimal hardware utilization), rather than the traditional factor of `4`.

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

### Dataset Loading
To avoid loading the entire dataset into RAM, the project uses NumPy's `memmap` to read the tokenized integer data directly from the binary files (`train.bin`, `val.bin`). This results in instantaneous startup times and zero memory overhead.
