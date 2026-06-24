# Architecture Details

This document outlines the design decisions and architectural components used in the `tiny_shakespeare_gpt` model. The model is built from scratch in PyTorch and incorporates several proven techniques from recent large language models to maximize efficiency and stability.

## Core Components

### 1. Tokenization
The project uses Byte-Pair Encoding (BPE) provided by OpenAI's `tiktoken` library. By default, it employs the `gpt2` vocabulary (vocab size 50257). This is highly efficient and standard for GPT-style implementations, providing a good balance between vocabulary size and sequence length.

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

### 6. Weight Tying
The input token embedding layer (`tok_emb`) and the final output linear projection (`lm_head`) share the same underlying weight matrix.
- **Why**: This saves a massive number of parameters (e.g., `vocab_size * n_embd`), acting as a regularizer and making the model more memory-efficient without sacrificing quality.

## Training Optimizations

### Optimizer Settings
The optimizer of choice is **AdamW**, specifically utilizing the fused implementation (`fused=True`) when running on CUDA. The weight decay is properly isolated: it is applied only to multi-dimensional weight matrices (like Linear layers and Embeddings) and explicitly disabled for 1D tensors (like RMSNorm weights).

### Dataset Loading
To avoid loading the entire dataset into RAM, the project uses NumPy's `memmap` to read the tokenized integer data directly from the binary files (`train.bin`, `val.bin`). This results in instantaneous startup times and zero memory overhead.
