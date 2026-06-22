"""
Modern GPT Architecture implementation.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257 # GPT-2 vocab size
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 2      # Grouped-Query Attention (fewer KV heads than query heads)
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False      # No biases in linear layers/norms for speed/stability

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    Strictly faster and more stable than standard LayerNorm.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for stable normalization, then cast back
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    SwiGLU FeedForward Network.
    Uses an expansion factor of 8/3, which is standard in Llama/modern LLMs.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = int(8 * config.n_embd / 3)
        # Ensure hidden_dim is a multiple of 256 for optimal performance
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation: (xW1 * sigmoid(xW1)) * xW3
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
