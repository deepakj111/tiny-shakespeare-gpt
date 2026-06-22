"""
Modern GPT Architecture implementation using 2026 best practices.
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
    bias: bool = False      # Modern practice: no biases in linear layers/norms for speed/stability

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    Strictly faster and more stable than standard LayerNorm. Used in Llama 3 / modern LLMs.
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
