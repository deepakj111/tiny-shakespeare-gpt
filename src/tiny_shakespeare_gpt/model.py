"""
GPT Architecture implementation.
"""
from dataclasses import dataclass
import inspect
import math
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
    Uses an expansion factor of 8/3, which is standard in Llama/other LLMs.
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

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor, 
    xk: torch.Tensor, 
    freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embeddings to query and key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # freqs_cis shape: (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self Attention with Grouped Query Attention (GQA)
    and Rotary Position Embeddings (RoPE).
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        
        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=config.bias)
        self.wk = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.wv = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.max_seq_len = config.block_size

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int = 0, use_cache: bool = False) -> torch.Tensor:
        B, T, C = x.shape
        
        xq = self.wq(x).view(B, T, self.n_head, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_head, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_head, self.head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        if use_cache:
            if start_pos == 0 or not hasattr(self, 'cache_k'):
                # Initialize cache lazily
                self.cache_k = torch.zeros((B, self.max_seq_len, self.n_kv_head, self.head_dim), dtype=xk.dtype, device=xk.device)
                self.cache_v = torch.zeros((B, self.max_seq_len, self.n_kv_head, self.head_dim), dtype=xv.dtype, device=xv.device)
                
            self.cache_k[:B, start_pos:start_pos+T] = xk
            self.cache_v[:B, start_pos:start_pos+T] = xv
            
            xk = self.cache_k[:B, :start_pos+T]
            xv = self.cache_v[:B, :start_pos+T]
        
        # Expand KV heads to match query heads for GQA
        if self.n_kv_head < self.n_head:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)
            
        # Transpose to (B, n_head, T, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Flash attention
        y = F.scaled_dot_product_attention(
            xq, xk, xv, 
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=(T > 1)
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.wo(y))

class Block(nn.Module):
    """Transformer Block containing Self-Attention and FeedForward networks."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int = 0, use_cache: bool = False) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs_cis, start_pos, use_cache)
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    """The main GPT Model."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying: share weights between token embeddings and lm head
        self.tok_emb.weight = self.lm_head.weight

        # Precompute rotary embeddings
        head_dim = config.n_embd // config.n_head
        freqs_cis = precompute_freqs_cis(head_dim, config.block_size)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        idx: torch.Tensor, 
        targets: torch.Tensor = None,
        start_pos: int = 0,
        use_cache: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        
        # Token embeddings
        x = self.tok_emb(idx)
        x = self.dropout(x)
        
        # Slice rotary embeddings for the current sequence length
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, freqs_cis, start_pos, use_cache)
            
        x = self.norm(x)
        
        if targets is not None:
            # If we are given some desired targets, calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference-time optimization: only forward the lm_head on the last position
            logits = self.lm_head(x[:, [-1], :]) # using list [-1] to preserve the time dim
            loss = None
            
        return logits, loss

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: tuple[float, float], device_type: str) -> torch.optim.Optimizer:
        # separate out all parameters to those that will and won't experience regularizing weight decay
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and RMSNorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        
        return optimizer

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, using KV caching.
        """
        B, T = idx.shape
        
        # Process the prompt (prefill)
        if T > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            T = idx.shape[1]
            
        logits, _ = self(idx, start_pos=0, use_cache=True)
        
        next_token_logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(next_token_logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        generated_ids = [idx_next]
        start_pos = T
        
        for _ in range(1, max_new_tokens):
            if start_pos >= self.config.block_size:
                break # exceeded maximum context length
                
            logits, _ = self(idx_next, start_pos=start_pos, use_cache=True)
            
            next_token_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(next_token_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            generated_ids.append(idx_next)
            start_pos += 1
            
        # Clear cache
        for block in self.blocks:
            if hasattr(block.attn, 'cache_k'):
                del block.attn.cache_k
                del block.attn.cache_v
                
        return torch.cat([idx] + generated_ids, dim=1)
