import torch
from tiny_shakespeare_gpt.model import GPTConfig, RMSNorm

def test_gpt_config():
    config = GPTConfig()
    
    # Check default modern attributes exist
    assert config.vocab_size == 50257
    assert config.n_layer == 6
    assert config.bias is False

def test_rmsnorm():
    torch.manual_seed(42)
    dim = 384
    norm = RMSNorm(dim)
    
    # Create batch of shape (batch_size, seq_len, dim)
    x = torch.randn(2, 128, dim)
    out = norm(x)
    
    # Output should preserve shape
    assert out.shape == x.shape
    
    # RMSNorm ensures the root mean square of the output is approx 1
    rms = torch.sqrt(out.pow(2).mean(-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

from tiny_shakespeare_gpt.model import FeedForward

def test_feedforward():
    torch.manual_seed(42)
    config = GPTConfig(n_embd=384, dropout=0.0, bias=False)
    ffn = FeedForward(config)
    
    # Create batch of shape (batch_size, seq_len, n_embd)
    x = torch.randn(2, 128, config.n_embd)
    out = ffn(x)
    
    # Output should preserve shape
    assert out.shape == x.shape

from tiny_shakespeare_gpt.model import precompute_freqs_cis, apply_rotary_emb, CausalSelfAttention

def test_rope():
    dim = 64
    seq_len = 128
    
    freqs_cis = precompute_freqs_cis(dim, seq_len)
    assert freqs_cis.shape == (seq_len, dim // 2)
    
    # Test applying rope
    xq = torch.randn(2, seq_len, 4, dim)
    xk = torch.randn(2, seq_len, 2, dim)
    
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)
    
    assert xq_out.shape == xq.shape
    assert xk_out.shape == xk.shape

def test_causal_self_attention():
    torch.manual_seed(42)
    # Use a small config
    config = GPTConfig(n_embd=128, n_head=4, n_kv_head=2, dropout=0.0, bias=False)
    attn = CausalSelfAttention(config)
    
    seq_len = 64
    x = torch.randn(2, seq_len, config.n_embd)
    freqs_cis = precompute_freqs_cis(config.n_embd // config.n_head, seq_len)
    
    out = attn(x, freqs_cis)
    
    assert out.shape == x.shape
