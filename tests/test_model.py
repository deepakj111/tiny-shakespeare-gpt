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
