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

from tiny_shakespeare_gpt.model import Block, GPT

def test_block():
    torch.manual_seed(42)
    config = GPTConfig(n_embd=128, n_head=4, n_kv_head=2, dropout=0.0, bias=False)
    block = Block(config)
    
    seq_len = 64
    x = torch.randn(2, seq_len, config.n_embd)
    freqs_cis = precompute_freqs_cis(config.n_embd // config.n_head, seq_len)
    
    out = block(x, freqs_cis)
    assert out.shape == x.shape

def test_gpt():
    torch.manual_seed(42)
    # small config for quick testing
    config = GPTConfig(vocab_size=100, n_embd=128, n_layer=2, n_head=4, n_kv_head=2)
    model = GPT(config)
    
    # Check weight tying
    assert torch.all(model.tok_emb.weight == model.lm_head.weight)
    
    # Test forward pass without targets
    idx = torch.randint(0, config.vocab_size, (2, 64))
    logits, loss = model(idx)
    
    # logits shape should be (B, 1, vocab_size) during inference
    assert logits.shape == (2, 1, config.vocab_size)
    assert loss is None
    
    # Test forward pass with targets
    targets = torch.randint(0, config.vocab_size, (2, 64))
    logits, loss = model(idx, targets=targets)
    
    # logits shape should be (B, T, vocab_size) during training
    assert logits.shape == (2, 64, config.vocab_size)
    assert loss is not None
    assert loss.item() > 0

def test_configure_optimizers():
    config = GPTConfig(vocab_size=100, n_embd=128, n_layer=2, n_head=4, n_kv_head=2, bias=True)
    model = GPT(config)
    
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.999), device_type='cpu')
    assert isinstance(optimizer, torch.optim.AdamW)
    
    # Check that there are two parameter groups (decay and no decay)
    assert len(optimizer.param_groups) == 2
    
def test_generate():
    torch.manual_seed(42)
    config = GPTConfig(vocab_size=100, n_embd=64, n_layer=1, n_head=2, n_kv_head=1)
    model = GPT(config)
    model.eval()
    
    # Provide a starting sequence of 4 tokens for 2 sequences in a batch
    idx = torch.randint(0, config.vocab_size, (2, 4))
    
    # Generate 5 new tokens
    out = model.generate(idx, max_new_tokens=5, top_k=5)
    
    # Output should have 4 + 5 = 9 tokens
    assert out.shape == (2, 9)
