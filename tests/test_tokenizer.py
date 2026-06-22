from tiny_shakespeare_gpt.tokenizer import BPETokenizer

def test_bpe_tokenizer():
    tokenizer = BPETokenizer("gpt2")
    
    # Vocabulary size should be large (gpt2 is 50257)
    assert tokenizer.vocab_size == 50257
    
    # Test encoding
    text = "hello world, this is a BPE test!"
    encoded = tokenizer.encode(text)
    assert isinstance(encoded, list)
    assert len(encoded) > 0
    assert all(isinstance(token, int) for token in encoded)
    
    # Test decoding roundtrip
    decoded = tokenizer.decode(encoded)
    assert decoded == text

def test_bpe_tokenizer_cl100k():
    tokenizer = BPETokenizer("cl100k_base")
    
    # Vocabulary size for cl100k_base (GPT-4)
    assert tokenizer.vocab_size == 100277
    
    text = "Testing GPT-4 tokenizer."
    assert tokenizer.decode(tokenizer.encode(text)) == text
