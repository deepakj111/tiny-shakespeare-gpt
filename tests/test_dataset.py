import numpy as np
import torch
from tiny_shakespeare_gpt.dataset import MemmapTokenDataset

def test_memmap_token_dataset(tmp_path):
    # Create a dummy binary file with tokens 0..99
    dummy_file = tmp_path / "dummy.bin"
    tokens = np.arange(100, dtype=np.uint32)
    tokens.tofile(dummy_file)
    
    block_size = 8
    dataset = MemmapTokenDataset(str(dummy_file), block_size=block_size)
    
    # Dataset length should be total tokens minus block_size
    assert len(dataset) == 100 - block_size
    
    # Check the first item
    x, y = dataset[0]
    
    # The output shapes should match the block size
    assert x.shape == (block_size,)
    assert y.shape == (block_size,)
    
    # x should be tokens 0..7, y should be tokens 1..8
    assert torch.all(x == torch.arange(8))
    assert torch.all(y == torch.arange(1, 9))
