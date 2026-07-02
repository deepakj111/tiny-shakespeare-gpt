"""
Data loading utilities for the GPT model.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

class MemmapTokenDataset(Dataset):
    """
    A PyTorch Dataset that reads tokenized integer data from a memory-mapped binary file.
    Using memory-mapping ensures that huge datasets load instantly and consume 
    almost zero RAM.
    """
    def __init__(self, file_path: str, block_size: int, dtype: np.dtype = np.uint32):
        self.data = np.memmap(file_path, dtype=dtype, mode='r')
        self.block_size = block_size

    def __len__(self) -> int:
        # Subtract block_size because we need to fetch x of length block_size
        # and y (targets) which is x shifted by 1.
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Grab a chunk of (block_size + 1) tokens from the memmap
        chunk = self.data[idx : idx + self.block_size + 1]
        
        # Convert to int64, which is required by PyTorch embedding layers
        chunk_tensor = torch.from_numpy(chunk.astype(np.int64))
        
        # x is the input sequence, y is the target sequence (shifted by 1)
        x = chunk_tensor[:-1]
        y = chunk_tensor[1:]
        
        return x, y
