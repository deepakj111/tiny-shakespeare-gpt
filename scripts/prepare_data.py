"""
Prepares the dataset by tokenizing the raw text and saving it to binary files.
"""

import os
import numpy as np
from tiny_shakespeare_gpt.tokenizer import BPETokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INPUT_FILE = os.path.join(DATA_DIR, "input.txt")
TRAIN_FILE = os.path.join(DATA_DIR, "train.bin")
VAL_FILE = os.path.join(DATA_DIR, "val.bin")


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Dataset not found at {INPUT_FILE}. Run download_dataset.py first."
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    # Split data: 90% train, 10% validation
    n = len(text)
    train_text = text[: int(n * 0.9)]
    val_text = text[int(n * 0.9) :]

    tokenizer = BPETokenizer()

    print("Encoding training data...")
    train_ids = tokenizer.encode(train_text)
    print(f"Train tokens: {len(train_ids):,}")

    print("Encoding validation data...")
    val_ids = tokenizer.encode(val_text)
    print(f"Validation tokens: {len(val_ids):,}")

    # Export to binary files
    # o200k_base vocab size is ~200k, so uint32 is required
    train_arr = np.array(train_ids, dtype=np.uint32)
    val_arr = np.array(val_ids, dtype=np.uint32)

    train_arr.tofile(TRAIN_FILE)
    val_arr.tofile(VAL_FILE)

    print(f"Saved {TRAIN_FILE} and {VAL_FILE}")


if __name__ == "__main__":
    main()
