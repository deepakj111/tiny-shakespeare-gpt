"""
Prepares the dataset by tokenizing the raw text and saving it to binary files.
"""

import numpy as np
from tiny_shakespeare_gpt.tokenizer import BPETokenizer
from tiny_shakespeare_gpt.utils import get_project_root, setup_logging


def main():
    logger = setup_logging(__name__)

    data_dir = get_project_root() / "data"
    input_file = data_dir / "input.txt"
    train_file = data_dir / "train.bin"
    val_file = data_dir / "val.bin"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at {input_file}. Run download_dataset.py first."
        )

    text = input_file.read_text(encoding="utf-8")

    # Split data: 90% train, 10% validation
    n = len(text)
    train_text = text[: int(n * 0.9)]
    val_text = text[int(n * 0.9) :]

    tokenizer = BPETokenizer()

    logger.info("Encoding training data...")
    train_ids = tokenizer.encode(train_text)
    logger.info(f"Train tokens: {len(train_ids):,}")

    logger.info("Encoding validation data...")
    val_ids = tokenizer.encode(val_text)
    logger.info(f"Validation tokens: {len(val_ids):,}")

    # Export to binary files
    # o200k_base vocab size is ~200k, so uint32 is required
    train_arr = np.array(train_ids, dtype=np.uint32)
    val_arr = np.array(val_ids, dtype=np.uint32)

    train_arr.tofile(train_file)
    val_arr.tofile(val_file)

    logger.info(f"Saved {train_file} and {val_file}")


if __name__ == "__main__":
    main()
