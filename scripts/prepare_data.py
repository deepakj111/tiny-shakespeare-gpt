"""
Prepares the dataset by tokenizing the raw text and saving it to binary files.
"""

import argparse
import numpy as np
from pathlib import Path
from tiny_shakespeare_gpt.tokenizer import BPETokenizer
from tiny_shakespeare_gpt.utils import get_project_root, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training.")
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(get_project_root() / "data" / "input.txt"),
        help="Path to input text file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(get_project_root() / "data"),
        help="Directory to save train.bin and val.bin",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (default: 0.1)",
    )
    args = parser.parse_args()

    logger = setup_logging(__name__)

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    train_file = output_dir / "train.bin"
    val_file = output_dir / "val.bin"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at {input_file}. Run download_dataset.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    text = input_file.read_text(encoding="utf-8")

    # Split data
    if not (0.0 <= args.val_split <= 1.0):
        raise ValueError("--val-split must be between 0.0 and 1.0")

    n = len(text)
    split_idx = int(n * (1.0 - args.val_split))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

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
