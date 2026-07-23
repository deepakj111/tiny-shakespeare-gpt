"""
Downloads a text dataset (default: Tiny Shakespeare).
"""

import argparse
from pathlib import Path
import requests
from tiny_shakespeare_gpt.utils import get_project_root, setup_logging

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    parser = argparse.ArgumentParser(description="Download a dataset.")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_DATA_URL,
        help="URL of the dataset to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(get_project_root() / "data"),
        help="Directory to save the dataset",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="input.txt",
        help="Filename for the downloaded dataset",
    )
    args = parser.parse_args()

    logger = setup_logging(__name__)

    data_dir = Path(args.output_dir)
    output_file = data_dir / args.filename

    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading dataset from {args.url} to {output_file}...")

    try:
        with requests.get(args.url, stream=True, timeout=15) as response:
            response.raise_for_status()
            downloaded_bytes = 0
            with open(output_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_bytes += len(chunk)

            logger.info(f"Successfully downloaded {downloaded_bytes} bytes.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download dataset: {e}")
        raise


if __name__ == "__main__":
    main()
