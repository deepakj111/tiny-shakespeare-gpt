"""
Downloads the Tiny Shakespeare dataset.
"""

import requests
from tiny_shakespeare_gpt.utils import get_project_root, setup_logging

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    logger = setup_logging(__name__)
    
    data_dir = get_project_root() / "data"
    output_file = data_dir / "input.txt"

    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Tiny Shakespeare dataset to {output_file}...")

    response = requests.get(DATA_URL)
    response.raise_for_status()

    output_file.write_text(response.text, encoding="utf-8")

    logger.info(f"Successfully downloaded {len(response.text)} characters.")


if __name__ == "__main__":
    main()
