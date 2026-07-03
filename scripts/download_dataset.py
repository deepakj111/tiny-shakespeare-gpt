"""
Downloads the Tiny Shakespeare dataset.
"""

import os
import requests

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "input.txt")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading Tiny Shakespeare dataset to {OUTPUT_FILE}...")

    response = requests.get(DATA_URL)
    response.raise_for_status()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Successfully downloaded {len(response.text)} characters.")


if __name__ == "__main__":
    main()
