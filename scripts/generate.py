"""
Script to generate text using a trained GPT model checkpoint.
"""

import argparse
import torch
from contextlib import nullcontext

from tiny_shakespeare_gpt.tokenizer import BPETokenizer
from tiny_shakespeare_gpt.utils import get_project_root, load_checkpoint, set_seed, setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using trained GPT model."
    )
    parser.add_argument(
        "--prompt", type=str, default="\n", help="Starting prompt for generation"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=500, help="Number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_k", type=int, default=200, help="Top-k sampling threshold"
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    logger = setup_logging(__name__)

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load checkpoint
    out_dir = get_project_root() / "out"
    
    try:
        model, meta = load_checkpoint(device, out_dir)
        logger.info(f"Loading checkpoint from {out_dir}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    model.eval()

    # Initialize tokenizer
    tokenizer = BPETokenizer()

    start_prompt = args.prompt
    # Encode prompt
    if start_prompt == "":
        start_prompt = "\n"

    start_ids = tokenizer.encode(start_prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # Generate
    logger.info(f"Generating {args.max_new_tokens} tokens...")
    print(start_prompt, end="", flush=True)

    with torch.no_grad():
        with (
            torch.autocast(
                device_type=device,
                dtype=torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16,
            )
            if device == "cuda"
            else nullcontext()
        ):
            y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)

    # Decode and print output
    output_text = tokenizer.decode(y[0].tolist())
    print(output_text[len(start_prompt) :])
    logger.info("Generation Complete")


if __name__ == "__main__":
    main()
