"""
Script to generate text using a trained GPT model checkpoint.
"""
import os
import torch
from tiny_shakespeare_gpt.model import GPT, GPTConfig
from tiny_shakespeare_gpt.tokenizer import BPETokenizer

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate text using trained GPT model.")
    parser.add_argument("--prompt", type=str, default="\n", help="Starting prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=200, help="Top-k sampling threshold")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    start_prompt = args.prompt
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_k = args.top_k
    seed = args.seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load checkpoint
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "out")
    model_ckpt_path = os.path.join(out_dir, "model.safetensors")
    meta_ckpt_path = os.path.join(out_dir, "ckpt_meta.pt")
    
    if not os.path.exists(model_ckpt_path) or not os.path.exists(meta_ckpt_path):
        print(f"Error: Checkpoint not found at {model_ckpt_path} or {meta_ckpt_path}")
        print("Please run scripts/train.py first.")
        return

    print(f"Loading checkpoint from {model_ckpt_path} and {meta_ckpt_path}")
    torch.serialization.add_safe_globals([GPTConfig])
    meta = torch.load(meta_ckpt_path, map_location=device, weights_only=True)
    
    # Initialize model from checkpoint config
    config = meta['config']
    model = GPT(config)
    
    # Load weights
    import safetensors.torch
    safetensors.torch.load_model(model, model_ckpt_path)
    model.eval()
    model.to(device)

    # Initialize tokenizer
    tokenizer = BPETokenizer()

    # Encode prompt
    if start_prompt == "":
        # if empty prompt, generate from the <|endoftext|> token equivalent or simply an empty start
        # since we don't have a specific SOS token in basic GPT-2 vocab, a newline or space is commonly used
        start_prompt = "\n"
        
    start_ids = tokenizer.encode(start_prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # Generate
    print(f"\n--- Generating {max_new_tokens} tokens ---\n")
    print(start_prompt, end="")
    
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device == 'cuda' else nullcontext():
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            
    # Decode and print output
    # y[0] gets the sequence from batch dim
    output_text = tokenizer.decode(y[0].tolist())
    print(output_text[len(start_prompt):])
    print("\n--- Generation Complete ---")

if __name__ == "__main__":
    from contextlib import nullcontext
    main()
