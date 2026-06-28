"""
Script to generate text using a trained GPT model checkpoint.
"""
import os
import torch
from tiny_shakespeare_gpt.model import GPT, GPTConfig
from tiny_shakespeare_gpt.tokenizer import BPETokenizer

def main():
    # Generation settings
    start_prompt = "\n"  # starting prompt (can be empty or custom string)
    max_new_tokens = 500 # number of tokens to generate
    temperature = 0.8    # 1.0 is default, lower is more conservative, higher is more diverse
    top_k = 200          # retain only the top_k most likely tokens
    seed = 1337

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load checkpoint
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "out")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        print("Please run scripts/train.py first.")
        return

    print(f"Loading checkpoint from {ckpt_path}")
    torch.serialization.add_safe_globals([GPTConfig])
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    # Initialize model from checkpoint config
    config = checkpoint['config']
    model = GPT(config)
    
    # Remove any unwanted prefix if model was trained with DDP or similar (though not used currently, good practice)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
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
