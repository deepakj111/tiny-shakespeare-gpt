"""
Training script for the GPT model.
"""
import os
import torch
from torch.utils.data import DataLoader

from tiny_shakespeare_gpt.model import GPT, GPTConfig
from tiny_shakespeare_gpt.dataset import MemmapTokenDataset

def get_batch(loader_iter, loader):
    try:
        x, y = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        x, y = next(loader_iter)
    return x, y, loader_iter

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters, device):
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        loader_iter = iter(loader)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y, loader_iter = get_batch(loader_iter, loader)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def main():
    # Basic configuration
    batch_size = 12
    block_size = 256
    max_iters = 500
    learning_rate = 1e-3
    eval_interval = 100
    eval_iters = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    # Dataset
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    train_data_path = os.path.join(data_dir, "train.bin")
    val_data_path = os.path.join(data_dir, "val.bin")

    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
        print("Data not found. Please run scripts/prepare_data.py first.")
        return

    train_dataset = MemmapTokenDataset(train_data_path, block_size)
    val_dataset = MemmapTokenDataset(val_data_path, block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Model
    config = GPTConfig(
        block_size=block_size,
        vocab_size=50257,
        n_layer=4,
        n_head=4,
        n_kv_head=2,
        n_embd=128,
        dropout=0.0,
    )
    model = GPT(config)
    model.to(device)

    # Optimizer
    optimizer = model.configure_optimizers(weight_decay=1e-1, learning_rate=learning_rate, betas=(0.9, 0.95), device_type=device)

    # Training loop
    train_iter = iter(train_loader)
    
    for step in range(max_iters):
        # Evaluation phase
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model, train_loader, val_loader, eval_iters, device)
            print(f"Step {step}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")

        # Training phase
        x, y, train_iter = get_batch(train_iter, train_loader)
        x, y = x.to(device), y.to(device)
        
        logits, loss = model(x, targets=y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Training complete.")

    # Save a simple checkpoint
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "out")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")

if __name__ == "__main__":
    main()
