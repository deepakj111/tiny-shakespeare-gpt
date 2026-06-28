"""
Training script for the GPT model.
"""
import os
import math
from contextlib import nullcontext
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
def estimate_loss(model, train_loader, val_loader, eval_iters, device, ctx):
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        loader_iter = iter(loader)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y, loader_iter = get_batch(loader_iter, loader)
            x, y = x.to(device), y.to(device)
            with ctx:
                _, loss = model(x, targets=y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(it, max_iters, learning_rate, warmup_iters, min_lr):
    if it < warmup_iters:
        return learning_rate * it / max(1, warmup_iters)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def main():
    # Performance settings
    torch.set_float32_matmul_precision('high')
    
    # Basic configuration
    batch_size = 12
    block_size = 256
    max_iters = 500
    eval_interval = 100
    eval_iters = 20
    
    # Optimization configurations
    learning_rate = 1e-3
    min_lr = 1e-4
    warmup_iters = 100
    grad_clip = 1.0
    gradient_accumulation_steps = 4 # simulate larger batch size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    else:
        dtype = 'float32'
    
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.autocast(device_type=device, dtype=ptdtype)
    
    print(f"Using device: {device}, dtype: {dtype}")

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
        n_layer=4,
        n_head=4,
        n_kv_head=2,
        n_embd=128,
        dropout=0.0,
    )
    model = GPT(config)
    model.to(device)
    
    if device == 'cuda':
        print("Compiling model...")
        model = torch.compile(model)

    # Optimizer
    optimizer = model.configure_optimizers(weight_decay=1e-1, learning_rate=learning_rate, betas=(0.9, 0.95), device_type=device)

    # Training loop
    train_iter = iter(train_loader)
    best_val_loss = float('inf')
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "out")
    os.makedirs(out_dir, exist_ok=True)
    
    for step in range(max_iters):
        # Update learning rate
        lr = get_lr(step, max_iters, learning_rate, warmup_iters, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Evaluation phase
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model, train_loader, val_loader, eval_iters, device, ctx)
            print(f"Step {step}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}, LR: {lr:.4e}")
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                ckpt_path = os.path.join(out_dir, "ckpt.pt")
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'iter_num': step,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, ckpt_path)
                print(f"Saved new best model with val loss {best_val_loss:.4f} to {ckpt_path}")

        # Training phase with gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        
        for micro_step in range(gradient_accumulation_steps):
            x, y, train_iter = get_batch(train_iter, train_loader)
            x, y = x.to(device), y.to(device)
            
            with ctx:
                logits, loss = model(x, targets=y)
                loss = loss / gradient_accumulation_steps
                
            loss.backward()
            
        # Gradient clipping
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        optimizer.step()

    print("Training complete.")

if __name__ == "__main__":
    main()
