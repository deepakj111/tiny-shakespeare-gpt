"""
Training script for the GPT model.
"""
import os
import math
import logging
from contextlib import nullcontext
import torch
import wandb
from torch.utils.data import DataLoader

from tiny_shakespeare_gpt.model import GPT, GPTConfig
from tiny_shakespeare_gpt.dataset import MemmapTokenDataset
from tiny_shakespeare_gpt.config import TrainConfig

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

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

def get_lr(it, train_config):
    if it < train_config.warmup_iters:
        return train_config.learning_rate * it / max(1, train_config.warmup_iters)
    if it > train_config.max_iters:
        return train_config.min_lr
    decay_ratio = (it - train_config.warmup_iters) / (train_config.max_iters - train_config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_config.min_lr + coeff * (train_config.learning_rate - train_config.min_lr)

def main():
    train_config = TrainConfig()
    
    if train_config.wandb_log:
        wandb.init(project=train_config.wandb_project, config=train_config.__dict__)

    # Performance settings
    torch.set_float32_matmul_precision('high')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    else:
        dtype = 'float32'
    
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.autocast(device_type=device, dtype=ptdtype)
    
    logger.info(f"Using device: {device}, dtype: {dtype}")

    # Dataset
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    train_data_path = os.path.join(data_dir, "train.bin")
    val_data_path = os.path.join(data_dir, "val.bin")

    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
        logger.error("Data not found. Please run scripts/prepare_data.py first.")
        return

    train_dataset = MemmapTokenDataset(train_data_path, train_config.block_size)
    val_dataset = MemmapTokenDataset(val_data_path, train_config.block_size)

    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=False, pin_memory=True)

    # Model
    model_config = GPTConfig(
        block_size=train_config.block_size,
        n_layer=train_config.n_layer,
        n_head=train_config.n_head,
        n_kv_head=train_config.n_kv_head,
        n_embd=train_config.n_embd,
        dropout=train_config.dropout,
    )
    model = GPT(model_config)
    model.to(device)
    
    if device == 'cuda':
        logger.info("Compiling model...")
        model = torch.compile(model)

    # Optimizer
    optimizer = model.configure_optimizers(weight_decay=train_config.weight_decay, learning_rate=train_config.learning_rate, betas=(0.9, 0.95), device_type=device)

    # Training loop
    train_iter = iter(train_loader)
    best_val_loss = float('inf')
    start_step = 0
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "out")
    os.makedirs(out_dir, exist_ok=True)
    
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if train_config.resume and os.path.exists(ckpt_path):
        logger.info(f"Resuming from checkpoint {ckpt_path}")
        torch.serialization.add_safe_globals([GPTConfig])
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint['iter_num'] + 1
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'])
        if 'cuda_rng_state' in checkpoint and device == 'cuda':
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        logger.info(f"Resumed from step {start_step - 1}")

    for step in range(start_step, train_config.max_iters):
        # Update learning rate
        lr = get_lr(step, train_config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Evaluation phase
        if step % train_config.eval_interval == 0 or step == train_config.max_iters - 1:
            losses = estimate_loss(model, train_loader, val_loader, train_config.eval_iters, device, ctx)
            logger.info(f"Step {step}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}, LR: {lr:.4e}")
            
            if train_config.wandb_log:
                wandb.log({
                    "iter": step,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                })
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                ckpt_path = os.path.join(out_dir, "ckpt.pt")
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': model_config,
                    'iter_num': step,
                    'best_val_loss': best_val_loss,
                    'rng_state': torch.get_rng_state(),
                }
                if device == 'cuda':
                    checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()
                torch.save(checkpoint, ckpt_path)
                logger.info(f"Saved new best model with val loss {best_val_loss:.4f} to {ckpt_path}")

        # Training phase with gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        
        for micro_step in range(train_config.gradient_accumulation_steps):
            x, y, train_iter = get_batch(train_iter, train_loader)
            x, y = x.to(device), y.to(device)
            
            with ctx:
                logits, loss = model(x, targets=y)
                loss = loss / train_config.gradient_accumulation_steps
                
            loss.backward()
            
        # Gradient clipping
        if train_config.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            
        optimizer.step()

    logger.info("Training complete.")
    if train_config.wandb_log:
        wandb.finish()

if __name__ == "__main__":
    main()
