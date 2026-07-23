"""
Training script for the GPT model.
"""

import time
import math
import argparse
import dataclasses
from contextlib import nullcontext
import torch
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from torch.utils.data.distributed import DistributedSampler
import safetensors.torch

from tiny_shakespeare_gpt.model import GPT, GPTConfig
from tiny_shakespeare_gpt.dataset import MemmapTokenDataset
from tiny_shakespeare_gpt.config import TrainConfig
from tiny_shakespeare_gpt.tokenizer import BPETokenizer
from tiny_shakespeare_gpt.utils import get_project_root, setup_logging, setup_ddp

logger = setup_logging(__name__)


def parse_args_into_config(config: TrainConfig) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train GPT model")

    for field in dataclasses.fields(config):
        arg_name = f"--{field.name}"
        if field.type is bool:
            group = parser.add_mutually_exclusive_group()
            group.add_argument(
                arg_name,
                action="store_true",
                default=field.default,
                help=f"Enable {field.name}",
            )
            group.add_argument(
                f"--no-{field.name}",
                dest=field.name,
                action="store_false",
                help=f"Disable {field.name}",
            )
        else:
            parser.add_argument(
                arg_name,
                type=field.type,
                default=field.default,
                help=f"{field.name} (default: {field.default})",
            )

    args = parser.parse_args()

    for field in dataclasses.fields(config):
        setattr(config, field.name, getattr(args, field.name))

    return config


def get_batch(loader_iter, loader, ddp, sampler, epoch):
    try:
        x, y = next(loader_iter)
    except StopIteration:
        epoch += 1
        if ddp and sampler is not None:
            sampler.set_epoch(epoch)
        loader_iter = iter(loader)
        x, y = next(loader_iter)
    return x, y, loader_iter, epoch


@torch.no_grad()
def estimate_loss(
    model,
    train_loader,
    val_loader,
    eval_iters,
    device,
    ctx,
    ddp,
    train_sampler,
    val_sampler,
):
    out = {}
    model.eval()

    # Temporary epoch counters for evaluation loaders to prevent breaking training epoch
    train_eval_epoch = 0
    val_eval_epoch = 0

    for split, loader, sampler, epoch in [
        ("train", train_loader, train_sampler, train_eval_epoch),
        ("val", val_loader, val_sampler, val_eval_epoch),
    ]:
        loader_iter = iter(loader)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y, loader_iter, epoch = get_batch(
                loader_iter, loader, ddp, sampler, epoch
            )
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
    decay_ratio = (it - train_config.warmup_iters) / (
        train_config.max_iters - train_config.warmup_iters
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_config.min_lr + coeff * (
        train_config.learning_rate - train_config.min_lr
    )


def main():
    train_config = TrainConfig()
    train_config = parse_args_into_config(train_config)

    # DDP Setup
    ddp_info = setup_ddp()
    ddp = ddp_info["ddp"]
    device = ddp_info["device"]
    master_process = ddp_info["master_process"]
    ddp_rank = ddp_info["ddp_rank"]
    ddp_local_rank = ddp_info["ddp_local_rank"]
    ddp_world_size = ddp_info["ddp_world_size"]

    if master_process and train_config.wandb_log:
        wandb.init(project=train_config.wandb_project, config=train_config.__dict__)

    # Performance settings
    torch.set_float32_matmul_precision("high")

    if device.startswith("cuda"):
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    else:
        dtype = "float32"

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.autocast(device_type=device.split(":")[0], dtype=ptdtype)
    )

    if master_process:
        logger.info(f"Using device: {device}, dtype: {dtype}, DDP: {ddp}")

    # Dataset
    data_dir = get_project_root() / "data"
    train_data_path = data_dir / "train.bin"
    val_data_path = data_dir / "val.bin"

    if not train_data_path.exists() or not val_data_path.exists():
        if master_process:
            logger.error("Data not found. Please run scripts/prepare_data.py first.")
        if ddp:
            destroy_process_group()
        return

    train_dataset = MemmapTokenDataset(str(train_data_path), train_config.block_size)
    val_dataset = MemmapTokenDataset(str(val_data_path), train_config.block_size)
    tokenizer = BPETokenizer()

    if ddp:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=False,
    )

    # Model
    model_config = GPTConfig(
        block_size=train_config.block_size,
        n_layer=train_config.n_layer,
        n_head=train_config.n_head,
        n_kv_head=train_config.n_kv_head,
        n_embd=train_config.n_embd,
        dropout=train_config.dropout,
        n_experts=train_config.n_experts,
        num_experts_per_tok=train_config.num_experts_per_tok,
    )
    model = GPT(model_config)
    model.to(device)

    if device.startswith("cuda"):
        if master_process:
            logger.info("Compiling model... (Skipped for stability)")
        # model = torch.compile(model)

    if ddp:
        if device == "cpu":
            model = DDP(model)
        else:
            model = DDP(model, device_ids=[ddp_local_rank % torch.cuda.device_count()])

    raw_model = model.module if ddp else model

    # Optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=train_config.weight_decay,
        learning_rate=train_config.learning_rate,
        betas=(0.9, 0.95),
        device_type=device.split(":")[0],
    )

    if master_process:
        start_time = time.time()
        history_iters = []
        history_train_loss = []
        history_val_loss = []
        history_lr = []
        sample_outputs = []

    # Training loop
    train_iter = iter(train_loader)
    best_val_loss = float("inf")
    start_step = 0
    epoch = 0
    out_dir = get_project_root() / "out"
    if master_process:
        out_dir.mkdir(parents=True, exist_ok=True)

    model_ckpt_path = out_dir / "model.safetensors"
    meta_ckpt_path = out_dir / "ckpt_meta.pt"

    if train_config.resume and model_ckpt_path.exists() and meta_ckpt_path.exists():
        if master_process:
            logger.info(
                f"Resuming from checkpoint {model_ckpt_path} and {meta_ckpt_path}"
            )

        # Load model weights via safetensors
        safetensors.torch.load_model(raw_model, str(model_ckpt_path))

        # Load metadata and optimizer state via standard PyTorch
        torch.serialization.add_safe_globals([GPTConfig])
        meta = torch.load(meta_ckpt_path, map_location=device, weights_only=True)

        optimizer.load_state_dict(meta["optimizer"])
        start_step = meta["iter_num"] + 1
        best_val_loss = meta.get("best_val_loss", best_val_loss)
        if "rng_state" in meta:
            torch.set_rng_state(meta["rng_state"])
        if "cuda_rng_state" in meta and device.startswith("cuda"):
            torch.cuda.set_rng_state(meta["cuda_rng_state"])
        if master_process:
            logger.info(f"Resumed from step {start_step - 1}")

    for step in range(start_step, train_config.max_iters):
        # Update learning rate
        lr = get_lr(step, train_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluation phase
        if step % train_config.eval_interval == 0 or step == train_config.max_iters - 1:
            losses = estimate_loss(
                raw_model,
                train_loader,
                val_loader,
                train_config.eval_iters,
                device,
                ctx,
                ddp,
                train_sampler,
                val_sampler,
            )
            if master_process:
                logger.info(
                    f"Step {step}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}, LR: {lr:.4e}"
                )
                history_iters.append(step)
                history_train_loss.append(losses["train"])
                history_val_loss.append(losses["val"])
                history_lr.append(lr)

                # Generate sample (use uncompiled model to avoid recompilation OOMs on every token)
                model_for_eval = (
                    raw_model._orig_mod
                    if hasattr(raw_model, "_orig_mod")
                    else raw_model
                )
                model_for_eval.eval()
                start_ids = tokenizer.encode(train_config.eval_generate_prompt)
                x_gen = torch.tensor(start_ids, dtype=torch.long, device=device)[
                    None, ...
                ]
                with torch.no_grad():
                    with ctx:
                        y_gen = model_for_eval.generate(
                            x_gen,
                            train_config.eval_generate_tokens,
                            temperature=0.8,
                            top_k=200,
                        )
                sample_text = tokenizer.decode(y_gen[0].tolist())
                logger.info(f"Sample Generation:\n{sample_text}\n{'-' * 30}")
                sample_outputs.append((step, sample_text))
                model_for_eval.train()

                if train_config.wandb_log:
                    wandb.log(
                        {
                            "iter": step,
                            "train/loss": losses["train"],
                            "val/loss": losses["val"],
                            "lr": lr,
                            "sample": wandb.Html(f"<pre>{sample_text}</pre>"),
                        }
                    )

                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    model_ckpt_path = out_dir / "model.safetensors"
                    meta_ckpt_path = out_dir / "ckpt_meta.pt"

                    # Save model weights via safetensors
                    model_to_save = (
                        raw_model._orig_mod
                        if hasattr(raw_model, "_orig_mod")
                        else raw_model
                    )
                    safetensors.torch.save_model(model_to_save, str(model_ckpt_path))

                    # Save training state via torch.save
                    meta = {
                        "optimizer": optimizer.state_dict(),
                        "config": model_config,
                        "iter_num": step,
                        "best_val_loss": best_val_loss,
                        "rng_state": torch.get_rng_state(),
                    }
                    if device.startswith("cuda"):
                        meta["cuda_rng_state"] = torch.cuda.get_rng_state()
                    torch.save(meta, str(meta_ckpt_path))
                    logger.info(
                        f"Saved new best model with val loss {best_val_loss:.4f} to {model_ckpt_path}"
                    )

        # Training phase with gradient accumulation
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(train_config.gradient_accumulation_steps):
            x, y, train_iter, epoch = get_batch(
                train_iter, train_loader, ddp, train_sampler, epoch
            )
            x, y = x.to(device), y.to(device)

            # Use no_sync to avoid syncing gradients on all but the last micro_step
            is_last_micro_step = (
                micro_step == train_config.gradient_accumulation_steps - 1
            )
            ctx_sync = (
                model.no_sync() if ddp and not is_last_micro_step else nullcontext()
            )

            with ctx_sync:
                with ctx:
                    logits, loss = model(x, targets=y)
                    loss = loss / train_config.gradient_accumulation_steps
                loss.backward()

        # Gradient clipping
        if train_config.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

        optimizer.step()

    if master_process:
        logger.info("Training complete.")

        end_time = time.time()
        training_time_seconds = end_time - start_time

        plt.figure(figsize=(10, 6))
        plt.plot(history_iters, history_train_loss, label="Train Loss")
        plt.plot(history_iters, history_val_loss, label="Val Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Curve")
        plt.legend()
        plt.grid(True)
        loss_curve_path = out_dir / "loss_curve.png"
        plt.savefig(str(loss_curve_path))
        plt.close()

        total_params = sum(p.numel() for p in raw_model.parameters())
        trainable_params = sum(
            p.numel() for p in raw_model.parameters() if p.requires_grad
        )

        final_train_loss = (
            f"{history_train_loss[-1]:.4f}" if history_train_loss else "N/A"
        )
        final_val_loss = f"{history_val_loss[-1]:.4f}" if history_val_loss else "N/A"

        report_md = f"""# Training Report

## Architecture
- **Block Size:** {model_config.block_size}
- **Layers:** {model_config.n_layer}
- **Heads:** {model_config.n_head}
- **KV Heads:** {model_config.n_kv_head}
- **Embedding Size:** {model_config.n_embd}
- **Dropout:** {model_config.dropout}
- **Total Experts (MoE):** {model_config.n_experts}
- **Experts Per Token:** {model_config.num_experts_per_tok}

## Parameters
- **Total Parameters:** {total_params:,}
- **Trainable Parameters:** {trainable_params:,}

## Training Details
- **Total Iterations:** {train_config.max_iters}
- **Training Time:** {training_time_seconds:.2f} seconds ({training_time_seconds / 60:.2f} minutes)
- **Final Train Loss:** {final_train_loss}
- **Final Val Loss:** {final_val_loss}
- **Best Val Loss:** {best_val_loss:.4f}

## Configuration
```python
{train_config.__dict__}
```

## Loss Curve
![Loss Curve](./loss_curve.png)

## Sample Outputs Generated During Training
"""
        for step_num, text in sample_outputs:
            report_md += f"\n### Step {step_num}\n```text\n{text}\n```\n"

        report_path = out_dir / "training_report.md"
        report_path.write_text(report_md)
        logger.info(f"Training report saved to {report_path}")

        if train_config.wandb_log:
            wandb.finish()

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
