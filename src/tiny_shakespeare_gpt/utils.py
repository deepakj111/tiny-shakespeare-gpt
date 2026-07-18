"""
Utility functions for the Tiny Shakespeare GPT project.
"""

import os
import logging
from pathlib import Path
import torch
import safetensors.torch
from torch.distributed import init_process_group

from tiny_shakespeare_gpt.model import GPT, GPTConfig


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup a standard logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_project_root() -> Path:
    """Get the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def setup_ddp() -> dict:
    """
    Setup Distributed Data Parallel (DDP) if available.
    Returns a dictionary with ddp settings.
    """
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        ddp_local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        if (
            torch.cuda.is_available()
            and torch.cuda.device_count() >= ddp_local_world_size
        ):
            backend = "nccl"
        else:
            backend = "gloo"

        init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])

        if backend == "nccl":
            device = f"cuda:{ddp_local_rank}"
            torch.cuda.set_device(device)
        else:
            device = "cpu"
        master_process = ddp_rank == 0
    else:
        master_process = True
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        "ddp": ddp,
        "device": device,
        "master_process": master_process,
        "ddp_rank": ddp_rank,
        "ddp_local_rank": ddp_local_rank,
        "ddp_world_size": ddp_world_size,
    }


def load_checkpoint(device: str, model_dir: Path) -> tuple[GPT, dict]:
    """
    Load a trained model checkpoint and metadata.
    """
    model_ckpt_path = model_dir / "model.safetensors"
    meta_ckpt_path = model_dir / "ckpt_meta.pt"

    if not model_ckpt_path.exists() or not meta_ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {model_ckpt_path} or {meta_ckpt_path}. "
            "Please run training first."
        )

    torch.serialization.add_safe_globals([GPTConfig])
    meta = torch.load(meta_ckpt_path, map_location=device, weights_only=True)

    config = meta["config"]
    model = GPT(config)

    safetensors.torch.load_model(model, str(model_ckpt_path))
    model.to(device)

    return model, meta


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
