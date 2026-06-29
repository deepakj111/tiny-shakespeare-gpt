from dataclasses import dataclass

@dataclass
class TrainConfig:
    # Basic configuration
    batch_size: int = 12
    block_size: int = 256
    max_iters: int = 500
    eval_interval: int = 100
    eval_iters: int = 20
    
    # Optimization configurations
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 100
    grad_clip: float = 1.0
    weight_decay: float = 1e-1
    gradient_accumulation_steps: int = 4
    
    # Model configuration
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 2
    n_embd: int = 128
    dropout: float = 0.0

    # Tracking configurations
    wandb_log: bool = False
    wandb_project: str = 'tiny-shakespeare-gpt'
    resume: bool = False

