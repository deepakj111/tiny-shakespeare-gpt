"""
FastAPI Server to serve the GPT model.
"""

import os
import json
import asyncio
import logging
import torch
import torch.nn.functional as F
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from contextlib import asynccontextmanager, nullcontext
from tiny_shakespeare_gpt.model import GPT, GPTConfig
from tiny_shakespeare_gpt.tokenizer import BPETokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t  %(message)s")
logger = logging.getLogger(__name__)

class ServerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    model_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "out")

    class Config:
        env_file = ".env"

settings = ServerSettings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model_ckpt_path = os.path.join(settings.model_dir, "model.safetensors")
    meta_ckpt_path = os.path.join(settings.model_dir, "ckpt_meta.pt")

    if not os.path.exists(model_ckpt_path) or not os.path.exists(meta_ckpt_path):
        raise RuntimeError(
            f"Checkpoint not found at {model_ckpt_path} or {meta_ckpt_path}. Run scripts/train.py first."
        )

    torch.serialization.add_safe_globals([GPTConfig])
    meta = torch.load(meta_ckpt_path, map_location=device, weights_only=True)

    config = meta["config"]
    model = GPT(config)

    import safetensors.torch
    safetensors.torch.load_model(model, model_ckpt_path)
    model.eval()
    model.to(device)

    tokenizer = BPETokenizer()

    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.device = device

    logger.info("Model and tokenizer loaded successfully.")
    yield
    # Teardown
    logger.info("Shutting down server.")


app = FastAPI(title="Tiny Shakespeare GPT API", lifespan=lifespan)

static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class GenerateRequest(BaseModel):
    prompt: str = "\n"
    max_new_tokens: int = 500
    temperature: float = 0.8
    top_k: int = 200
    stream: bool = False
    seed: int = 1337


class GenerateResponse(BaseModel):
    text: str


def get_model(request: Request) -> GPT:
    if not hasattr(request.app.state, "model"):
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return request.app.state.model


def get_tokenizer(request: Request) -> BPETokenizer:
    if not hasattr(request.app.state, "tokenizer"):
        raise HTTPException(status_code=503, detail="Tokenizer not loaded yet")
    return request.app.state.tokenizer


def get_device(request: Request) -> str:
    if not hasattr(request.app.state, "device"):
        return "cpu"
    return request.app.state.device


async def generate_stream(
    request: GenerateRequest, model: GPT, tokenizer: BPETokenizer, device: str
) -> AsyncGenerator[str, None]:
    torch.manual_seed(request.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed(request.seed)

    start_prompt = request.prompt if request.prompt else "\n"
    start_ids = tokenizer.encode(start_prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    B, T = x.shape
    if T > model.config.block_size:
        x = x[:, -model.config.block_size :]
        T = x.shape[1]

    ctx = (
        torch.autocast(
            device_type=device,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        if device.startswith("cuda")
        else nullcontext()
    )

    with torch.no_grad():
        with ctx:
            # Initialize KV cache
            kv_caches = []
            for _ in model.blocks:
                cache_k = torch.zeros(
                    (B, model.config.block_size, model.config.n_kv_head, model.config.n_embd // model.config.n_head),
                    dtype=model.tok_emb.weight.dtype,
                    device=device,
                )
                cache_v = torch.zeros_like(cache_k)
                kv_caches.append((cache_k, cache_v))

            logits, _, kv_caches = model(x, start_pos=0, kv_caches=kv_caches)

            next_token_logits = logits[:, -1, :] / request.temperature
            if request.top_k is not None:
                v, _ = torch.topk(
                    next_token_logits, min(request.top_k, next_token_logits.size(-1))
                )
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(next_token_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            start_pos = T

            # Yield first token
            token_str = tokenizer.decode(idx_next[0].tolist())
            yield f"data: {json.dumps({'text': token_str})}\n\n"
            await asyncio.sleep(0)

            for _ in range(1, request.max_new_tokens):
                if start_pos >= model.config.block_size:
                    break

                logits, _, kv_caches = model(idx_next, start_pos=start_pos, kv_caches=kv_caches)

                next_token_logits = logits[:, -1, :] / request.temperature
                if request.top_k is not None:
                    v, _ = torch.topk(
                        next_token_logits,
                        min(request.top_k, next_token_logits.size(-1)),
                    )
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(next_token_logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                start_pos += 1

                token_str = tokenizer.decode(idx_next[0].tolist())
                yield f"data: {json.dumps({'text': token_str})}\n\n"
                await asyncio.sleep(0)

    yield "data: [DONE]\n\n"


@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    model: GPT = Depends(get_model),
    tokenizer: BPETokenizer = Depends(get_tokenizer),
    device: str = Depends(get_device),
):
    if request.stream:
        return StreamingResponse(
            generate_stream(request, model, tokenizer, device),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming fallback
        torch.manual_seed(request.seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed(request.seed)

        start_prompt = request.prompt if request.prompt else "\n"
        start_ids = tokenizer.encode(start_prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

        ctx = (
            torch.autocast(
                device_type=device,
                dtype=torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16,
            )
            if device.startswith("cuda")
            else nullcontext()
        )

        with torch.no_grad():
            with ctx:
                y = model.generate(
                    x,
                    request.max_new_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                )

        output_text = tokenizer.decode(y[0].tolist())
        return GenerateResponse(text=output_text[len(start_prompt) :])


def main():
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
