"""
FastAPI Server to serve the GPT model.
"""

import os
import json
import asyncio
import torch
import torch.nn.functional as F
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, RedirectResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager, nullcontext
from tiny_shakespeare_gpt.model import GPT, GPTConfig
from tiny_shakespeare_gpt.tokenizer import BPETokenizer

# Global state
app_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "out")
    model_ckpt_path = os.path.join(out_dir, "model.safetensors")
    meta_ckpt_path = os.path.join(out_dir, "ckpt_meta.pt")

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

    app_state["model"] = model
    app_state["tokenizer"] = tokenizer
    app_state["device"] = device

    yield
    # Teardown
    app_state.clear()


app = FastAPI(title="Tiny Shakespeare GPT API", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = "\n"
    max_new_tokens: int = 500
    temperature: float = 0.8
    top_k: int = 200
    stream: bool = False
    seed: int = 1337


class GenerateResponse(BaseModel):
    text: str


async def generate_stream(request: GenerateRequest, model: GPT, tokenizer, device: str):
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
            logits, _ = model(x, start_pos=0, use_cache=True)

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

                logits, _ = model(idx_next, start_pos=start_pos, use_cache=True)

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

            # Clear cache
            for block in model.blocks:
                if hasattr(block.attn, "cache_k"):
                    del block.attn.cache_k  # type: ignore
                    del block.attn.cache_v  # type: ignore

    yield "data: [DONE]\n\n"


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    model = app_state.get("model")
    tokenizer = app_state.get("tokenizer")
    device = str(app_state.get("device", "cpu"))

    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
