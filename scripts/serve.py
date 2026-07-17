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
from fastapi.middleware.cors import CORSMiddleware
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
    cors_origins: list[str] = ["*"]

    class Config:
        env_file = ".env"

settings = ServerSettings()


class InferenceEngine:
    def __init__(self, model_dir: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        model_ckpt_path = os.path.join(model_dir, "model.safetensors")
        meta_ckpt_path = os.path.join(model_dir, "ckpt_meta.pt")

        if not os.path.exists(model_ckpt_path) or not os.path.exists(meta_ckpt_path):
            raise RuntimeError(
                f"Checkpoint not found at {model_ckpt_path} or {meta_ckpt_path}. Run scripts/train.py first."
            )

        torch.serialization.add_safe_globals([GPTConfig])
        meta = torch.load(meta_ckpt_path, map_location=self.device, weights_only=True)

        config = meta["config"]
        self.model = GPT(config)

        import safetensors.torch
        safetensors.torch.load_model(self.model, model_ckpt_path)
        self.model.eval()
        self.model.to(self.device)

        self.tokenizer = BPETokenizer()
        logger.info("Model and tokenizer loaded successfully.")

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_k: int, seed: int) -> str:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        start_prompt = prompt if prompt else "\n"
        start_ids = self.tokenizer.encode(start_prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]

        ctx = (
            torch.autocast(
                device_type=self.device,
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
            if self.device.startswith("cuda")
            else nullcontext()
        )

        with torch.no_grad():
            with ctx:
                y = self.model.generate(
                    x,
                    max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    generator=generator,
                )

        output_text = self.tokenizer.decode(y[0].tolist())
        return output_text[len(start_prompt) :]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup
    engine = InferenceEngine(settings.model_dir)
    app.state.engine = engine
    yield
    # Teardown
    logger.info("Shutting down server.")
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="Tiny Shakespeare GPT API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


def get_engine(request: Request) -> InferenceEngine:
    if not hasattr(request.app.state, "engine"):
        raise HTTPException(status_code=503, detail="Inference engine not loaded yet")
    return request.app.state.engine


async def generate_stream(
    request: GenerateRequest, engine: InferenceEngine
) -> AsyncGenerator[str, None]:
    torch.manual_seed(request.seed)
    if engine.device.startswith("cuda"):
        torch.cuda.manual_seed(request.seed)

    start_prompt = request.prompt if request.prompt else "\n"
    start_ids = engine.tokenizer.encode(start_prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=engine.device)[None, ...]

    B, T = x.shape
    if T > engine.model.config.block_size:
        x = x[:, -engine.model.config.block_size :]
        T = x.shape[1]

    ctx = (
        torch.autocast(
            device_type=engine.device,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        if engine.device.startswith("cuda")
        else nullcontext()
    )

    with torch.no_grad():
        with ctx:
            # Initialize KV cache
            kv_caches = []
            for _ in engine.model.blocks:
                cache_k = torch.zeros(
                    (B, engine.model.config.block_size, engine.model.config.n_kv_head, engine.model.config.n_embd // engine.model.config.n_head),
                    dtype=engine.model.tok_emb.weight.dtype,
                    device=engine.device,
                )
                cache_v = torch.zeros_like(cache_k)
                kv_caches.append((cache_k, cache_v))

            logits, _, kv_caches = engine.model(x, start_pos=0, kv_caches=kv_caches)

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
            token_str = engine.tokenizer.decode(idx_next[0].tolist())
            yield f"data: {json.dumps({'text': token_str})}\n\n"
            await asyncio.sleep(0)

            for _ in range(1, request.max_new_tokens):
                if start_pos >= engine.model.config.block_size:
                    break

                logits, _, kv_caches = engine.model(idx_next, start_pos=start_pos, kv_caches=kv_caches)

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

                token_str = engine.tokenizer.decode(idx_next[0].tolist())
                yield f"data: {json.dumps({'text': token_str})}\n\n"
                await asyncio.sleep(0)

    yield "data: [DONE]\n\n"


@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}


import anyio

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    engine: InferenceEngine = Depends(get_engine),
):
    if request.stream:
        return StreamingResponse(
            generate_stream(request, engine),
            media_type="text/event-stream",
        )
    else:
        # Offload non-streaming inference to a background thread
        output_text = await anyio.to_thread.run_sync(
            engine.generate,
            request.prompt,
            request.max_new_tokens,
            request.temperature,
            request.top_k,
            request.seed,
        )
        return GenerateResponse(text=output_text)


def main():
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
