"""
FastAPI Server to serve the GPT model.
"""

import os
import json
import asyncio
import queue
import threading
import anyio
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
from tiny_shakespeare_gpt.tokenizer import BPETokenizer
from tiny_shakespeare_gpt.utils import get_project_root, setup_logging, load_checkpoint

# Setup logging
logger = setup_logging(__name__)


class ServerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    model_dir: str = str(get_project_root() / "out")
    cors_origins: list[str] = ["*"]
    max_concurrent_requests: int = 2

    class Config:
        env_file = ".env"


settings = ServerSettings()


class InferenceEngine:
    def __init__(self, model_dir: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        from pathlib import Path

        model_dir_path = Path(model_dir)

        try:
            self.model, _ = load_checkpoint(self.device, model_dir_path)
            logger.info("Model loaded successfully.")
        except FileNotFoundError as e:
            raise RuntimeError(str(e))

        self.tokenizer = BPETokenizer()
        logger.info("Tokenizer loaded successfully.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        seed: int,
    ) -> str:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        start_prompt = prompt if prompt else "\n"
        start_ids = self.tokenizer.encode(start_prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]

        ctx = (
            torch.autocast(
                device_type=self.device,
                dtype=torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16,
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

    def generate_stream_worker(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        seed: int,
        q: queue.Queue,
    ):
        try:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

            start_prompt = prompt if prompt else "\n"
            start_ids = self.tokenizer.encode(start_prompt)
            x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]

            B, T = x.shape
            if T > self.model.config.block_size:
                x = x[:, -self.model.config.block_size :]
                T = x.shape[1]

            ctx = (
                torch.autocast(
                    device_type=self.device,
                    dtype=torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16,
                )
                if self.device.startswith("cuda")
                else nullcontext()
            )

            with torch.no_grad():
                with ctx:
                    # Initialize KV cache
                    kv_caches = []
                    for _ in self.model.blocks:
                        cache_k = torch.zeros(
                            (
                                B,
                                self.model.config.block_size,
                                self.model.config.n_kv_head,
                                self.model.config.n_embd // self.model.config.n_head,
                            ),
                            dtype=self.model.tok_emb.weight.dtype,
                            device=self.device,
                        )
                        cache_v = torch.zeros_like(cache_k)
                        kv_caches.append((cache_k, cache_v))

                    logits, _, kv_caches = self.model(
                        x, start_pos=0, kv_caches=kv_caches
                    )

                    next_token_logits = logits[:, -1, :] / temperature
                    if top_k is not None:
                        v, _ = torch.topk(
                            next_token_logits, min(top_k, next_token_logits.size(-1))
                        )
                        next_token_logits[next_token_logits < v[:, [-1]]] = -float(
                            "Inf"
                        )
                    probs = F.softmax(next_token_logits, dim=-1)
                    idx_next = torch.multinomial(
                        probs, num_samples=1, generator=generator
                    )

                    start_pos = T

                    # Yield first token
                    token_str = self.tokenizer.decode(idx_next[0].tolist())
                    q.put(token_str)

                    for _ in range(1, max_new_tokens):
                        if start_pos >= self.model.config.block_size:
                            break

                        logits, _, kv_caches = self.model(
                            idx_next, start_pos=start_pos, kv_caches=kv_caches
                        )

                        next_token_logits = logits[:, -1, :] / temperature
                        if top_k is not None:
                            v, _ = torch.topk(
                                next_token_logits,
                                min(top_k, next_token_logits.size(-1)),
                            )
                            next_token_logits[next_token_logits < v[:, [-1]]] = -float(
                                "Inf"
                            )
                        probs = F.softmax(next_token_logits, dim=-1)
                        idx_next = torch.multinomial(
                            probs, num_samples=1, generator=generator
                        )

                        start_pos += 1

                        token_str = self.tokenizer.decode(idx_next[0].tolist())
                        q.put(token_str)
        except Exception as e:
            logger.error(f"Error during stream generation: {e}")
        finally:
            q.put(None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup
    engine = InferenceEngine(settings.model_dir)
    app.state.engine = engine
    app.state.generate_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
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

static_dir = str(get_project_root() / "static")
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


def get_semaphore(request: Request) -> asyncio.Semaphore:
    if not hasattr(request.app.state, "generate_semaphore"):
        raise HTTPException(status_code=503, detail="Semaphore not loaded yet")
    return request.app.state.generate_semaphore


async def generate_stream(
    request: GenerateRequest, engine: InferenceEngine
) -> AsyncGenerator[str, None]:
    q = queue.Queue()
    thread = threading.Thread(
        target=engine.generate_stream_worker,
        args=(
            request.prompt,
            request.max_new_tokens,
            request.temperature,
            request.top_k,
            request.seed,
            q,
        ),
    )
    thread.start()

    while True:
        token = await asyncio.to_thread(q.get)
        if token is None:
            break
        yield f"data: {json.dumps({'text': token})}\n\n"

    yield "data: [DONE]\n\n"


@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}


async def stream_with_semaphore(
    generator: AsyncGenerator[str, None], sem: asyncio.Semaphore
) -> AsyncGenerator[str, None]:
    try:
        async for chunk in generator:
            yield chunk
    finally:
        sem.release()


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    engine: InferenceEngine = Depends(get_engine),
    semaphore: asyncio.Semaphore = Depends(get_semaphore),
):
    await semaphore.acquire()

    if request.stream:
        return StreamingResponse(
            stream_with_semaphore(generate_stream(request, engine), semaphore),
            media_type="text/event-stream",
        )
    else:
        try:
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
        finally:
            semaphore.release()


def main():
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
