FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Install basic build tools just in case
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install dependencies first for better caching (without installing the project itself yet)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy the application code
COPY . .
# Now install the project
RUN uv sync --frozen --no-dev

# Ensure entrypoint is executable
RUN chmod +x scripts/docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["scripts/docker-entrypoint.sh"]
