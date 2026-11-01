FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies first (cached unless pyproject.toml changes)
COPY pyproject.toml .
RUN uv pip install --system .

# Pre-download the embedding model (cached unless deps change)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

# Copy source code last (changes here don't invalidate dep/model cache)
COPY config.yaml .
COPY src/ src/
RUN uv pip install --system .

EXPOSE 8080

CMD ["python", "-m", "vault_mcp.server"]
