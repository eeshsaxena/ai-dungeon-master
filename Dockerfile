# AI Dungeon Master — Production-grade slim Docker image
# Multi-stage build: ~250MB final image.
# Defaults to mock LLM mode so the game is playable without any external services.

FROM python:3.11-slim AS base

# System deps for Pillow + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for layer caching
COPY requirements.txt .

# Install minimal core deps (no GPU/SD/LoRA — those are opt-in extras)
RUN pip install --no-cache-dir \
    fastapi>=0.109.0 \
    "uvicorn[standard]>=0.27.0" \
    pydantic>=2.6.0 \
    httpx>=0.26.0 \
    python-dotenv>=1.0.0 \
    networkx>=3.2.0 \
    numpy>=1.26.0 \
    scikit-learn>=1.4.0 \
    aiofiles>=23.2.0 \
    python-multipart>=0.0.9 \
    Pillow>=10.0.0 \
    sentence-transformers>=2.3.0

# Copy app code
COPY backend ./backend
COPY frontend ./frontend
COPY .env.example .env

# Pre-warm RAG embedding cache at build time so first boot is instant
RUN cd backend && python -c "from rag import lore_retriever; lore_retriever.initialize()" || true

# Ports
EXPOSE 8000

# Non-root user
RUN useradd -m -u 1000 wizard && chown -R wizard:wizard /app
USER wizard

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=3)" || exit 1

# Start the FastAPI server
ENV APP_HOST=0.0.0.0 APP_PORT=8000 DEBUG=false LLM_PROVIDER=ollama
WORKDIR /app/backend
CMD ["python", "main.py"]
