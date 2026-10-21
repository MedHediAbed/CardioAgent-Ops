FROM python:3.12-slim

WORKDIR /app

# CPU-only PyTorch keeps the image ~2 GB smaller than the default CUDA build
RUN pip install --no-cache-dir --default-timeout=300 --retries 5 torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=300 --retries 5 -r requirements.txt

# Download the embedding model at build time so the container starts offline.
# Change it with: docker build --build-arg EMBEDDING_MODEL=<huggingface-id> .
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    PORT=8000
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY *.py .

EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
