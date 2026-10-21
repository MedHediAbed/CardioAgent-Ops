# rag-engine — RAG Pipeline over Service 2 (vector-db)

Indexes the ESC (European Society of Cardiology) guidelines into Qdrant and
returns the passages most relevant to a clinical question. Service 4
(agent-orchestrator) calls `/search` and puts the results in the LLM prompt,
so the generated report is grounded in real guidelines instead of invented.

```
ESC PDFs ──> chunking ──> embeddings ──> Qdrant (vector-db)      [/ingest, once]
question ──> embedding ──> cosine search ──> top-k passages      [/search]
```

## Pipeline
1. **Load** — `pypdf` extracts the text of each PDF page by page
   (`.txt` and `.md` files are also accepted).
2. **Clean** — joins hyphenated line breaks, collapses whitespace.
3. **Chunk** — 200-word windows with a 40-word overlap (so a sentence cut at a
   boundary is still whole in one of the two chunks). Chunks < 20 words are dropped.
4. **Embed** — `sentence-transformers/all-MiniLM-L6-v2` (384 dims, normalized).
5. **Index** — upsert into the `esc_guidelines` collection (cosine distance).
   Each chunk has a stable ID (`source:page:chunk`), so re-indexing a PDF
   overwrites it instead of creating duplicates.

## Endpoints
| Method | Path | Description |
|---|---|---|
| GET | `/health` | Qdrant status + number of indexed chunks |
| POST | `/ingest` | Index every `.pdf` / `.txt` / `.md` in `docs/` |
| POST | `/search` | `{"query": "...", "top_k": 4}` → `{"results": [{text, source, page, score}]}` |

## Configuration
Everything is set through environment variables. With Docker, copy
`.env.example` to `.env` and change what you need. Every value has a default,
so the `.env` is optional.

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` (`http://vector-db:6333` in compose) | Qdrant address |
| `QDRANT_API_KEY` | empty | Only for Qdrant Cloud / secured instances |
| `QDRANT_COLLECTION` | `esc_guidelines` | Collection name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Any sentence-transformers model from HuggingFace |
| `DOCS_DIR` | `docs` | Folder indexed by `/ingest` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `200` / `40` | In words |
| `TOP_K` | `4` | Default number of passages returned |
| `PORT` | `8000` | Port inside the container |
| `RAG_PORT` / `QDRANT_PORT` | `8003` / `6333` | Ports on your machine (compose only) |

**Changing the embedding model:** in Docker it is downloaded at build time,
so rebuild (`docker compose up --build`). Also use a new `QDRANT_COLLECTION`,
because each model produces vectors of a different size.

**Joining the main platform compose:** copy the `vector-db` and `rag-engine`
services into the root `docker-compose.yml` (adjust `build.context` to
`./rag-engine`). Other services reach it at `http://rag-engine:8000`.

## Run
1. Put the ESC guideline PDFs in `docs/` (PDFs are not committed: copyright + size).
2. Start Qdrant + the service:
   ```bash
   docker compose up --build
   ```
3. Index, then search:
   ```bash
   curl -X POST http://localhost:8003/ingest
   curl -X POST http://localhost:8003/search \
     -H "Content-Type: application/json" \
     -d '{"query": "systolic murmur, which exam is recommended?", "top_k": 3}'
   ```

Without Docker: `pip install -r requirements.txt`, start Qdrant, then
`python ingest.py` and `uvicorn main:app --port 8003`.

## Tests
```bash
pytest test_rag_processing.py
```
