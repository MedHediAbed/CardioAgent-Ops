from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_processing import COLLECTION_NAME, EMBEDDING_MODEL, TOP_K, count_chunks, index_documents, search


app = FastAPI(
    title="CardioAgent-Ops RAG Engine",
    description="RAG service - ESC guideline indexing and retrieval over Service 2 (vector-db)",
    version="1.0.0"
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(TOP_K, ge=1, le=20)


@app.get("/")
def root():
    return {
        "service": "rag-engine",
        "status": "running"
    }


@app.get("/health")
def health_check():
    try:
        chunks = count_chunks()
    except Exception:
        return {
            "status": "degraded",
            "vector_db": "unreachable"
        }

    return {
        "status": "healthy",
        "vector_db": "connected",
        "collection": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "indexed_chunks": chunks
    }


@app.post("/ingest")
def ingest_documents():
    """
    Index every document (.pdf, .txt, .md) found in the docs folder into Qdrant.
    """

    try:
        documents = index_documents()
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {str(error)}"
        )

    if not documents:
        raise HTTPException(
            status_code=404,
            detail="No document found in the docs folder"
        )

    return {
        "status": "success",
        "documents": documents,
        "total_chunks": sum(doc["chunks"] for doc in documents)
    }


@app.post("/search")
def search_guidelines(request: SearchRequest):
    """
    Return the ESC guideline chunks most relevant to the query.
    Used by Service 4 (agent-orchestrator) to ground the LLM.
    """

    try:
        results = search(request.query, request.top_k)
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Vector DB unavailable: {str(error)}"
        )

    return {
        "query": request.query,
        "results": results
    }
