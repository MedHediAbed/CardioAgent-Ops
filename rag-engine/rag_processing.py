import os
import re
import uuid
from pathlib import Path

from pypdf import PdfReader


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None   # only for Qdrant Cloud / secured instances
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "esc_guidelines")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DOCS_DIR = os.getenv("DOCS_DIR", "docs")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))        # words per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))   # words shared between chunks
MIN_CHUNK_WORDS = 20                                    # skip tiny chunks (titles, page numbers)
TOP_K = int(os.getenv("TOP_K", "4"))

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")

# Loaded lazily: the model and the client are heavy and not needed for chunking
_model = None
_client = None


def get_model():
    """
    Load the sentence-transformers embedding model once.
    """

    global _model

    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)

    return _model


def get_client():
    """
    Connect to Service 2 (vector-db / Qdrant) once.
    """

    global _client

    if _client is None:
        from qdrant_client import QdrantClient

        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    return _client


def clean_text(text: str):
    """
    Remove PDF artefacts: hyphenated line breaks and repeated whitespace.
    """

    # "valvu-\nlar" -> "valvular"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_pdf(file_path: str):
    """
    Read a PDF and return its text page by page.
    """

    reader = PdfReader(file_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")

        if text:
            pages.append({"page": page_number, "text": text})

    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Split a text into overlapping chunks of `chunk_size` words.
    """

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]

        if len(chunk) >= MIN_CHUNK_WORDS:
            chunks.append(" ".join(chunk))

        # The last chunk already reaches the end of the text
        if start + chunk_size >= len(words):
            break

    return chunks


def load_text_file(file_path: str):
    """
    Read a .txt / .md file as a single page.
    """

    text = clean_text(Path(file_path).read_text(encoding="utf-8"))

    return [{"page": 1, "text": text}] if text else []


def load_document(file_path: str):
    """
    Load any supported document and return its text page by page.
    """

    if Path(file_path).suffix.lower() == ".pdf":
        return load_pdf(file_path)

    return load_text_file(file_path)


def chunk_document(file_path: str):
    """
    Load a document and turn it into chunks with their metadata.
    """

    source = Path(file_path).name
    chunks = []

    for page in load_document(file_path):
        for index, text in enumerate(chunk_text(page["text"])):
            chunks.append({
                "text": text,
                "source": source,
                "page": page["page"],
                "chunk_index": index
            })

    return chunks


def chunk_id(chunk: dict):
    """
    Stable ID so that indexing the same document twice overwrites it
    instead of creating duplicates.
    """

    key = f"{chunk['source']}:{chunk['page']}:{chunk['chunk_index']}"

    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def create_collection():
    """
    Create the Qdrant collection if it does not exist yet (cosine similarity).
    """

    from qdrant_client.models import Distance, VectorParams

    client = get_client()

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=get_model().get_embedding_dimension(),
                distance=Distance.COSINE
            )
        )


def index_chunks(chunks: list[dict]):
    """
    Embed the chunks and store them in Qdrant.
    """

    from qdrant_client.models import PointStruct

    if not chunks:
        return 0

    create_collection()

    vectors = get_model().encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    points = [
        PointStruct(id=chunk_id(chunk), vector=vector.tolist(), payload=chunk)
        for chunk, vector in zip(chunks, vectors)
    ]

    # Upload in batches to keep requests small
    for start in range(0, len(points), 256):
        get_client().upsert(
            collection_name=COLLECTION_NAME,
            points=points[start:start + 256]
        )

    return len(points)


def index_documents(docs_dir: str = DOCS_DIR):
    """
    Complete indexing pipeline: every supported document in `docs_dir` -> chunks -> Qdrant.
    """

    if not Path(docs_dir).is_dir():
        return []

    files = sorted(
        path for path in Path(docs_dir).iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.name != "README.md"
    )
    documents = []

    for file in files:
        chunks = chunk_document(str(file))
        count = index_chunks(chunks)

        documents.append({"source": file.name, "chunks": count})

    return documents


def search(query: str, top_k: int = TOP_K):
    """
    Return the `top_k` guideline chunks most similar to the query.
    """

    client = get_client()

    if not client.collection_exists(COLLECTION_NAME):
        return []

    vector = get_model().encode(query, normalize_embeddings=True)

    points = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector.tolist(),
        limit=top_k
    ).points

    return [
        {
            "text": point.payload["text"],
            "source": point.payload["source"],
            "page": point.payload["page"],
            "score": round(point.score, 4)
        }
        for point in points
    ]


def count_chunks():
    """
    Number of chunks stored in the collection (0 if it does not exist).
    """

    client = get_client()

    if not client.collection_exists(COLLECTION_NAME):
        return 0

    return client.count(collection_name=COLLECTION_NAME).count
