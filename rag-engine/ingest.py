"""
Index the ESC guidelines (.pdf, .txt, .md) from the command line:

    python ingest.py            # uses DOCS_DIR (default: docs/)
    python ingest.py path/to/docs
"""
import sys

from rag_processing import COLLECTION_NAME, DOCS_DIR, index_documents


docs_dir = sys.argv[1] if len(sys.argv) > 1 else DOCS_DIR

documents = index_documents(docs_dir)

if not documents:
    print(f"No document found in {docs_dir}/")
    sys.exit(1)

for doc in documents:
    print(f"{doc['source']}: {doc['chunks']} chunks")

print(f"Total: {sum(doc['chunks'] for doc in documents)} chunks in '{COLLECTION_NAME}'")
