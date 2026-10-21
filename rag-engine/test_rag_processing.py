import pytest

from rag_processing import (
    MIN_CHUNK_WORDS,
    chunk_document,
    chunk_id,
    chunk_text,
    clean_text,
    index_documents
)


def test_clean_text_joins_hyphenated_words():
    assert clean_text("aortic valvu-\nlar  disease\n\n") == "aortic valvular disease"


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(500))

    chunks = chunk_text(text, chunk_size=200, overlap=40)

    assert all(len(chunk.split()) <= 200 for chunk in chunks)
    # The last 40 words of a chunk start the next one
    assert chunks[0].split()[-40:] == chunks[1].split()[:40]
    # No word is lost
    assert chunks[-1].split()[-1] == "w499"


def test_chunk_text_skips_tiny_text():
    assert chunk_text("Table 3", chunk_size=200, overlap=40) == []


def test_chunk_text_keeps_short_paragraph():
    text = " ".join(["word"] * MIN_CHUNK_WORDS)

    assert chunk_text(text, chunk_size=200, overlap=40) == [text]


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=50, overlap=50)


def test_chunk_id_is_stable():
    chunk = {"source": "esc_vhd.pdf", "page": 12, "chunk_index": 0}

    assert chunk_id(chunk) == chunk_id(dict(chunk))
    assert chunk_id(chunk) != chunk_id({**chunk, "page": 13})


def test_chunk_document_reads_text_files(tmp_path):
    file = tmp_path / "esc_notes.md"
    file.write_text(" ".join(["murmur"] * 50), encoding="utf-8")

    chunks = chunk_document(str(file))

    assert len(chunks) == 1
    assert chunks[0]["source"] == "esc_notes.md"
    assert chunks[0]["page"] == 1


def test_index_documents_missing_folder_returns_empty(tmp_path):
    assert index_documents(str(tmp_path / "does_not_exist")) == []
