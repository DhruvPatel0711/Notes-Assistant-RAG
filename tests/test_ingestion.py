"""
Tests for Phase 2: Ingestion — loader, chunker, embeddings, and build_index.

These tests cover:
    - Document loading (28 files, correct metadata)
    - Metadata structure validation
    - Chunking strategy (whole-unit)
    - BGE embedding dimensions (1024)
    - Embedding consistency (same text → same vector)
"""

import json
from pathlib import Path

import pytest

from src.config import settings


# ═══════════════════════════════════════════════════════════════════
# Document Loader Tests
# ═══════════════════════════════════════════════════════════════════


class TestDocumentLoader:
    """Tests for src.ingestion.loader.load_documents."""

    def test_loads_all_28_documents(self):
        """Should load exactly 28 documents (one per law)."""
        from src.ingestion.loader import load_documents

        docs = load_documents()
        assert len(docs) == 28

    def test_documents_sorted_by_law_number(self):
        """Documents should be returned in law-number order."""
        from src.ingestion.loader import load_documents

        docs = load_documents()
        law_numbers = [doc.metadata["law"] for doc in docs]
        assert law_numbers == sorted(law_numbers)

    def test_document_has_page_content(self):
        """Each document must have non-empty text content."""
        from src.ingestion.loader import load_documents

        docs = load_documents()
        for doc in docs:
            assert doc.page_content, f"Empty content for {doc.metadata.get('id')}"
            assert len(doc.page_content) > 50, (
                f"Suspiciously short content for {doc.metadata.get('id')}: "
                f"{len(doc.page_content)} chars"
            )

    def test_page_content_matches_txt_file(self):
        """page_content should be the raw text from the .txt file, not the JSON summary."""
        from src.ingestion.loader import load_documents

        docs = load_documents()
        # Check law 1 specifically
        law_1 = [d for d in docs if d.metadata["law"] == 1][0]
        txt_content = (settings.data_path / "1.txt").read_text(encoding="utf-8")
        assert law_1.page_content == txt_content

    def test_raises_on_missing_data_dir(self, tmp_path):
        """Should raise FileNotFoundError for non-existent data directory."""
        from src.ingestion.loader import load_documents

        with pytest.raises(FileNotFoundError, match="Data directory"):
            load_documents(data_dir=tmp_path / "nonexistent")

    def test_raises_on_missing_output_dir(self, tmp_path):
        """Should raise FileNotFoundError for non-existent output directory."""
        from src.ingestion.loader import load_documents

        with pytest.raises(FileNotFoundError, match="Output directory"):
            load_documents(output_dir=tmp_path / "nonexistent")


# ═══════════════════════════════════════════════════════════════════
# Metadata Tests
# ═══════════════════════════════════════════════════════════════════


class TestMetadata:
    """Tests for document metadata structure."""

    def test_metadata_has_required_fields(self):
        """Every document must have: law, title, source, id."""
        from src.ingestion.loader import load_documents

        docs = load_documents()
        required = {"law", "title", "source", "id"}
        for doc in docs:
            missing = required - set(doc.metadata.keys())
            assert not missing, (
                f"Document {doc.metadata.get('id', '?')} missing metadata: {missing}"
            )

    def test_law_number_is_int(self):
        """'law' metadata should be an integer, not a string."""
        from src.ingestion.loader import load_documents

        docs = load_documents()
        for doc in docs:
            assert isinstance(doc.metadata["law"], int), (
                f"law should be int, got {type(doc.metadata['law'])} "
                f"for {doc.metadata['id']}"
            )

    def test_law_numbers_cover_1_to_28(self):
        """All 28 law numbers should be present."""
        from src.ingestion.loader import load_documents

        docs = load_documents()
        law_numbers = {doc.metadata["law"] for doc in docs}
        expected = set(range(1, 29))
        assert law_numbers == expected, f"Missing laws: {expected - law_numbers}"

    def test_id_format(self):
        """ID should follow 'law_N' format."""
        from src.ingestion.loader import load_documents

        docs = load_documents()
        for doc in docs:
            expected_id = f"law_{doc.metadata['law']}"
            assert doc.metadata["id"] == expected_id

    def test_source_is_filename(self):
        """Source should be just the filename, not a full path."""
        from src.ingestion.loader import load_documents

        docs = load_documents()
        for doc in docs:
            assert doc.metadata["source"].endswith(".txt")
            assert "/" not in doc.metadata["source"]
            assert "\\" not in doc.metadata["source"]

    def test_title_is_non_empty(self):
        """Every law should have a non-empty title."""
        from src.ingestion.loader import load_documents

        docs = load_documents()
        for doc in docs:
            assert doc.metadata["title"], (
                f"Empty title for {doc.metadata['id']}"
            )


# ═══════════════════════════════════════════════════════════════════
# Chunker Tests
# ═══════════════════════════════════════════════════════════════════


class TestChunker:
    """Tests for src.ingestion.chunker.chunk_documents."""

    def test_whole_strategy_returns_same_documents(self):
        """'whole' strategy should return the exact same documents."""
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.loader import load_documents

        docs = load_documents()
        chunks = chunk_documents(docs, strategy="whole")
        assert len(chunks) == len(docs)
        assert chunks is docs  # Same object, not a copy

    def test_default_strategy_is_whole(self):
        """Default strategy should be 'whole'."""
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.loader import load_documents

        docs = load_documents()
        chunks = chunk_documents(docs)  # no strategy argument
        assert len(chunks) == len(docs)

    def test_invalid_strategy_raises(self):
        """Unknown strategy should raise ValueError with clear message."""
        from src.ingestion.chunker import chunk_documents

        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunk_documents([], strategy="semantic")


# ═══════════════════════════════════════════════════════════════════
# BGE Embedding Tests
# ═══════════════════════════════════════════════════════════════════


class TestBGEEmbeddings:
    """Tests for src.embeddings.bge.BGEEmbeddings.

    Note: These tests load the actual BGE model (~1.3GB).
    First run may be slow due to model download.
    """

@pytest.fixture(scope="module")
def embeddings():
    """Shared embedding instance across all tests."""
    from src.embeddings.bge import BGEEmbeddings
    return BGEEmbeddings()

    def test_dimension_is_1024(self, embeddings):
        """BGE-large should produce 1024-dimensional vectors."""
        assert embeddings.dimension == 1024

    def test_embed_query_returns_list_of_floats(self, embeddings):
        """embed_query should return a flat list of floats."""
        vector = embeddings.embed_query("test query")
        assert isinstance(vector, list)
        assert len(vector) == 1024
        assert all(isinstance(v, float) for v in vector)

    def test_embed_documents_returns_list_of_lists(self, embeddings):
        """embed_documents should return a list of vectors."""
        vectors = embeddings.embed_documents(["text one", "text two"])
        assert isinstance(vectors, list)
        assert len(vectors) == 2
        assert all(len(v) == 1024 for v in vectors)

    def test_same_text_produces_same_embedding(self, embeddings):
        """Determinism: same input should always produce same output."""
        text = "How to avoid offending the wrong person"
        v1 = embeddings.embed_query(text)
        v2 = embeddings.embed_query(text)
        assert v1 == v2

    def test_different_texts_produce_different_embeddings(self, embeddings):
        """Semantically different texts should have different vectors."""
        v1 = embeddings.embed_query("power and strategy")
        v2 = embeddings.embed_query("pizza and ice cream")
        assert v1 != v2
