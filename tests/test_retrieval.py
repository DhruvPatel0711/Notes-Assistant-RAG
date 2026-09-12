"""
Tests for Phase 3: Retrieval — vectorstore, retriever, and search quality.

These tests require a built ChromaDB index (run build_index first).
They load the actual BGE model, so first run may be slow.
"""

import pytest

from src.config import settings


# ═══════════════════════════════════════════════════════════════════
# Vectorstore Tests
# ═══════════════════════════════════════════════════════════════════


class TestVectorstore:
    """Tests for get_vectorstore()."""

    def test_vectorstore_loads(self):
        """Vectorstore should connect to the persisted ChromaDB."""
        from src.retrieval.retriever import get_vectorstore

        vs = get_vectorstore()
        assert vs is not None

    def test_vectorstore_has_28_documents(self):
        """Collection should contain exactly 28 documents."""
        from src.retrieval.retriever import get_vectorstore

        vs = get_vectorstore()
        assert vs._collection.count() == 28

    def test_vectorstore_is_cached(self):
        """Calling get_vectorstore() twice should return the same instance."""
        from src.retrieval.retriever import get_vectorstore

        vs1 = get_vectorstore()
        vs2 = get_vectorstore()
        assert vs1 is vs2


# ═══════════════════════════════════════════════════════════════════
# Retriever Tests
# ═══════════════════════════════════════════════════════════════════


class TestRetriever:
    """Tests for get_retriever() and retrieve()."""

    def test_retriever_returns_k_documents(self):
        """Retriever should return exactly k documents."""
        from src.retrieval.retriever import retrieve

        docs = retrieve("How to avoid offending the wrong person", k=3)
        assert len(docs) == 3

    def test_retriever_returns_5_by_default(self):
        """Default k should come from settings.top_k (5)."""
        from src.retrieval.retriever import retrieve

        docs = retrieve("How to deal with a more powerful opponent")
        assert len(docs) == settings.top_k

    def test_retrieved_docs_have_metadata(self):
        """Every retrieved document should have law/title/source/id metadata."""
        from src.retrieval.retriever import retrieve

        docs = retrieve("Change identity and reputation", k=3)
        required = {"law", "title", "source", "id"}
        for doc in docs:
            missing = required - set(doc.metadata.keys())
            assert not missing, f"Missing metadata: {missing}"

    def test_retrieved_docs_have_content(self):
        """Retrieved documents should have non-empty page_content."""
        from src.retrieval.retriever import retrieve

        docs = retrieve("How to use selective honesty", k=3)
        for doc in docs:
            assert doc.page_content
            assert len(doc.page_content) > 50

    def test_law_19_retrieved_for_offending_question(self):
        """'Avoid offending wrong person' should retrieve Law 19 in top 3."""
        from src.retrieval.retriever import retrieve

        docs = retrieve("How do you avoid offending the wrong person", k=3)
        retrieved_laws = [doc.metadata["law"] for doc in docs]
        assert 19 in retrieved_laws, (
            f"Law 19 not in top 3. Got: {retrieved_laws}"
        )

    def test_law_25_retrieved_for_reputation_question(self):
        """'Change identity/reputation' should retrieve Law 25 in top 5."""
        from src.retrieval.retriever import retrieve

        docs = retrieve("How do I change how people perceive me", k=5)
        retrieved_laws = [doc.metadata["law"] for doc in docs]
        assert 25 in retrieved_laws, (
            f"Law 25 not in top 5. Got: {retrieved_laws}"
        )

    def test_law_22_retrieved_for_surrender_question(self):
        """'Surrender to gain leverage' should retrieve Law 22 in top 3."""
        from src.retrieval.retriever import retrieve

        docs = retrieve(
            "How can surrendering gain leverage against a stronger opponent",
            k=3,
        )
        retrieved_laws = [doc.metadata["law"] for doc in docs]
        assert 22 in retrieved_laws, (
            f"Law 22 not in top 3. Got: {retrieved_laws}"
        )

    def test_different_k_values_return_different_counts(self):
        """k=1 and k=10 should return different numbers of documents."""
        from src.retrieval.retriever import retrieve

        docs_1 = retrieve("power", k=1)
        docs_10 = retrieve("power", k=10)
        assert len(docs_1) == 1
        assert len(docs_10) == 10

    def test_no_duplicate_laws_in_results(self):
        """Each retrieved document should be a different law (no duplicates)."""
        from src.retrieval.retriever import retrieve

        docs = retrieve("How to deal with enemies", k=5)
        law_numbers = [doc.metadata["law"] for doc in docs]
        assert len(law_numbers) == len(set(law_numbers)), (
            f"Duplicate laws in results: {law_numbers}"
        )
