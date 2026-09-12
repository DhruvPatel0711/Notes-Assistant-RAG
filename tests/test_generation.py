"""
Tests for Phases 4+5: Generation — prompts, context formatting,
source extraction, and LCEL chain.

Split into:
    - Unit tests: format_docs, extract_sources, prompt structure (no LLM)
    - Integration tests: full chain with live LLM (marked with pytest.mark.integration)
"""

import re

import pytest

from src.config import settings


# ═══════════════════════════════════════════════════════════════════
# format_docs Tests (unit — no LLM)
# ═══════════════════════════════════════════════════════════════════


class TestFormatDocs:
    """Tests for context formatting."""

    def test_format_docs_produces_labeled_blocks(self):
        """Each doc should become a '--- Law N: Title ---' block."""
        from langchain_core.documents import Document
        from src.generation.chain import format_docs

        docs = [
            Document(
                page_content="Content of law one.",
                metadata={"law": 1, "title": "Never Outshine the Master",
                          "source": "1.txt", "id": "law_1"},
            ),
            Document(
                page_content="Content of law two.",
                metadata={"law": 2, "title": "Use Former Enemies",
                          "source": "2.txt", "id": "law_2"},
            ),
        ]
        result = format_docs(docs)

        assert "--- Law 1: Never Outshine the Master ---" in result
        assert "--- Law 2: Use Former Enemies ---" in result
        assert "Content of law one." in result
        assert "Content of law two." in result

    def test_format_docs_separates_with_blank_lines(self):
        """Laws should be separated by double newlines."""
        from langchain_core.documents import Document
        from src.generation.chain import format_docs

        docs = [
            Document(page_content="A", metadata={"law": 1, "title": "T1"}),
            Document(page_content="B", metadata={"law": 2, "title": "T2"}),
        ]
        result = format_docs(docs)
        assert "\n\n" in result

    def test_format_docs_empty_list(self):
        """Empty input should produce empty string."""
        from src.generation.chain import format_docs

        assert format_docs([]) == ""

    def test_format_docs_with_real_documents(self):
        """Should work with actual loaded documents."""
        from src.ingestion.loader import load_documents
        from src.generation.chain import format_docs

        docs = load_documents()[:3]
        result = format_docs(docs)
        assert "--- Law" in result
        assert len(result) > 100


# ═══════════════════════════════════════════════════════════════════
# extract_sources Tests (unit — no LLM)
# ═══════════════════════════════════════════════════════════════════


class TestExtractSources:
    """Tests for source extraction."""

    def test_extract_sources_structure(self):
        """Each source should have law, title, source keys."""
        from langchain_core.documents import Document
        from src.generation.chain import extract_sources

        docs = [
            Document(
                page_content="text",
                metadata={"law": 19, "title": "Know Who You Deal With",
                          "source": "19.txt", "id": "law_19"},
            ),
        ]
        sources = extract_sources(docs)
        assert len(sources) == 1
        assert sources[0]["law"] == 19
        assert sources[0]["title"] == "Know Who You Deal With"
        assert sources[0]["source"] == "19.txt"

    def test_extract_sources_deduplicates(self):
        """Same law appearing twice should only produce one source entry."""
        from langchain_core.documents import Document
        from src.generation.chain import extract_sources

        docs = [
            Document(page_content="chunk 1", metadata={"law": 5, "title": "T", "source": "5.txt"}),
            Document(page_content="chunk 2", metadata={"law": 5, "title": "T", "source": "5.txt"}),
            Document(page_content="chunk 3", metadata={"law": 10, "title": "T2", "source": "10.txt"}),
        ]
        sources = extract_sources(docs)
        assert len(sources) == 2
        law_numbers = [s["law"] for s in sources]
        assert 5 in law_numbers
        assert 10 in law_numbers

    def test_extract_sources_empty(self):
        """Empty input should produce empty list."""
        from src.generation.chain import extract_sources

        assert extract_sources([]) == []

    def test_extract_sources_preserves_order(self):
        """Sources should maintain retrieval order (most relevant first)."""
        from langchain_core.documents import Document
        from src.generation.chain import extract_sources

        docs = [
            Document(page_content="a", metadata={"law": 19, "title": "T1", "source": "19.txt"}),
            Document(page_content="b", metadata={"law": 1, "title": "T2", "source": "1.txt"}),
            Document(page_content="c", metadata={"law": 12, "title": "T3", "source": "12.txt"}),
        ]
        sources = extract_sources(docs)
        assert [s["law"] for s in sources] == [19, 1, 12]


# ═══════════════════════════════════════════════════════════════════
# Prompt Template Tests (unit — no LLM)
# ═══════════════════════════════════════════════════════════════════


class TestPromptTemplates:
    """Tests for prompt template structure."""

    def test_rag_prompt_has_required_variables(self):
        """RAG prompt must accept {question} and {context}."""
        from src.generation.prompts import RAG_PROMPT

        variables = RAG_PROMPT.input_variables
        assert "question" in variables
        assert "context" in variables

    def test_situation_prompt_has_required_variables(self):
        """Situation prompt must accept {question} and {context}."""
        from src.generation.prompts import SITUATION_PROMPT

        variables = SITUATION_PROMPT.input_variables
        assert "question" in variables
        assert "context" in variables

    def test_rag_prompt_contains_grounding_instructions(self):
        """Prompt must instruct LLM to use ONLY retrieved context."""
        from src.generation.prompts import RAG_PROMPT

        template_text = RAG_PROMPT.messages[0].prompt.template
        assert "ONLY" in template_text
        assert "MUST NOT" in template_text
        assert "prior knowledge" in template_text.lower()

    def test_rag_prompt_specifies_output_format(self):
        """Prompt must specify the expected output structure."""
        from src.generation.prompts import RAG_PROMPT

        template_text = RAG_PROMPT.messages[0].prompt.template
        assert "Most Relevant Law:" in template_text
        assert "Why it applies:" in template_text

    def test_situation_prompt_has_practical_section(self):
        """Situation prompt should ask for actionable advice."""
        from src.generation.prompts import SITUATION_PROMPT

        template_text = SITUATION_PROMPT.messages[0].prompt.template
        assert "What you should do:" in template_text


# ═══════════════════════════════════════════════════════════════════
# Integration Tests — Full Chain (need live LLM)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestRAGChainIntegration:
    """Integration tests for the full LCEL chain.

    These call the actual Gemini API and require:
        - GOOGLE_API_KEY set in .env
        - ChromaDB index built

    Run with: pytest -m integration
    """

    def test_ask_returns_answer_and_sources(self):
        """ask() should return dict with 'answer' and 'sources' keys."""
        from src.generation.chain import ask

        result = ask("How to avoid offending the wrong person")
        assert "answer" in result
        assert "sources" in result
        assert isinstance(result["answer"], str)
        assert isinstance(result["sources"], list)
        assert len(result["answer"]) > 50
        assert len(result["sources"]) > 0

    def test_answer_mentions_relevant_law(self):
        """Answer should mention a law number from the retrieved sources."""
        from src.generation.chain import ask

        result = ask("How to avoid offending the wrong person")
        # Should mention at least one law number
        assert re.search(r"Law\s+\d+", result["answer"]), (
            "Answer doesn't mention any law number"
        )

    def test_answer_follows_format(self):
        """Answer should follow the prescribed format."""
        from src.generation.chain import ask

        result = ask("What principle applies to guarding your reputation")
        answer = result["answer"]
        assert "Most Relevant Law:" in answer
        assert "Why it applies:" in answer or "why it applies:" in answer.lower()

    def test_sources_have_required_fields(self):
        """Each source in the response should have law, title, source."""
        from src.generation.chain import ask

        result = ask("How to deal with a more powerful opponent")
        for source in result["sources"]:
            assert "law" in source
            assert "title" in source
            assert "source" in source

    def test_ask_situation_returns_practical_advice(self):
        """ask_situation() should return response with practical advice."""
        from src.generation.chain import ask_situation

        result = ask_situation(
            "My manager is insecure about their technical skills "
            "while I clearly excel. How should I act around them?"
        )
        assert "answer" in result
        assert "sources" in result
        assert len(result["answer"]) > 50
