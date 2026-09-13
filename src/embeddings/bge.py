"""
BGE Embeddings — LangChain-compatible wrapper around SentenceTransformer.

Uses BAAI/bge-large-en-v1.5 (1024-dim) for both document and query embeddings.
This is the SAME model for indexing and search — using different models
would cause a vector-space mismatch and silently degrade retrieval.

Why BGE-large specifically:
    - 1024-dim vectors provide strong semantic resolution
    - BAAI/bge-large-en-v1.5 is consistently ranked among the strongest
      open-source embedding models for retrieval (MTEB benchmarks, 2024–2026)
    - Self-hosted: no API cost, no rate limits on embedding calls
    - The LangChain Embeddings interface makes it swappable later if needed

Usage:
    from src.embeddings.bge import BGEEmbeddings

    embeddings = BGEEmbeddings()
    doc_vectors = embeddings.embed_documents(["text1", "text2"])
    query_vector = embeddings.embed_query("search query")
"""

import logging
from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)

# Module-level singleton to avoid reloading the ~1.3GB model on every call.
# SentenceTransformer is thread-safe for inference.
_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    """Load and cache the SentenceTransformer model."""
    if model_name not in _model_cache:
        logger.info("Loading embedding model: %s (this may take a moment)...", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")
    return _model_cache[model_name]


class BGEEmbeddings(Embeddings):
    """LangChain-compatible embedding wrapper for BGE models.

    Implements the Embeddings interface so it plugs directly into
    LangChain's Chroma integration (and any other vectorstore).

    Attributes:
        model_name: HuggingFace model ID (default from settings).
    """

    def __init__(self, model_name: str | None = None):
        """Initialize with model name from settings or override.

        Args:
            model_name: HuggingFace model ID. Defaults to settings.embedding_model.
        """
        self.model_name = model_name or settings.embedding_model
        # Eagerly load the model so any download/loading errors surface
        # at initialization time, not during a query.
        self._model = _get_model(self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (law texts).

        Args:
            texts: List of document strings to embed.

        Returns:
            List of embedding vectors (each is a list of 1024 floats).
        """
        return self._model.encode(texts, show_progress_bar=len(texts) > 5).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string.

        Args:
            text: The user's question or search query.

        Returns:
            A single embedding vector (list of 1024 floats).
        """
        return self._model.encode(text).tolist()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._model.get_sentence_embedding_dimension()
