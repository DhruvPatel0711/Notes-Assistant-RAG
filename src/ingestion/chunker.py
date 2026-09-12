"""
Document chunker — controls how documents are split for embedding.

Default strategy: WHOLE-UNIT chunking (one law = one chunk).

Why this is the default (and NOT a compromise):
    A NAACL 2025 peer-reviewed benchmark found that fixed-size / whole-unit
    chunking consistently outperformed semantic chunking on realistic document
    sets across both retrieval and generation metrics. Since each of the 28
    laws is already a self-contained semantic unit (one topic, one argument),
    splitting them would fragment that coherence without measurable benefit.

    Sub-chunking can be added later by swapping in a different chunker
    function — the pipeline is designed as:
        raw law → chunker → chunks → embeddings
    so only this module needs to change.

Usage:
    from src.ingestion.chunker import chunk_documents
    chunks = chunk_documents(documents)  # by default, returns input unchanged
"""

import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: List[Document],
    strategy: str = "whole",
) -> List[Document]:
    """Apply a chunking strategy to a list of Documents.

    Args:
        documents: List of loaded LangChain Documents (one per law).
        strategy: Chunking strategy to use. Currently supported:
            - "whole": One document per law (default, evidence-backed).

    Returns:
        List of Document chunks. With "whole" strategy, this is the
        same list passed in (identity operation).

    Raises:
        ValueError: If an unsupported strategy is specified.
    """
    if strategy == "whole":
        logger.info(
            "Chunking strategy: whole-unit (1 law = 1 chunk). "
            "%d documents → %d chunks.",
            len(documents),
            len(documents),
        )
        return documents

    # Future: add "recursive", "semantic", etc. here
    raise ValueError(
        f"Unknown chunking strategy: '{strategy}'. "
        f"Supported: 'whole'. "
        f"Add new strategies here if evaluation shows they're needed."
    )
