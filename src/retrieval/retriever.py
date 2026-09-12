"""
Retriever — LangChain retriever abstraction over ChromaDB.

Provides a configurable retriever that searches the persisted ChromaDB
collection for semantically similar law documents.

The retriever returns LangChain Document objects with metadata intact,
which downstream components (context formatter, source extractor) depend on.

Usage:
    from src.retrieval.retriever import get_retriever, get_vectorstore

    # Get a retriever with default k from settings
    retriever = get_retriever()
    docs = retriever.invoke("How to avoid offending the wrong person")

    # Get a retriever with custom k
    retriever = get_retriever(k=10)

    # Direct vectorstore access (for evaluation, debugging)
    vs = get_vectorstore()
    results = vs.similarity_search_with_score("query", k=5)
"""

import logging

from langchain_chroma import Chroma

from src.config import settings
from src.embeddings.bge import BGEEmbeddings

logger = logging.getLogger(__name__)

# Module-level cache for the vectorstore instance.
# The embedding model inside it is also cached (see bge.py),
# so this is cheap after first initialization.
_vectorstore_cache: Chroma | None = None


def get_vectorstore() -> Chroma:
    """Get (or create) the ChromaDB vectorstore instance.

    The vectorstore connects to the persisted ChromaDB collection
    using the same embedding model and collection name from settings.

    Returns:
        A Chroma vectorstore instance backed by the persisted index.

    Raises:
        FileNotFoundError: If chroma_path doesn't exist (run build_index first).
        RuntimeError: If the collection is empty (run build_index first).
    """
    global _vectorstore_cache

    if _vectorstore_cache is not None:
        return _vectorstore_cache

    if not settings.chroma_path.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {settings.chroma_path}. "
            f"Run 'python -m src.ingestion.build_index' first."
        )

    embeddings = BGEEmbeddings()

    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_path),
    )

    doc_count = vectorstore._collection.count()
    if doc_count == 0:
        raise RuntimeError(
            f"ChromaDB collection '{settings.collection_name}' is empty. "
            f"Run 'python -m src.ingestion.build_index' first."
        )

    logger.info(
        "Vectorstore ready: %d documents in '%s'",
        doc_count,
        settings.collection_name,
    )

    _vectorstore_cache = vectorstore
    return vectorstore


def get_retriever(k: int | None = None):
    """Get a LangChain retriever over the ChromaDB vectorstore.

    The retriever wraps ChromaDB's similarity search and returns
    the top-k most relevant Document objects for any query.

    Args:
        k: Number of documents to retrieve. Defaults to settings.top_k.

    Returns:
        A LangChain VectorStoreRetriever (callable with .invoke(query)).
    """
    top_k = k if k is not None else settings.top_k
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": top_k},
    )

    logger.info("Retriever created: k=%d", top_k)
    return retriever


def retrieve(query: str, k: int | None = None):
    """Convenience function: retrieve documents for a query.

    This is a shortcut for get_retriever(k).invoke(query).
    Useful for evaluation and debugging.

    Args:
        query: The user's question or search text.
        k: Number of documents to retrieve. Defaults to settings.top_k.

    Returns:
        List of LangChain Document objects, ranked by relevance.
    """
    retriever = get_retriever(k=k)
    return retriever.invoke(query)
