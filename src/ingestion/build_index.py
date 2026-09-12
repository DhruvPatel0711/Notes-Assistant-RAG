"""
Index builder — ingests documents into ChromaDB.

This is the ONLY entry point for populating the vector database.
Run it whenever the source .txt files change:

    python -m src.ingestion.build_index

What it does:
    1. Loads all 28 law .txt files + .json metadata    (loader.py)
    2. Applies chunking strategy (default: whole-unit)  (chunker.py)
    3. Embeds chunks with BGE-large-en-v1.5             (bge.py)
    4. Stores in ChromaDB with cosine similarity        (here)

The vector database persists to disk (settings.chroma_path), so the
application can start up without re-embedding every time.

Flags:
    --force    Delete existing collection and rebuild from scratch.
"""

import argparse
import logging
import sys
import time

from langchain_chroma import Chroma

from src.config import settings
from src.embeddings.bge import BGEEmbeddings
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_documents

# Configure logging for CLI usage
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_index(force: bool = False) -> None:
    """Build or rebuild the ChromaDB vector index.

    Args:
        force: If True, delete the existing collection before rebuilding.
               If False, skip if the collection already has the expected
               number of documents.
    """
    logger.info("=" * 60)
    logger.info("Laws of Power — Index Builder")
    logger.info("=" * 60)

    # Step 1: Load documents
    logger.info("Step 1/4: Loading documents...")
    documents = load_documents()
    logger.info("  → %d documents loaded.", len(documents))

    # Step 2: Chunk documents
    logger.info("Step 2/4: Chunking documents...")
    chunks = chunk_documents(documents)
    logger.info("  → %d chunks produced.", len(chunks))

    # Step 3: Initialize embeddings
    logger.info("Step 3/4: Initializing embedding model (%s)...", settings.embedding_model)
    embeddings = BGEEmbeddings()
    logger.info("  → Embedding model ready (dim=%d).", embeddings.dimension)

    # Step 4: Build ChromaDB collection
    logger.info("Step 4/4: Building ChromaDB index...")
    logger.info("  → Collection: %s", settings.collection_name)
    logger.info("  → Persist dir: %s", settings.chroma_path)

    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_path),
        collection_metadata={"hnsw:space": "cosine"},
    )

    existing_count = vectorstore._collection.count()

    if force and existing_count > 0:
        logger.warning("  → --force: Deleting existing collection (%d docs)...", existing_count)
        vectorstore.delete_collection()
        # Re-create after deletion
        vectorstore = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_path),
            collection_metadata={"hnsw:space": "cosine"},
        )
        existing_count = 0

    if existing_count >= len(chunks):
        logger.info(
            "  → Collection already has %d documents (expected %d). "
            "Skipping. Use --force to rebuild.",
            existing_count,
            len(chunks),
        )
    else:
        logger.info("  → Embedding and inserting %d chunks...", len(chunks))
        start_time = time.time()

        # Use document IDs from metadata to ensure idempotency
        ids = [doc.metadata["id"] for doc in chunks]
        vectorstore.add_documents(documents=chunks, ids=ids)

        elapsed = time.time() - start_time
        logger.info("  → Index built in %.1f seconds.", elapsed)

    # Verify
    final_count = vectorstore._collection.count()
    logger.info("")
    logger.info("✅ Index ready: %d documents in '%s'", final_count, settings.collection_name)
    logger.info("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build the ChromaDB vector index for Laws of Power."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing collection and rebuild from scratch.",
    )
    args = parser.parse_args()

    try:
        build_index(force=args.force)
    except FileNotFoundError as e:
        logger.error("❌ %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("❌ Unexpected error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
