"""
Document loader — reads law .txt files and enrichment .json metadata.

Produces LangChain Document objects with metadata attached.
Each Document has:
    - page_content: the full text of the law notes (from .txt)
    - metadata: law number, title, source filename, id (from .json)

Why two file reads per law:
    The .txt files are the user's original handwritten notes — the source
    of truth that gets embedded and retrieved. The .json files are
    LLM-generated enrichments (title, summary) created in an earlier step.
    We ONLY use .json for metadata extraction (law number, title).
    The actual content embedded is always the raw .txt text.

Usage:
    from src.ingestion.loader import load_documents
    docs = load_documents()  # returns list of 28 LangChain Documents
"""

import json
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.config import settings

logger = logging.getLogger(__name__)


def load_documents(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> List[Document]:
    """Load all law documents with metadata.

    Reads each .txt file from data_dir and its corresponding .json
    from output_dir. Pairs them by filename number (e.g. 1.txt ↔ 1.json).

    Args:
        data_dir: Path to .txt files. Defaults to settings.data_path.
        output_dir: Path to .json files. Defaults to settings.output_path.

    Returns:
        List of LangChain Document objects, sorted by law number.

    Raises:
        FileNotFoundError: If data_dir or output_dir doesn't exist.
        ValueError: If a .txt file has no matching .json file.
    """
    data_path = data_dir or settings.data_path
    output_path = output_dir or settings.output_path

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    if not output_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_path}")

    # Find and sort .txt files by law number
    txt_files = sorted(
        data_path.glob("*.txt"),
        key=lambda p: int(p.stem),
    )

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {data_path}")

    documents = []

    for txt_path in txt_files:
        # Read the raw law text (source of truth)
        text = txt_path.read_text(encoding="utf-8")

        # Find matching JSON for metadata
        json_path = output_path / f"{txt_path.stem}.json"
        if not json_path.exists():
            raise ValueError(
                f"No matching .json metadata for {txt_path.name}. "
                f"Expected: {json_path}"
            )

        with open(json_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        # Build the Document with structured metadata
        doc = Document(
            page_content=text,
            metadata={
                "law": meta_data["law"],
                "title": meta_data["title"],
                "source": txt_path.name,
                "id": f"law_{meta_data['law']}",
            },
        )
        documents.append(doc)

        logger.debug(
            "Loaded: Law %d — %s (%d chars)",
            meta_data["law"],
            meta_data["title"],
            len(text),
        )

    logger.info("Loaded %d documents from %s", len(documents), data_path)
    return documents
