"""
LCEL RAG chain — end-to-end question → answer pipeline.

Architecture (Section 12 of the spec):
    User question
        → RunnableParallel(retriever, RunnablePassthrough)
        → context formatting + source extraction
        → prompt template
        → LLM (with fallback chain)
        → StrOutputParser
        → Answer + Sources

Key design decisions:
    - Single retrieval per request (Section 13): the retrieved docs are
      reused for context formatting, source extraction, and diagnostics.
    - format_docs() converts Document objects into readable context blocks.
    - extract_sources() pulls structured source info for the API response.
    - The chain returns {"answer": str, "sources": list} — not just a string.

Usage:
    from src.generation.chain import ask, ask_situation

    result = ask("How to avoid offending the wrong person")
    print(result["answer"])
    print(result["sources"])
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

from src.config import settings
from src.generation.prompts import RAG_PROMPT, SITUATION_PROMPT
from src.llm import get_llm_with_fallbacks
from src.retrieval.retriever import get_retriever

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Context Formatter
# ═══════════════════════════════════════════════════════════════════


def format_docs(retrieved_docs: List[Document]) -> str:
    """Convert retrieved Document objects into a readable context string.

    Each document becomes a labeled block:
        --- Law 19: Know Who You Are Dealing With ---
        [full text of the law notes]

    This format makes it clear to the LLM which law each block belongs to,
    and ensures it can cite law numbers accurately.

    Args:
        retrieved_docs: List of LangChain Documents with metadata.

    Returns:
        A formatted string with all retrieved laws, separated by blank lines.
    """
    context_chunks = [
        f"--- Law {doc.metadata['law']}: {doc.metadata['title']} ---\n{doc.page_content}"
        for doc in retrieved_docs
    ]
    return "\n\n".join(context_chunks)


# ═══════════════════════════════════════════════════════════════════
# Source Extractor
# ═══════════════════════════════════════════════════════════════════


def extract_sources(retrieved_docs: List[Document]) -> list[dict]:
    """Extract structured source info from retrieved documents.

    Deduplicates by law number (in case chunking ever produces
    multiple chunks from the same law).

    Args:
        retrieved_docs: List of LangChain Documents with metadata.

    Returns:
        List of dicts with keys: law, title, source. Example:
        [{"law": 19, "title": "Know Who You...", "source": "19.txt"}]
    """
    sources = []
    seen = set()
    for doc in retrieved_docs:
        law_num = doc.metadata.get("law")
        if law_num not in seen:
            seen.add(law_num)
            sources.append({
                "law": doc.metadata.get("law"),
                "title": doc.metadata.get("title"),
                "source": doc.metadata.get("source"),
            })
    return sources


# ═══════════════════════════════════════════════════════════════════
# LCEL Chain Builder
# ═══════════════════════════════════════════════════════════════════


def _build_rag_chain(prompt_template=None, k: int | None = None):
    """Build the LCEL RAG chain.

    Pipeline:
        question → RunnableParallel(retrieved_docs, question)
                 → format context + extract sources
                 → prompt → LLM → parse → {answer, sources}

    Args:
        prompt_template: The ChatPromptTemplate to use.
                        Defaults to RAG_PROMPT.
        k: Number of documents to retrieve. Defaults to settings.top_k.

    Returns:
        An LCEL Runnable that takes a question string and returns
        {"answer": str, "sources": list[dict]}.
    """
    prompt = prompt_template or RAG_PROMPT
    retriever = get_retriever(k=k)
    llm = get_llm_with_fallbacks()
    output_parser = StrOutputParser()

    # Step 1: Retrieve docs + pass question through
    retrieve_and_pass = RunnableParallel({
        "retrieved_docs": retriever,
        "question": RunnablePassthrough(),
    })

    # Step 2: Format context and generate answer, extract sources
    # We use RunnableParallel again to produce both answer and sources
    # from the same retrieved_docs (single retrieval, dual use).
    generate_and_extract = RunnableParallel({
        "answer": (
            RunnableLambda(lambda x: {
                "context": format_docs(x["retrieved_docs"]),
                "question": x["question"],
            })
            | prompt
            | llm
            | output_parser
        ),
        "sources": RunnableLambda(lambda x: extract_sources(x["retrieved_docs"])),
    })

    # Full chain: question → retrieve → format + generate → {answer, sources}
    chain = retrieve_and_pass | generate_and_extract
    return chain


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════


def ask(question: str, k: int | None = None) -> dict:
    """Ask a question and get a grounded answer with sources.

    Args:
        question: The user's question about Laws of Power.
        k: Number of documents to retrieve. Defaults to settings.top_k.

    Returns:
        Dict with keys:
            - "answer": The LLM's response (string)
            - "sources": List of source dicts [{law, title, source}, ...]
    """
    chain = _build_rag_chain(prompt_template=RAG_PROMPT, k=k)
    result = chain.invoke(question)
    logger.info("Question answered: '%s' → %d sources", question[:50], len(result["sources"]))
    return result


def ask_situation(situation: str, k: int | None = None) -> dict:
    """Describe a situation and get relevant law + practical advice.

    Uses the SITUATION_PROMPT which adds a 'What you should do' section.

    Args:
        situation: Description of the user's real situation.
        k: Number of documents to retrieve. Defaults to settings.top_k.

    Returns:
        Dict with keys:
            - "answer": The LLM's response (string)
            - "sources": List of source dicts [{law, title, source}, ...]
    """
    chain = _build_rag_chain(prompt_template=SITUATION_PROMPT, k=k)
    result = chain.invoke(situation)
    logger.info("Situation answered: '%s' → %d sources", situation[:50], len(result["sources"]))
    return result
