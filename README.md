# Laws of Power — Evaluated RAG Assistant

A production-grade Retrieval-Augmented Generation (RAG) system built to answer questions based on a personal collection of notes on the 48 Laws of Power.

## Architecture

This project is built using a modern, modular RAG architecture:

1.  **Ingestion (`src/ingestion/`)**:
    *   Reads 28 `.txt` law files (content) and 28 `.json` files (metadata).
    *   Uses a **Whole-Unit Chunking** strategy (1 law = 1 document) to preserve semantic coherence.
    *   Embeds documents using **BAAI/bge-large-en-v1.5** (1024-dim), running locally via `sentence-transformers`.
    *   Stores embeddings in a persistent **ChromaDB** vector database.

2.  **Retrieval (`src/retrieval/`)**:
    *   Exposes a LangChain retriever over ChromaDB.
    *   Uses **Cosine Similarity** to fetch the top-k most relevant laws.

3.  **Generation (`src/generation/`)**:
    *   Implements an **LCEL (LangChain Expression Language)** chain.
    *   Features two modes: Standard (RAG_PROMPT) and Situation Advice (SITUATION_PROMPT).
    *   **LLM Fallback Chain**: Automatically cascades through `gemini-3.8-flash → 3.7 → 3.6 → 3.5 → 3.5-flash-lite` if rate limits (429/503) are hit.
    *   Enforces strict grounding constraints to prevent hallucination.

4.  **Evaluation (`src/evaluation/`)**:
    *   **Retrieval@K**: Tests accuracy against a 60-question benchmark dataset.
    *   **RAGAS**: Measures generation quality (Faithfulness, Answer Relevancy).
    *   **Eval Cache**: Caches LLM responses to avoid burning API tokens during testing loops.

5.  **Deployment (`src/api/` & `app/`)**:
    *   **FastAPI Backend**: Provides `/ask` endpoint with in-memory rate limiting.
    *   **Streamlit UI**: Decoupled frontend for interactive chatting and source inspection.

## Quick Start

1.  **Set up environment**:
    Copy `.env.example` to `.env` and add your Google API key:
    ```bash
    cp .env.example .env
    # Edit .env and set GOOGLE_API_KEY
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Build the Vector Index**:
    This embeds the 28 laws and stores them in `chroma_db/`:
    ```bash
    python -m src.ingestion.build_index
    ```

4.  **Run the API Backend**:
    ```bash
    uvicorn src.api.main:app --reload
    ```

5.  **Run the Streamlit UI**:
    In a separate terminal:
    ```bash
    streamlit run app/streamlit_app.py
    ```

## Running Tests

Run the full test suite (unit + retrieval integration):
```bash
pytest
```

Run only unit tests (fast, no API key needed):
```bash
pytest -m "not integration"
```

Run integration tests (requires built index and valid API key):
```bash
pytest -m "integration"
```

Run evaluations:
```bash
python -m src.evaluation.retrieval_eval
python -m src.evaluation.ragas_eval
```
