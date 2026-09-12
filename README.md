# Laws of Power — Evaluated RAG Assistant

> A production-grade Retrieval-Augmented Generation system over 28 personal Laws of Power notes, with full retrieval and generation evaluation.

## 🚧 Under Construction

This project is being built phase by phase. See the development log below.

---

## Architecture

```
User Question
    │
    ▼
┌──────────────┐    ┌───────────────────┐    ┌──────────────┐
│  BGE-large   │───▶│   ChromaDB        │───▶│  Top-K Docs  │
│  Embeddings  │    │   (cosine sim)    │    │  Retrieved   │
└──────────────┘    └───────────────────┘    └──────┬───────┘
                                                    │
                                                    ▼
                                            ┌───────────────┐
                                            │  Prompt +     │
                                            │  Context      │
                                            │  Formatting   │
                                            └──────┬────────┘
                                                   │
                                                   ▼
                                            ┌───────────────┐
                                            │  Gemini LLM   │
                                            │  (fallback    │
                                            │   chain)      │
                                            └──────┬────────┘
                                                   │
                                                   ▼
                                            ┌───────────────┐
                                            │  Grounded     │
                                            │  Answer +     │
                                            │  Sources      │
                                            └───────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | BAAI/bge-large-en-v1.5 (1024-dim) |
| Vector DB | ChromaDB (persistent, cosine similarity) |
| LLM | Gemini (5-model fallback chain) |
| Orchestration | LangChain LCEL |
| API | FastAPI + rate limiting |
| UI | Streamlit |
| Evaluation | RAGAS + custom retrieval/generation split |

## Development Log

- [ ] Phase 1: Foundation
- [ ] Phase 2: Ingestion
- [ ] Phase 3: Retrieval
- [ ] Phase 4: Generation
- [ ] Phase 5: LCEL Chain
- [ ] Phase 6: Evaluation
- [ ] Phase 7: Application (API + UI)
- [ ] Phase 8: Finalization

---

*Built as a portfolio project demonstrating evaluated RAG system design.*
