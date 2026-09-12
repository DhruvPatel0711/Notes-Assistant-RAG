"""
Centralized configuration for the Laws of Power RAG Assistant.

All settings are loaded from environment variables (via .env file).
Uses pydantic-settings for validation and type coercion.

Usage:
    from src.config import settings
    print(settings.top_k)
    print(settings.data_path)
"""

import sys
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env file.

    Attributes:
        google_api_key: Google AI API key (required).
        embedding_model: HuggingFace model ID for embeddings.
        llm_models: Ordered list of Gemini models for fallback chain.
        chroma_dir: Directory name for ChromaDB persistence.
        collection_name: ChromaDB collection name.
        top_k: Number of documents to retrieve.
        temperature: LLM temperature for evaluation (deterministic).
        temperature_interactive: LLM temperature for interactive use.
        data_dir: Directory containing source .txt law files.
        output_dir: Directory containing enrichment .json files.
    """

    google_api_key: str = ""

    embedding_model: str = "BAAI/bge-large-en-v1.5"

    # Stored as comma-separated string because pydantic-settings parses
    # List[str] env vars as JSON before validators run, which breaks
    # plain comma-separated strings like "model-a,model-b".
    llm_models_str: str = "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-3.5-flash-lite"

    chroma_dir: str = "chroma_db"
    collection_name: str = "laws_of_power"
    top_k: int = 5
    temperature: float = 0.0
    temperature_interactive: float = 0.3
    data_dir: str = "data"
    output_dir: str = "output"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("google_api_key", mode="after")
    @classmethod
    def validate_api_key(cls, v):
        """Warn if API key is missing — don't crash at import time,
        but fail loudly when actually needed."""
        if not v or v == "your_api_key_here":
            print(
                "⚠️  WARNING: GOOGLE_API_KEY is not set. "
                "LLM and evaluation features will fail. "
                "Set it in your .env file.",
                file=sys.stderr,
            )
        return v

    # ── Computed properties ──────────────────────────────────────────

    @property
    def llm_models(self) -> list[str]:
        """Parse comma-separated model string into an ordered list."""
        return [m.strip() for m in self.llm_models_str.split(",") if m.strip()]

    @property
    def project_root(self) -> Path:
        """Project root: parent directory of the src/ package."""
        return Path(__file__).resolve().parent.parent

    @property
    def data_path(self) -> Path:
        """Absolute path to the law .txt files."""
        return self.project_root / self.data_dir

    @property
    def output_path(self) -> Path:
        """Absolute path to the enrichment .json files."""
        return self.project_root / self.output_dir

    @property
    def chroma_path(self) -> Path:
        """Absolute path to ChromaDB persistence directory."""
        return self.project_root / self.chroma_dir

    @property
    def primary_llm_model(self) -> str:
        """The first (preferred) model in the fallback chain."""
        return self.llm_models[0]


# ── Singleton instance ──────────────────────────────────────────────
# Import this everywhere: `from src.config import settings`
settings = Settings()
