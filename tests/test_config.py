"""
Tests for src/config.py — validates configuration loading and path resolution.

These tests do NOT require a live API key or any external services.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Helpers ─────────────────────────────────────────────────────────

def _get_project_root() -> Path:
    """Returns the project root (parent of tests/)."""
    return Path(__file__).resolve().parent.parent


# ── Test: Settings load with defaults ───────────────────────────────

def test_settings_loads_with_defaults():
    """Config should load without crashing, even with a placeholder API key."""
    from src.config import settings

    assert settings is not None
    assert isinstance(settings.top_k, int)
    assert isinstance(settings.temperature, float)
    assert isinstance(settings.llm_models, list)


def test_default_embedding_model():
    """Default embedding model should be BGE-large."""
    from src.config import settings

    assert settings.embedding_model == "BAAI/bge-large-en-v1.5"


def test_default_llm_models_is_list():
    """LLM models should be a list of strings, not a raw comma-separated string."""
    from src.config import settings

    assert isinstance(settings.llm_models, list)
    assert len(settings.llm_models) >= 1
    for model in settings.llm_models:
        assert isinstance(model, str)
        assert "gemini" in model


def test_default_top_k():
    """Default TOP_K should be 5."""
    from src.config import settings

    assert settings.top_k == 5


def test_default_temperature():
    """Default temperature should be 0 (deterministic for evaluation)."""
    from src.config import settings

    assert settings.temperature == 0.0


def test_interactive_temperature():
    """Interactive temperature should be > 0."""
    from src.config import settings

    assert settings.temperature_interactive > 0.0


# ── Test: Path resolution ───────────────────────────────────────────

def test_project_root_resolution():
    """project_root should point to the actual project directory."""
    from src.config import settings

    root = settings.project_root
    assert root.exists()
    assert (root / "src").is_dir()
    assert (root / "src" / "config.py").is_file()


def test_data_path_resolution():
    """data_path should resolve relative to project root."""
    from src.config import settings

    assert settings.data_path == settings.project_root / "data"


def test_output_path_resolution():
    """output_path should resolve relative to project root."""
    from src.config import settings

    assert settings.output_path == settings.project_root / "output"


def test_chroma_path_resolution():
    """chroma_path should resolve relative to project root."""
    from src.config import settings

    assert settings.chroma_path == settings.project_root / "chroma_db"


# ── Test: Primary model accessor ───────────────────────────────────

def test_primary_llm_model():
    """primary_llm_model should return the first model in the chain."""
    from src.config import settings

    assert settings.primary_llm_model == settings.llm_models[0]


# ── Test: LLM_MODELS parsing from comma-separated string ──────────

def test_llm_models_parsed_from_string():
    """When LLM_MODELS env var is a comma-separated string, it should
    be parsed into a list."""
    from src.config import Settings

    with patch.dict(os.environ, {
        "GOOGLE_API_KEY": "test-key",
        "LLM_MODELS_STR": "model-a,model-b,model-c",
    }):
        s = Settings()
        assert s.llm_models == ["model-a", "model-b", "model-c"]


def test_llm_models_handles_whitespace():
    """Comma-separated model string with spaces should be trimmed."""
    from src.config import Settings

    with patch.dict(os.environ, {
        "GOOGLE_API_KEY": "test-key",
        "LLM_MODELS_STR": " model-a , model-b , model-c ",
    }):
        s = Settings()
        assert s.llm_models == ["model-a", "model-b", "model-c"]


# ── Test: Data files exist ─────────────────────────────────────────

def test_data_files_exist():
    """All 28 law .txt files should be present in data/."""
    from src.config import settings

    if not settings.data_path.exists():
        pytest.skip("Data files not yet copied")

    txt_files = sorted(settings.data_path.glob("*.txt"))
    assert len(txt_files) == 28, f"Expected 28 .txt files, found {len(txt_files)}"


def test_output_files_exist():
    """All 28 enrichment .json files should be present in output/."""
    from src.config import settings

    if not settings.output_path.exists():
        pytest.skip("Output files not yet copied")

    json_files = sorted([f for f in settings.output_path.glob("*.json") if not f.name.endswith("cache.json")])
    assert len(json_files) == 28, f"Expected 28 .json files, found {len(json_files)}"
