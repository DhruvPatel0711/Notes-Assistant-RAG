"""
Evaluation cache — avoids redundant LLM calls during evaluation.

Stores LLM responses based on a hash of the input parameters:
    - Question
    - Context (retrieved docs)
    - Prompt template version
    - LLM model name

This ensures we don't burn API tokens or rate limits when re-running
evaluations if the pipeline hasn't changed.
"""

import hashlib
import json
import logging
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)


class EvalCache:
    """Disk-backed cache for evaluation LLM calls."""
    
    def __init__(self, cache_name: str = "eval_cache.json"):
        self.cache_path = settings.output_path / cache_name
        self.cache = self._load()
        
    def _load(self) -> dict:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Cache file corrupted. Starting fresh.")
        return {}
        
    def _save(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)
            
    def _hash(self, kwargs: dict) -> str:
        """Create a deterministic hash of the input parameters."""
        serialized = json.dumps(kwargs, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
        
    def get(self, **kwargs) -> str | None:
        """Get cached response if it exists."""
        key = self._hash(kwargs)
        return self.cache.get(key)
        
    def set(self, response: str, **kwargs):
        """Save response to cache."""
        key = self._hash(kwargs)
        self.cache[key] = response
        self._save()
