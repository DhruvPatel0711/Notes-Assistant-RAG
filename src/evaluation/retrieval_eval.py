"""
Retrieval evaluation — measures Retrieval@k accuracy.

Runs the 60-question benchmark and calculates per-question
retrieval hit/miss, plus aggregate accuracy at k=1,3,5,10.

Usage:
    python -m src.evaluation.retrieval_eval
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

from src.config import settings
from src.retrieval.retriever import get_retriever

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_benchmark() -> list[dict]:
    """Load the 60-question benchmark dataset."""
    benchmark_path = settings.data_path / "benchmark_60.json"
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")
    
    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Convert dict of dicts to list of dicts
    return list(data.values())


def evaluate_retrieval(k_values: list[int] = [1, 3, 5, 10]) -> dict:
    """Evaluate retrieval accuracy at multiple k values."""
    benchmark = load_benchmark()
    logger.info("Loaded %d benchmark questions.", len(benchmark))
    
    results = {k: {"hits": 0, "total": len(benchmark)} for k in k_values}
    
    # Run retrieval at max k, then we can slice for smaller k values
    max_k = max(k_values)
    retriever = get_retriever(k=max_k)
    
    logger.info("Running retrieval evaluation (max_k=%d)...", max_k)
    
    for item in benchmark:
        question = item["Question"]
        expected_law = item["Expected"]
        
        # Retrieve top max_k documents
        docs = retriever.invoke(question)
        retrieved_laws = [doc.metadata.get("law") for doc in docs]
        
        # Calculate hits for each k
        for k in k_values:
            k_laws = retrieved_laws[:k]
            if expected_law in k_laws:
                results[k]["hits"] += 1
                
    # Calculate percentages
    for k in k_values:
        hits = results[k]["hits"]
        total = results[k]["total"]
        accuracy = (hits / total) * 100
        results[k]["accuracy"] = accuracy
        logger.info("Retrieval@%02d: %.1f%% (%d/%d)", k, accuracy, hits, total)
        
    return results


if __name__ == "__main__":
    evaluate_retrieval()
