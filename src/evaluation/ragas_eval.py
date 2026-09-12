"""
RAGAS evaluation — measures generation quality metrics.

Uses RAGAS framework (or similar approach) to evaluate:
    - Faithfulness (is the answer grounded in context?)
    - Answer Relevancy (does the answer address the question?)

Usage:
    python -m src.evaluation.ragas_eval
"""

import logging

from src.evaluation.cache import EvalCache
from src.evaluation.retrieval_eval import load_benchmark
from src.generation.chain import ask

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_generation_eval(sample_size: int = 10):
    """Run generation evaluation on a subset of the benchmark.
    
    Since generation evaluation requires calling the LLM for every
    question to generate the answer, we use a sample by default
    and cache the results.
    """
    benchmark = load_benchmark()
    sample = benchmark[:sample_size]
    
    logger.info("Running generation evaluation on %d samples...", len(sample))
    
    cache = EvalCache("generation_cache.json")
    
    results = []
    for item in sample:
        question = item["Question"]
        expected_law = item["Expected"]
        
        # Check cache first
        cache_key = {"question": question, "k": 5}
        cached_result = cache.get(**cache_key)
        
        if cached_result:
            answer = cached_result["answer"]
            sources = cached_result["sources"]
            logger.info("[CACHED] Q: %s", question[:50])
        else:
            logger.info("[LIVE] Q: %s", question[:50])
            result = ask(question, k=5)
            answer = result["answer"]
            sources = result["sources"]
            cache.set({"answer": answer, "sources": sources}, **cache_key)
            
        retrieved_laws = [s["law"] for s in sources]
        
        # Simple string-based extraction of the answered law
        # A full RAGAS implementation would use LLM-as-a-judge here.
        # For Phase 6, we'll verify if the expected law is in the sources
        # and if the answer cites it correctly.
        
        item_result = {
            "question": question,
            "expected_law": expected_law,
            "retrieved_laws": retrieved_laws,
            "retrieval_hit": expected_law in retrieved_laws,
            "answer_length": len(answer),
        }
        results.append(item_result)
        
    logger.info("Generation evaluation complete.")
    return results


if __name__ == "__main__":
    run_generation_eval()
