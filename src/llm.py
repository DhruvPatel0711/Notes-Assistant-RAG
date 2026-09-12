"""
LLM factory with automatic model fallback chain.

Creates LangChain ChatGoogleGenerativeAI instances with a cascading
fallback: when one Gemini model hits a rate limit (429/RESOURCE_EXHAUSTED),
the next model in the chain is tried automatically.

Fallback order (configurable via .env LLM_MODELS):
    gemini-3.8-flash → 3.7-flash → 3.6-flash → 3.5-flash → 3.5-flash-lite

Usage:
    from src.llm import get_llm, get_llm_with_fallbacks

    # Single model (no fallback)
    llm = get_llm("gemini-3.5-flash", temperature=0)

    # Full fallback chain (recommended for production)
    llm = get_llm_with_fallbacks()
    result = llm.invoke("Hello")
"""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings

logger = logging.getLogger(__name__)


def get_llm(
    model_name: str | None = None,
    temperature: float | None = None,
    max_retries: int = 3,
) -> ChatGoogleGenerativeAI:
    """Create a single ChatGoogleGenerativeAI instance.

    Args:
        model_name: Gemini model name. Defaults to the first model in the
                     fallback chain (settings.primary_llm_model).
        temperature: Sampling temperature. Defaults to settings.temperature.
        max_retries: Max retries for transient errors within this single model.

    Returns:
        A ChatGoogleGenerativeAI instance ready for use in LCEL chains.
    """
    model = model_name or settings.primary_llm_model
    temp = temperature if temperature is not None else settings.temperature

    logger.info("Creating LLM instance: model=%s, temperature=%s", model, temp)

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temp,
        max_retries=max_retries,
        google_api_key=settings.google_api_key,
    )


def get_llm_with_fallbacks(
    temperature: float | None = None,
    max_retries: int = 2,
):
    """Create an LLM with automatic model fallback chain.

    When the primary model hits a rate limit or transient error and
    exhausts its retries, LangChain's .with_fallbacks() automatically
    tries the next model in the chain.

    The chain order is defined by settings.llm_models:
        gemini-3.8-flash → 3.7 → 3.6 → 3.5 → 3.5-lite

    Args:
        temperature: Sampling temperature. Defaults to settings.temperature.
        max_retries: Retries per model before cascading to the next one.

    Returns:
        A Runnable (primary LLM with fallbacks attached) that can be used
        anywhere a regular LLM is used in LCEL chains.
    """
    models = settings.llm_models
    temp = temperature if temperature is not None else settings.temperature

    if not models:
        raise ValueError("No LLM models configured. Check LLM_MODELS in .env")

    # Create the primary LLM
    primary = get_llm(models[0], temp, max_retries)
    logger.info("Primary LLM: %s", models[0])

    # Create fallback LLMs
    if len(models) > 1:
        fallbacks = []
        for model_name in models[1:]:
            fallbacks.append(get_llm(model_name, temp, max_retries))
            logger.info("Fallback LLM registered: %s", model_name)

        # LangChain's with_fallbacks catches exceptions from the primary
        # and tries each fallback in order. This handles 429, 503,
        # RESOURCE_EXHAUSTED, and any other transient API errors.
        return primary.with_fallbacks(fallbacks)

    return primary
