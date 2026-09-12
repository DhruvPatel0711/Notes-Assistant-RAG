"""
FastAPI backend — serves the RAG Assistant via REST API.

Endpoints:
    - GET /health: Basic health check
    - POST /ask: Query the RAG system

Includes a simple in-memory rate limiter to protect the LLM API
from excessive requests if deployed publicly.

Run the server with:
    uvicorn src.api.main:app --reload
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from src.config import settings
from src.generation.chain import ask, ask_situation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Rate Limiter
# ═══════════════════════════════════════════════════════════════════


class RateLimiter:
    """Simple in-memory rate limiter using a sliding window.
    
    In a real production environment, this would be backed by Redis.
    """
    
    def __init__(self, requests_per_minute: int = 10):
        self.limit = requests_per_minute
        # IP -> list of timestamps
        self.requests: Dict[str, List[float]] = defaultdict(list)
        
    def check_rate_limit(self, client_ip: str) -> bool:
        """Return True if request is allowed, False if rate limited."""
        now = time.time()
        window_start = now - 60.0
        
        # Clean up old requests
        self.requests[client_ip] = [t for t in self.requests[client_ip] if t > window_start]
        
        if len(self.requests[client_ip]) >= self.limit:
            return False
            
        self.requests[client_ip].append(now)
        return True


rate_limiter = RateLimiter(requests_per_minute=20)


# ═══════════════════════════════════════════════════════════════════
# FastAPI App setup
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Laws of Power RAG API",
    description="Evaluated RAG Assistant over the 48 Laws of Power notes.",
    version="1.0.0",
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Middleware to enforce rate limits."""
    # Skip rate limiting for health check
    if request.url.path == "/health":
        return await call_next(request)
        
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.check_rate_limit(client_ip):
        return Response(content="Rate limit exceeded", status_code=429)
        
    response = await call_next(request)
    return response


# ═══════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════


class QueryRequest(BaseModel):
    """Request payload for the /ask endpoint."""
    query: str
    mode: str = "standard"  # "standard" or "situation"
    k: Optional[int] = None


class SourceMetadata(BaseModel):
    law: int
    title: str
    source: str


class QueryResponse(BaseModel):
    """Response payload for the /ask endpoint."""
    answer: str
    sources: List[SourceMetadata]


# ═══════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Query the RAG Assistant."""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")
            
        logger.info("API Request: query='%s', mode='%s', k=%s", 
                    request.query[:50], request.mode, request.k)
        
        if request.mode == "situation":
            result = ask_situation(request.query, k=request.k)
        else:
            result = ask(request.query, k=request.k)
            
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        logger.error("Error processing query: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error processing query.")
