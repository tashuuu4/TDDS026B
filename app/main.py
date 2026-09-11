from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from app.core.config import settings
from app.api.v1.router import api_v1_router

tags_metadata = [
    {
        "name": "Optimization Engine",
        "description": "Core endpoints to optimize LLM calls via semantic caching, token compression, and model cascading.",
    },
    {
        "name": "Cache Management",
        "description": "Inspect and manage exact hash and semantic vector cache layers.",
    },
    {
        "name": "Model Cascading & Routing",
        "description": "Complexity classification heuristics and tiered model selection.",
    },
    {
        "name": "Batch Processing",
        "description": "Concurrent execution with rate-limit and semaphore management.",
    },
    {
        "name": "Benchmarking & Dataset",
        "description": "Run comparative simulations on built-in multi-domain datasets.",
    },
    {
        "name": "Analytics & Telemetry",
        "description": "System-wide tracking of token savings, latency reductions, and cost ($) avoided.",
    },
]

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
# 🚀 LLM API Call Optimizer Backend

A high-performance optimization gateway designed for efficient Generative AI workloads.

### Key Capabilities:
- **Multi-Tier Semantic Caching**: Sub-millisecond exact hash matching and cosine-similarity vector caching.
- **Prompt Token Pruning & Compression**: Eliminates whitespace bloat, conversational filler, and redundant boilerplate.
- **Smart Model Cascading**: Routes simple/factual queries to high-throughput lightweight SLMs (Tier 1) and heavy algorithmic problems to frontier LLMs (Tier 2).
- **Batch Processing**: Concurrency-controlled micro-batching.
- **Automated Benchmarking**: Compare unoptimized naive baseline vs optimized pipeline on built-in multi-domain prompts.
- **Telemetry & Cost Accounting**: Tracks real-time token reduction, latency improvements, and financial dollar savings ($).
    """,
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Enable CORS for flexible integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API V1
app.include_router(api_v1_router)


@app.get("/", include_in_schema=False)
async def root():
    """Redirects root to Swagger UI documentation."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    tags=["System"],
    summary="Service Health Check",
    description="Returns service status and active configuration."
)
async def health_check():
    return {
        "status": "healthy",
        "version": settings.app_version,
        "provider": settings.default_llm_provider,
        "exact_cache_enabled": settings.cache_exact_enabled,
        "semantic_cache_enabled": settings.cache_semantic_enabled,
        "prompt_compression_enabled": settings.enable_prompt_compression,
        "model_cascading_enabled": settings.enable_model_cascading
    }
