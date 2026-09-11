import os
from typing import Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env file if available
load_dotenv()


class ModelTierPricing(BaseModel):
    name: str
    description: str
    input_cost_per_1m: float  # In USD
    output_cost_per_1m: float  # In USD
    avg_latency_ms: float  # In milliseconds (simulated/baseline)


class Settings(BaseModel):
    # App Settings
    app_name: str = "LLM API Call Optimizer"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # LLM Settings
    default_llm_provider: str = os.getenv("DEFAULT_LLM_PROVIDER", "mock")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    mock_simulate_latency: bool = os.getenv("MOCK_SIMULATE_LATENCY", "true").lower() == "true"

    # Pricing Models (Industry standard reference rates per 1M tokens)
    # Tier 1: Fast & lightweight model (e.g., GPT-4o-mini, Gemini 1.5 Flash)
    tier1_model: ModelTierPricing = ModelTierPricing(
        name="tier-1-fast",
        description="Lightweight SLM for high-throughput, low-latency, and simpler queries",
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
        avg_latency_ms=180.0
    )

    # Tier 2: Advanced reasoning model (e.g., GPT-4o, Claude 3.5 Sonnet, Gemini Pro)
    tier2_model: ModelTierPricing = ModelTierPricing(
        name="tier-2-reasoning",
        description="Frontier LLM for complex logic, multi-step math, and architectural code reasoning",
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00,
        avg_latency_ms=1200.0
    )

    # Baseline Model (Used to calculate savings when client would have naively sent everything to heavy model)
    baseline_model: ModelTierPricing = ModelTierPricing(
        name="baseline-unoptimized",
        description="Naive direct call to heavy frontier LLM without caching or compression",
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00,
        avg_latency_ms=1200.0
    )

    # Cache Configuration
    cache_exact_enabled: bool = os.getenv("CACHE_EXACT_ENABLED", "true").lower() == "true"
    cache_semantic_enabled: bool = os.getenv("CACHE_SEMANTIC_ENABLED", "true").lower() == "true"
    cache_semantic_similarity_threshold: float = float(os.getenv("CACHE_SEMANTIC_SIMILARITY_THRESHOLD", "0.70"))
    cache_max_entries: int = int(os.getenv("CACHE_MAX_ENTRIES", "1000"))

    # Optimization Configuration
    enable_prompt_compression: bool = os.getenv("ENABLE_PROMPT_COMPRESSION", "true").lower() == "true"
    enable_model_cascading: bool = os.getenv("ENABLE_MODEL_CASCADING", "true").lower() == "true"
    compression_max_length: int = int(os.getenv("COMPRESSION_MAX_LENGTH", "4000"))


settings = Settings()
