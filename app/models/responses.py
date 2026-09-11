from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class OptimizationDetails(BaseModel):
    cache_status: Literal["exact_hit", "semantic_hit", "miss"] = Field(
        ...,
        description="Indicates whether the request was served from exact cache, semantic vector cache, or freshly generated."
    )
    cache_similarity_score: Optional[float] = Field(
        default=None,
        description="Cosine similarity score if served via semantic cache."
    )
    original_prompt_tokens: int = Field(
        ...,
        description="Token count of the input prompt before compression."
    )
    compressed_prompt_tokens: int = Field(
        ...,
        description="Token count of the prompt sent to the LLM after compression."
    )
    tokens_pruned: int = Field(
        ...,
        description="Number of prompt tokens removed via compression (0 if served from cache)."
    )
    model_tier_used: str = Field(
        ...,
        description="The model tier executed (e.g. 'tier-1-fast' or 'tier-2-reasoning', or 'cached')."
    )
    model_routing_reason: str = Field(
        ...,
        description="Explanation of why this model tier was selected."
    )
    latency_ms: float = Field(
        ...,
        description="Total round-trip latency in milliseconds."
    )
    actual_cost_usd: float = Field(
        ...,
        description="Actual cost incurred for this request in USD."
    )
    baseline_cost_usd: float = Field(
        ...,
        description="Hypothetical cost if sent naively to uncompressed heavy model."
    )
    cost_saved_usd: float = Field(
        ...,
        description="Net dollar savings achieved by optimization."
    )
    cost_savings_percentage: float = Field(
        ...,
        description="Percentage of cost reduced compared to naive baseline."
    )


class OptimizedChatResponse(BaseModel):
    response: str = Field(
        ...,
        description="The generated or retrieved LLM response."
    )
    optimization: OptimizationDetails = Field(
        ...,
        description="Detailed breakdown of efficiency gains, cache status, and cost savings."
    )


class CompressionResponse(BaseModel):
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    compression_percentage: float
    removed_filler_count: int


class RoutingClassificationResponse(BaseModel):
    prompt: str
    complexity_score: float = Field(
        ...,
        description="Normalized complexity score from 0.0 (simplest) to 1.0 (most complex)."
    )
    complexity_label: Literal["simple", "moderate", "complex"]
    recommended_tier: str
    model_name: str
    estimated_input_cost_per_1m: float
    reasons: List[str]


class CacheLookupResponse(BaseModel):
    found: bool
    match_type: Optional[str] = None
    similarity_score: Optional[float] = None
    cached_prompt: Optional[str] = None
    cached_response: Optional[str] = None


class CacheStatsResponse(BaseModel):
    exact_cache_entries: int
    semantic_cache_entries: int
    exact_hits: int
    semantic_hits: int
    misses: int
    total_lookups: int
    hit_ratio_percentage: float


class BenchmarkItemResult(BaseModel):
    prompt: str
    category: str
    cache_status: str
    model_tier: str
    latency_ms: float
    cost_usd: float
    tokens_used: int


class BenchmarkComparisonResponse(BaseModel):
    total_prompts: int
    exact_cache_hits: int
    semantic_cache_hits: int
    cache_hit_ratio_percentage: float
    tier1_fast_routed: int
    tier2_reasoning_routed: int
    baseline_total_latency_ms: float
    optimized_total_latency_ms: float
    latency_reduction_percentage: float
    baseline_total_tokens: int
    optimized_total_tokens: int
    tokens_saved: int
    token_reduction_percentage: float
    baseline_total_cost_usd: float
    optimized_total_cost_usd: float
    cost_saved_usd: float
    cost_reduction_percentage: float
    details: List[BenchmarkItemResult]


class AnalyticsOverviewResponse(BaseModel):
    total_requests: int
    total_cache_hits: int
    exact_cache_hits: int
    semantic_cache_hits: int
    cache_hit_rate_percentage: float
    total_tokens_processed: int
    total_tokens_saved: int
    total_actual_cost_usd: float
    total_baseline_cost_usd: float
    total_cost_saved_usd: float
    overall_cost_reduction_percentage: float
    average_latency_ms: float
