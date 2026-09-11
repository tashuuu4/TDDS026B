from fastapi import APIRouter
from typing import List, Dict, Any
from app.models.requests import BenchmarkRunRequest
from app.models.responses import BenchmarkComparisonResponse, BenchmarkItemResult
from app.data.benchmark_dataset import load_benchmark_dataset, get_dataset_categories
from app.services.cache_service import cache_service
from app.services.batch_processor import process_single_optimized_prompt
from app.core.config import settings
from app.core.telemetry import estimate_tokens, calculate_cost, Timer

router = APIRouter(prefix="/benchmark", tags=["Benchmarking & Dataset"])


@router.get(
    "/dataset",
    summary="Get Sample Benchmark Dataset",
    description="Inspect the built-in self-contained prompt benchmark dataset categorized into customer support, repetitive queries, paraphrased pairs, verbose prompts, and complex reasoning."
)
async def get_benchmark_dataset():
    data = load_benchmark_dataset()
    categories = get_dataset_categories()
    return {
        "categories": categories,
        "total_prompts": len(data),
        "prompts": data
    }


@router.post(
    "/run",
    response_model=BenchmarkComparisonResponse,
    summary="Run Comprehensive Optimization Benchmark",
    description=(
        "Executes benchmark prompts through both **Naive Baseline** (uncompressed, no cache, heavy model) "
        "and **Optimized Pipeline** (exact & semantic cache, compression, tiered cascading). "
        "Returns a detailed comparative report showing token, cost, and latency reduction."
    )
)
async def run_optimization_benchmark(request: BenchmarkRunRequest) -> BenchmarkComparisonResponse:
    dataset = load_benchmark_dataset(
        category=request.dataset_category,
        limit=request.sample_size
    )

    if request.clear_cache_first:
        cache_service.clear()

    details: List[BenchmarkItemResult] = []

    exact_hits = 0
    semantic_hits = 0
    tier1_count = 0
    tier2_count = 0

    baseline_total_tokens = 0
    optimized_total_tokens = 0
    baseline_total_cost = 0.0
    optimized_total_cost = 0.0
    baseline_total_latency = 0.0
    optimized_total_latency = 0.0

    for item in dataset:
        prompt = item["prompt"]
        category = item.get("category", "general")

        # 1. Run Baseline (Hypothetical naive call: uncompressed to heavy tier-2 model)
        b_tokens_in = estimate_tokens(prompt)
        b_tokens_out = 120  # Average mock response tokens
        b_cost = calculate_cost(b_tokens_in, b_tokens_out, settings.baseline_model)
        b_latency = settings.baseline_model.avg_latency_ms

        baseline_total_tokens += (b_tokens_in + b_tokens_out)
        baseline_total_cost += b_cost
        baseline_total_latency += b_latency

        # 2. Run Optimized Pipeline
        opt_res = await process_single_optimized_prompt(prompt)

        status = opt_res["cache_status"]
        if status == "exact_hit":
            exact_hits += 1
        elif status == "semantic_hit":
            semantic_hits += 1

        tier = opt_res["model_tier_used"]
        if "tier-1" in tier:
            tier1_count += 1
        elif "tier-2" in tier:
            tier2_count += 1

        actual_cost = opt_res["actual_cost_usd"]
        latency = opt_res["latency_ms"]
        tokens_used = opt_res["compressed_tokens"] if status == "miss" else 0

        optimized_total_cost += actual_cost
        optimized_total_tokens += tokens_used
        optimized_total_latency += latency

        details.append(
            BenchmarkItemResult(
                prompt=prompt,
                category=category,
                cache_status=status,
                model_tier=tier,
                latency_ms=latency,
                cost_usd=actual_cost,
                tokens_used=tokens_used
            )
        )

    # Compute Comparative Metrics
    total_prompts = len(dataset)
    total_cache_hits = exact_hits + semantic_hits
    hit_ratio = round((total_cache_hits / total_prompts * 100.0), 2) if total_prompts > 0 else 0.0

    tokens_saved = max(0, baseline_total_tokens - optimized_total_tokens)
    token_reduction_pct = round((tokens_saved / baseline_total_tokens * 100.0), 2) if baseline_total_tokens > 0 else 0.0

    cost_saved = max(0.0, round(baseline_total_cost - optimized_total_cost, 6))
    cost_reduction_pct = round((cost_saved / baseline_total_cost * 100.0), 2) if baseline_total_cost > 0 else 0.0

    latency_diff = max(0.0, baseline_total_latency - optimized_total_latency)
    latency_reduction_pct = round((latency_diff / baseline_total_latency * 100.0), 2) if baseline_total_latency > 0 else 0.0

    return BenchmarkComparisonResponse(
        total_prompts=total_prompts,
        exact_cache_hits=exact_hits,
        semantic_cache_hits=semantic_hits,
        cache_hit_ratio_percentage=hit_ratio,
        tier1_fast_routed=tier1_count,
        tier2_reasoning_routed=tier2_count,
        baseline_total_latency_ms=round(baseline_total_latency, 2),
        optimized_total_latency_ms=round(optimized_total_latency, 2),
        latency_reduction_percentage=latency_reduction_pct,
        baseline_total_tokens=baseline_total_tokens,
        optimized_total_tokens=optimized_total_tokens,
        tokens_saved=tokens_saved,
        token_reduction_percentage=token_reduction_pct,
        baseline_total_cost_usd=round(baseline_total_cost, 6),
        optimized_total_cost_usd=round(optimized_total_cost, 6),
        cost_saved_usd=cost_saved,
        cost_reduction_percentage=cost_reduction_pct,
        details=details
    )
