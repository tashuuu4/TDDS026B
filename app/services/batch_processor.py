import asyncio
from typing import List, Dict, Any, Optional
from app.services.cache_service import cache_service
from app.services.prompt_compressor import prompt_compressor
from app.services.model_router import model_router
from app.services.llm_gateway import llm_gateway
from app.core.config import settings
from app.core.telemetry import estimate_tokens, calculate_cost, Timer


# In-flight task registry for request coalescing (singleflight)
_in_flight_tasks: Dict[str, asyncio.Future] = {}


async def process_single_optimized_prompt(
    prompt: str,
    system_prompt: Optional[str] = None,
    bypass_cache: bool = False,
    force_model_tier: Optional[str] = None
) -> Dict[str, Any]:
    """Processes a single prompt through the complete optimization pipeline."""
    with Timer() as full_timer:
        orig_tokens = estimate_tokens(prompt)
        dedup_key = cache_service._hash_key(prompt, system_prompt)

        # 1. Cache Check
        if not bypass_cache:
            entry, match_type, sim_score = cache_service.lookup(prompt, system_prompt)
            if entry is not None:
                resp_tokens = estimate_tokens(entry.response)
                base_cost = calculate_cost(orig_tokens, resp_tokens, settings.baseline_model)
                actual_cost = 0.0  # 0 cost for cache hits!

                return {
                    "prompt": prompt,
                    "response": entry.response,
                    "cache_status": match_type,
                    "cache_similarity_score": sim_score,
                    "original_tokens": orig_tokens,
                    "compressed_tokens": orig_tokens,
                    "tokens_pruned": 0,
                    "model_tier_used": "cached",
                    "model_routing_reason": f"Retrieved from {match_type} (similarity: {sim_score or 1.0})",
                    "latency_ms": full_timer.elapsed_ms,
                    "actual_cost_usd": actual_cost,
                    "baseline_cost_usd": base_cost,
                    "cost_saved_usd": round(base_cost - actual_cost, 6),
                    "savings_percentage": 100.0
                }

            # 1b. In-Flight Request Coalescing (Singleflight)
            # If an identical prompt is already executing concurrently, await its completion
            if dedup_key in _in_flight_tasks:
                shared_result = await _in_flight_tasks[dedup_key]
                return {
                    **shared_result,
                    "cache_status": "exact_hit",
                    "model_tier_used": "cached",
                    "model_routing_reason": "Coalesced with concurrent in-flight request",
                    "latency_ms": full_timer.elapsed_ms,
                    "actual_cost_usd": 0.0,
                    "cost_saved_usd": shared_result["baseline_cost_usd"],
                    "savings_percentage": 100.0
                }

        # Register this in-flight request
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        _in_flight_tasks[dedup_key] = future

        try:
            # 2. Prompt Compression
            if settings.enable_prompt_compression:
                compressed_text, _, comp_tokens, pruned_count = prompt_compressor.compress(prompt)
            else:
                compressed_text = prompt
                comp_tokens = orig_tokens
                pruned_count = 0

            # 3. Model Routing
            if force_model_tier == "tier-1-fast":
                selected_pricing = settings.tier1_model
                routing_reason = "Manual override to Tier 1 Fast SLM"
            elif force_model_tier == "tier-2-reasoning":
                selected_pricing = settings.tier2_model
                routing_reason = "Manual override to Tier 2 Reasoning LLM"
            else:
                _, _, reasons, selected_pricing = model_router.analyze_complexity(compressed_text, system_prompt or "")
                routing_reason = "; ".join(reasons)

            # 4. LLM Generation
            response_text, output_tokens, _ = await llm_gateway.call_llm(
                compressed_text,
                system_prompt,
                selected_pricing
            )

            # 5. Store in Cache
            if not bypass_cache:
                cache_service.store(prompt, response_text, selected_pricing.name, system_prompt)

            # 6. Cost and Savings Computation
            actual_cost = calculate_cost(comp_tokens, output_tokens, selected_pricing)
            baseline_cost = calculate_cost(orig_tokens, output_tokens, settings.baseline_model)
            cost_saved = max(0.0, round(baseline_cost - actual_cost, 6))
            savings_pct = round((cost_saved / baseline_cost * 100.0), 2) if baseline_cost > 0 else 0.0

            result = {
                "prompt": prompt,
                "response": response_text,
                "cache_status": "miss",
                "cache_similarity_score": None,
                "original_tokens": orig_tokens,
                "compressed_tokens": comp_tokens,
                "tokens_pruned": pruned_count,
                "model_tier_used": selected_pricing.name,
                "model_routing_reason": routing_reason,
                "latency_ms": full_timer.elapsed_ms,
                "actual_cost_usd": actual_cost,
                "baseline_cost_usd": baseline_cost,
                "cost_saved_usd": cost_saved,
                "savings_percentage": savings_pct
            }

            if not future.done():
                future.set_result(result)
            return result
        except Exception as e:
            if not future.done():
                future.set_exception(e)
            raise
        finally:
            _in_flight_tasks.pop(dedup_key, None)


async def process_batch_prompts(
    prompts: List[str],
    system_prompt: Optional[str] = None,
    max_concurrency: int = 5
) -> List[Dict[str, Any]]:
    """Executes a batch of prompts concurrently using a bounded semaphore pool."""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _worker(p: str):
        async with semaphore:
            return await process_single_optimized_prompt(p, system_prompt)

    tasks = [_worker(p) for p in prompts]
    return await asyncio.gather(*tasks)
