from fastapi import APIRouter, HTTPException
from app.models.requests import ChatPromptRequest, CompressionRequest
from app.models.responses import (
    OptimizedChatResponse,
    OptimizationDetails,
    CompressionResponse
)
from app.services.cache_service import cache_service
from app.services.prompt_compressor import prompt_compressor
from app.services.model_router import model_router
from app.services.llm_gateway import llm_gateway
from app.services.analytics_service import analytics_service
from app.core.config import settings
from app.core.telemetry import estimate_tokens, calculate_cost, Timer

router = APIRouter(prefix="/optimize", tags=["Optimization Engine"])


@router.post(
    "/chat",
    response_model=OptimizedChatResponse,
    summary="Execute Optimized LLM Completion",
    description=(
        "Processes a prompt through the end-to-end optimization pipeline:\n"
        "1. **Exact & Semantic Vector Cache**: Reuses prior responses for identical/paraphrased queries (0 API tokens, ~1ms).\n"
        "2. **Prompt Compression**: Prunes conversational bloat, whitespace, and filler phrases.\n"
        "3. **Model Cascading**: Automatically routes simple queries to fast Tier 1 SLMs and reserved complex tasks to Tier 2 LLMs.\n"
        "4. **Telemetry & Savings Tracking**: Returns dollar savings, latency saved, and token metrics."
    )
)
async def chat_optimize(request: ChatPromptRequest) -> OptimizedChatResponse:
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    with Timer() as total_timer:
        orig_tokens = estimate_tokens(request.prompt)

        # Step 1: Cache Lookup (unless bypassed)
        if not request.bypass_cache:
            entry, match_type, sim_score = cache_service.lookup(
                request.prompt,
                request.system_prompt
            )
            if entry is not None:
                resp_tokens = estimate_tokens(entry.response)
                baseline_cost = calculate_cost(orig_tokens, resp_tokens, settings.baseline_model)
                actual_cost = 0.0  # Zero cost on cache hit

                analytics_service.record_request(
                    cache_status=match_type,
                    input_tokens=orig_tokens,
                    tokens_saved=orig_tokens,
                    actual_cost=actual_cost,
                    baseline_cost=baseline_cost,
                    latency_ms=total_timer.elapsed_ms
                )

                return OptimizedChatResponse(
                    response=entry.response,
                    optimization=OptimizationDetails(
                        cache_status=match_type,
                        cache_similarity_score=sim_score,
                        original_prompt_tokens=orig_tokens,
                        compressed_prompt_tokens=orig_tokens,
                        tokens_pruned=0,
                        model_tier_used="cached",
                        model_routing_reason=f"Served instantly from {match_type.replace('_', ' ').title()} (score: {sim_score or 1.0})",
                        latency_ms=total_timer.elapsed_ms,
                        actual_cost_usd=actual_cost,
                        baseline_cost_usd=baseline_cost,
                        cost_saved_usd=baseline_cost,
                        cost_savings_percentage=100.0
                    )
                )

        # Step 2: Prompt Compression
        enable_comp = request.enable_compression if request.enable_compression is not None else settings.enable_prompt_compression
        if enable_comp:
            compressed_prompt, _, comp_tokens, tokens_pruned = prompt_compressor.compress(request.prompt)
        else:
            compressed_prompt = request.prompt
            comp_tokens = orig_tokens
            tokens_pruned = 0

        # Step 3: Model Cascading / Complexity Routing
        if request.force_model_tier == "tier-1-fast":
            selected_pricing = settings.tier1_model
            routing_reason = "Manual override: forced Tier 1 Fast SLM"
        elif request.force_model_tier == "tier-2-reasoning":
            selected_pricing = settings.tier2_model
            routing_reason = "Manual override: forced Tier 2 Reasoning LLM"
        else:
            _, _, reasons, selected_pricing = model_router.analyze_complexity(
                compressed_prompt,
                request.system_prompt or ""
            )
            routing_reason = "; ".join(reasons)

        # Step 4: LLM Generation
        response_text, output_tokens, _ = await llm_gateway.call_llm(
            prompt=compressed_prompt,
            system_prompt=request.system_prompt,
            pricing=selected_pricing,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        # Step 5: Store in Cache
        if not request.bypass_cache:
            cache_service.store(
                prompt=request.prompt,
                response=response_text,
                model_tier=selected_pricing.name,
                system_prompt=request.system_prompt
            )

        # Step 6: Cost, Telemetry & Savings
        actual_cost = calculate_cost(comp_tokens, output_tokens, selected_pricing)
        baseline_cost = calculate_cost(orig_tokens, output_tokens, settings.baseline_model)
        cost_saved = max(0.0, round(baseline_cost - actual_cost, 6))
        savings_pct = round((cost_saved / baseline_cost * 100.0), 2) if baseline_cost > 0 else 0.0

        analytics_service.record_request(
            cache_status="miss",
            input_tokens=orig_tokens,
            tokens_saved=tokens_pruned,
            actual_cost=actual_cost,
            baseline_cost=baseline_cost,
            latency_ms=total_timer.elapsed_ms
        )

        return OptimizedChatResponse(
            response=response_text,
            optimization=OptimizationDetails(
                cache_status="miss",
                cache_similarity_score=None,
                original_prompt_tokens=orig_tokens,
                compressed_prompt_tokens=comp_tokens,
                tokens_pruned=tokens_pruned,
                model_tier_used=selected_pricing.name,
                model_routing_reason=routing_reason,
                latency_ms=total_timer.elapsed_ms,
                actual_cost_usd=actual_cost,
                baseline_cost_usd=baseline_cost,
                cost_saved_usd=cost_saved,
                cost_savings_percentage=savings_pct
            )
        )


@router.post(
    "/compress",
    response_model=CompressionResponse,
    summary="Test Prompt Compression",
    description="Analyzes and compresses a prompt, stripping conversational filler, verbose pleasantries, and redundant whitespace."
)
async def compress_prompt_endpoint(request: CompressionRequest) -> CompressionResponse:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    compressed_text, orig_tokens, comp_tokens, tokens_saved = prompt_compressor.compress(
        request.text,
        aggressiveness=request.aggressiveness
    )

    ratio = round((tokens_saved / orig_tokens * 100.0), 2) if orig_tokens > 0 else 0.0

    return CompressionResponse(
        original_text=request.text,
        compressed_text=compressed_text,
        original_tokens=orig_tokens,
        compressed_tokens=comp_tokens,
        tokens_saved=tokens_saved,
        compression_percentage=ratio,
        removed_filler_count=max(0, orig_tokens - comp_tokens)
    )
