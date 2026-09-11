from fastapi import APIRouter
from typing import List, Dict, Any
from app.models.requests import BatchPromptRequest
from app.services.batch_processor import process_batch_prompts

router = APIRouter(prefix="/batch", tags=["Batch Processing"])


@router.post(
    "/process",
    summary="Process Batch of Prompts Concurrently",
    description="Processes multiple prompts concurrently with bounded worker concurrency, caching identical/paraphrased items and optimizing token costs."
)
async def process_prompt_batch(request: BatchPromptRequest) -> Dict[str, Any]:
    results = await process_batch_prompts(
        prompts=request.prompts,
        system_prompt=request.system_prompt,
        max_concurrency=request.max_concurrency
    )

    total_orig_tokens = sum(r["original_tokens"] for r in results)
    total_comp_tokens = sum(r["compressed_tokens"] for r in results)
    total_cost_saved = sum(r["cost_saved_usd"] for r in results)
    cache_hits = sum(1 for r in results if r["cache_status"] != "miss")

    return {
        "total_prompts": len(results),
        "cache_hits": cache_hits,
        "total_original_tokens": total_orig_tokens,
        "total_compressed_tokens": total_comp_tokens,
        "total_tokens_saved": max(0, total_orig_tokens - total_comp_tokens),
        "total_cost_saved_usd": round(total_cost_saved, 6),
        "results": results
    }
