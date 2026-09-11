from fastapi import APIRouter
from app.models.requests import CacheLookupRequest
from app.models.responses import CacheStatsResponse, CacheLookupResponse
from app.services.cache_service import cache_service

router = APIRouter(prefix="/cache", tags=["Cache Management"])


@router.get(
    "/stats",
    response_model=CacheStatsResponse,
    summary="Get Cache Performance Statistics",
    description="Returns metrics on exact and semantic cache size, hits, misses, and aggregate hit ratio."
)
async def get_cache_statistics() -> CacheStatsResponse:
    stats = cache_service.get_stats()
    return CacheStatsResponse(**stats)


@router.post(
    "/lookup",
    response_model=CacheLookupResponse,
    summary="Lookup Prompt in Exact & Semantic Cache",
    description="Tests if a prompt exists in cache without triggering generation or mutating state."
)
async def test_cache_lookup(request: CacheLookupRequest) -> CacheLookupResponse:
    entry, match_type, score = cache_service.lookup(
        prompt=request.prompt,
        threshold=request.threshold
    )
    if entry is not None:
        return CacheLookupResponse(
            found=True,
            match_type=match_type,
            similarity_score=score,
            cached_prompt=entry.prompt,
            cached_response=entry.response
        )
    return CacheLookupResponse(
        found=False,
        match_type="miss",
        similarity_score=None,
        cached_prompt=None,
        cached_response=None
    )


@router.delete(
    "/clear",
    summary="Flush Cache",
    description="Clears all exact hash and semantic vector cache entries and resets hit counters."
)
async def clear_cache_endpoint():
    cache_service.clear()
    return {"message": "Cache successfully cleared."}
