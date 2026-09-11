from fastapi import APIRouter
from app.models.responses import AnalyticsOverviewResponse
from app.services.analytics_service import analytics_service

router = APIRouter(prefix="/analytics", tags=["Analytics & Telemetry"])


@router.get(
    "/overview",
    response_model=AnalyticsOverviewResponse,
    summary="Get System Analytics & Efficiency Metrics",
    description="Returns aggregate metrics on total requests, cache hit rate, token savings, dollar savings, and latency improvements."
)
async def get_analytics_overview() -> AnalyticsOverviewResponse:
    summary = analytics_service.get_summary()
    return AnalyticsOverviewResponse(**summary)


@router.post(
    "/reset",
    summary="Reset Analytics Telemetry",
    description="Resets all recorded telemetry metrics to zero."
)
async def reset_analytics():
    analytics_service.reset()
    return {"message": "Analytics telemetry reset successfully."}
