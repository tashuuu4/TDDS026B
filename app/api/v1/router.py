from fastapi import APIRouter
from app.api.v1.endpoints.optimize import router as optimize_router
from app.api.v1.endpoints.cache import router as cache_router
from app.api.v1.endpoints.routing import router as routing_router
from app.api.v1.endpoints.batch import router as batch_router
from app.api.v1.endpoints.benchmark import router as benchmark_router
from app.api.v1.endpoints.analytics import router as analytics_router

api_v1_router = APIRouter(prefix="/v1")

api_v1_router.include_router(optimize_router)
api_v1_router.include_router(cache_router)
api_v1_router.include_router(routing_router)
api_v1_router.include_router(batch_router)
api_v1_router.include_router(benchmark_router)
api_v1_router.include_router(analytics_router)
