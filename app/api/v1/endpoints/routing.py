from fastapi import APIRouter
from app.models.requests import ModelClassificationRequest
from app.models.responses import RoutingClassificationResponse
from app.services.model_router import model_router

router = APIRouter(prefix="/routing", tags=["Model Cascading & Routing"])


@router.post(
    "/classify",
    response_model=RoutingClassificationResponse,
    summary="Classify Prompt Complexity & Select Model Tier",
    description=(
        "Evaluates the prompt's structural and semantic complexity (identifying algorithms, mathematical proofs, "
        "concurrency, or distributed systems vs standard FAQ/factual lookup). Recommends either Tier 1 Fast SLM or Tier 2 Reasoning LLM."
    )
)
async def classify_prompt_routing(request: ModelClassificationRequest) -> RoutingClassificationResponse:
    score, label, reasons, pricing = model_router.analyze_complexity(
        prompt=request.prompt,
        system_prompt=request.system_prompt or ""
    )

    return RoutingClassificationResponse(
        prompt=request.prompt,
        complexity_score=score,
        complexity_label=label,
        recommended_tier=pricing.name,
        model_name=pricing.name,
        estimated_input_cost_per_1m=pricing.input_cost_per_1m,
        reasons=reasons
    )
