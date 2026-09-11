import pytest
from app.services.model_router import ModelRouter


def test_simple_query_routing():
    simple_prompt = "What is the capital city of Japan?"
    score, label, reasons, pricing = ModelRouter.analyze_complexity(simple_prompt)

    assert label == "simple"
    assert score < 0.35
    assert "tier-1" in pricing.name


def test_complex_distributed_systems_routing():
    complex_prompt = (
        "Design a distributed message broker with write-ahead logging, "
        "Raft consensus for leader election, and two-phase locking for concurrency."
    )
    score, label, reasons, pricing = ModelRouter.analyze_complexity(complex_prompt)

    assert label == "complex"
    assert score >= 0.50
    assert "tier-2" in pricing.name
    assert any("domain concepts" in r for r in reasons)
