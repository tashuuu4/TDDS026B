import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.cache_service import cache_service

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_teardown():
    cache_service.clear()
    yield
    cache_service.clear()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_optimize_chat_flow_and_caching():
    prompt = "What is the speed of light in a vacuum?"

    # 1. First call -> Cache Miss
    resp1 = client.post("/v1/optimize/chat", json={"prompt": prompt})
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["optimization"]["cache_status"] == "miss"
    assert data1["optimization"]["actual_cost_usd"] > 0
    assert len(data1["response"]) > 0

    # 2. Second identical call -> Exact Cache Hit
    resp2 = client.post("/v1/optimize/chat", json={"prompt": prompt})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["optimization"]["cache_status"] == "exact_hit"
    assert data2["optimization"]["actual_cost_usd"] == 0.0
    assert data2["optimization"]["cost_savings_percentage"] == 100.0


def test_compress_endpoint():
    payload = {
        "text": "Hello there! Could you please be so kind as to explain Python? Thank you so much!",
        "aggressiveness": "moderate"
    }
    response = client.post("/v1/optimize/compress", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["tokens_saved"] > 0
    assert "Python" in data["compressed_text"]


def test_routing_classify_endpoint():
    payload = {
        "prompt": "Design an eventual consistency distributed database with Raft consensus and vector clocks."
    }
    response = client.post("/v1/routing/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["complexity_label"] == "complex"
    assert "tier-2" in data["recommended_tier"]


def test_cache_stats_and_clear():
    # Insert entry
    client.post("/v1/optimize/chat", json={"prompt": "What is Python?"})

    stats_resp = client.get("/v1/cache/stats")
    assert stats_resp.status_code == 200
    assert stats_resp.json()["exact_cache_entries"] >= 1

    clear_resp = client.delete("/v1/cache/clear")
    assert clear_resp.status_code == 200

    stats_resp2 = client.get("/v1/cache/stats")
    assert stats_resp2.json()["exact_cache_entries"] == 0


def test_semantic_cache_api_flow():
    prompt1 = "How do I reset my account password if I forgot my email?"
    prompt2 = "How do I recover my password when I no longer have access to my email?"

    # 1. Send first prompt -> cache miss
    resp1 = client.post("/v1/optimize/chat", json={"prompt": prompt1})
    assert resp1.status_code == 200
    assert resp1.json()["optimization"]["cache_status"] == "miss"

    # 2. Send paraphrased prompt -> semantic cache hit
    resp2 = client.post("/v1/optimize/chat", json={"prompt": prompt2})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["optimization"]["cache_status"] == "semantic_hit"
    assert data2["optimization"]["cache_similarity_score"] is not None
    assert data2["optimization"]["actual_cost_usd"] == 0.0
    assert data2["optimization"]["cost_savings_percentage"] == 100.0


def test_batch_processing_endpoint():
    payload = {
        "prompts": [
            "What is Python?",
            "What is Python?",
            "How do I reset my account password if I forgot my email?"
        ],
        "max_concurrency": 2
    }
    response = client.post("/v1/batch/process", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_prompts"] == 3
    assert data["cache_hits"] >= 1
    assert "total_cost_saved_usd" in data


def test_benchmark_run():
    payload = {
        "dataset_category": "repetitive_queries",
        "sample_size": 4,
        "clear_cache_first": True
    }
    response = client.post("/v1/benchmark/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_prompts"] > 0
    assert "cost_saved_usd" in data
    assert "token_reduction_percentage" in data
    assert len(data["details"]) > 0
