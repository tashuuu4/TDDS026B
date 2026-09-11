import pytest
from app.services.cache_service import DualTierCache


def test_exact_cache_hit():
    cache = DualTierCache()
    prompt = "What is the capital of France?"
    response = "The capital of France is Paris."

    # Initial lookup should miss
    entry, match_type, score = cache.lookup(prompt)
    assert entry is None
    assert match_type == "miss"

    # Store entry
    cache.store(prompt, response, "tier-1-fast")

    # Identical lookup should be exact hit
    entry, match_type, score = cache.lookup(prompt)
    assert entry is not None
    assert match_type == "exact_hit"
    assert entry.response == response
    assert score == 1.0


def test_semantic_cache_hit():
    cache = DualTierCache()
    prompt1 = "How do I reset my account password if I forgot my email?"
    response = "Contact support with your phone number to reset password."

    cache.store(prompt1, response, "tier-1-fast")

    # Paraphrased query
    prompt2 = "How do I recover my password when I no longer have access to my email?"
    entry, match_type, score = cache.lookup(prompt2, threshold=0.70)

    assert entry is not None
    assert match_type == "semantic_hit"
    assert score is not None and score >= 0.70
    assert entry.response == response


def test_cache_miss_on_unrelated_query():
    cache = DualTierCache()
    cache.store("How do I boil an egg?", "Boil in water for 7 minutes.", "tier-1-fast")

    entry, match_type, score = cache.lookup("Explain quantum mechanics", threshold=0.85)
    assert entry is None
    assert match_type == "miss"


def test_cache_clear():
    cache = DualTierCache()
    cache.store("Hello", "Hi there", "tier-1-fast")
    assert len(cache.exact_cache) == 1

    cache.clear()
    assert len(cache.exact_cache) == 0
    assert len(cache.semantic_entries) == 0
    assert cache.total_lookups == 0
