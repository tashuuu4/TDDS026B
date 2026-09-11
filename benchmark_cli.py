"""CLI tool to run optimization benchmark and print rich statistics."""
import sys
import json
from fastapi.testclient import TestClient
from app.main import app

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run():
    client = TestClient(app)
    resp = client.post('/v1/benchmark/run', json={'sample_size': 15, 'clear_cache_first': True})
    data = resp.json()

    print("\n" + "=" * 70)
    print("      📊 LLM API CALL OPTIMIZER - BENCHMARK REPORT")
    print("=" * 70)
    print(f"Total Prompts Evaluated:       {data['total_prompts']}")
    print(f"Exact Cache Hits:              {data['exact_cache_hits']}")
    print(f"Semantic Cache Hits:           {data['semantic_cache_hits']}")
    print(f"Overall Cache Hit Ratio:       {data['cache_hit_ratio_percentage']:.1f}%")
    print(f"Tier 1 Fast SLM Routed:        {data['tier1_fast_routed']}")
    print(f"Tier 2 Reasoning LLM Routed:   {data['tier2_reasoning_routed']}")
    print("-" * 70)
    print(f"Baseline Latency (Naive):      {data['baseline_total_latency_ms']:.1f} ms")
    print(f"Optimized Latency:             {data['optimized_total_latency_ms']:.1f} ms")
    print(f"⚡ Latency Reduction:           {data['latency_reduction_percentage']:.1f}%")
    print("-" * 70)
    print(f"Baseline Tokens:               {data['baseline_total_tokens']}")
    print(f"Optimized Tokens:              {data['optimized_total_tokens']}")
    print(f"📉 Tokens Saved:               {data['tokens_saved']} ({data['token_reduction_percentage']:.1f}% reduction)")
    print("-" * 70)
    print(f"Baseline Cost (Naive):         ${data['baseline_total_cost_usd']:.6f}")
    print(f"Optimized Cost:                ${data['optimized_total_cost_usd']:.6f}")
    print(f"💰 Cost Saved:                 ${data['cost_saved_usd']:.6f} ({data['cost_reduction_percentage']:.1f}% reduction)")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run()
