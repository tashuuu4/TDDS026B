# 🚀 LLM API Call Optimizer for Efficient Generative AI

A production-grade, API-first backend built with **FastAPI** and documented via interactive **Swagger UI** (`/docs`). It tackles the primary bottlenecks of Generative AI systems—API costs, latency spikes, and token rate-limit (TPM/RPM) exhaustion—through four core optimization layers:

1. **Multi-Tier Semantic & Exact Caching**: Sub-millisecond exact hash matching (SHA-256) and cosine similarity vector caching. Completely avoids redundant LLM calls (100% cost reduction, ~1ms latency).
2. **Prompt Token Pruning & Compression**: Eliminates whitespace bloat, conversational filler, polite boilerplate, and low-information tokens, reducing input token expenditures by 20% to 50%.
3. **Complexity-Aware Model Cascading (SLM vs. LLM)**: Intelligently classifies query complexity to route simple QA, factual lookups, and customer support queries to low-cost, fast Small Language Models (Tier 1 @ $0.15/1M tokens) while preserving costly frontier reasoning models (Tier 2 @ $2.50/1M tokens) for complex architectures, formal proofs, and multi-step logic.
4. **Micro-Batching & Async Workers**: Concurrency-controlled execution queues to handle high-throughput workloads without exceeding provider rate limits.
5. **Self-Contained Multi-Domain Dataset**: Built-in benchmark suite covering customer support, repetitive queries, paraphrased pairs, bloated prompts, and complex reasoning for immediate zero-dependency evaluation.

---

## 🏛️ Architecture Overview

```
                      Client Request
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │           FastAPI Gateway & Router           │
     │      Interactive Swagger UI at /docs         │
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │ 1. Multi-Tier Cache Layer                    │
     │    ├── Exact Hash Cache (SHA-256)            │
     │    └── Semantic Vector Cache (Cosine Sim)    │
     └──────────────┬───────────────────────────────┘
                    │
         ┌──────────┴──────────┐
         │ Cache Miss          │ Cache Hit
         ▼                     ▼
┌──────────────────┐   ┌────────────────────────────┐
│ 2. Prompt        │   │ Return Instant Response    │
│    Compressor    │   │ (0 API Tokens, <5ms, $0)   │
│ - Filler Removal │   └────────────────────────────┘
│ - Whitespace Opt │
└────────┬─────────┘
         │
         ▼
┌───────────────────────────────────────────────────┐
│ 3. Model Cascade Router                           │
│    - Analyze query complexity & intent            │
│    - Route: Tier 1 Fast SLM vs Tier 2 Reasoning   │
└────────┬──────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────┐
│ 4. Execution Gateway & Telemetry                  │
│    - Zero-dependency Mock LLM / Real OpenAI API   │
│    - Compute token & dollar ($) savings           │
│    - Update aggregate analytics engine            │
└───────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Requirements & Setup

Ensure you have Python 3.10+ installed.

```bash
cd C:\Users\tashu\.gemini\antigravity\scratch\llm-api-optimizer

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Backend & Streamlit Frontend

#### Launch FastAPI Backend (Swagger UI):
```bash
python run.py
```
- **Interactive Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc UI**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **OpenAPI JSON**: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

#### Launch Interactive Streamlit Dashboard:
```bash
python run_streamlit.py
```
- **Streamlit Dashboard**: [http://localhost:8501](http://localhost:8501)

The Streamlit UI connects to the FastAPI backend with automatic fallback to in-process execution, offering 5 interactive tabs:
1. **💬 Chat & Optimization Playground**: Real-time prompt optimization with visual KPI delta cards, side-by-side prompt diff, and routing reasons.
2. **📊 Live Benchmark Suite & ROI Analytics**: One-click benchmark runner comparing naive baseline vs. optimized pipeline with comparative efficiency meters and itemized results.
3. **✂️ Prompt Compressor Lab**: Live slider testing for polite filler removal, whitespace reduction, and token reduction metrics.
4. **🗄️ Cache Explorer & Cosine Similarity Inspector**: Real-time cosine similarity calculator between arbitrary query pairs and cache flush button.
5. **💰 Enterprise ROI Calculator**: Interactive sliders to estimate annual organizational cost savings ($) at scale.

---

## 📖 Interactive Swagger UI Endpoints

| Category | Method | Endpoint | Description |
|---|---|---|---|
| **Optimization** | `POST` | `/v1/optimize/chat` | End-to-end optimization pipeline (Cache -> Compress -> Route -> LLM -> Analytics) |
| **Optimization** | `POST` | `/v1/optimize/compress` | Test prompt token compression and filler removal |
| **Routing** | `POST` | `/v1/routing/classify` | Inspect query complexity score and tier selection rationale |
| **Cache** | `GET` | `/v1/cache/stats` | View exact/semantic cache entries, hits, and hit ratio |
| **Cache** | `POST` | `/v1/cache/lookup` | Query cache directly without triggering generation |
| **Cache** | `DELETE` | `/v1/cache/clear` | Flush all cache entries |
| **Batch** | `POST` | `/v1/batch/process` | Micro-batch multiple prompts with concurrency controls |
| **Benchmarking** | `GET` | `/v1/benchmark/dataset` | Inspect self-contained sample prompts and categories |
| **Benchmarking** | `POST` | `/v1/benchmark/run` | Execute comparative benchmark (Naive Baseline vs. Optimized) |
| **Analytics** | `GET` | `/v1/analytics/overview` | View cumulative server token and dollar ($) savings |
| **System** | `GET` | `/health` | Server health check and active configuration |

---

## 📊 Benchmark Example: Naive Baseline vs. Optimized Pipeline

When you run `POST /v1/benchmark/run`, the system runs a batch of queries through both the naive baseline (uncompressed direct calls to heavy model) and the optimized pipeline:

```json
{
  "total_prompts": 12,
  "exact_cache_hits": 2,
  "semantic_cache_hits": 2,
  "cache_hit_ratio_percentage": 33.33,
  "tier1_fast_routed": 6,
  "tier2_reasoning_routed": 2,
  "tokens_saved": 680,
  "token_reduction_percentage": 42.15,
  "cost_saved_usd": 0.003480,
  "cost_reduction_percentage": 81.25,
  "latency_reduction_percentage": 68.50
}
```

---

## 🧪 Running Automated Tests

Run the complete test suite with `pytest`:

```bash
pytest -v
```

All unit tests for caching, compression, routing, and API endpoints are included under `tests/`.
