import os
import sys
import json
import time
import urllib.request
import urllib.error
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="LLM API Call Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def check_api_health():
    """Checks if the FastAPI backend is running."""
    try:
        req = urllib.request.Request(f"{API_BASE_URL}/health", headers={"User-Agent": "Streamlit"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                return True, json.loads(resp.read().decode())
    except Exception:
        pass
    return False, None


def api_post(endpoint: str, payload: dict):
    """Sends POST request to the FastAPI backend with fallback to direct service execution."""
    try:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{API_BASE_URL}{endpoint}",
            data=data_bytes,
            headers={"Content-Type": "application/json", "User-Agent": "Streamlit"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        # Fallback to in-process service execution if backend server is not running
        try:
            import asyncio
            from app.services.batch_processor import process_single_optimized_prompt
            from app.services.prompt_compressor import prompt_compressor
            from app.services.model_router import model_router
            from app.services.cache_service import cache_service

            if endpoint == "/v1/optimize/chat":
                res = asyncio.run(process_single_optimized_prompt(
                    prompt=payload.get("prompt", ""),
                    system_prompt=payload.get("system_prompt"),
                    bypass_cache=payload.get("bypass_cache", False),
                    force_model_tier=payload.get("force_model_tier")
                ))
                return {
                    "response": res["response"],
                    "optimization": {
                        "cache_status": res["cache_status"],
                        "cache_similarity_score": res["cache_similarity_score"],
                        "original_prompt_tokens": res["original_tokens"],
                        "compressed_prompt_tokens": res["compressed_tokens"],
                        "tokens_pruned": res["tokens_pruned"],
                        "model_tier_used": res["model_tier_used"],
                        "model_routing_reason": res["model_routing_reason"],
                        "latency_ms": res["latency_ms"],
                        "actual_cost_usd": res["actual_cost_usd"],
                        "baseline_cost_usd": res["baseline_cost_usd"],
                        "cost_saved_usd": res["cost_saved_usd"],
                        "cost_savings_percentage": res["savings_percentage"]
                    }
                }
            elif endpoint == "/v1/optimize/compress":
                comp_text, orig_t, comp_t, saved_t = prompt_compressor.compress(
                    payload.get("text", ""),
                    aggressiveness=payload.get("aggressiveness", "moderate")
                )
                ratio = round((saved_t / orig_t * 100.0), 2) if orig_t > 0 else 0.0
                return {
                    "original_text": payload.get("text", ""),
                    "compressed_text": comp_text,
                    "original_tokens": orig_t,
                    "compressed_tokens": comp_t,
                    "tokens_saved": saved_t,
                    "compression_percentage": ratio,
                    "removed_filler_count": saved_t
                }
            elif endpoint == "/v1/routing/classify":
                score, label, reasons, pricing = model_router.analyze_complexity(
                    payload.get("prompt", ""),
                    payload.get("system_prompt", "")
                )
                return {
                    "prompt": payload.get("prompt", ""),
                    "complexity_score": score,
                    "complexity_label": label,
                    "recommended_tier": pricing.name,
                    "model_name": pricing.name,
                    "estimated_input_cost_per_1m": pricing.input_cost_per_1m,
                    "reasons": reasons
                }
            elif endpoint == "/v1/cache/clear":
                cache_service.clear()
                return {"message": "Cache successfully cleared."}
            elif endpoint == "/v1/benchmark/run":
                from app.api.v1.endpoints.benchmark import run_optimization_benchmark
                from app.models.requests import BenchmarkRunRequest
                req_obj = BenchmarkRunRequest(**payload)
                res = asyncio.run(run_optimization_benchmark(req_obj))
                return res.model_dump()
        except Exception as fallback_err:
            st.error(f"Error calling backend ({e}) and fallback failed: {fallback_err}")
            return None
        return None


def api_get(endpoint: str):
    """Sends GET request to backend with fallback to direct service import."""
    try:
        req = urllib.request.Request(f"{API_BASE_URL}{endpoint}", headers={"User-Agent": "Streamlit"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        try:
            from app.services.cache_service import cache_service
            from app.services.analytics_service import analytics_service
            from app.data.benchmark_dataset import load_benchmark_dataset, get_dataset_categories

            if endpoint == "/v1/cache/stats":
                return cache_service.get_stats()
            elif endpoint == "/v1/analytics/overview":
                return analytics_service.get_summary()
            elif endpoint == "/v1/benchmark/dataset":
                return {"categories": get_dataset_categories(), "prompts": load_benchmark_dataset()}
        except Exception:
            return None
    return None


def render_markdown_table(rows: list) -> str:
    """Renders a list of dicts as a clean Markdown table without external dependencies."""
    if not rows:
        return "*No records to display.*"
    
    headers = [
        "Prompt", "Category", "Cache Status", "Model Tier", "Latency (ms)", "Tokens", "Cost ($)"
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for r in rows:
        prompt_snippet = r.get("prompt", "")[:45] + "..." if len(r.get("prompt", "")) > 45 else r.get("prompt", "")
        status = r.get("cache_status", "miss")
        if status == "exact_hit":
            status_badge = "🟢 `exact_hit`"
        elif status == "semantic_hit":
            status_badge = "🔵 `semantic_hit`"
        else:
            status_badge = "⚪ `miss`"

        lines.append(
            f"| {prompt_snippet} | `{r.get('category', 'general')}` | {status_badge} | `{r.get('model_tier', '')}` | {r.get('latency_ms', 0):.1f} | {r.get('tokens_used', 0)} | ${r.get('cost_usd', 0):.6f} |"
        )
    return "\n".join(lines)


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/artificial-intelligence.png", width=70)
    st.title("API Optimizer")
    st.caption("GenAI Cost & Latency Optimization Gateway")

    is_online, health_data = check_api_health()
    if is_online:
        st.success(f"🟢 **FastAPI Gateway**: Connected\n`{API_BASE_URL}`")
    else:
        st.warning("🟡 **FastAPI Gateway**: Offline (Using in-process fallback engine)")
        st.caption("Run `python run.py` in your terminal to start the live Swagger UI backend on port 8000.")

    st.markdown("---")
    st.markdown("### ⚙️ Quick Documentation")
    st.markdown("- [Swagger UI Docs](http://127.0.0.1:8000/docs)")
    st.markdown("- [ReDoc Specifications](http://127.0.0.1:8000/redoc)")
    st.markdown("- [OpenAPI Schema](http://127.0.0.1:8000/openapi.json)")

    st.markdown("---")
    if st.button("🧹 Flush Cache Now", use_container_width=True):
        res = api_post("/v1/cache/clear", {})
        st.toast("Cache flushed successfully!", icon="🧹")


# Top Header
st.title("⚡ LLM API Call Optimizer Dashboard")
st.markdown(
    "Intelligent Gateway for Large Language Model calls: **Multi-Tier Semantic Caching**, "
    "**Prompt Token Pruning**, **Complexity-Aware Model Cascading**, and **Real-Time Savings Telemetry**."
)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Chat & Optimization Playground",
    "📊 Live Benchmark & ROI Analytics",
    "✂️ Prompt Compressor Lab",
    "🗄️ Cache Explorer & Cosine Similarity",
    "💰 Enterprise ROI Calculator"
])


# -------------------------------------------------------------
# TAB 1: Chat & Optimization Playground
# -------------------------------------------------------------
with tab1:
    st.subheader("Interactive Optimization Playground")
    st.caption("Submit a prompt to observe real-time semantic caching, token compression, and model cascading.")

    PRESETS = {
        "Custom Prompt": "",
        "Exact Cache Test (Identical Prompt)": "What is the difference between synchronous and asynchronous programming in Python?",
        "Semantic Cache Test (Paraphrased Query)": "Could you explain the difference between sync and async execution in Python?",
        "Verbose / Bloated Prompt (Token Pruning)": "Hello dear AI assistant! I hope you are having a wonderful day. Could you please be so kind as to do me a small favor and provide me with a brief summary of how Git rebase works? Thank you very much in advance!",
        "Simple Factual Lookup (Tier 1 Fast SLM)": "What is the capital city of Japan?",
        "Complex Architectural Reasoning (Tier 2 Heavy LLM)": "Design a high-throughput distributed message broker with write-ahead logging, Raft consensus for leader election, and two-phase locking."
    }

    selected_preset = st.selectbox("💡 Select a Benchmark Preset or write your own:", list(PRESETS.keys()))
    default_text = PRESETS[selected_preset] if selected_preset != "Custom Prompt" else ""

    col_input, col_ctrls = st.columns([3, 1])

    with col_input:
        user_prompt = st.text_area(
            "User Prompt",
            value=default_text,
            height=130,
            placeholder="Type your prompt here..."
        )
        system_prompt = st.text_input(
            "System Prompt (Optional)",
            placeholder="e.g. You are a senior DevOps engineer and cloud architect."
        )

    with col_ctrls:
        st.markdown("**Optimization Options**")
        bypass_cache = st.checkbox("Bypass Cache", value=False, help="Force fresh generation, ignoring exact and semantic cache.")
        enable_compression = st.checkbox("Enable Compression", value=True, help="Prune polite filler and whitespace.")
        force_tier = st.selectbox(
            "Model Cascading",
            ["Auto-Detect Complexity", "Force Tier 1 (Fast SLM)", "Force Tier 2 (Reasoning LLM)"],
            help="Auto selects tier based on algorithmic complexity heuristic."
        )

    submit_btn = st.button("🚀 Execute Optimized LLM Call", type="primary", use_container_width=True)

    if submit_btn:
        if not user_prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            tier_arg = None
            if "Tier 1" in force_tier:
                tier_arg = "tier-1-fast"
            elif "Tier 2" in force_tier:
                tier_arg = "tier-2-reasoning"

            payload = {
                "prompt": user_prompt,
                "system_prompt": system_prompt if system_prompt.strip() else None,
                "bypass_cache": bypass_cache,
                "enable_compression": enable_compression,
                "force_model_tier": tier_arg
            }

            with st.spinner("Processing through optimization gateway..."):
                res = api_post("/v1/optimize/chat", payload)

            if res and "optimization" in res:
                opt = res["optimization"]
                cache_status = opt["cache_status"]

                st.markdown("---")
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)

                with kpi1:
                    if cache_status == "exact_hit":
                        st.metric("🎯 Cache Status", "Exact Hash Hit", delta="100% saved (0 tokens)", delta_color="normal")
                    elif cache_status == "semantic_hit":
                        sim = opt.get("cache_similarity_score", 1.0)
                        st.metric("🧬 Cache Status", f"Semantic Hit ({sim:.2f})", delta="100% saved (0 tokens)", delta_color="normal")
                    else:
                        st.metric("⚡ Cache Status", "Cache Miss (Fresh)", delta="Cached for next time", delta_color="off")

                with kpi2:
                    st.metric("⏱️ Latency", f"{opt['latency_ms']:.1f} ms", delta=f"{opt['model_tier_used']}")

                with kpi3:
                    orig_tok = opt["original_prompt_tokens"]
                    comp_tok = opt["compressed_prompt_tokens"]
                    pruned = opt["tokens_pruned"]
                    st.metric("📉 Tokens Processed", f"{comp_tok} tokens", delta=f"-{pruned} pruned" if pruned > 0 else "0 pruned")

                with kpi4:
                    cost_saved = opt["cost_saved_usd"]
                    savings_pct = opt["cost_savings_percentage"]
                    st.metric("💰 Cost Savings", f"${cost_saved:.6f}", delta=f"{savings_pct:.1f}% vs baseline", delta_color="normal")

                st.markdown("#### 🤖 Generated / Cached LLM Response")
                st.info(res["response"])

                with st.expander("🔍 Optimization Pipeline Breakdown & Routing Rationale", expanded=True):
                    d_col1, d_col2 = st.columns(2)
                    with d_col1:
                        st.markdown("**Model Routing Decision:**")
                        st.write(f"- **Tier Assigned**: `{opt['model_tier_used']}`")
                        st.write(f"- **Rationale**: {opt['model_routing_reason']}")
                    with d_col2:
                        st.markdown("**Financial Accounting:**")
                        st.write(f"- **Actual Incurred Cost**: `${opt['actual_cost_usd']:.6f}`")
                        st.write(f"- **Naive Baseline Cost**: `${opt['baseline_cost_usd']:.6f}`")
                        st.write(f"- **Dollar Savings**: `${opt['cost_saved_usd']:.6f}` ({opt['cost_savings_percentage']:.1f}%)")


# -------------------------------------------------------------
# TAB 2: Live Benchmark & ROI Analytics
# -------------------------------------------------------------
with tab2:
    st.subheader("📊 Live Optimization Benchmark")
    st.caption("Runs the built-in multi-domain dataset through both Naive Baseline and Optimized Pipeline to evaluate overall performance.")

    b_col1, b_col2, b_col3 = st.columns([2, 1, 1])
    with b_col1:
        category_choice = st.selectbox(
            "Dataset Category Filter",
            ["all", "customer_support", "repetitive_queries", "paraphrased_pairs", "complex_reasoning", "verbose_prompts"]
        )
    with b_col2:
        sample_size = st.slider("Sample Size", min_value=3, max_value=15, value=15)
    with b_col3:
        st.write("")
        st.write("")
        run_bench_btn = st.button("🚀 Run Benchmark Suite", type="primary", use_container_width=True)

    if run_bench_btn:
        with st.spinner("Executing comparative simulation..."):
            bench_res = api_post("/v1/benchmark/run", {
                "dataset_category": category_choice,
                "sample_size": sample_size,
                "clear_cache_first": True
            })

        if bench_res:
            st.markdown("---")
            st.success("✅ Benchmark Simulation Completed Successfully!")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Prompts Evaluated", bench_res["total_prompts"])
            m2.metric("Cache Hit Ratio", f"{bench_res['cache_hit_ratio_percentage']:.1f}%", f"{bench_res['exact_cache_hits']} exact, {bench_res['semantic_cache_hits']} semantic")
            m3.metric("Token Reduction", f"{bench_res['token_reduction_percentage']:.1f}%", f"{bench_res['tokens_saved']} tokens saved")
            m4.metric("Cost Reduction", f"{bench_res['cost_reduction_percentage']:.1f}%", f"${bench_res['cost_saved_usd']:.6f} saved")

            st.markdown("---")
            st.markdown("#### 📈 Comparative Efficiency: Baseline vs. Optimized Pipeline")

            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.markdown("**💰 Total Cost Incurred ($ USD)**")
                st.markdown(f"- **Naive Baseline**: `${bench_res['baseline_total_cost_usd']:.6f}`")
                st.markdown(f"- **Optimized Pipeline**: `${bench_res['optimized_total_cost_usd']:.6f}`")
                st.progress(max(0.01, min(1.0, 1.0 - (bench_res['cost_reduction_percentage'] / 100.0))))
                st.caption(f"🎉 **{bench_res['cost_reduction_percentage']:.1f}% Cost Reduction** achieved")

            with v_col2:
                st.markdown("**📉 Total Tokens Consumed**")
                st.markdown(f"- **Naive Baseline**: `{bench_res['baseline_total_tokens']}` tokens")
                st.markdown(f"- **Optimized Pipeline**: `{bench_res['optimized_total_tokens']}` tokens")
                st.progress(max(0.01, min(1.0, 1.0 - (bench_res['token_reduction_percentage'] / 100.0))))
                st.caption(f"📉 **{bench_res['token_reduction_percentage']:.1f}% Tokens Saved** across pipeline")

            st.markdown("---")
            st.markdown("#### 📋 Itemized Benchmark Results")
            if "details" in bench_res and bench_res["details"]:
                table_md = render_markdown_table(bench_res["details"])
                st.markdown(table_md)


# -------------------------------------------------------------
# TAB 3: Prompt Compressor Lab
# -------------------------------------------------------------
with tab3:
    st.subheader("✂️ Prompt Token Pruning & Compression Lab")
    st.caption("Observe how polite conversational filler, whitespace bloat, and boilerplate are stripped to reduce input tokens.")

    comp_input = st.text_area(
        "Input Prompt to Compress",
        value="Hello dear assistant! I hope this message finds you well. Could you please be so kind as to do me a small favor and explain how Git rebase works? Thank you very much in advance!",
        height=120
    )

    agg_level = st.radio(
        "Compression Aggressiveness",
        ["conservative", "moderate", "aggressive"],
        index=1,
        horizontal=True,
        help="Conservative: whitespace only; Moderate: filler & pleasantries; Aggressive: concise token replacement."
    )

    if st.button("Prune & Compress Prompt"):
        comp_res = api_post("/v1/optimize/compress", {"text": comp_input, "aggressiveness": agg_level})
        if comp_res:
            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric("Original Tokens", comp_res["original_tokens"])
            col_res2.metric("Compressed Tokens", comp_res["compressed_tokens"])
            col_res3.metric("Tokens Saved", f"{comp_res['tokens_saved']} ({comp_res['compression_percentage']:.1f}%)")

            st.markdown("#### Cleaned Compressed Prompt:")
            st.code(comp_res["compressed_text"], language="text")


# -------------------------------------------------------------
# TAB 4: Cache Explorer & Cosine Similarity Inspector
# -------------------------------------------------------------
with tab4:
    st.subheader("🗄️ Cache Explorer & Semantic Similarity Inspector")
    st.caption("Inspect live cache entries and test the cosine distance between arbitrary prompt pairs.")

    stats = api_get("/v1/cache/stats")
    if stats:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Exact Cache Entries", stats.get("exact_cache_entries", 0))
        s2.metric("Semantic Cache Entries", stats.get("semantic_cache_entries", 0))
        s3.metric("Exact Hits", stats.get("exact_hits", 0))
        s4.metric("Semantic Hits", stats.get("semantic_hits", 0))

    st.markdown("---")
    st.markdown("#### 🔬 Real-Time Semantic Similarity Tester")
    st.caption("Test how the semantic vector engine compares two query phrasings.")

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        query_a = st.text_input("Query A (Base Prompt)", value="How do I reset my account password if I forgot my email?")
    with sim_col2:
        query_b = st.text_input("Query B (Candidate Paraphrase)", value="How do I recover my password when I no longer have access to my email?")

    if st.button("Calculate Semantic Similarity"):
        try:
            from app.services.cache_service import _get_embedding_vector, _cosine_similarity
            v_a = _get_embedding_vector(query_a)
            v_b = _get_embedding_vector(query_b)
            score = _cosine_similarity(v_a, v_b)

            st.markdown(f"### Cosine Similarity Score: `{score:.4f}`")
            st.progress(float(score))

            if score >= 0.70:
                st.success(f"✅ **Semantic Cache Hit!** (Score {score:.4f} >= 0.70 threshold). This query will be served from cache at 0 API cost.")
            else:
                st.info(f"ℹ️ **Cache Miss** (Score {score:.4f} < 0.70 threshold). This query will generate a fresh response.")
        except Exception as err:
            st.error(f"Error computing similarity: {err}")


# -------------------------------------------------------------
# TAB 5: Enterprise ROI Calculator
# -------------------------------------------------------------
with tab5:
    st.subheader("💰 Projected Enterprise ROI & Cost Savings Calculator")
    st.caption("Calculate how much money your organization saves annually using this LLM API Optimizer.")

    roi_c1, roi_c2 = st.columns(2)
    with roi_c1:
        monthly_requests = st.slider("Monthly LLM API Requests", min_value=50000, max_value=5000000, value=500000, step=50000, format="%d")
        avg_input_tokens = st.slider("Average Prompt Tokens", min_value=50, max_value=2000, value=350, step=25)
        avg_output_tokens = st.slider("Average Response Tokens", min_value=50, max_value=2000, value=250, step=25)

    with roi_c2:
        cache_hit_rate = st.slider("Estimated Cache Hit Rate (%)", min_value=10, max_value=80, value=35, step=5)
        tier1_routing_share = st.slider("Share of Simple Queries Routed to Tier 1 SLM (%)", min_value=30, max_value=95, value=75, step=5)
        compression_saving_rate = st.slider("Average Prompt Compression Savings (%)", min_value=5, max_value=40, value=20, step=5)

    heavy_in = 2.50
    heavy_out = 10.00
    fast_in = 0.15
    fast_out = 0.60

    # 1. Baseline Monthly Cost
    base_cost_month = (monthly_requests * (avg_input_tokens / 1_000_000) * heavy_in) + (monthly_requests * (avg_output_tokens / 1_000_000) * heavy_out)

    # 2. Optimized Monthly Cost
    cache_requests = monthly_requests * (cache_hit_rate / 100.0)
    remaining_requests = monthly_requests - cache_requests
    compressed_in_tokens = avg_input_tokens * (1.0 - (compression_saving_rate / 100.0))

    t1_requests = remaining_requests * (tier1_routing_share / 100.0)
    t2_requests = remaining_requests * (1.0 - (tier1_routing_share / 100.0))

    cost_t1 = (t1_requests * (compressed_in_tokens / 1_000_000) * fast_in) + (t1_requests * (avg_output_tokens / 1_000_000) * fast_out)
    cost_t2 = (t2_requests * (compressed_in_tokens / 1_000_000) * heavy_in) + (t2_requests * (avg_output_tokens / 1_000_000) * heavy_out)

    opt_cost_month = cost_t1 + cost_t2
    month_saved = base_cost_month - opt_cost_month
    year_saved = month_saved * 12.0
    savings_pct = (month_saved / base_cost_month) * 100.0

    st.markdown("---")
    st.markdown("### 💵 Estimated Enterprise Savings Projection")

    res_kpi1, res_kpi2, res_kpi3 = st.columns(3)
    res_kpi1.metric("Naive Baseline Cost", f"${base_cost_month:,.2f} / month", f"${base_cost_month * 12:,.2f} / year")
    res_kpi2.metric("Optimized Gateway Cost", f"${opt_cost_month:,.2f} / month", f"${opt_cost_month * 12:,.2f} / year", delta_color="inverse")
    res_kpi3.metric("🎉 Net Enterprise Savings", f"${month_saved:,.2f} / month", f"${year_saved:,.2f} / year ({savings_pct:.1f}% reduction)")
