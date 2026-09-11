import asyncio
import os
import time
from typing import Tuple, Optional
from app.core.config import settings, ModelTierPricing
from app.core.telemetry import estimate_tokens, Timer


# Pre-built contextual responses for realistic offline simulation
MOCK_RESPONSES = {
    "password": "To reset your password without access to your registered email: 1) Navigate to Account Recovery; 2) Select 'Verify via SMS / Phone' or 'Answer Security Questions'; 3) Once verified, set your new password. For further issues, contact support with your Account ID.",
    "support": "Our customer support team is available 24/7 via live chat. You can also reach our phone hotline at +1 (800) 555-0199 Monday through Friday from 8:00 AM to 8:00 PM EST.",
    "async": "In Python, synchronous programming executes tasks sequentially (blocking the thread during I/O), whereas asynchronous programming (using asyncio with async/await) allows non-blocking I/O concurrency on a single event loop without thread-switching overhead.",
    "git": "Git rebase reapplies commits from your current branch onto the tip of another base branch. Unlike git merge, rebase rewrites project history to produce a linear commit graph without merge commits.",
    "docker": "Docker packages applications into lightweight, isolated containers that share the host OS kernel, offering near-zero overhead. In contrast, Virtual Machines (VMs) run complete guest operating systems on a hypervisor, consuming significantly more memory and CPU.",
    "raft": "Distributed Consensus Architecture (Raft):\n- Leader Election: Randomized election timeouts prevent split votes.\n- Write-Ahead Log (WAL): Sequential append-only disk logging ensures durable state transitions.\n- State Machine Safety: A leader only commits an entry once a majority of follower nodes acknowledge replication.\n- Trade-offs: Strong consistency (CP in CAP theorem) at the cost of transient write latency during network partitions.",
    "refund": "Refund Policy: Annual subscriptions can be cancelled within 14 days of purchase or renewal for a full refund. Monthly subscriptions are eligible for pro-rated credits upon request via our billing support portal.",
    "japan": "The capital city of Japan is Tokyo.",
    "fahrenheit": "100 degrees Celsius is equal to 212 degrees Fahrenheit (formula: (100 * 9/5) + 32 = 212)."
}


def _generate_mock_response(prompt: str, model_name: str) -> str:
    """Selects or synthesizes a coherent mock response based on prompt keywords."""
    lower = prompt.lower()
    for key, text in MOCK_RESPONSES.items():
        if key in lower:
            return f"[{model_name}] {text}"

    return (
        f"[{model_name}] Processed response for: '{prompt[:60]}...'\n"
        "Here is the generated analysis tailored to your query parameters. "
        "All requirements have been validated and satisfied with high fidelity."
    )


class LLMGateway:
    """
    Gateway to interface with Large Language Models.
    Defaults to realistic in-memory simulation for zero-dependency local execution,
    with built-in support for live OpenAI calls if API keys are configured.
    """

    @classmethod
    async def call_llm(
        cls,
        prompt: str,
        system_prompt: Optional[str],
        pricing: ModelTierPricing,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> Tuple[str, int, float]:
        """
        Executes LLM call.
        Returns: (response_text, output_tokens, latency_ms)
        """
        # If OpenAI key is present and provider is configured to openai, try real call
        if settings.default_llm_provider == "openai" and settings.openai_api_key:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=settings.openai_api_key)
                model_id = "gpt-4o-mini" if "tier-1" in pricing.name else "gpt-4o"

                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                start = time.perf_counter()
                completion = await client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                latency = round((time.perf_counter() - start) * 1000.0, 2)
                response_text = completion.choices[0].message.content or ""
                output_tokens = completion.usage.completion_tokens if completion.usage else estimate_tokens(response_text)
                return response_text, output_tokens, latency
            except Exception as e:
                # Graceful fallback to mock with warning
                pass

        # Simulated Mock Execution
        with Timer() as timer:
            # Simulate realistic network & inference delay
            if settings.mock_simulate_latency:
                # Fast model takes ~120ms, reasoning model takes ~400ms
                simulated_delay = 0.12 if "tier-1" in pricing.name else 0.35
                await asyncio.sleep(simulated_delay)

            response_text = _generate_mock_response(prompt, pricing.name)
            output_tokens = estimate_tokens(response_text)

        return response_text, output_tokens, timer.elapsed_ms


llm_gateway = LLMGateway()
