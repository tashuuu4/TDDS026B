from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class ChatPromptRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="The primary user prompt or question to process.",
        json_schema_extra={"example": "Could you please explain what is Docker and how does containerization differ from a Virtual Machine?"}
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system instruction / role context for the LLM.",
        json_schema_extra={"example": "You are a senior DevOps engineer and cloud architect."}
    )
    bypass_cache: bool = Field(
        default=False,
        description="If True, bypasses both exact and semantic caches and forces a fresh generation."
    )
    force_model_tier: Optional[Literal["tier-1-fast", "tier-2-reasoning"]] = Field(
        default=None,
        description="Optionally override automatic complexity cascading and force a specific model tier."
    )
    enable_compression: Optional[bool] = Field(
        default=None,
        description="Override global setting for prompt token pruning/compression."
    )
    max_tokens: int = Field(
        default=512,
        ge=1,
        le=4096,
        description="Maximum tokens to generate."
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the response."
    )


class CompressionRequest(BaseModel):
    text: str = Field(
        ...,
        description="The prompt text to compress and prune.",
        json_schema_extra={"example": "Hello there! Could you please be so kind as to tell me what is the speed of light in a vacuum? Thank you very much in advance!"}
    )
    aggressiveness: Literal["conservative", "moderate", "aggressive"] = Field(
        default="moderate",
        description="Compression aggressiveness level. Conservative removes only whitespace/newlines; Moderate removes conversational filler and boilerplate; Aggressive also prunes low-information words."
    )


class CacheLookupRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="Query text to check in the semantic vector and exact hash cache.",
        json_schema_extra={"example": "Explain Docker vs virtual machines"}
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Custom cosine similarity threshold (defaults to server configuration if omitted)."
    )


class ModelClassificationRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="Prompt to analyze for query complexity and model routing.",
        json_schema_extra={"example": "Write a distributed Raft consensus implementation in Rust with leader election and log compaction."}
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt to consider during routing evaluation."
    )


class BatchPromptRequest(BaseModel):
    prompts: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of prompts to process concurrently using the optimizer pipeline.",
        json_schema_extra={"example": [
            "What is Python?",
            "Can you explain the Python programming language?",
            "Design an enterprise database schema for e-commerce orders with ACID guarantees."
        ]}
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional shared system prompt."
    )
    max_concurrency: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum parallel async workers."
    )


class BenchmarkRunRequest(BaseModel):
    dataset_category: Optional[Literal["all", "customer_support", "paraphrased_pairs", "repetitive_queries", "complex_reasoning", "verbose_prompts"]] = Field(
        default="all",
        description="Filter specific benchmark category or run all."
    )
    sample_size: Optional[int] = Field(
        default=12,
        ge=1,
        le=50,
        description="Number of benchmark prompts to evaluate."
    )
    clear_cache_first: bool = Field(
        default=True,
        description="Clear cache before starting benchmark to ensure a clean, reproducible run."
    )
