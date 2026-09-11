import re
import time
from contextlib import contextmanager
from typing import Generator
from app.core.config import ModelTierPricing, settings


def estimate_tokens(text: str) -> int:
    """
    Estimates token count for LLMs using an enhanced regex tokenizer
    that closely approximates BPE tokenization (~4 chars/token on average,
    splitting on word boundaries, numbers, and punctuation).
    """
    if not text:
        return 0

    # Match words, numbers, individual punctuations, and sequences of whitespace
    tokens = re.findall(r"\w+|[^\w\s]|\s+", text)
    # Most subword tokenizers average ~0.75-0.8 tokens per matched group or ~3.8-4 characters
    char_len = len(text)
    heuristic_tokens = max(1, int(char_len / 3.8))
    regex_tokens = max(1, len(tokens))
    # Return weighted average rounded to int
    return max(1, int((heuristic_tokens + regex_tokens) / 2))


def calculate_cost(tokens_in: int, tokens_out: int, pricing: ModelTierPricing) -> float:
    """
    Calculates cost in USD for a given number of input and output tokens.
    """
    cost_in = (tokens_in / 1_000_000.0) * pricing.input_cost_per_1m
    cost_out = (tokens_out / 1_000_000.0) * pricing.output_cost_per_1m
    return round(cost_in + cost_out, 6)


class Timer:
    """Context manager and helper for timing operations in milliseconds."""

    def __init__(self):
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = round((time.perf_counter() - self.start_time) * 1000.0, 2)
