import re
from typing import Dict, Any, List, Tuple
from app.core.config import settings, ModelTierPricing

# Keyword sets for complexity detection
COMPLEX_KEYWORDS = [
    "architect", "architecture", "distributed", "concurrency", "mutex", "deadlock",
    "consensus", "raft", "paxos", "byzantine", "recurrence", "theorem", "proof",
    "derive", "formal", "trade-off", "tradeoff", "edge failure", "serializability",
    "kadane", "divide and conquer", "dynamic programming", "in-memory database",
    "two-phase locking", "vector clock", "gossip protocol", "zero-copy", "optimize time complexity"
]

SIMPLE_PATTERNS = [
    r"(?i)\b(what is|what are|define|meaning of|who is|when was|where is)\b",
    r"(?i)\b(how do i reset|customer support|working hours|phone number|refund policy)\b",
    r"(?i)\b(convert \d+|translate|yes or no|true or false)\b",
    r"(?i)\b(capital city of|spell check|format as json)\b"
]


class ModelRouter:
    """
    Analyzes prompt complexity and routes requests to the optimal model tier:
    - Tier 1 (Fast SLM): high speed, minimal cost for simple tasks.
    - Tier 2 (Reasoning LLM): deep analytical power for high complexity tasks.
    """

    @classmethod
    def analyze_complexity(
        cls,
        prompt: str,
        system_prompt: str = ""
    ) -> Tuple[float, str, List[str], ModelTierPricing]:
        """
        Calculates complexity score (0.0 to 1.0) and selects model tier.
        Returns: (complexity_score, label, reasons, selected_pricing)
        """
        combined = f"{system_prompt} {prompt}".lower()
        reasons: List[str] = []
        score = 0.20  # Base neutral score

        # 1. Check for complex domain keywords
        matched_complex = [kw for kw in COMPLEX_KEYWORDS if kw in combined]
        if matched_complex:
            increment = min(0.60, len(matched_complex) * 0.20)
            score += increment
            reasons.append(f"Identified high-complexity domain concepts: {', '.join(matched_complex[:4])}")

        # 2. Check for code blocks or technical syntax
        if "```" in prompt or "class " in prompt or "def " in prompt:
            score += 0.25
            reasons.append("Detected code implementation or snippet analysis")

        # 3. Check for multi-step reasoning instructions
        if any(marker in combined for marker in ["step by step", "formal proof", "trade-offs", "edge cases"]):
            score += 0.20
            reasons.append("Detected explicit request for multi-step reasoning or formal analysis")

        # 4. Prompt length penalty / bonus
        if len(prompt) > 300:
            score += 0.15
            reasons.append(f"Long comprehensive prompt length ({len(prompt)} chars)")
        elif len(prompt) < 80:
            score -= 0.15
            reasons.append(f"Concise prompt length ({len(prompt)} chars)")

        # 5. Check for simple patterns
        matched_simple = any(re.search(pat, combined) for pat in SIMPLE_PATTERNS)
        if matched_simple:
            score -= 0.25
            reasons.append("Matched standard factual retrieval, FAQ, or conversion pattern")

        # Clamp score between 0.05 and 0.99
        final_score = round(max(0.05, min(0.99, score)), 2)

        if final_score >= 0.50:
            label = "complex"
            reasons.append("Assigned to Tier 2 Reasoning LLM due to high algorithmic or conceptual complexity")
            selected_model = settings.tier2_model
        elif final_score >= 0.35:
            label = "moderate"
            reasons.append("Assigned to Tier 1 Fast SLM (capable of handling moderate tasks at 90%+ lower cost)")
            selected_model = settings.tier1_model
        else:
            label = "simple"
            reasons.append("Assigned to Tier 1 Fast SLM for instantaneous response and minimal token expenditure")
            selected_model = settings.tier1_model

        return final_score, label, reasons, selected_model


model_router = ModelRouter()
