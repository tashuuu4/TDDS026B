import hashlib
import math
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any, Set
from app.core.config import settings

# Common English stopwords to focus similarity on informational content
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few",
    "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your", "yours", "yourself", "yourselves", "please", "kindly", "longer", "access"
}

# Semantic synonym bridges for key domain intent concepts
SYNONYMS_MAP: Dict[str, str] = {
    "reset": "recover",
    "recover": "reset",
    "password": "credentials",
    "credentials": "password",
    "sync": "synchronous",
    "synchronous": "sync",
    "async": "asynchronous",
    "asynchronous": "async",
    "refund": "cancellation",
    "cancel": "cancellation",
    "cancellation": "refund",
    "invert": "reverse",
    "reverse": "invert",
    "difference": "distinct",
    "differ": "distinct",
    "distinct": "differ"
}


def _normalize_text(text: str) -> str:
    """Lowercases, normalizes whitespace and removes punctuation."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(cleaned.split())


def _extract_content_tokens(text: str) -> List[str]:
    """Extracts non-stopword content tokens."""
    normalized = _normalize_text(text)
    tokens = [w for w in normalized.split() if w and w not in STOPWORDS]
    # Fallback to all tokens if all were stopwords
    return tokens if tokens else normalized.split()


def _get_embedding_vector(text: str) -> Dict[str, float]:
    """
    Generates a normalized sparse semantic feature vector based on:
    - Content word unigrams (high weight)
    - Domain synonym expansions
    - Character 3-grams of content words (stem and morphological tolerance)
    """
    content_tokens = _extract_content_tokens(text)
    features = Counter()

    for w in content_tokens:
        # High weight for primary content keywords
        features[f"w:{w}"] += 2.0

        # Expand synonym bridge
        if w in SYNONYMS_MAP:
            features[f"w:{SYNONYMS_MAP[w]}"] += 2.0

        # Subword 3-grams for morphological variations (e.g. cancel, cancelling, cancellation)
        if len(w) >= 4:
            for i in range(len(w) - 2):
                features[f"ng:{w[i:i+3]}"] += 0.5

    # Compute L2 norm
    total_sq = sum(v * v for v in features.values())
    norm = math.sqrt(total_sq) if total_sq > 0 else 1.0

    return {k: round(v / norm, 5) for k, v in features.items()}


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Computes cosine similarity between two normalized sparse vectors."""
    if not vec_a or not vec_b:
        return 0.0

    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a

    dot_product = sum(val * vec_b.get(k, 0.0) for k, val in vec_a.items())
    return round(min(1.0, max(0.0, dot_product)), 4)


class CacheEntry:
    def __init__(self, prompt: str, response: str, model_tier: str, system_prompt: Optional[str] = None):
        self.prompt = prompt
        self.system_prompt = system_prompt or ""
        self.response = response
        self.model_tier = model_tier
        self.created_at = time.time()
        self.hits = 0
        self.vector = _get_embedding_vector(f"{self.system_prompt} {prompt}".strip())


class DualTierCache:
    """
    A two-tier caching engine:
    Tier 1: Exact Hash Cache (SHA-256) - O(1) lookup, 0 cost.
    Tier 2: Semantic Vector Cache - Cosine similarity over content embeddings.
    """

    def __init__(self):
        self.exact_cache: Dict[str, CacheEntry] = {}
        self.semantic_entries: List[CacheEntry] = []
        self.exact_hits: int = 0
        self.semantic_hits: int = 0
        self.misses: int = 0
        self.total_lookups: int = 0

    def _hash_key(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        content = f"{system_prompt or ''}::{_normalize_text(prompt)}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def lookup(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> Tuple[Optional[CacheEntry], str, Optional[float]]:
        """
        Looks up prompt in cache.
        Returns: (entry, match_type, similarity_score)
        match_type is one of 'exact_hit', 'semantic_hit', or 'miss'.
        """
        self.total_lookups += 1
        sim_threshold = threshold if threshold is not None else settings.cache_semantic_similarity_threshold

        # 1. Check Exact Cache
        if settings.cache_exact_enabled:
            key = self._hash_key(prompt, system_prompt)
            if key in self.exact_cache:
                entry = self.exact_cache[key]
                entry.hits += 1
                self.exact_hits += 1
                return entry, "exact_hit", 1.0

        # 2. Check Semantic Vector Cache
        if settings.cache_semantic_enabled and self.semantic_entries:
            query_vec = _get_embedding_vector(f"{system_prompt or ''} {prompt}".strip())
            best_score = 0.0
            best_entry: Optional[CacheEntry] = None

            for entry in self.semantic_entries:
                score = _cosine_similarity(query_vec, entry.vector)
                if score > best_score:
                    best_score = score
                    best_entry = entry

            if best_score >= sim_threshold and best_entry is not None:
                best_entry.hits += 1
                self.semantic_hits += 1
                return best_entry, "semantic_hit", best_score

        self.misses += 1
        return None, "miss", None

    def store(
        self,
        prompt: str,
        response: str,
        model_tier: str,
        system_prompt: Optional[str] = None
    ) -> CacheEntry:
        """Stores a new prompt-response pair in both exact and semantic cache tiers."""
        entry = CacheEntry(prompt, response, model_tier, system_prompt)

        # Enforce max entries eviction (FIFO)
        if len(self.semantic_entries) >= settings.cache_max_entries:
            removed = self.semantic_entries.pop(0)
            removed_key = self._hash_key(removed.prompt, removed.system_prompt)
            self.exact_cache.pop(removed_key, None)

        key = self._hash_key(prompt, system_prompt)
        self.exact_cache[key] = entry
        self.semantic_entries.append(entry)
        return entry

    def clear(self) -> None:
        """Clears all cached entries and resets hit counters."""
        self.exact_cache.clear()
        self.semantic_entries.clear()
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0
        self.total_lookups = 0

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics for cache size, hits, and hit ratio."""
        total_hits = self.exact_hits + self.semantic_hits
        hit_ratio = round((total_hits / self.total_lookups * 100.0), 2) if self.total_lookups > 0 else 0.0
        return {
            "exact_cache_entries": len(self.exact_cache),
            "semantic_cache_entries": len(self.semantic_entries),
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "total_lookups": self.total_lookups,
            "hit_ratio_percentage": hit_ratio
        }


# Global cache singleton instance
cache_service = DualTierCache()
