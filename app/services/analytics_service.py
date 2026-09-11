import threading
from typing import Dict, Any


class AnalyticsService:
    """Thread-safe telemetry and metrics aggregator for API optimization."""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_requests: int = 0
        self.exact_cache_hits: int = 0
        self.semantic_cache_hits: int = 0
        self.total_tokens_processed: int = 0
        self.total_tokens_saved: int = 0
        self.total_actual_cost_usd: float = 0.0
        self.total_baseline_cost_usd: float = 0.0
        self.total_latency_ms: float = 0.0

    def record_request(
        self,
        cache_status: str,
        input_tokens: int,
        tokens_saved: int,
        actual_cost: float,
        baseline_cost: float,
        latency_ms: float
    ) -> None:
        """Records an individual request's telemetry metrics."""
        with self._lock:
            self.total_requests += 1
            if cache_status == "exact_hit":
                self.exact_cache_hits += 1
            elif cache_status == "semantic_hit":
                self.semantic_cache_hits += 1

            self.total_tokens_processed += input_tokens
            self.total_tokens_saved += tokens_saved
            self.total_actual_cost_usd = round(self.total_actual_cost_usd + actual_cost, 6)
            self.total_baseline_cost_usd = round(self.total_baseline_cost_usd + baseline_cost, 6)
            self.total_latency_ms += latency_ms

    def get_summary(self) -> Dict[str, Any]:
        """Returns aggregated dashboard statistics."""
        with self._lock:
            total_hits = self.exact_cache_hits + self.semantic_cache_hits
            hit_rate = round((total_hits / self.total_requests * 100.0), 2) if self.total_requests > 0 else 0.0
            avg_latency = round((self.total_latency_ms / self.total_requests), 2) if self.total_requests > 0 else 0.0
            cost_saved = round(max(0.0, self.total_baseline_cost_usd - self.total_actual_cost_usd), 6)
            savings_pct = round((cost_saved / self.total_baseline_cost_usd * 100.0), 2) if self.total_baseline_cost_usd > 0 else 0.0

            return {
                "total_requests": self.total_requests,
                "total_cache_hits": total_hits,
                "exact_cache_hits": self.exact_cache_hits,
                "semantic_cache_hits": self.semantic_cache_hits,
                "cache_hit_rate_percentage": hit_rate,
                "total_tokens_processed": self.total_tokens_processed,
                "total_tokens_saved": self.total_tokens_saved,
                "total_actual_cost_usd": self.total_actual_cost_usd,
                "total_baseline_cost_usd": self.total_baseline_cost_usd,
                "total_cost_saved_usd": cost_saved,
                "overall_cost_reduction_percentage": savings_pct,
                "average_latency_ms": avg_latency
            }

    def reset(self) -> None:
        """Resets all metrics."""
        with self._lock:
            self.total_requests = 0
            self.exact_cache_hits = 0
            self.semantic_cache_hits = 0
            self.total_tokens_processed = 0
            self.total_tokens_saved = 0
            self.total_actual_cost_usd = 0.0
            self.total_baseline_cost_usd = 0.0
            self.total_latency_ms = 0.0


analytics_service = AnalyticsService()
