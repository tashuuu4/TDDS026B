import json
import os
from typing import List, Dict, Any, Optional

DATASET_PATH = os.path.join(os.path.dirname(__file__), "sample_prompts.json")


def load_benchmark_dataset(
    category: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Loads benchmark prompts from the self-contained JSON dataset.
    Supports filtering by category and slicing with a limit.
    """
    if not os.path.exists(DATASET_PATH):
        return []

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    if category and category != "all":
        data = [item for item in data if item.get("category") == category]

    if limit and limit > 0:
        data = data[:limit]

    return data


def get_dataset_categories() -> List[str]:
    """Returns a list of unique categories available in the benchmark dataset."""
    items = load_benchmark_dataset()
    categories = sorted(list(set(item["category"] for item in items if "category" in item)))
    return ["all"] + categories
