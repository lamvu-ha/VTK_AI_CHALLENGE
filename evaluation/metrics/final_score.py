"""
Final Score metric theo mục 2.2 thể lệ AIC 2026.
Final Score = average R@{1, 5, 20, 50, 100}
"""
from typing import List, Dict, Any
from evaluation.metrics.r_score import calculate_r_score

_K_VALS = [1, 5, 20, 50, 100]


def r_at_k(
    query_type: str,
    ground_truth: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    k: int,
) -> float:
    """
    R@k = max R-score trong top-k predictions.
    """
    if not predictions:
        return 0.0
    top_k = predictions[:k]
    return max(calculate_r_score(query_type, ground_truth, predictions, rank=i + 1)
               for i in range(len(top_k)))


def final_score(
    query_type: str,
    ground_truth: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    k_vals: List[int] = _K_VALS,
) -> Dict[str, float]:
    """
    Tính R@k cho từng k và Final Score.
    Returns: {"R@1": ..., "R@5": ..., "final": ...}
    """
    scores = {f"R@{k}": r_at_k(query_type, ground_truth, predictions, k) for k in k_vals}
    scores["final"] = sum(scores.values()) / len(k_vals) if k_vals else 0.0
    return scores


def calculate_final_score(r_k_dict: Dict[int, float]) -> float:
    """
    Tính Final Score từ dict {k: R@k}.
    Tương thích với placeholder cũ.
    """
    scores = [r_k_dict.get(k, 0.0) for k in _K_VALS]
    return sum(scores) / len(_K_VALS) if _K_VALS else 0.0
