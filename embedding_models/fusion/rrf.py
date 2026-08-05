"""
Reciprocal Rank Fusion: gộp nhiều ranked list thành 1 ranking duy nhất.
"""
from typing import List, Dict, Tuple


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
) -> Dict[str, float]:
    """
    Gộp nhiều ranked list bằng RRF.
    
    Args:
        ranked_lists: list of [(doc_id, score), ...] đã sắp theo điểm giảm dần
        k: hằng số RRF (default=60)
    Returns:
        {doc_id: rrf_score} — score càng cao càng tốt
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def rrf_sort(ranked_lists: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
    """Trả về danh sách (doc_id, rrf_score) đã sắp xếp giảm dần."""
    scores = reciprocal_rank_fusion(ranked_lists, k)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
