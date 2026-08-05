"""
Weighted score fusion: gộp ranked list bằng trọng số điểm cosine similarity.
"""
from typing import List, Dict, Tuple


def weighted_score_fusion(
    scored_lists: List[Tuple[List[Tuple[str, float]], float]],
) -> Dict[str, float]:
    """
    Gộp nhiều list bằng trọng số.
    
    Args:
        scored_lists: [([(doc_id, score), ...], weight), ...]
    Returns:
        {doc_id: combined_score}
    """
    combined: Dict[str, float] = {}
    total_weight = sum(w for _, w in scored_lists)
    if total_weight == 0:
        return combined
    for ranked, weight in scored_lists:
        for doc_id, score in ranked:
            combined[doc_id] = combined.get(doc_id, 0.0) + weight * score / total_weight
    return combined


def weighted_sort(
    scored_lists: List[Tuple[List[Tuple[str, float]], float]],
) -> List[Tuple[str, float]]:
    """Trả về danh sách (doc_id, score) đã sắp giảm dần."""
    scores = weighted_score_fusion(scored_lists)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
