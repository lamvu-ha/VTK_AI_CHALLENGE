"""
Ranking Optimizer for AIC 2026 submission.

Final Score formula (from competition rules):
  Final_Score = 1/5 * (R@1 + R@5 + R@20 + R@50 + R@100)

Score impact table:
  Rank 1        → Final Score contribution: 1.00
  Rank 2-5      → 0.80
  Rank 6-20     → 0.60
  Rank 21-50    → 0.40
  Rank 51-100   → 0.20

Strategy:
1. R@1 Protection: Top-1 must be the highest-confidence prediction (by combined score)
2. MMR Diversity (Maximal Marginal Relevance): Positions 2-100 diversified to maximize coverage
3. Video-level Coverage Boost: Ensure multiple distinct videos appear in top-20
4. Deduplication: No duplicate (video_id, frame_id) pairs
"""
from typing import List, Dict, Any, Set, Tuple
import math


def _normalize_scores(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Min-max normalize scores to [0, 1] range."""
    if not candidates:
        return candidates
    scores = [c.get("score", 0.0) for c in candidates]
    min_s, max_s = min(scores), max(scores)
    span = max_s - min_s
    if span < 1e-9:
        return candidates
    for c in candidates:
        c["norm_score"] = (c.get("score", 0.0) - min_s) / span
    return candidates


def _mmr_rerank(
    candidates: List[Dict[str, Any]],
    lambda_mmr: float = 0.7,
    max_items: int = 100,
) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance (MMR) re-ranking to balance relevance and diversity.
    
    MMR(d) = lambda * relevance(d) - (1 - lambda) * max_sim_to_selected(d)
    
    We use video_id as the "similarity" measure: a document is similar to selected
    documents if it shares the same video_id (already in the output list).
    
    Args:
        lambda_mmr: trade-off between relevance (higher) and diversity (lower)
        max_items: output list length
    """
    if not candidates:
        return []

    candidates = _normalize_scores(candidates)

    # Sort by score descending — first pick is always highest score (R@1 protection)
    sorted_cands = sorted(candidates, key=lambda x: x.get("norm_score", x.get("score", 0.0)), reverse=True)

    selected: List[Dict[str, Any]] = []
    selected_video_ids: List[str] = []
    remaining = list(sorted_cands)

    while remaining and len(selected) < max_items:
        if len(selected) == 0:
            # Always pick top-score as rank 1 (R@1 protection)
            best = remaining[0]
        else:
            # MMR scoring for remaining candidates
            best = None
            best_mmr = -math.inf
            for cand in remaining:
                rel = cand.get("norm_score", cand.get("score", 0.0))
                # Diversity penalty: count how many selected have same video_id
                same_video_count = selected_video_ids.count(cand["video_id"])
                # Simulate redundancy as fraction of selected already from same video
                n_selected = len(selected)
                redundancy = same_video_count / n_selected if n_selected > 0 else 0.0
                mmr_score = lambda_mmr * rel - (1.0 - lambda_mmr) * redundancy
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best = cand

        if best is None:
            break

        selected.append(best)
        selected_video_ids.append(best["video_id"])
        remaining.remove(best)

    return selected


class RankingOptimizer:
    """
    Final submission ranking optimizer for AIC 2026.
    
    Pipeline:
    1. Deduplicate candidates
    2. Normalize scores
    3. Protect R@1 (highest confidence always at rank 1)
    4. Apply MMR diversity for ranks 2-100
    5. Return exactly max_items entries
    """

    def __init__(self, lambda_mmr: float = 0.7):
        self.lambda_mmr = lambda_mmr

    def optimize_ranking(
        self,
        candidates: List[Dict[str, Any]],
        max_items: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Optimized ranking pipeline for AIC 2026 Final Score maximization.
        
        Args:
            candidates: raw result list from solvers (with 'score' field)
            max_items: max 100 per AIC rules

        Returns:
            Re-ranked, deduplicated list of up to max_items candidates
        """
        if not candidates:
            return []

        # Step 1: Deduplicate based on unique (video_id, frame_id) or (video_id, frame_ids)
        deduped = self._deduplicate(candidates)

        # Step 2: MMR re-ranking with R@1 protection
        ranked = _mmr_rerank(deduped, lambda_mmr=self.lambda_mmr, max_items=max_items)

        return ranked[:max_items]

    def _deduplicate(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove exact duplicates while preserving rank order (highest score first)."""
        sorted_cands = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
        seen: Set[Tuple] = set()
        deduped: List[Dict[str, Any]] = []

        for c in sorted_cands:
            if "answer" in c:
                key = (c["video_id"], c.get("frame_id", -1), str(c["answer"]).strip().lower())
            elif "frame_ids" in c:
                key = (c["video_id"], tuple(c["frame_ids"]))
            else:
                key = (c["video_id"], c.get("frame_id", -1))

            if key not in seen:
                seen.add(key)
                deduped.append(c)

        return deduped
