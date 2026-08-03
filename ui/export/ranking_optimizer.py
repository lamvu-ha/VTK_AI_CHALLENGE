from typing import List, Dict, Any, Set, Tuple
import math


def _normalize_scores(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    if not candidates:
        return []

    candidates = _normalize_scores(candidates)

    sorted_cands = sorted(candidates, key=lambda x: x.get("norm_score", x.get("score", 0.0)), reverse=True)

    selected: List[Dict[str, Any]] = []
    selected_video_ids: List[str] = []
    remaining = list(sorted_cands)

    while remaining and len(selected) < max_items:
        if len(selected) == 0:
            best = remaining[0]
        else:
            best = None
            best_mmr = -math.inf
            for cand in remaining:
                rel = cand.get("norm_score", cand.get("score", 0.0))
                same_video_count = selected_video_ids.count(cand["video_id"])
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
    """

    def __init__(self, lambda_mmr: float = 0.7):
        self.lambda_mmr = lambda_mmr

    def optimize_ranking(
        self,
        candidates: List[Dict[str, Any]],
        max_items: int = 100
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        deduped = self._deduplicate(candidates)
        ranked = _mmr_rerank(deduped, lambda_mmr=self.lambda_mmr, max_items=max_items)
        return ranked[:max_items]

    def _deduplicate(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
