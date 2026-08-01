from typing import List, Dict, Any

class RankingOptimizer:
    """
    Submission Ranking Optimizer to maximize Final Score defined in AIC 2026 rules.
    Formula: Final Score = 1/5 * sum_{k in {1, 5, 20, 50, 100}} R@k
    Ensures highest-confidence candidates are strictly placed at top rank positions (rank 1).
    """

    def __init__(self):
        pass

    def optimize_ranking(self, candidates: List[Dict[str, Any]], max_items: int = 100) -> List[Dict[str, Any]]:
        """
        Sorts candidates by confidence score descending and trims to exactly max_items (<= 100).
        """
        if not candidates:
            return []

        # Sort strictly descending by score
        sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)

        # Deduplicate while preserving rank order
        seen = set()
        deduped = []
        for c in sorted_candidates:
            # Identifier based on task type
            if "answer" in c:
                key = (c["video_id"], c["frame_id"], str(c["answer"]).strip().lower())
            elif "frame_ids" in c:
                key = (c["video_id"], tuple(c["frame_ids"]))
            else:
                key = (c["video_id"], c["frame_id"])

            if key not in seen:
                seen.add(key)
                deduped.append(c)

        return deduped[:max_items]
