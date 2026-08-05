"""
Video selector: tìm 1 video khớp nhất với toàn bộ chuỗi event TRAKE.
Dùng max-pooling hoặc embedding trung bình qua các event.
"""
import numpy as np
from typing import List, Dict, Any, Optional


class VideoSelector:
    """
    Tìm video khớp nhất với chuỗi event embeddings.
    """
    def __init__(self, feature_indexer):
        self.feature_indexer = feature_indexer

    def select_best_video(
        self,
        event_embeddings: List[np.ndarray],
        top_k_videos: int = 10,
        strategy: str = "max_pool",  # "max_pool" | "avg_pool"
    ) -> List[Dict[str, Any]]:
        """
        Tổng hợp event embeddings → query embedding → tìm video.
        
        Args:
            event_embeddings: list embedding cho từng event
            top_k_videos: số video trả về
            strategy: cách gộp embedding
        Returns:
            [{video_id, score}] sắp theo score giảm dần
        """
        if not event_embeddings:
            return []

        stacked = np.stack(event_embeddings, axis=0)
        if strategy == "max_pool":
            query_emb = stacked.max(axis=0)
        else:
            query_emb = stacked.mean(axis=0)
        norm = np.linalg.norm(query_emb)
        if norm > 1e-9:
            query_emb = query_emb / norm

        raw = self.feature_indexer.search(query_emb, top_k=top_k_videos * 10)

        # Aggregate per video (max score)
        video_scores: Dict[str, float] = {}
        for info, score in raw:
            vid = info["video_id"]
            if vid not in video_scores or video_scores[vid] < score:
                video_scores[vid] = float(score)

        ranked = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"video_id": v, "score": s} for v, s in ranked[:top_k_videos]]
