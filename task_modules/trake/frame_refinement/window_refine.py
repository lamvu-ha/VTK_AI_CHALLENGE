"""
Window refinement: trượt cửa sổ ±N frame quanh keyframe ứng viên,
chọn frame khớp nhất với mô tả event.
"""
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class WindowRefiner:
    """
    Với mỗi keyframe ứng viên, slide window ±window_size frame
    và chọn frame có cosine similarity cao nhất với event embedding.
    """
    def __init__(self, encoder, window_size: int = 5):
        """
        Args:
            encoder: object có phương thức encode_image(image_path) → np.ndarray
            window_size: số frame mỗi chiều để search
        """
        self.encoder = encoder
        self.window_size = window_size

    def refine(
        self,
        keyframes_dir: str,
        video_id: str,
        candidate_frame_id: int,
        event_embedding: np.ndarray,
        total_frames: Optional[int] = None,
    ) -> Tuple[int, float]:
        """
        Tìm frame tốt nhất trong cửa sổ [candidate - window, candidate + window].
        Returns: (best_frame_id, best_score)
        """
        lo = max(0, candidate_frame_id - self.window_size)
        hi = candidate_frame_id + self.window_size
        if total_frames:
            hi = min(hi, total_frames - 1)

        best_fid, best_score = candidate_frame_id, -1.0
        for fid in range(lo, hi + 1):
            img_path = os.path.join(keyframes_dir, video_id, f"{fid:06d}.jpg")
            if not os.path.exists(img_path):
                continue
            try:
                emb = self.encoder.encode_image(img_path)
                score = float(np.dot(emb, event_embedding) /
                              (np.linalg.norm(emb) * np.linalg.norm(event_embedding) + 1e-9))
                if score > best_score:
                    best_score = score
                    best_fid = fid
            except Exception:
                continue
        return best_fid, best_score

    def refine_sequence(
        self,
        keyframes_dir: str,
        video_id: str,
        candidate_frame_ids: List[int],
        event_embeddings: List[np.ndarray],
    ) -> List[Tuple[int, float]]:
        """Refine từng frame trong chuỗi event."""
        return [
            self.refine(keyframes_dir, video_id, fid, emb)
            for fid, emb in zip(candidate_frame_ids, event_embeddings)
        ]
