"""
Beam search aligner: tìm chuỗi frame khớp nhất với thứ tự event trong video.
Temporal decay phạt các lựa chọn phá vỡ trật tự thời gian.
"""
import numpy as np
from typing import List, Dict, Any, Tuple


class BeamSearchAligner:
    """
    Beam search qua shot/keyframe để tìm chuỗi frame khớp với event sequence.
    """
    def __init__(self, beam_width: int = 5, min_gap: int = 3, max_gap: int = 500, decay: float = 0.15):
        self.beam_width = beam_width
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.decay = decay

    def _penalty(self, gap: int) -> float:
        if gap < self.min_gap:
            return (self.min_gap - gap) / self.min_gap
        if gap > self.max_gap:
            return (gap - self.max_gap) / self.max_gap
        return 0.0

    def align(
        self,
        per_event_candidates: List[List[Tuple[int, float]]],
    ) -> Tuple[List[int], float]:
        """
        Args:
            per_event_candidates: [(frame_id, score)] cho từng event, đã lọc theo video
        Returns:
            (best_frame_sequence, best_score)
        """
        if not per_event_candidates or not per_event_candidates[0]:
            return [], 0.0

        # Beam: [(score, [frame_ids])]
        beam: List[Tuple[float, List[int]]] = [
            (score, [fid]) for fid, score in per_event_candidates[0][:self.beam_width]
        ]

        for event_cands in per_event_candidates[1:]:
            new_beam: List[Tuple[float, List[int]]] = []
            for curr_fid, curr_score in event_cands:
                for prev_score, prev_path in beam:
                    prev_fid = prev_path[-1]
                    if curr_fid <= prev_fid:
                        continue
                    gap = curr_fid - prev_fid
                    pen = self._penalty(gap)
                    score = prev_score + curr_score - self.decay * pen
                    new_beam.append((score, prev_path + [curr_fid]))

            if new_beam:
                new_beam.sort(key=lambda x: x[0], reverse=True)
                beam = new_beam[:self.beam_width]

        if not beam:
            return [], 0.0
        best_score, best_path = beam[0]
        return best_path, best_score
