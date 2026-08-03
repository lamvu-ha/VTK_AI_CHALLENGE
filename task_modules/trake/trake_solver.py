from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from retrieval.fusion.hybrid_search import HybridSearchEngine
from retrieval.query_processing.query_processor import QueryProcessor

_MIN_GAP = 3
_MAX_GAP = 500
_LAMBDA = 0.15


def _time_gap_penalty(gap: int, min_gap: int = _MIN_GAP, max_gap: int = _MAX_GAP) -> float:
    if gap < min_gap:
        return float(min_gap - gap) / min_gap
    if gap > max_gap:
        return float(gap - max_gap) / max_gap
    return 0.0


class TRAKESolver:
    """
    Solver for Task 1.3: Temporal Retrieval and Alignment of Key Events (TRAKE).
    """

    def __init__(
        self,
        search_engine: HybridSearchEngine,
        query_processor: QueryProcessor,
        min_gap: int = _MIN_GAP,
        max_gap: int = _MAX_GAP,
        penalty_lambda: float = _LAMBDA,
    ):
        self.search_engine = search_engine
        self.query_processor = query_processor
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.penalty_lambda = penalty_lambda

    def solve(
        self,
        trake_query_text: str,
        event_embeddings: List[np.ndarray],
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        event_descriptions = self.query_processor.parse_trake_query(trake_query_text)
        num_events = len(event_embeddings)

        if num_events == 0:
            return []

        event_candidates: List[List[Dict[str, Any]]] = []
        video_hit_sets: List[set] = []

        for event_idx in range(num_events):
            emb = event_embeddings[event_idx]
            desc = (event_descriptions[event_idx]
                    if event_idx < len(event_descriptions)
                    else trake_query_text)
            parsed = self.query_processor.extract_keywords_and_objects(desc)

            cands = self.search_engine.search_candidates(
                query_embedding=emb,
                query_keywords=parsed["extracted_keywords"],
                query_text=desc,
                top_k=200,
                vec_search_k=300,
            )
            event_candidates.append(cands)
            video_hit_sets.append({c["video_id"] for c in cands})

        if video_hit_sets:
            common_videos = video_hit_sets[0]
            for s in video_hit_sets[1:]:
                common_videos = common_videos & s
        else:
            common_videos = set()

        if not common_videos:
            video_count: Dict[str, int] = {}
            for hit_set in video_hit_sets:
                for vid in hit_set:
                    video_count[vid] = video_count.get(vid, 0) + 1
            threshold = max(1, num_events // 2)
            common_videos = {v for v, cnt in video_count.items() if cnt >= threshold}

        video_agg_score: Dict[str, float] = {}
        for cands in event_candidates:
            for c in cands:
                if c["video_id"] in common_videos:
                    video_agg_score[c["video_id"]] = (
                        video_agg_score.get(c["video_id"], 0.0) + c["score"]
                    )

        ranked_videos = sorted(
            common_videos,
            key=lambda v: video_agg_score.get(v, 0.0),
            reverse=True
        )

        results = []
        for video_id in ranked_videos[:min(top_k * 3, 150)]:
            per_event_frames: List[List[Tuple[int, float]]] = []
            for event_idx in range(num_events):
                frames = [
                    (c["frame_id"], c["score"])
                    for c in event_candidates[event_idx]
                    if c["video_id"] == video_id
                ]
                frames.sort(key=lambda x: x[0])
                per_event_frames.append(frames)

            aligned_frames, seq_score = self._dp_align(per_event_frames, num_events)

            if len(aligned_frames) == num_events:
                results.append({
                    "video_id": video_id,
                    "frame_ids": aligned_frames,
                    "score": seq_score + video_agg_score.get(video_id, 0.0) * 0.5,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _dp_align(
        self,
        per_event_frames: List[List[Tuple[int, float]]],
        num_events: int
    ) -> Tuple[List[int], float]:
        if not per_event_frames or not per_event_frames[0]:
            return [], 0.0

        first_frames = per_event_frames[0]
        dp: List[Tuple[float, List[int]]] = [
            (score, [fid]) for fid, score in first_frames
        ]

        for event_idx in range(1, num_events):
            curr_frames = per_event_frames[event_idx]
            if not curr_frames:
                new_dp = []
                for prev_score, prev_path in dp:
                    last_frame = prev_path[-1] if prev_path else 0
                    fallback_fid = last_frame + self.min_gap
                    new_dp.append((prev_score, prev_path + [fallback_fid]))
                dp = new_dp
                continue

            new_dp: List[Tuple[float, List[int]]] = []

            for curr_fid, curr_sim in curr_frames:
                best_score = -np.inf
                best_path: List[int] = []

                for prev_score, prev_path in dp:
                    prev_fid = prev_path[-1]
                    if curr_fid <= prev_fid:
                        continue

                    gap = curr_fid - prev_fid
                    penalty = _time_gap_penalty(gap, self.min_gap, self.max_gap)
                    candidate_score = prev_score + curr_sim - self.penalty_lambda * penalty

                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_path = prev_path + [curr_fid]

                if best_path:
                    new_dp.append((best_score, best_path))

            if new_dp:
                dp = new_dp
            else:
                new_dp = []
                for prev_score, prev_path in dp:
                    last_frame = prev_path[-1] if prev_path else 0
                    new_dp.append((prev_score, prev_path + [last_frame + self.min_gap]))
                dp = new_dp

        if not dp:
            return [], 0.0

        best_score, best_path = max(dp, key=lambda x: x[0])
        return best_path, best_score
