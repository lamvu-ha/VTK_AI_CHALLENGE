"""
Enhanced TRAKE Solver using Dynamic Programming with Time-Gap Penalty.

Design doc spec:
- Decompose query into N sub-events
- Run N independent searches → candidate lists R_1 ... R_N
- Filter videos appearing in all N lists
- For each candidate video, find frame sequence f_1 < f_2 < ... < f_N
  using DP with time-gap penalty:
    score(v) = sum_i sim(e_i, f_i) - λ * sum_i penalty(gap(f_i, f_{i+1}))
- Penalty: gap too small (< 3 frames) or too large (> 500 frames) is penalized

Final output: up to top_k (video_id, frame_ids, score) tuples, sorted by score.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.query_processor import QueryProcessor

# Time-gap penalty config (per design doc)
_MIN_GAP = 3       # minimum reasonable frame gap between events
_MAX_GAP = 500     # maximum reasonable frame gap between events
_LAMBDA = 0.15     # penalty coefficient


def _time_gap_penalty(gap: int, min_gap: int = _MIN_GAP, max_gap: int = _MAX_GAP) -> float:
    """
    Computes time-gap penalty for a frame gap between consecutive events.
    Returns 0 for reasonable gaps, positive penalty for violations.
    """
    if gap < min_gap:
        # Too close: possibly same scene, not distinct events
        return float(min_gap - gap) / min_gap
    if gap > max_gap:
        # Too far: exceeds natural motion temporal window
        return float(gap - max_gap) / max_gap
    return 0.0


class TRAKESolver:
    """
    Solver for Task 1.3: Temporal Retrieval and Alignment of Key Events (TRAKE).
    
    Improvements over baseline:
    - Dynamic Programming for optimal sequence alignment (vs greedy)
    - Time-gap penalty discourages implausible frame distance
    - RRF-based video scoring (via HybridSearchEngine)
    - Overlap-intersection filtering: only considers videos present in ALL event lists
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
        """
        Solves TRAKE query for a sequence of events.
        
        Args:
            trake_query_text: full TRAKE query string
            event_embeddings: list of N embeddings, one per sub-event
            top_k: max candidates to return

        Returns:
            List of dicts with video_id, frame_ids (List[int]), score
        """
        event_descriptions = self.query_processor.parse_trake_query(trake_query_text)
        num_events = len(event_embeddings)

        if num_events == 0:
            return []

        # ── Step 1: Search per-event candidate lists ──────────────────────
        event_candidates: List[List[Dict[str, Any]]] = []
        video_hit_sets: List[set] = []   # which videos appear in each event's results

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

        # ── Step 2: Find videos appearing in ALL event candidate lists ──────
        # (intersection filter from design doc)
        if video_hit_sets:
            common_videos = video_hit_sets[0]
            for s in video_hit_sets[1:]:
                common_videos = common_videos & s
        else:
            common_videos = set()

        # If intersection is empty, fall back to videos in majority of lists
        if not common_videos:
            # Count how many event lists each video appears in
            video_count: Dict[str, int] = {}
            for hit_set in video_hit_sets:
                for vid in hit_set:
                    video_count[vid] = video_count.get(vid, 0) + 1
            # Use videos appearing in at least ceil(N/2) lists
            threshold = max(1, num_events // 2)
            common_videos = {v for v, cnt in video_count.items() if cnt >= threshold}

        # ── Step 3: Score videos by aggregate RRF score ──────────────────
        video_agg_score: Dict[str, float] = {}
        for cands in event_candidates:
            for c in cands:
                if c["video_id"] in common_videos:
                    video_agg_score[c["video_id"]] = (
                        video_agg_score.get(c["video_id"], 0.0) + c["score"]
                    )

        # Sort candidate videos by aggregate score
        ranked_videos = sorted(
            common_videos,
            key=lambda v: video_agg_score.get(v, 0.0),
            reverse=True
        )

        # ── Step 4: DP alignment per video ────────────────────────────────
        results = []
        for video_id in ranked_videos[:min(top_k * 3, 150)]:
            # Collect frame candidates per event for this video
            per_event_frames: List[List[Tuple[int, float]]] = []
            for event_idx in range(num_events):
                frames = [
                    (c["frame_id"], c["score"])
                    for c in event_candidates[event_idx]
                    if c["video_id"] == video_id
                ]
                frames.sort(key=lambda x: x[0])
                per_event_frames.append(frames)

            # DP alignment with time-gap penalty
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
        """
        Dynamic Programming for optimal monotonically-increasing frame sequence.
        Maximizes: sum(sim_scores) - lambda * sum(time_gap_penalties)
        
        State: dp[i][j] = (best_score, chosen_frame) for event i choosing frame j
        Transition: dp[i][j] = max over all k where frame_k < frame_j:
                      dp[i-1][k].score + sim(event_i, frame_j) - lambda * penalty(frame_j - frame_k)
        """
        if not per_event_frames or not per_event_frames[0]:
            return [], 0.0

        # Initialize DP for first event
        # dp[j] = (best cumulative score, [chosen frame ids so far])
        first_frames = per_event_frames[0]
        dp: List[Tuple[float, List[int]]] = [
            (score, [fid]) for fid, score in first_frames
        ]

        for event_idx in range(1, num_events):
            curr_frames = per_event_frames[event_idx]
            if not curr_frames:
                # No candidates for this event – use fallback
                new_dp = []
                for prev_score, prev_path in dp:
                    last_frame = prev_path[-1] if prev_path else 0
                    fallback_fid = last_frame + self.min_gap
                    new_dp.append((prev_score, prev_path + [fallback_fid]))
                dp = new_dp
                continue

            new_dp: List[Tuple[float, List[int]]] = []
            # For each current frame candidate, find best previous state
            prev_frame_arr = np.array([fid for fid, _ in first_frames if event_idx == 1] or
                                       [path[-1] for _, path in dp])

            for curr_fid, curr_sim in curr_frames:
                best_score = -np.inf
                best_path: List[int] = []

                for prev_score, prev_path in dp:
                    prev_fid = prev_path[-1]
                    if curr_fid <= prev_fid:
                        continue  # must be strictly increasing

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
                # Fallback: extend each previous path with a dummy frame
                new_dp = []
                for prev_score, prev_path in dp:
                    last_frame = prev_path[-1] if prev_path else 0
                    new_dp.append((prev_score, prev_path + [last_frame + self.min_gap]))
                dp = new_dp

        if not dp:
            return [], 0.0

        # Pick the path with highest score
        best_score, best_path = max(dp, key=lambda x: x[0])
        return best_path, best_score
