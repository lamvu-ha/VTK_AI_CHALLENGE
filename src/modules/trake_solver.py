from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.query_processor import QueryProcessor

class TRAKESolver:
    """
    Solver for Task 1.3: Temporal Retrieval and Alignment of Key Events (TRAKE).
    Stage 1: Video Retrieval - Find best candidate video containing the event sequence.
    Stage 2: Keyframe Alignment - Find ordered semantic keyframes <frame_id_1, ..., frame_id_n>.
    """

    def __init__(self, search_engine: HybridSearchEngine, query_processor: QueryProcessor):
        self.search_engine = search_engine
        self.query_processor = query_processor

    def solve(self, trake_query_text: str, event_embeddings: List[np.ndarray], top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Solves TRAKE query for a sequence of events.
        `event_embeddings` is a list of N vectors corresponding to the N sub-events in the sequence.
        Returns top_k candidates, each containing: video_id, keyframe_sequence: List[int], score.
        """
        event_descriptions = self.query_processor.parse_trake_query(trake_query_text)
        num_events = len(event_embeddings)

        # Step 1: Collect candidates for each event in the sequence
        event_candidates: List[List[Dict[str, Any]]] = []
        video_scores: Dict[str, float] = {}

        for event_idx in range(num_events):
            emb = event_embeddings[event_idx]
            desc = event_descriptions[event_idx] if event_idx < len(event_descriptions) else trake_query_text
            parsed = self.query_processor.extract_keywords_and_objects(desc)

            cands = self.search_engine.search_candidates(
                query_embedding=emb,
                query_keywords=parsed["extracted_keywords"],
                top_k=200
            )
            event_candidates.append(cands)

            # Accumulate scores per video to find candidate videos containing all events
            for c in cands:
                v_id = c["video_id"]
                video_scores[v_id] = video_scores.get(v_id, 0.0) + c["score"]

        # Sort videos by total aggregate score
        ranked_videos = sorted(video_scores.keys(), key=lambda v: video_scores[v], reverse=True)

        results = []
        # Step 2: Temporal alignment within top candidate videos
        for video_id in ranked_videos[:top_k]:
            # Filter candidates belonging to this video for each event stage
            per_event_frames: List[List[Tuple[int, float]]] = []
            for event_idx in range(num_events):
                frames = [
                    (c["frame_id"], c["score"]) 
                    for c in event_candidates[event_idx] 
                    if c["video_id"] == video_id
                ]
                # Sort by frame_id for monotonic alignment search
                frames.sort(key=lambda x: x[0])
                per_event_frames.append(frames)

            # Simple Greedy Sequential Alignment: find frame_id_1 < frame_id_2 < ... < frame_id_N
            aligned_frames, seq_score = self._align_monotonic_sequence(per_event_frames, num_events)

            if len(aligned_frames) == num_events:
                results.append({
                    "video_id": video_id,
                    "frame_ids": aligned_frames,
                    "score": seq_score + video_scores[video_id]
                })

        # Sort final TRAKE submissions by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _align_monotonic_sequence(self, per_event_frames: List[List[Tuple[int, float]]], num_events: int) -> Tuple[List[int], float]:
        """
        Greedy/Dynamic Search for strictly increasing frame IDs: f1 < f2 < ... < fN with max score.
        """
        chosen_frames = []
        total_score = 0.0
        last_frame = -1

        for event_idx in range(num_events):
            candidates = per_event_frames[event_idx]
            valid_candidates = [c for c in candidates if c[0] > last_frame]

            if not valid_candidates:
                # Fallback: pick last_frame + 1 if no candidate frame strictly greater
                next_f = last_frame + 1
                chosen_frames.append(next_f)
                last_frame = next_f
            else:
                # Pick highest scoring candidate that satisfies f > last_frame
                best_cand = max(valid_candidates, key=lambda x: x[1])
                chosen_frames.append(best_cand[0])
                total_score += best_cand[1]
                last_frame = best_cand[0]

        return chosen_frames, total_score
