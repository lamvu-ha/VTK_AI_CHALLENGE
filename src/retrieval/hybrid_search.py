"""
Hybrid Search Engine with Reciprocal Rank Fusion (RRF) for AIC 2026.

Architecture (from design doc):
  Stage 1 (Broad Retrieval):
    - Dense vector search (CLIP/SigLIP features via FeatureIndexer)
    - Sparse BM25 text search (video metadata via MetadataIndexer/BM25Engine)
    - Merge via RRF: score(d) = sum_r 1/(k + rank_r(d)), k=60

  Stage 2 (Reranking) [future: Cross-Encoder / VLM]:
    - Candidates re-scored by VLM when available

Final Score formula: Final = 1/5 * (R@1 + R@5 + R@20 + R@50 + R@100)
→ Critical to place correct answer at Rank 1.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.data.feature_indexer import FeatureIndexer
from src.data.metadata_indexer import MetadataIndexer

# RRF constant (standard k=60 from research)
_RRF_K = 60


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],  # each list is [(video_id, score), ...]
    k: int = _RRF_K
) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.
    RRF_score(d) = sum_r 1/(k + rank_r(d))
    where rank_r(d) is 1-indexed position of document d in ranked list r.
    """
    rrf_scores: Dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank_idx, (doc_id, _) in enumerate(ranked_list):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank_idx + 1)
    return rrf_scores


class HybridSearchEngine:
    """
    Hybrid Search Engine combining:
    1. Dense vector search (CLIP/SigLIP cosine similarity)
    2. Sparse BM25 keyword search (video title/description/tags)
    Merged via Reciprocal Rank Fusion (RRF).
    
    Returns candidates with combined RRF score + per-source scores for debugging.
    """

    def __init__(
        self,
        feature_indexer: FeatureIndexer,
        metadata_indexer: Optional[MetadataIndexer] = None,
        rrf_k: int = _RRF_K
    ):
        self.feature_indexer = feature_indexer
        self.metadata_indexer = metadata_indexer
        self.rrf_k = rrf_k

    def search_candidates(
        self,
        query_embedding: np.ndarray,
        query_keywords: Optional[List[str]] = None,
        query_text: Optional[str] = None,
        top_k: int = 100,
        vec_search_k: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid RRF search and returns ranked candidate keyframe objects.
        
        Args:
            query_embedding: normalized query vector (512-dim CLIP/SigLIP)
            query_keywords: list of extracted keywords for BM25 search
            query_text: raw query text (used if query_keywords is None)
            top_k: number of final candidates to return
            vec_search_k: how many candidates to retrieve from vector search

        Returns:
            List of dicts with: video_id, frame_id, score (RRF), vector_score, bm25_score, pts_time, fps
        """
        # ── Step 1: Dense vector search ──────────────────────────────────────
        vec_results = self.feature_indexer.search(query_embedding, top_k=vec_search_k)
        
        # Build video-level ranked list for RRF (use best score per video first)
        # Also keep frame-level info for final output
        frame_vec_scores: Dict[str, Dict[str, Any]] = {}  # (video_id, frame_id) -> info
        vec_video_ranked: List[Tuple[str, float]] = []     # [(video_id, best_score), ...]
        seen_videos_vec: Dict[str, float] = {}

        for keyframe_info, vec_score in vec_results:
            v_id = keyframe_info["video_id"]
            f_id = keyframe_info["frame_id"]
            key = f"{v_id}|{f_id}"
            frame_vec_scores[key] = {
                "video_id": v_id,
                "frame_id": f_id,
                "vector_score": vec_score,
                "pts_time": keyframe_info.get("pts_time", 0.0),
                "fps": keyframe_info.get("fps", 30.0),
            }
            if v_id not in seen_videos_vec or seen_videos_vec[v_id] < vec_score:
                seen_videos_vec[v_id] = vec_score

        # Rank videos by their best frame score for RRF
        vec_video_ranked = sorted(seen_videos_vec.items(), key=lambda x: x[1], reverse=True)

        # ── Step 2: BM25 / keyword search ──────────────────────────────────
        bm25_video_ranked: List[Tuple[str, float]] = []
        bm25_scores_map: Dict[str, float] = {}

        if self.metadata_indexer is not None:
            # Prefer query_text for BM25 (richer signal), fall back to keywords
            search_text = query_text or (" ".join(query_keywords) if query_keywords else "")
            if search_text:
                bm25_scores_map = self.metadata_indexer.search_bm25(search_text, top_k=vec_search_k)
                bm25_video_ranked = sorted(bm25_scores_map.items(), key=lambda x: x[1], reverse=True)

        # ── Step 3: Reciprocal Rank Fusion ──────────────────────────────────
        ranked_lists = [vec_video_ranked]
        if bm25_video_ranked:
            ranked_lists.append(bm25_video_ranked)

        rrf_video_scores = _reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)

        if not rrf_video_scores:
            return []

        # ── Step 4: Expand to frame-level candidates ────────────────────────
        # For each video in RRF results, take all matching frames from vector search
        candidates: List[Dict[str, Any]] = []
        for key, frame_info in frame_vec_scores.items():
            v_id = frame_info["video_id"]
            rrf_score = rrf_video_scores.get(v_id, 0.0)
            bm25_score = bm25_scores_map.get(v_id, 0.0)

            candidates.append({
                "video_id": v_id,
                "frame_id": frame_info["frame_id"],
                "score": rrf_score,
                "vector_score": frame_info["vector_score"],
                "bm25_score": bm25_score,
                "pts_time": frame_info["pts_time"],
                "fps": frame_info["fps"],
                "path": "",
            })

        # Sort by RRF score desc, then by vector_score as tiebreaker
        candidates.sort(key=lambda x: (x["score"], x["vector_score"]), reverse=True)
        return candidates[:top_k]

    def search_candidates_multi(
        self,
        query_embeddings: List[np.ndarray],
        query_text: Optional[str] = None,
        query_keywords: Optional[List[str]] = None,
        top_k: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Multi-embedding search: run search for each embedding, merge via RRF.
        Used when multiple query variants (multi-query expansion) are provided.
        """
        if not query_embeddings:
            return []

        # Collect per-embedding ranked lists
        all_ranked_lists: List[List[Tuple[str, float]]] = []
        # For frame info, use the first embedding's results as reference
        primary_candidates = None

        for i, emb in enumerate(query_embeddings):
            vec_results = self.feature_indexer.search(emb, top_k=200)
            # Build video-ranked list for this embedding
            seen: Dict[str, float] = {}
            for kf_info, score in vec_results:
                vid = kf_info["video_id"]
                if vid not in seen or seen[vid] < score:
                    seen[vid] = score
            ranked = sorted(seen.items(), key=lambda x: x[1], reverse=True)
            all_ranked_lists.append(ranked)

        # BM25 list
        bm25_scores_map: Dict[str, float] = {}
        if self.metadata_indexer is not None:
            search_text = query_text or (" ".join(query_keywords) if query_keywords else "")
            if search_text:
                bm25_scores_map = self.metadata_indexer.search_bm25(search_text, top_k=200)
                bm25_ranked = sorted(bm25_scores_map.items(), key=lambda x: x[1], reverse=True)
                if bm25_ranked:
                    all_ranked_lists.append(bm25_ranked)

        rrf_scores = _reciprocal_rank_fusion(all_ranked_lists, k=self.rrf_k)

        # Use primary embedding (first) to get frame-level info
        primary_results = self.feature_indexer.search(query_embeddings[0], top_k=200)
        seen_keys = set()
        candidates = []
        for kf_info, vec_score in primary_results:
            v_id = kf_info["video_id"]
            f_id = kf_info["frame_id"]
            key = f"{v_id}|{f_id}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append({
                "video_id": v_id,
                "frame_id": f_id,
                "score": rrf_scores.get(v_id, 0.0),
                "vector_score": vec_score,
                "bm25_score": bm25_scores_map.get(v_id, 0.0),
                "pts_time": kf_info.get("pts_time", 0.0),
                "fps": kf_info.get("fps", 30.0),
                "path": "",
            })

        candidates.sort(key=lambda x: (x["score"], x["vector_score"]), reverse=True)
        return candidates[:top_k]
