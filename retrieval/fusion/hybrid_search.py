from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from indexing.vector_db.faiss_backup.feature_indexer import FeatureIndexer
from indexing.text_search.metadata_indexer import MetadataIndexer

_RRF_K = 60


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = _RRF_K
) -> Dict[str, float]:
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
    """

    def __init__(
        self,
        feature_indexer: FeatureIndexer,
        metadata_indexer: Optional[MetadataIndexer] = None,
        rrf_k: int = _RRF_K,
        clip_encoder: Optional[Any] = None,
    ):
        self.feature_indexer = feature_indexer
        self.metadata_indexer = metadata_indexer
        self.rrf_k = rrf_k
        self.clip_encoder = clip_encoder

    def search_candidates(
        self,
        query_embedding: np.ndarray,
        query_keywords: Optional[List[str]] = None,
        query_text: Optional[str] = None,
        top_k: int = 100,
        vec_search_k: int = 200,
    ) -> List[Dict[str, Any]]:
        vec_results = self.feature_indexer.search(query_embedding, top_k=vec_search_k)
        
        frame_vec_scores: Dict[str, Dict[str, Any]] = {}
        vec_video_ranked: List[Tuple[str, float]] = []
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

        vec_video_ranked = sorted(seen_videos_vec.items(), key=lambda x: x[1], reverse=True)

        bm25_video_ranked: List[Tuple[str, float]] = []
        bm25_scores_map: Dict[str, float] = {}

        if self.metadata_indexer is not None:
            search_text = query_text or (" ".join(query_keywords) if query_keywords else "")
            if search_text:
                bm25_scores_map = self.metadata_indexer.search_bm25(search_text, top_k=vec_search_k)
                bm25_video_ranked = sorted(bm25_scores_map.items(), key=lambda x: x[1], reverse=True)

        ranked_lists = [vec_video_ranked]
        if bm25_video_ranked:
            ranked_lists.append(bm25_video_ranked)

        rrf_video_scores = _reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)

        if not rrf_video_scores:
            return []

        candidates: List[Dict[str, Any]] = []
        seen_frame_keys = set(frame_vec_scores.keys())

        # Scale RRF score so rank-1 RRF = 1.0 instead of 0.016
        max_rrf_possible = 1.0 / (self.rrf_k + 1)

        # Include keyframes from FAISS vector search
        for key, frame_info in frame_vec_scores.items():
            v_id = frame_info["video_id"]
            rrf_raw = rrf_video_scores.get(v_id, 0.0)
            norm_rrf = rrf_raw / max_rrf_possible if max_rrf_possible > 0 else 0.0
            bm25_score = bm25_scores_map.get(v_id, 0.0)
            vec_score = frame_info["vector_score"]

            # Hybrid score: vector similarity boosted up to +25% by text RRF rank
            final_score = vec_score * (1.0 + 0.25 * norm_rrf)

            candidates.append({
                "video_id": v_id,
                "frame_id": frame_info["frame_id"],
                "score": final_score,
                "vector_score": vec_score,
                "bm25_score": bm25_score,
                "pts_time": frame_info["pts_time"],
                "fps": frame_info["fps"],
                "path": "",
            })

        # Also include keyframes from top BM25 videos that weren't in vector top-k
        if self.feature_indexer and hasattr(self.feature_indexer, "keyframe_map") and self.feature_indexer.keyframe_map:
            video_to_kfs: Dict[str, List[Dict[str, Any]]] = {}
            for kf in self.feature_indexer.keyframe_map:
                vid = kf["video_id"]
                if vid not in video_to_kfs:
                    video_to_kfs[vid] = []
                video_to_kfs[vid].append(kf)

            top_rrf_vids = sorted(rrf_video_scores.items(), key=lambda x: x[1], reverse=True)[:30]
            for v_id, rrf_sc in top_rrf_vids:
                if any(c["video_id"] == v_id for c in candidates):
                    continue
                kfs = video_to_kfs.get(v_id, [])
                if kfs:
                    mid_kf = kfs[len(kfs) // 2]
                    key = f"{v_id}|{mid_kf['frame_id']}"
                    if key not in seen_frame_keys:
                        seen_frame_keys.add(key)
                        bm25_sc = bm25_scores_map.get(v_id, 0.0)
                        final_score = 0.25 * rrf_sc + 0.1 * (bm25_sc / 100.0 if bm25_sc else 0.0)
                        candidates.append({
                            "video_id": v_id,
                            "frame_id": mid_kf["frame_id"],
                            "score": final_score,
                            "vector_score": 0.0,
                            "bm25_score": bm25_sc,
                            "pts_time": mid_kf.get("pts_time", 0.0),
                            "fps": mid_kf.get("fps", 25.0),
                            "path": "",
                        })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def search_candidates_multi(
        self,
        query_embeddings: List[np.ndarray],
        query_text: Optional[str] = None,
        query_keywords: Optional[List[str]] = None,
        top_k: int = 100,
    ) -> List[Dict[str, Any]]:
        if not query_embeddings:
            return []

        all_ranked_lists: List[List[Tuple[str, float]]] = []

        for i, emb in enumerate(query_embeddings):
            vec_results = self.feature_indexer.search(emb, top_k=200)
            seen: Dict[str, float] = {}
            for kf_info, score in vec_results:
                vid = kf_info["video_id"]
                if vid not in seen or seen[vid] < score:
                    seen[vid] = score
            ranked = sorted(seen.items(), key=lambda x: x[1], reverse=True)
            all_ranked_lists.append(ranked)

        bm25_scores_map: Dict[str, float] = {}
        if self.metadata_indexer is not None:
            search_text = query_text or (" ".join(query_keywords) if query_keywords else "")
            if search_text:
                bm25_scores_map = self.metadata_indexer.search_bm25(search_text, top_k=200)
                bm25_ranked = sorted(bm25_scores_map.items(), key=lambda x: x[1], reverse=True)
                if bm25_ranked:
                    all_ranked_lists.append(bm25_ranked)

        rrf_scores = _reciprocal_rank_fusion(all_ranked_lists, k=self.rrf_k)

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
            
            rrf_sc = rrf_scores.get(v_id, 0.0)
            final_score = 0.75 * vec_score + 0.25 * rrf_sc

            candidates.append({
                "video_id": v_id,
                "frame_id": f_id,
                "score": final_score,
                "vector_score": vec_score,
                "bm25_score": bm25_scores_map.get(v_id, 0.0),
                "pts_time": kf_info.get("pts_time", 0.0),
                "fps": kf_info.get("fps", 30.0),
                "path": "",
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
