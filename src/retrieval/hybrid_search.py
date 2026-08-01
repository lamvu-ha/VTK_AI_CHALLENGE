from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from src.data.feature_indexer import FeatureIndexer
from src.data.metadata_indexer import MetadataIndexer

class HybridSearchEngine:
    """
    Hybrid Search Engine combining Vector Embeddings (CLIP), Video Metadata,
    and Object Detection features for high-precision retrieval.
    """

    def __init__(self, feature_indexer: FeatureIndexer, metadata_indexer: Optional[MetadataIndexer] = None):
        self.feature_indexer = feature_indexer
        self.metadata_indexer = metadata_indexer

    def search_candidates(self, query_embedding: np.ndarray, query_keywords: Optional[List[str]] = None, top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Executes hybrid search and returns ranked candidate keyframe objects.
        Each candidate contains: video_id, frame_id, score, path.
        """
        # Step 1: Initial Vector Search
        raw_results = self.feature_indexer.search(query_embedding, top_k=top_k * 2)

        # Step 2: Metadata keyword boosting if available
        meta_scores: Dict[str, float] = {}
        if self.metadata_indexer and query_keywords:
            meta_scores = self.metadata_indexer.search_metadata_by_keywords(query_keywords)

        candidates = []
        for keyframe_info, vec_score in raw_results:
            v_id = keyframe_info["video_id"]
            f_id = keyframe_info["frame_id"]
            
            # Combine vector similarity score + metadata boost
            meta_boost = meta_scores.get(v_id, 0.0) * 0.1
            combined_score = vec_score + meta_boost

            candidates.append({
                "video_id": v_id,
                "frame_id": f_id,
                "score": combined_score,
                "vector_score": vec_score,
                "meta_boost": meta_boost,
                "path": keyframe_info.get("path", "")
            })

        # Step 3: Sort by combined score descending and trim to top_k
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
