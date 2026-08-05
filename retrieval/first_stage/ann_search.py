"""
First-stage ANN search: wrapper gọi Milvus hoặc FAISS tìm top-K candidates.
"""
import numpy as np
from typing import List, Dict, Any, Optional


class ANNSearch:
    """
    Unified ANN search interface — tự chọn backend: Milvus hoặc FAISS.
    """
    def __init__(self, feature_indexer=None, milvus_builder=None):
        """
        Args:
            feature_indexer: FeatureIndexer (FAISS) instance
            milvus_builder: MilvusIndexBuilder instance (nếu có)
        """
        self.feature_indexer = feature_indexer
        self.milvus = milvus_builder

    def search_topk(
        self,
        query_embedding: np.ndarray,
        k: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Tìm top-k frame tương tự nhất với query_embedding.
        Ưu tiên FAISS (đã có), fallback về Milvus nếu cấu hình.
        
        Returns:
            [{video_id, frame_id, score, pts_time, fps}]
        """
        if self.feature_indexer is not None:
            raw = self.feature_indexer.search(query_embedding, top_k=k)
            return [
                {
                    "video_id": info["video_id"],
                    "frame_id": info["frame_id"],
                    "score": float(score),
                    "pts_time": info.get("pts_time", 0.0),
                    "fps": info.get("fps", 25.0),
                }
                for info, score in raw
            ]

        if self.milvus is not None:
            return self._milvus_search(query_embedding, k)

        return []

    def _milvus_search(self, query_embedding: np.ndarray, k: int) -> List[Dict]:
        try:
            from pymilvus import Collection  # type: ignore
            from indexing.vector_db.milvus.schema import MILVUS_SCHEMA
            col = Collection(MILVUS_SCHEMA["collection_name"])
            col.load()
            results = col.search(
                data=[query_embedding.tolist()],
                anns_field="embedding_vector",
                param={"metric_type": "IP", "params": {"nprobe": 16}},
                limit=k,
                output_fields=["video_id", "frame_id"],
            )
            return [
                {"video_id": hit.entity.get("video_id"), "frame_id": hit.entity.get("frame_id"),
                 "score": float(hit.score), "pts_time": 0.0, "fps": 25.0}
                for hit in results[0]
            ]
        except Exception as e:
            print(f"[!] Milvus search lỗi: {e}")
            return []
