import numpy as np
from typing import List, Tuple, Dict, Any, Optional

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class FeatureIndexer:
    """
    Optimized Indexer and Vector Search Engine for Keyframe Embeddings (CLIP / SigLIP).
    
    Improvements over baseline:
    - Uses FAISS IndexHNSWFlat for sub-linear approximate nearest neighbor search
    - Falls back to numpy dot-product brute-force if FAISS is unavailable
    - Supports multi-vector ensemble queries (average pooling before search)
    - Returns per-result similarity scores for RRF merging
    """

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.features: Optional[np.ndarray] = None
        self.keyframe_map: List[Dict[str, Any]] = []
        self._faiss_index: Optional[Any] = None

    def build_index(self, features: np.ndarray, keyframe_map: List[Dict[str, Any]]) -> None:
        """
        Build index from feature matrix and corresponding keyframe metadata list.
        `features` should be of shape (N, embedding_dim).
        Features are L2-normalized for cosine similarity via dot product.
        """
        if features.shape[0] != len(keyframe_map):
            raise ValueError(
                f"Feature count ({features.shape[0]}) does not match "
                f"keyframe map count ({len(keyframe_map)})"
            )

        # L2-normalize for cosine similarity
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        norm_features = (features / norms).astype(np.float32)

        self.features = norm_features
        self.keyframe_map = keyframe_map

        # Build FAISS index if available
        if _HAS_FAISS:
            try:
                # HNSW for fast ANN search (no GPU needed)
                # M=32: number of connections per layer (higher = more accurate, more memory)
                index = faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = 200
                index.add(norm_features)
                index.hnsw.efSearch = 128
                self._faiss_index = index
                print(f"[+] FAISS HNSW index built for {features.shape[0]} keyframes (dim={self.embedding_dim}).")
            except Exception as e:
                print(f"[!] FAISS index build failed: {e}. Falling back to numpy.")
                self._faiss_index = None
        else:
            print(f"[!] faiss-cpu not installed. Using numpy brute-force search.")

    def search(
        self,
        query_feature: np.ndarray,
        top_k: int = 100
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for top_k most similar keyframes to query_feature.
        Returns list of (keyframe_metadata, similarity_score) sorted by score descending.
        """
        if self.features is None or len(self.keyframe_map) == 0:
            return []

        # L2-normalize query
        q_vec = query_feature.squeeze().astype(np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        actual_k = min(top_k, len(self.keyframe_map))

        if _HAS_FAISS and self._faiss_index is not None:
            # FAISS search
            q_vec_2d = q_vec.reshape(1, -1)
            scores_arr, indices = self._faiss_index.search(q_vec_2d, actual_k)
            results = []
            for score, idx in zip(scores_arr[0], indices[0]):
                if idx >= 0:
                    results.append((self.keyframe_map[int(idx)], float(score)))
            return results
        else:
            # Numpy brute-force cosine similarity
            scores = np.dot(self.features, q_vec)
            top_indices = np.argpartition(scores, -actual_k)[-actual_k:]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
            return [(self.keyframe_map[i], float(scores[i])) for i in top_indices]

    def search_multi(
        self,
        query_features: List[np.ndarray],
        top_k: int = 100
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Multi-vector ensemble search: average-pool multiple query vectors then search.
        Used for multi-query expansion in the CLIP encoder.
        """
        if not query_features:
            return []

        stacked = np.stack([q.squeeze().astype(np.float32) for q in query_features], axis=0)
        # Average pooling
        avg_vec = stacked.mean(axis=0)
        # Re-normalize
        norm = np.linalg.norm(avg_vec)
        if norm > 0:
            avg_vec = avg_vec / norm

        return self.search(avg_vec, top_k=top_k)
