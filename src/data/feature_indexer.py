import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class FeatureIndexer:
    """
    Indexer and Vector Search Engine for Keyframe Embeddings (CLIP / SigLIP).
    Supports Cosine Similarity search over precomputed feature matrices.
    """

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.features: Optional[np.ndarray] = None
        self.keyframe_map: List[Dict[str, Any]] = []

    def build_index(self, features: np.ndarray, keyframe_map: List[Dict[str, Any]]) -> None:
        """
        Build index from normalized feature matrix and corresponding keyframe metadata list.
        `features` should be of shape (N, embedding_dim).
        """
        if features.shape[0] != len(keyframe_map):
            raise ValueError(f"Feature count ({features.shape[0]}) does not match keyframe map count ({len(keyframe_map)})")

        # L2-normalize features for fast cosine similarity via dot product
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        self.features = features / norms
        self.keyframe_map = keyframe_map

    def search(self, query_feature: np.ndarray, top_k: int = 100) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for top_k most similar keyframes to the query_feature vector.
        Returns a list of tuples: (keyframe_metadata, similarity_score).
        """
        if self.features is None or len(self.keyframe_map) == 0:
            return []

        # Ensure query vector is L2-normalized
        q_norm = np.linalg.norm(query_feature)
        if q_norm > 0:
            q_vec = query_feature / q_norm
        else:
            q_vec = query_feature

        # Compute cosine similarity scores
        scores = np.dot(self.features, q_vec.squeeze())
        
        # Get top-k indices
        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        # Sort top-k by score descending
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for idx in top_indices:
            results.append((self.keyframe_map[idx], float(scores[idx])))
        return results
