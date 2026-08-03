"""
Multi-Encoder Fusion via Reciprocal Rank Fusion (RRF).
Combines CLIP + BEiT-3 + SigLIP2 (or any subset) for more robust retrieval.
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

_RRF_K = 60


class MultiEncoderFusion:
    """
    Wraps multiple text encoders and fuses their search results via RRF.
    Each encoder must expose: encode_text_ensemble(texts: List[str]) -> np.ndarray

    Usage:
        fusion = MultiEncoderFusion([clip_enc, siglip_enc, beit3_enc], weights=[1, 1, 1])
        combined_vec = fusion.encode_weighted(query_variants)   # for single-index search
        rrf_scores   = fusion.rrf_search(query_variants, indexers)  # for multi-index RRF
    """

    def __init__(self, encoders: list, weights: Optional[List[float]] = None):
        self.encoders = encoders
        self.weights = weights if weights else [1.0] * len(encoders)
        assert len(self.encoders) == len(self.weights)

    def encode_weighted(self, texts: List[str]) -> np.ndarray:
        """Weighted average-pool embeddings from all encoders (same dim required)."""
        vecs = []
        for enc, w in zip(self.encoders, self.weights):
            try:
                v = enc.encode_text_ensemble(texts).astype(np.float32)
                vecs.append(v * w)
            except Exception:
                continue
        if not vecs:
            return np.zeros(512, dtype=np.float32)
        combined = np.sum(vecs, axis=0)
        n = np.linalg.norm(combined)
        return combined / n if n > 1e-9 else combined

    def rrf_search(
        self,
        query_texts: List[str],
        feature_indexers: Dict[str, Any],  # {"clip": FeatureIndexer, "siglip": ..., ...}
        top_k: int = 100,
        vec_k: int = 200,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Run ANN search per encoder on its own index, then RRF-merge.
        feature_indexers keys must match encoder order (or use list of tuples).
        Falls back to first indexer with first encoder if dict unmatched.
        """
        ranked_lists: List[List[Tuple[str, float]]] = []

        for enc, (idx_name, indexer) in zip(self.encoders, feature_indexers.items()):
            try:
                q_vec = enc.encode_text_ensemble(query_texts)
                results = indexer.search(q_vec, top_k=vec_k)
                # Build video-level ranked list (best frame score per video)
                seen: Dict[str, float] = {}
                for kf, score in results:
                    vid = kf["video_id"]
                    if vid not in seen or seen[vid] < score:
                        seen[vid] = score
                ranked_lists.append(sorted(seen.items(), key=lambda x: x[1], reverse=True))
            except Exception:
                continue

        # RRF
        rrf: Dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, (vid, _) in enumerate(ranked):
                rrf[vid] = rrf.get(vid, 0.0) + 1.0 / (_RRF_K + rank + 1)

        return sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
