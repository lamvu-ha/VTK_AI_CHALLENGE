from typing import List, Dict, Any, Optional
import numpy as np
from retrieval.fusion.hybrid_search import HybridSearchEngine
from retrieval.query_processing.query_processor import QueryProcessor


class TextualKISSolver:
    """
    SOTA Solver for Task 1.1: Textual Known Item Search (Textual KIS).
    Uses Two-Stage Retrieval with Contrastive Color Discrimination:
    Stage 1: Dense FAISS + BM25 + Object Index Coarse Search (Top-200)
    Stage 2: Negative Color Contrast Penalization & Object Detection Reranking
    """

    def __init__(self, search_engine: HybridSearchEngine, query_processor: QueryProcessor):
        self.search_engine = search_engine
        self.query_processor = query_processor

    def solve(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        # Stage 1: Coarse Candidate Retrieval (Top-200)
        parsed_query = self.query_processor.extract_keywords_and_objects(query_text)
        keywords = parsed_query["extracted_keywords"]

        candidates = self.search_engine.search_candidates(
            query_embedding=query_embedding,
            query_keywords=keywords,
            query_text=query_text,
            top_k=top_k * 2,
            vec_search_k=top_k * 3,
        )

        if not candidates:
            return []

        # Stage 2: Fine-Grained Attribute & Object Verification Reranking
        query_lower = query_text.lower()
        clip_encoder = getattr(self.search_engine, "clip_encoder", None)
        feat_indexer = getattr(self.search_engine, "feature_indexer", None)

        # Detect target color
        color_map = {
            "trắng": "white shirt", "white": "white shirt",
            "đỏ": "red shirt", "red": "red shirt",
            "xanh": "blue shirt", "blue": "blue shirt",
            "vàng": "yellow shirt", "yellow": "yellow shirt",
            "đen": "black shirt", "black": "black shirt",
        }
        target_col_name = None
        for vi_c, en_prompt in color_map.items():
            if vi_c in query_lower:
                target_col_name = en_prompt
                break

        target_emb = None
        competing_embs: List[np.ndarray] = []
        if target_col_name and clip_encoder and hasattr(clip_encoder, "encode_text"):
            target_emb = clip_encoder.encode_text(f"a photo of a person wearing a {target_col_name}")
            other_colors = ["white shirt", "red shirt", "blue shirt", "yellow shirt", "black shirt"]
            for col in other_colors:
                if col != target_col_name:
                    competing_embs.append(clip_encoder.encode_text(f"a photo of a person wearing a {col}"))

        # Build fast feature lookup for top candidate keyframes
        kf_feat_map: Dict[str, np.ndarray] = {}
        if target_emb is not None and feat_indexer and hasattr(feat_indexer, "features") and feat_indexer.features is not None:
            if hasattr(feat_indexer, "keyframe_map") and feat_indexer.keyframe_map:
                for idx, kf in enumerate(feat_indexer.keyframe_map):
                    key = f"{kf['video_id']}|{kf['frame_id']}"
                    kf_feat_map[key] = feat_indexer.features[idx]

        results = []
        for cand in candidates:
            v_id = cand["video_id"]
            f_id = cand["frame_id"]
            score = cand["score"]

            # 1. Object detection verification bonus
            obj_bonus = 1.0
            meta_indexer = getattr(self.search_engine, "metadata_indexer", None)
            if meta_indexer and hasattr(meta_indexer, "keyframe_objects"):
                kf_objs = meta_indexer.keyframe_objects.get(v_id, {}).get(f_id, set())
                if kf_objs:
                    if any(w in query_lower for w in ["người", "diễn giả", "nam", "nữ", "man", "woman", "person"]) and any(o in kf_objs for o in ["person", "man", "woman"]):
                        obj_bonus += 0.15
                    if any(w in query_lower for w in ["xe", "car", "vehicle"]) and any(o in kf_objs for o in ["car", "vehicle", "land vehicle", "automobile"]):
                        obj_bonus += 0.15
                    if any(w in query_lower for w in ["sân khấu", "hội nghị", "tòa nhà", "stage", "building"]) and any(o in kf_objs for o in ["stage", "building", "skyscraper"]):
                        obj_bonus += 0.10

            # 2. Contrastive Color Discrimination
            color_multiplier = 1.0
            key = f"{v_id}|{f_id}"
            if target_emb is not None and key in kf_feat_map:
                img_feat = kf_feat_map[key]
                target_sim = float(np.dot(img_feat, target_emb))
                max_competing = max(float(np.dot(img_feat, c_emb)) for c_emb in competing_embs) if competing_embs else 0.0

                if target_sim > max_competing:
                    color_multiplier += 0.25  # True color match bonus
                else:
                    color_multiplier -= 0.20  # False color match penalty

            final_score = score * obj_bonus * color_multiplier

            results.append({
                "video_id": v_id,
                "frame_id": f_id,
                "score": final_score,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
