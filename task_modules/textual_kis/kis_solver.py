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

        # Detect target color and context object
        color_names = {
            "trắng": "white", "white": "white",
            "đỏ": "red", "red": "red",
            "xanh": "blue", "blue": "blue",
            "vàng": "yellow", "yellow": "yellow",
            "đen": "black", "black": "black",
        }
        target_color = None
        for vi_c, en_c in color_names.items():
            if vi_c in query_lower:
                target_color = en_c
                break

        if any(w in query_lower for w in ["áo", "mặc", "trang phục", "shirt", "clothes"]):
            obj_type = "clothing"
        elif any(w in query_lower for w in ["xe", "ô tô", "car", "vehicle"]):
            obj_type = "vehicle"
        else:
            obj_type = "object"

        target_emb = None
        competing_embs: List[np.ndarray] = []
        if target_color and clip_encoder and hasattr(clip_encoder, "encode_text"):
            target_emb = clip_encoder.encode_text(f"a photo of a {target_color} {obj_type}")
            all_cols = ["white", "red", "blue", "yellow", "black"]
            for col in all_cols:
                if col != target_color:
                    competing_embs.append(clip_encoder.encode_text(f"a photo of a {col} {obj_type}"))

        # Build fast feature lookup ONLY for top candidates (O(candidates) instead of O(177,321))
        cand_keys = {f"{c['video_id']}|{c['frame_id']}" for c in candidates}
        kf_feat_map: Dict[str, np.ndarray] = {}
        if target_emb is not None and feat_indexer and hasattr(feat_indexer, "features") and feat_indexer.features is not None:
            if hasattr(feat_indexer, "keyframe_map") and feat_indexer.keyframe_map:
                for idx, kf in enumerate(feat_indexer.keyframe_map):
                    key = f"{kf['video_id']}|{kf['frame_id']}"
                    if key in cand_keys:
                        kf_feat_map[key] = feat_indexer.features[idx]

        results = []
        for cand in candidates:
            v_id = cand["video_id"]
            f_id = cand["frame_id"]
            score = cand["score"]

            # 1. Strict Object Detection Verification & Penalization
            obj_bonus = 1.0
            meta_indexer = getattr(self.search_engine, "metadata_indexer", None)
            if meta_indexer and hasattr(meta_indexer, "keyframe_objects"):
                kf_objs = meta_indexer.keyframe_objects.get(v_id, {}).get(f_id, set())

                # Person check
                requires_person = any(w in query_lower for w in ["người", "diễn giả", "nam", "nữ", "phụ nữ", "đàn ông", "man", "woman", "person", "speaker"])
                if requires_person:
                    if kf_objs and any(o in kf_objs for o in ["person", "man", "woman", "human", "guy", "boy", "girl"]):
                        obj_bonus += 0.25
                    elif kf_objs:  # Objects detected for this frame, but NO person!
                        obj_bonus *= 0.3  # Heavy false-positive penalty

                # Vehicle check
                requires_vehicle = any(w in query_lower for w in ["xe", "ô tô", "xe máy", "car", "vehicle"])
                if requires_vehicle:
                    if kf_objs and any(o in kf_objs for o in ["car", "vehicle", "land vehicle", "automobile", "bus", "truck"]):
                        obj_bonus += 0.25
                    elif kf_objs:
                        obj_bonus *= 0.3

                # Stage / Building check
                requires_stage = any(w in query_lower for w in ["sân khấu", "hội nghị", "tòa nhà", "stage", "building"])
                if requires_stage and kf_objs and any(o in kf_objs for o in ["stage", "building", "skyscraper", "house"]):
                    obj_bonus += 0.15

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
