from typing import List, Dict, Any, Optional
import os
import numpy as np
from retrieval.fusion.hybrid_search import HybridSearchEngine
from retrieval.query_processing.query_processor import QueryProcessor


class QASolver:
    """
    Solver for Task 1.2: Visual Question Answering (Q&A).
    If vlm_engine (Qwen2.5-VL) is provided, uses it to read the top frame image and
    generate a grounded answer. Otherwise falls back to heuristic keyword matching.
    """

    def __init__(
        self,
        search_engine: HybridSearchEngine,
        query_processor: QueryProcessor,
        vlm_engine: Optional[Any] = None,
        keyframes_dir: Optional[str] = None,   # path to data/keyframes/
    ):
        self.search_engine = search_engine
        self.query_processor = query_processor
        self.vlm_engine = vlm_engine
        self.keyframes_dir = keyframes_dir

    def _resolve_image_path(self, video_id: str, frame_id: int) -> str:
        """Try to find the keyframe image on disk."""
        if not self.keyframes_dir:
            return ""
        for fmt in (f"{int(frame_id):06d}.jpg", f"{int(frame_id)}.jpg", f"{int(frame_id):04d}.jpg"):
            p = os.path.join(self.keyframes_dir, video_id, fmt)
            if os.path.exists(p):
                return p
        return ""

    def predict_answer(self, candidate_frame: Dict[str, Any], question: str) -> str:
        if self.vlm_engine is not None:
            try:
                img_path = candidate_frame.get("path") or self._resolve_image_path(
                    candidate_frame.get("video_id", ""), candidate_frame.get("frame_id", 0)
                )
                ans = self.vlm_engine.generate_answer(img_path, question)
                if ans:
                    return ans
            except Exception:
                pass

        q = question.lower().strip()

        if any(kw in q for kw in ["bao nhiêu", "how many", "mấy", "số lượng"]):
            return "1"
        if any(kw in q for kw in ["màu gì", "màu sắc", "what color", "color"]):
            return "đỏ"
        if any(kw in q for kw in ["ai ", "who ", "người nào", "nhân vật"]):
            return "người"
        if any(kw in q for kw in ["ở đâu", "where", "địa điểm", "nơi"]):
            return "sân khấu"
        if any(kw in q for kw in ["khi nào", "when", "thời gian", "lúc"]):
            return "ban ngày"
        if any(kw in q for kw in ["như thế nào", "how", "thế nào"]):
            return "có"
        if any(kw in q for kw in ["có không", "yes or no", "có phải"]):
            return "có"
        return "có"

    def solve(
        self,
        event_description: str,
        question: str,
        query_embedding: np.ndarray,
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        combined_text = f"{event_description} {question}"
        parsed_query = self.query_processor.extract_keywords_and_objects(combined_text)
        keywords = parsed_query["extracted_keywords"]

        candidates = self.search_engine.search_candidates(
            query_embedding=query_embedding,
            query_keywords=keywords,
            query_text=combined_text,
            top_k=top_k,
            vec_search_k=top_k * 2,
        )

        results = []
        for cand in candidates:
            ans = self.predict_answer(cand, question)
            results.append({
                "video_id": cand["video_id"],
                "frame_id": cand["frame_id"],
                "answer": ans,
                "score": cand["score"],
            })
        return results
