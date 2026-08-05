"""
LLM reranker: gửi top-N candidate kèm ảnh cho VLM để chấm điểm lại.
Dùng cho các câu truy vấn khó cần hiểu ngữ nghĩa sâu.
"""
import os
from typing import List, Dict, Any, Optional


class LLMReranker:
    """
    Dùng Gemini Vision API chấm lại top-N candidates.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.client = None
        self._init(model)

    def _init(self, model: str):
        if not self.api_key:
            print("[!] Không có API key — LLMReranker bị tắt.")
            return
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(model)
        except Exception as e:
            print(f"[!] LLMReranker không khả dụng: {e}")

    def _score_one(self, image_path: str, query: str) -> float:
        """Cho VLM chấm mức độ phù hợp [0.0 – 1.0]."""
        if self.client is None or not os.path.exists(image_path):
            return 0.0
        try:
            from PIL import Image  # type: ignore
            img = Image.open(image_path)
            prompt = (
                f"Query: {query}\n"
                "On a scale from 0.0 to 1.0, how relevant is this image to the query? "
                "Respond with only a decimal number."
            )
            response = self.client.generate_content([prompt, img])
            return float(response.text.strip())
        except Exception:
            return 0.0

    def rerank(
        self,
        candidates: List[Dict[str, Any]],
        query: str,
        keyframes_dir: str = "",
        top_n: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Chấm lại top_n candidates đầu bằng VLM.
        Trả về toàn bộ list, top_n đầu được rescored.
        """
        rescored = []
        for cand in candidates[:top_n]:
            img_path = cand.get("path", "")
            if not img_path and keyframes_dir:
                vid, fid = cand.get("video_id", ""), cand.get("frame_id", 0)
                img_path = os.path.join(keyframes_dir, vid, f"{int(fid):06d}.jpg")
            vlm_score = self._score_one(img_path, query)
            rescored.append({**cand, "score": 0.6 * cand["score"] + 0.4 * vlm_score})

        rescored.sort(key=lambda x: x["score"], reverse=True)
        return rescored + candidates[top_n:]
