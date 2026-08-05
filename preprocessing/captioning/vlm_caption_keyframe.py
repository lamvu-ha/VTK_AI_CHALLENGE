"""
VLM caption cho keyframe: sinh caption mô tả tự nhiên.
Dùng làm text index bổ sung và hỗ trợ Q&A fallback.
"""
import os
from typing import Optional, List, Dict


class VLMCaptionRunner:
    """
    Gọi VLM (Qwen2.5-VL hoặc Gemini) để sinh caption cho keyframe.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.model_name = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.client = None
        self._init()

    def _init(self):
        if not self.api_key:
            print("[!] Không có API key — VLM caption bị tắt.")
            return
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
            print(f"[+] VLMCaption: {self.model_name}")
        except Exception as e:
            print(f"[!] VLMCaption không khả dụng: {e}")

    def caption(self, image_path: str, prompt: str = "Describe this image briefly in English.") -> str:
        if self.client is None or not os.path.exists(image_path):
            return ""
        try:
            from PIL import Image  # type: ignore
            img = Image.open(image_path)
            response = self.client.generate_content([prompt, img])
            return response.text.strip() if response.text else ""
        except Exception as e:
            print(f"[!] Caption lỗi {image_path}: {e}")
            return ""

    def batch_caption(self, keyframes: List[Dict]) -> List[Dict]:
        """
        Sinh caption cho danh sách keyframes.
        Args: [{video_id, frame_id, path, ...}]
        Returns: same list với thêm trường 'caption'
        """
        results = []
        for kf in keyframes:
            cap = self.caption(kf.get("path", ""))
            results.append({**kf, "caption": cap})
            if cap:
                print(f"  caption [{kf['video_id']}@{kf['frame_id']}]: {cap[:80]}")
        return results
