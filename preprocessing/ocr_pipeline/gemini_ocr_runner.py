"""
Gemini OCR: dùng Gemini API cho text khó (chữ nghệ thuật, góc nghiêng).
Fallback khi PaddleOCR không xử lý được.
"""
import os
import json
import base64
from typing import Optional


def _encode_image_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class GeminiOCRRunner:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.client = None
        self._init()

    def _init(self):
        if not self.api_key:
            print("[!] GEMINI_API_KEY không có. GeminiOCR bị tắt.")
            return
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
            print(f"[+] GeminiOCR khởi tạo: {self.model}")
        except Exception as e:
            print(f"[!] google-generativeai không khả dụng: {e}")

    def run_ocr(self, image_path: str) -> str:
        """Trả về text từ ảnh bằng Gemini Vision."""
        if self.client is None or not os.path.exists(image_path):
            return ""
        try:
            import google.generativeai as genai  # type: ignore
            from PIL import Image  # type: ignore
            img = Image.open(image_path)
            prompt = "Extract all visible text from this image. Return only the raw text, no explanations."
            response = self.client.generate_content([prompt, img])
            return response.text.strip() if response.text else ""
        except Exception as e:
            print(f"[!] GeminiOCR lỗi {image_path}: {e}")
            return ""

    def batch_ocr(self, image_dir: str, output_json: str):
        results = {}
        for root, _, files in os.walk(image_dir):
            for fname in sorted(files):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, image_dir)
                text = self.run_ocr(fpath)
                if text:
                    results[rel] = text
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[+] GeminiOCR: {len(results)} frames → {output_json}")
        return results
