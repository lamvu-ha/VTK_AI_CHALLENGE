"""
LLM query expander: sinh 3–4 câu truy vấn đồng nghĩa/biến thể bằng LLM.
"""
import os
from typing import List, Optional


class LLMQueryExpander:
    """
    Dùng Gemini hoặc bất kỳ LLM nào để mở rộng query.
    Nếu không có API key, trả về query gốc đơn độc.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.model_name = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.client = None
        self._init()

    def _init(self):
        if not self.api_key:
            return
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
        except Exception as e:
            print(f"[!] LLMExpander không khả dụng: {e}")

    def expand(self, query: str, n: int = 3) -> List[str]:
        """
        Sinh n câu truy vấn biến thể cho query gốc.
        Luôn bao gồm query gốc ở đầu danh sách.
        """
        if self.client is None:
            return [query]
        prompt = (
            f"Generate {n} alternative search queries (synonyms/rephrasings) for:\n"
            f'"{query}"\n'
            f"Return only the queries, one per line, no numbering."
        )
        try:
            response = self.client.generate_content(prompt)
            lines = [l.strip() for l in response.text.strip().splitlines() if l.strip()]
            variants = [query] + [l for l in lines if l != query][:n]
            return variants[:n + 1]
        except Exception as e:
            print(f"[!] LLM expand lỗi: {e}")
            return [query]
