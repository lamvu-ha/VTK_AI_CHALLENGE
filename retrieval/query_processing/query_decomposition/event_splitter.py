"""
Event splitter: tách 1 mô tả TRAKE thành danh sách event có nhãn thứ tự (E1, E2, ...).
"""
import re
import os
from typing import List, Optional


class EventSplitter:
    """
    Tách truy vấn TRAKE thành chuỗi event bằng LLM hoặc heuristic.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.client = None
        self._init()

    def _init(self):
        if not self.api_key:
            return
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(model)
        except Exception:
            pass

    def split(self, trake_query: str) -> List[str]:
        """
        Tách query thành danh sách event theo thứ tự thời gian.
        Ưu tiên dùng LLM; fallback về heuristic tách dấu phân cách.
        """
        if self.client is not None:
            return self._split_with_llm(trake_query)
        return self._split_heuristic(trake_query)

    def _split_with_llm(self, query: str) -> List[str]:
        prompt = (
            f"Split this video retrieval query into sequential events (E1, E2, ...):\n"
            f'"{query}"\n'
            f"Return only the event descriptions, one per line, in temporal order."
        )
        try:
            response = self.client.generate_content(prompt)
            lines = [l.strip() for l in response.text.strip().splitlines() if l.strip()]
            # Loại bỏ prefix E1: E2: nếu có
            return [re.sub(r"^E\d+[\s:.]+", "", l) for l in lines]
        except Exception:
            return self._split_heuristic(query)

    def _split_heuristic(self, query: str) -> List[str]:
        """Tách bằng dấu phân cách phổ biến: 'then', 'sau đó', 'tiếp theo', 'rồi'."""
        pattern = r"\s+(?:then|sau\s+đó|tiếp\s+theo|rồi|and\s+then|và\s+sau\s+đó)\s+"
        parts = re.split(pattern, query, flags=re.IGNORECASE)
        return [p.strip() for p in parts if p.strip()] or [query]
