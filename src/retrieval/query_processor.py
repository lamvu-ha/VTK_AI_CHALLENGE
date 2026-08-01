import re
from typing import List, Dict, Any

class QueryProcessor:
    """
    Query Processor for parsing natural language queries into structured search requests.
    Handles language normalization, entity extraction, and TRAKE sequential event decomposition.
    """

    def __init__(self):
        pass

    def normalize_text(self, text: str) -> str:
        """Normalizes text by removing redundant whitespaces and lowercasing."""
        if not text:
            return ""
        text = text.strip().lower()
        text = re.sub(r'\s+', ' ', text)
        return text

    def extract_keywords_and_objects(self, query: str) -> Dict[str, Any]:
        """
        Extracts key noun phrases and potential object classes from query.
        """
        normalized = self.normalize_text(query)
        words = normalized.split()
        
        return {
            "raw_query": query,
            "normalized_query": normalized,
            "words": words,
            "extracted_keywords": [w for w in words if len(w) > 2]
        }

    def parse_trake_query(self, trake_text: str) -> List[str]:
        """
        Parses a TRAKE query describing a sequence of events into individual event descriptions.
        Example input: "(1) giậm nhảy, (2) bay qua xà, (3) tiếp đất, (4) đứng dậy"
        Returns: ["giậm nhảy", "bay qua xà", "tiếp đất", "đứng dậy"]
        """
        events = []
        # Pattern to split by (1), (2), Event 1, Event 2, etc.
        parts = re.split(r'\(\d+\)|\bEvent\s+\d+[:\-]?|\bKhoảnh khắc\s+\d+[:\-]?', trake_text, flags=re.IGNORECASE)
        for part in parts:
            cleaned = part.strip(" :,.-")
            if cleaned:
                events.append(cleaned)
        
        if not events:
            # Fallback split by commas or semi-colons
            events = [p.strip() for p in re.split(r'[,;]', trake_text) if p.strip()]

        return events
