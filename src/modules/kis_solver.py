from typing import List, Dict, Any, Optional
import numpy as np
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.query_processor import QueryProcessor


class TextualKISSolver:
    """
    Solver for Task 1.1: Textual Known Item Search (Textual KIS).
    
    Improvements:
    - Passes raw query_text to HybridSearchEngine for BM25 search
    - Uses weighted keywords from QueryProcessor
    - Supports multi-query ensemble embedding from CLIPTextEncoder
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
        """
        Solves a Textual KIS query.
        Returns up to top_k candidates formatted for submission.
        """
        parsed_query = self.query_processor.extract_keywords_and_objects(query_text)
        keywords = parsed_query["extracted_keywords"]

        candidates = self.search_engine.search_candidates(
            query_embedding=query_embedding,
            query_keywords=keywords,
            query_text=query_text,  # pass raw text for BM25
            top_k=top_k,
            vec_search_k=top_k * 2,
        )

        results = []
        for cand in candidates:
            results.append({
                "video_id": cand["video_id"],
                "frame_id": cand["frame_id"],
                "score": cand["score"],
            })
        return results
