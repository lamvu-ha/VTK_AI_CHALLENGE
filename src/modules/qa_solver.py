from typing import List, Dict, Any, Optional
import numpy as np
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.query_processor import QueryProcessor

class QASolver:
    """
    Solver for Task 1.2: Visual Question Answering (Q&A).
    Given event description + question, returns top ranked <video_id, frame_id, answer> tuples.
    """

    def __init__(self, search_engine: HybridSearchEngine, query_processor: QueryProcessor, vlm_engine: Optional[Any] = None):
        self.search_engine = search_engine
        self.query_processor = query_processor
        self.vlm_engine = vlm_engine

    def predict_answer(self, candidate_frame: Dict[str, Any], question: str) -> str:
        """
        Uses VLM or rule-based heuristics to answer the question based on the candidate frame.
        """
        if self.vlm_engine is not None:
            # Delegate to VLM model inference
            return self.vlm_engine.generate_answer(candidate_frame["path"], question)
        
        # Heuristic / Fallback answer extraction if VLM is not loaded
        question_norm = question.lower()
        if "bao nhiêu" in question_norm or "how many" in question_norm:
            return "1"
        elif "màu gì" in question_norm or "what color" in question_norm:
            return "đỏ"
        elif "ai" in question_norm or "who" in question_norm:
            return "người"
        return "có"

    def solve(self, event_description: str, question: str, query_embedding: np.ndarray, top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Solves a Q&A query.
        Returns up to top_k candidates formatted with video_id, frame_id, and answer.
        """
        combined_text = f"{event_description} {question}"
        parsed_query = self.query_processor.extract_keywords_and_objects(combined_text)
        keywords = parsed_query["extracted_keywords"]

        candidates = self.search_engine.search_candidates(
            query_embedding=query_embedding,
            query_keywords=keywords,
            top_k=top_k
        )

        results = []
        for cand in candidates:
            ans = self.predict_answer(cand, question)
            results.append({
                "video_id": cand["video_id"],
                "frame_id": cand["frame_id"],
                "answer": ans,
                "score": cand["score"]
            })
        return results
