"""
BM25 Sparse Search Engine for AIC 2026.
Provides keyword-based retrieval using BM25 ranking over video metadata
(title, description, tags) and object detection labels.
Falls back to simple TF-based matching if rank_bm25 is not installed.
"""
import re
import math
import unicodedata
from typing import List, Dict, Any, Optional, Tuple

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer with Vietnamese support."""
    text = text.lower().strip()
    # Remove punctuation, keep Vietnamese chars
    text = re.sub(r'[^\w\sàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def _remove_accent(text: str) -> str:
    nfkd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


class BM25Engine:
    """
    BM25 sparse text search engine for video metadata.
    
    Indexes: video title, description, tags, object labels
    Supports: BM25Okapi (rank_bm25) with TF-IDF fallback
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.video_ids: List[str] = []
        self.corpus_tokens: List[List[str]] = []
        self._bm25: Optional[Any] = None  # BM25Okapi instance
        # Fallback: TF-IDF style data
        self._idf: Dict[str, float] = {}
        self._tf_matrix: List[Dict[str, float]] = []
        self._is_built = False

    def _build_document_text(self, video_id: str, metadata: Dict[str, Any], objects: Optional[List[str]] = None) -> str:
        """Merge all text sources for a video into a single indexed document."""
        parts = []
        if "title" in metadata and isinstance(metadata["title"], str):
            # Title is most important — repeat for weight boost
            parts.append(metadata["title"])
            parts.append(metadata["title"])
        if "description" in metadata and isinstance(metadata["description"], str):
            parts.append(metadata["description"][:500])  # limit to 500 chars
        if "tags" in metadata and isinstance(metadata["tags"], list):
            parts.append(" ".join(str(t) for t in metadata["tags"][:50]))
        if objects:
            parts.append(" ".join(objects[:100]))
        return " ".join(parts)

    def add_document(self, video_id: str, metadata: Dict[str, Any], objects: Optional[List[str]] = None) -> None:
        """Add a single video's metadata to the index (call before build())."""
        doc_text = self._build_document_text(video_id, metadata, objects)
        tokens = _tokenize(doc_text)
        # Also add no-accent version for robustness
        no_accent_tokens = _tokenize(_remove_accent(doc_text))
        combined = tokens + [t for t in no_accent_tokens if t not in tokens]
        self.video_ids.append(video_id)
        self.corpus_tokens.append(combined)

    def build_index(self) -> None:
        """Build BM25 or fallback TF-IDF index over all added documents."""
        if not self.corpus_tokens:
            return

        if _HAS_BM25:
            self._bm25 = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)
            print(f"[+] BM25 index built over {len(self.video_ids)} videos.")
        else:
            # Fallback: TF-IDF style
            print(f"[!] rank_bm25 not installed. Using TF-IDF fallback for {len(self.video_ids)} videos.")
            self._build_tfidf_fallback()
        self._is_built = True

    def _build_tfidf_fallback(self) -> None:
        """Build simple TF-IDF index as fallback."""
        N = len(self.corpus_tokens)
        df: Dict[str, int] = {}
        for tokens in self.corpus_tokens:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log((N + 1) / (cnt + 1)) + 1 for t, cnt in df.items()}

        self._tf_matrix = []
        for tokens in self.corpus_tokens:
            total = len(tokens) if tokens else 1
            tf: Dict[str, float] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1.0 / total
            self._tf_matrix.append(tf)

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Search for top_k videos matching the query.
        Returns list of (video_id, score) sorted by score descending.
        """
        if not self._is_built or not self.video_ids:
            return []

        query_tokens = _tokenize(query)
        no_accent_tokens = _tokenize(_remove_accent(query))
        all_tokens = list(set(query_tokens + no_accent_tokens))

        if not all_tokens:
            return []

        if _HAS_BM25 and self._bm25 is not None:
            scores = self._bm25.get_scores(all_tokens)
            ranked = sorted(
                ((self.video_ids[i], float(scores[i])) for i in range(len(self.video_ids))),
                key=lambda x: x[1], reverse=True
            )
        else:
            ranked = []
            for i, tf in enumerate(self._tf_matrix):
                score = sum(tf.get(t, 0) * self._idf.get(t, 0) for t in all_tokens)
                if score > 0:
                    ranked.append((self.video_ids[i], score))
            ranked.sort(key=lambda x: x[1], reverse=True)

        # Filter zero scores and return top_k
        return [(vid, sc) for vid, sc in ranked if sc > 0][:top_k]
