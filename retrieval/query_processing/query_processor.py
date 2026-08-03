import re
import unicodedata
from typing import List, Dict, Any, Tuple

_VI_STOPWORDS = {
    "là", "và", "của", "có", "trong", "được", "cho", "với", "một", "các",
    "này", "đó", "những", "khi", "thì", "từ", "về", "trên", "để", "đã",
    "như", "không", "vào", "ra", "lên", "xuống", "hay", "hoặc", "mà",
    "tại", "sau", "trước", "giữa", "đang", "sẽ", "vẫn", "cũng", "nên",
    "rồi", "nữa", "rất", "quá", "bị", "theo", "qua", "bằng", "gì", "ai",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "do",
    "does", "did", "will", "would", "can", "could", "in", "on", "at",
    "to", "of", "for", "with", "by", "from", "as", "or", "and", "but",
}

_ACTION_PATTERNS = re.compile(
    r'(nhảy|chạy|đứng|ngồi|đi|đến|giơ|vẫy|cầm|mặc|đeo|phát biểu|trao|nhận|'
    r'bay|tiếp đất|giậm|leo|nhìn|quay|bắt|ném|đá|đấm|kick|jump|run|walk|'
    r'hold|carry|wear|stand|sit|talk|speak|award|receive|wave)',
    re.IGNORECASE
)

_COLOR_PATTERNS = re.compile(
    r'(đỏ|xanh|vàng|trắng|đen|tím|cam|hồng|xám|nâu|'
    r'red|blue|green|yellow|white|black|purple|orange|pink|gray|brown)',
    re.IGNORECASE
)

_LOCATION_PATTERNS = re.compile(
    r'(sân khấu|phòng|ngoài trời|trong nhà|sân vận động|hội trường|'
    r'đường phố|biển|núi|trường|stage|outdoor|indoor|hall|street)',
    re.IGNORECASE
)

# ── Vietnamese → English quick-sub for CLIP prompts (offline) ─────────────────
_VI_EN = {
    "người": "person", "nam": "man", "nữ": "woman", "trẻ em": "child", "em bé": "baby",
    "áo đỏ": "red shirt", "áo xanh": "blue shirt", "áo trắng": "white shirt",
    "áo vàng": "yellow shirt", "áo đen": "black shirt", "áo tím": "purple shirt",
    "áo cam": "orange shirt", "áo hồng": "pink shirt",
    "mặc": "wearing", "đeo": "wearing",
    "phát biểu": "giving a speech", "diễn thuyết": "speaking at podium",
    "trao giải": "awarding prize", "nhận giải": "receiving award",
    "sân khấu": "stage", "hội nghị": "conference", "hội trường": "auditorium",
    "ngoài trời": "outdoor", "trong nhà": "indoor",
    "đám đông": "crowd", "khán giả": "audience",
    "cầu thủ": "athlete", "vận động viên": "athlete",
    "nhảy": "jumping", "chạy": "running", "đứng": "standing", "ngồi": "sitting",
    "múa": "dancing", "hát": "singing", "biểu diễn": "performing",
    "lá cờ": "flag", "bục phát biểu": "podium", "micro": "microphone",
}

_CLIP_TEMPLATES = [
    "{}",
    "a photo of {}",
    "a picture of {}",
]


def _vi_to_en_quick(text: str) -> str:
    """Apply Vi→En substitution for CLIP prompt quality (works offline)."""
    result = text
    for vi, en in sorted(_VI_EN.items(), key=lambda x: -len(x[0])):
        result = re.sub(re.escape(vi), en, result, flags=re.IGNORECASE)
    return result


def _remove_accent_vi(text: Any) -> str:
    if not text:
        return ""
    if isinstance(text, (list, tuple)):
        text = " ".join(str(item) for item in text if item)
    elif not isinstance(text, str):
        text = str(text)
    nfkd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


class QueryProcessor:
    """
    Enhanced Query Processor for AIC 2026.
    """

    def __init__(self):
        self.stopwords = _VI_STOPWORDS

    def normalize_text(self, text: Any) -> str:
        if not text:
            return ""
        if isinstance(text, (list, tuple)):
            text = " ".join(str(item) for item in text if item)
        elif not isinstance(text, str):
            text = str(text)
        text = text.strip().lower()
        text = re.sub(r'\s+', ' ', text)
        return text

    def normalize_no_accent(self, text: Any) -> str:
        return _remove_accent_vi(self.normalize_text(text))

    def extract_keywords_and_objects(self, query: Any) -> Dict[str, Any]:
        if not isinstance(query, str):
            query = str(query) if query is not None else ""

        normalized = self.normalize_text(query)
        words = normalized.split()

        weighted_keywords: List[Tuple[str, float]] = []
        for w in words:
            if w in self.stopwords:
                continue
            if len(w) <= 1:
                continue
            weight = 1.0
            if _ACTION_PATTERNS.search(w):
                weight = 2.5
            elif _COLOR_PATTERNS.search(w):
                weight = 2.0
            elif _LOCATION_PATTERNS.search(w):
                weight = 1.8
            elif len(w) >= 4:
                weight = 1.2
            weighted_keywords.append((w, weight))

        keywords = [w for w, _ in weighted_keywords]

        no_accent = self.normalize_no_accent(query)
        no_accent_keywords = [
            w for w in no_accent.split()
            if w not in self.stopwords and len(w) > 1
        ]

        return {
            "raw_query": query,
            "normalized_query": normalized,
            "words": words,
            "extracted_keywords": keywords,
            "weighted_keywords": weighted_keywords,
            "no_accent_keywords": no_accent_keywords,
        }

    def expand_queries(self, query: Any) -> List[str]:
        if not isinstance(query, str):
            query = str(query) if query is not None else ""

        normalized = self.normalize_text(query)
        no_accent = self.normalize_no_accent(query)
        variants = [query]

        if normalized != query.lower():
            variants.append(normalized)

        if no_accent not in [v.lower() for v in variants]:
            variants.append(no_accent)

        # Quick Vi→En substitution for CLIP (offline, no network needed)
        en_quick = _vi_to_en_quick(query)
        if en_quick.lower() != query.lower():
            for tmpl in _CLIP_TEMPLATES:
                v = tmpl.format(en_quick)
                if v.lower() not in [x.lower() for x in variants]:
                    variants.append(v)

        # Online Google Translate as best-effort
        try:
            from deep_translator import GoogleTranslator
            en_translation = GoogleTranslator(source='auto', target='en').translate(query)
            if en_translation and isinstance(en_translation, str):
                en_clean = en_translation.strip()
                if en_clean and en_clean.lower() not in [v.lower() for v in variants]:
                    variants.append(en_clean)
                    variants.append(f"a photo of {en_clean.lower()}")
        except Exception:
            pass

        simplified = re.sub(r'\b(đang|đã|sẽ|được|bị|vẫn|cũng|rất|quá)\b', '', normalized)
        simplified = re.sub(r'\s+', ' ', simplified).strip()
        if simplified and simplified not in variants:
            variants.append(simplified)

        return list(dict.fromkeys(variants))

    def parse_trake_query(self, trake_text: Any) -> List[str]:
        if not trake_text:
            return []
        if isinstance(trake_text, (list, tuple)):
            trake_text = " ".join(str(item) for item in trake_text if item)
        elif not isinstance(trake_text, str):
            trake_text = str(trake_text)

        events = []
        parts = re.split(
            r'\(\d+\)|\bEvent\s+\d+[:\-]?|\bKhoảnh\s+khắc\s+\d+[:\-]?|\bBước\s+\d+[:\-]?|\b\d+[\.\\)]\s*',
            trake_text, flags=re.IGNORECASE
        )
        for part in parts:
            cleaned = part.strip(" :,.-→▶")
            if cleaned and len(cleaned) > 1:
                events.append(cleaned)

        if not events:
            events = [
                p.strip() for p in re.split(r'[,;]|→|▶', trake_text)
                if p.strip() and len(p.strip()) > 1
            ]

        return events
