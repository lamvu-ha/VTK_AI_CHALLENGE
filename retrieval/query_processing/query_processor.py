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
    # Full phrases first to prevent partial phrase corruption
    "người mặc áo màu trắng": "person wearing a white shirt",
    "người mặc áo trắng": "person wearing a white shirt",
    "người mặc áo màu đỏ": "person wearing a red shirt",
    "người mặc áo đỏ": "person wearing a red shirt",
    "người mặc áo màu xanh": "person wearing a blue shirt",
    "người mặc áo xanh": "person wearing a blue shirt",
    "người mặc áo màu vàng": "person wearing a yellow shirt",
    "người mặc áo vàng": "person wearing a yellow shirt",
    "người mặc áo màu đen": "person wearing a black shirt",
    "người mặc áo đen": "person wearing a black shirt",
    "người mặc áo màu tím": "person wearing a purple shirt",
    "người mặc áo màu hồng": "person wearing a pink shirt",
    
    "áo màu trắng": "white shirt",
    "áo màu đỏ": "red shirt",
    "áo màu xanh": "blue shirt",
    "áo màu vàng": "yellow shirt",
    "áo màu đen": "black shirt",
    "áo màu tím": "purple shirt",
    "áo màu hồng": "pink shirt",
    "áo màu cam": "orange shirt",
    "áo màu xám": "gray shirt",
    "áo màu nâu": "brown shirt",

    "áo phông": "t-shirt", "áo thun": "t-shirt", "áo sơ mi": "shirt",
    "áo khoác": "jacket", "áo vest": "suit jacket", "áo đầm": "dress", "váy": "skirt",
    "quần dài": "pants", "quần đùi": "shorts", "quần jean": "jeans",
    "nón": "hat", "mũ": "hat", "kính": "glasses", "khẩu trang": "mask",

    "màu trắng": "white", "màu đỏ": "red", "màu xanh": "blue", "màu vàng": "yellow",
    "màu đen": "black", "màu tím": "purple", "màu hồng": "pink", "màu cam": "orange",
    "màu xám": "gray", "màu nâu": "brown",

    "người": "person", "nam": "man", "nữ": "woman", "đàn ông": "man", "phụ nữ": "woman",
    "trẻ em": "child", "em bé": "baby", "cậu bé": "boy", "cô gái": "girl",

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
    "a video frame of {}",
]


def _vi_to_en_quick(text: str) -> str:
    """Apply Vi→En substitution for CLIP prompt quality (works offline)."""
    result = text
    # Sort phrases by length descending to replace longest phrases first
    for vi, en in sorted(_VI_EN.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(vi), re.IGNORECASE)
        result = pattern.sub(en, result)
    # Clean up duplicate whitespace
    return re.sub(r'\s+', ' ', result).strip()


def _translate_vi_to_en_nmt(query_text: str) -> str:
    """
    Zero-dependency Neural Machine Translation (NMT) via standard Python urllib.
    Translates ANY arbitrary Vietnamese sentence into fluent natural English.
    """
    if not query_text or not query_text.strip():
        return ""
    try:
        import urllib.request
        import urllib.parse
        import json
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q=" + urllib.parse.quote(query_text.strip())
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data and isinstance(data, list) and len(data) > 0 and data[0]:
                translated = "".join(s[0] for s in data[0] if s and isinstance(s, list) and len(s) > 0 and s[0])
                return translated.strip()
    except Exception:
        pass
    return ""


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

        # 1. Primary: Automatic Full-Sentence Neural Machine Translation (Google NMT)
        nmt_translation = _translate_vi_to_en_nmt(query)
        if nmt_translation and nmt_translation.lower() != query.lower():
            for tmpl in _CLIP_TEMPLATES:
                v = tmpl.format(nmt_translation)
                if v.lower() not in [x.lower() for x in variants]:
                    variants.append(v)

        # 2. Deep-translator fallback
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

        # 3. Quick Vi→En substitution fallback (offline backup)
        en_quick = _vi_to_en_quick(query)
        if en_quick.lower() != query.lower():
            for tmpl in _CLIP_TEMPLATES:
                v = tmpl.format(en_quick)
                if v.lower() not in [x.lower() for x in variants]:
                    variants.append(v)

        simplified = re.sub(r'\b(đang|đã|sẽ|được|bị|vẫn|cũng|rất|quá)\b', '', normalized)
        simplified = re.sub(r'\s+', ' ', simplified).strip()
        if simplified and simplified not in variants:
            variants.append(simplified)

        return list(dict.fromkeys(variants))

    def extract_attribute_prompts(self, query: Any) -> Dict[str, List[str]]:
        """
        SOTA Query Decomposition: Decomposes natural query into (Global, Attribute/Color, Subject) prompts.
        Allows fine-grained multi-prompt vector ensembling and reranking.
        """
        if not isinstance(query, str):
            query = str(query) if query is not None else ""

        nmt_en = _translate_vi_to_en_nmt(query)
        if not nmt_en:
            nmt_en = _vi_to_en_quick(query)

        global_prompts = [
            f"a photo of {nmt_en}",
            f"a picture of {nmt_en}",
            nmt_en,
        ]

        attribute_prompts: List[str] = []
        subject_prompts: List[str] = []

        query_lower = query.lower()
        nmt_lower = nmt_en.lower()

        # Color & clothing extraction
        colors = {
            "trắng": "white", "đỏ": "red", "xanh": "blue", "vàng": "yellow",
            "đen": "black", "tím": "purple", "hồng": "pink", "cam": "orange",
            "xám": "gray", "nâu": "brown"
        }
        for vi_col, en_col in colors.items():
            if vi_col in query_lower or en_col in nmt_lower:
                attribute_prompts.append(f"a photo of {en_col} clothing")
                attribute_prompts.append(f"a person wearing a {en_col} shirt")
                attribute_prompts.append(f"{en_col} shirt")

        # Subject extraction
        if any(w in query_lower for w in ["người", "diễn giả", "nam", "nữ", "phụ nữ", "đàn ông", "cầu thủ"]):
            subject_prompts.append("a photo of a person")
        if any(w in query_lower for w in ["xe", "ô tô", "xe máy"]):
            subject_prompts.append("a photo of a vehicle")
        if any(w in query_lower for w in ["sân khấu", "hội nghị"]):
            subject_prompts.append("a photo of a stage")

        return {
            "global": list(dict.fromkeys(global_prompts)),
            "attribute": list(dict.fromkeys(attribute_prompts)),
            "subject": list(dict.fromkeys(subject_prompts)),
        }

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
