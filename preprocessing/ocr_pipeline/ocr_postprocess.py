"""
OCR postprocess: chuẩn hoá text OCR (bỏ ký tự rác, gộp dòng), gắn timestamp.
"""
import re
from typing import Dict, List, Optional


def clean_text(text: str) -> str:
    """Bỏ ký tự rác, chuẩn hoá khoảng trắng."""
    # Loại bỏ ký tự không phải chữ/số/dấu câu phổ biến
    text = re.sub(r"[^\w\s.,!?:;()\-\u00C0-\u024F\u1E00-\u1EFF]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def merge_lines(lines: List[str], separator: str = " ") -> str:
    """Gộp các dòng OCR thành 1 chuỗi."""
    return separator.join(l.strip() for l in lines if l.strip())


def postprocess_ocr_result(
    ocr_raw: Dict[str, str],
    fps: float = 25.0,
    frame_id_from_path: bool = True,
) -> List[Dict]:
    """
    Chuẩn hoá kết quả OCR và gắn timestamp.
    
    Args:
        ocr_raw: {relative_path: raw_text} từ paddleocr_runner hoặc gemini_ocr_runner
        fps: fps của video để tính timestamp
        frame_id_from_path: nếu True, parse frame_id từ tên file
    Returns:
        [{frame_id, timestamp, text}]
    """
    import os
    results = []
    for rel_path, text in ocr_raw.items():
        cleaned = clean_text(text)
        if not cleaned:
            continue
        frame_id = 0
        if frame_id_from_path:
            stem = os.path.splitext(os.path.basename(rel_path))[0]
            try:
                frame_id = int(re.search(r"\d+", stem).group())
            except Exception:
                pass
        results.append({
            "frame_id": frame_id,
            "timestamp": round(frame_id / fps, 3) if fps > 0 else 0.0,
            "text": cleaned,
        })
    results.sort(key=lambda x: x["frame_id"])
    return results
