"""
Hợp nhất kết quả object detection từ nhiều nguồn (BTC Faster R-CNN + YOLOv8 bổ sung).
"""
from typing import List, Dict


def merge_object_sources(
    btc_detections: List[Dict],
    extra_detections: List[Dict],
    iou_threshold: float = 0.5,
) -> List[Dict]:
    """
    Gộp detection từ 2 nguồn, loại bỏ trùng lặp theo IoU.
    
    Args:
        btc_detections: kết quả từ model BTC cấp (Faster R-CNN)
        extra_detections: kết quả từ YOLOv8/DINO bổ sung
        iou_threshold: ngưỡng IoU để coi là trùng
    Returns:
        Danh sách detection hợp nhất, chuẩn hoá về {label, confidence, bbox}.
    """
    def iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    merged = list(btc_detections)
    for det in extra_detections:
        bbox = det.get("bbox", [])
        is_dup = any(
            d.get("label") == det.get("label") and iou(d.get("bbox", [0,0,0,0]), bbox) > iou_threshold
            for d in merged
        )
        if not is_dup:
            merged.append(det)
    return merged


def normalize_detection(det: Dict, source: str = "btc") -> Dict:
    """Chuẩn hoá format về {label, confidence, bbox, source}."""
    return {
        "label": det.get("label") or det.get("class") or det.get("name", "unknown"),
        "confidence": float(det.get("confidence") or det.get("score", 1.0)),
        "bbox": det.get("bbox") or det.get("box", []),
        "source": source,
    }
