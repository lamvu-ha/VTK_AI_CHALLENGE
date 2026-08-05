"""
Tiện ích chung cho shot detection: merge shot ngắn, convert frame ↔ timestamp.
"""
from typing import List, Dict, Tuple


def merge_short_shots(shots: List[Dict], min_frames: int = 10) -> List[Dict]:
    """Gộp các shot quá ngắn với shot trước đó."""
    merged = []
    for shot in shots:
        if merged and (shot["end_frame"] - shot["start_frame"]) < min_frames:
            merged[-1]["end_frame"] = shot["end_frame"]
        else:
            merged.append(dict(shot))
    return merged


def frame_to_timestamp(frame_idx: int, fps: float) -> float:
    """Chuyển frame index sang timestamp (giây)."""
    return round(frame_idx / fps, 3) if fps > 0 else 0.0


def timestamp_to_frame(timestamp: float, fps: float) -> int:
    """Chuyển timestamp (giây) sang frame index gần nhất."""
    return int(round(timestamp * fps))


def shots_with_timestamps(shots: List[Dict], fps: float) -> List[Dict]:
    """Thêm trường timestamp vào danh sách shot."""
    result = []
    for shot in shots:
        result.append({
            **shot,
            "start_time": frame_to_timestamp(shot["start_frame"], fps),
            "end_time": frame_to_timestamp(shot["end_frame"], fps),
        })
    return result
