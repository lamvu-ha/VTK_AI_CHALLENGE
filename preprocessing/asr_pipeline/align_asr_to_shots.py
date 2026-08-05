"""
ASR alignment: map đoạn transcript vào shot/keyframe tương ứng theo timestamp.
"""
from typing import List, Dict


def align_asr_to_shots(
    transcript: List[Dict],
    shots: List[Dict],
    fps: float = 25.0,
) -> List[Dict]:
    """
    Map transcript segments vào shot tương ứng theo timestamp.
    
    Args:
        transcript: [{start, end, text}] từ whisper
        shots: [{start_frame, end_frame, ...}] 
        fps: frames per second
    Returns:
        shots với thêm trường 'asr_text'
    """
    result = []
    for shot in shots:
        shot_start_t = shot["start_frame"] / fps
        shot_end_t = shot["end_frame"] / fps
        texts = []
        for seg in transcript:
            # Overlap với shot
            if seg["start"] < shot_end_t and seg["end"] > shot_start_t:
                texts.append(seg["text"])
        result.append({**shot, "asr_text": " ".join(texts).strip()})
    return result


def align_asr_to_keyframes(
    transcript: List[Dict],
    keyframes: List[Dict],
    fps: float = 25.0,
    window_sec: float = 2.0,
) -> List[Dict]:
    """
    Gắn đoạn ASR gần nhất vào từng keyframe (trong cửa sổ ±window_sec).
    """
    result = []
    for kf in keyframes:
        kf_time = kf["frame_id"] / fps
        texts = []
        for seg in transcript:
            if abs((seg["start"] + seg["end"]) / 2 - kf_time) <= window_sec:
                texts.append(seg["text"])
        result.append({**kf, "asr_text": " ".join(texts).strip()})
    return result
