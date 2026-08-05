"""
DAKE Lightweight: phát hiện shot boundary bằng biến thiên kích thước JPEG.
Không cần model, nhanh, phù hợp batch lớn.
"""
import os
import io
import json
from typing import List, Dict


def _jpeg_size(frame_bgr) -> int:
    """Nén frame sang JPEG buffer và trả về kích thước bytes."""
    import cv2
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return len(buf) if ok else 0


def detect_shots_lightweight(
    video_path: str,
    threshold: float = 0.4,
    min_shot_frames: int = 10,
) -> List[Dict]:
    """
    Phân tích biến thiên kích thước JPEG giữa các frame liên tiếp.
    Trả về [{start_frame, end_frame}].
    """
    try:
        import cv2
    except ImportError:
        print("[!] OpenCV không khả dụng.")
        return []

    cap = cv2.VideoCapture(video_path)
    shots = []
    prev_size = None
    shot_start = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        size = _jpeg_size(frame)
        if prev_size and prev_size > 0:
            ratio = abs(size - prev_size) / prev_size
            if ratio > threshold and (frame_idx - shot_start) >= min_shot_frames:
                shots.append({"start_frame": shot_start, "end_frame": frame_idx - 1})
                shot_start = frame_idx
        prev_size = size
        frame_idx += 1

    cap.release()
    # Đóng shot cuối
    if shot_start < frame_idx:
        shots.append({"start_frame": shot_start, "end_frame": frame_idx - 1})
    return shots


def batch_detect(video_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(video_dir)):
        if not fname.lower().endswith((".mp4", ".avi", ".mkv")):
            continue
        vid_id = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, f"{vid_id}.json")
        if os.path.exists(out_path):
            continue
        shots = detect_shots_lightweight(os.path.join(video_dir, fname))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(shots, f, indent=2)
        print(f"  DAKE [{vid_id}]: {len(shots)} shots")
