"""
Keyframe extraction: với mỗi shot, chọn 1–3 frame đại diện.
Ưu tiên frame sắc nét nhất (Laplacian variance); fallback về frame giữa.
"""
import os
import json
from typing import List, Dict, Optional


def _sharpness(frame_bgr) -> float:
    """Laplacian variance — độ sắc nét của frame."""
    import cv2
    import numpy as np
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_keyframes_from_shots(
    video_path: str,
    shots: List[Dict],
    output_dir: str,
    num_per_shot: int = 1,
) -> List[Dict]:
    """
    Với mỗi shot, lưu num_per_shot frame tốt nhất ra output_dir/<video_id>/.
    Trả về [{video_id, frame_id, path, shot_idx}].
    """
    try:
        import cv2
    except ImportError:
        print("[!] OpenCV không khả dụng.")
        return []

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(output_dir, video_id)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    keyframes = []

    for shot_idx, shot in enumerate(shots):
        s, e = shot["start_frame"], shot["end_frame"]
        total = e - s + 1
        # Chọn các vị trí candidate trong shot
        if num_per_shot == 1:
            candidates = [s + total // 2]
        else:
            step = max(1, total // (num_per_shot + 1))
            candidates = [s + step * (i + 1) for i in range(num_per_shot)]
            candidates = [min(c, e) for c in candidates]

        best_frames = []  # (sharpness, frame_idx, frame_bgr)
        for fid in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            if ok:
                best_frames.append((_sharpness(frame), fid, frame))

        for _, fid, frame in sorted(best_frames, key=lambda x: -x[0])[:num_per_shot]:
            fname = f"{fid:06d}.jpg"
            fpath = os.path.join(save_dir, fname)
            cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            keyframes.append({"video_id": video_id, "frame_id": fid, "path": fpath, "shot_idx": shot_idx})

    cap.release()
    print(f"[+] Keyframes [{video_id}]: {len(keyframes)} frames → {save_dir}")
    return keyframes


def batch_extract(video_dir: str, shots_dir: str, output_dir: str, num_per_shot: int = 1):
    for fname in sorted(os.listdir(video_dir)):
        if not fname.lower().endswith((".mp4", ".avi", ".mkv")):
            continue
        vid_id = os.path.splitext(fname)[0]
        shots_path = os.path.join(shots_dir, f"{vid_id}.json")
        if not os.path.exists(shots_path):
            print(f"  [skip] Không có shots cho {vid_id}")
            continue
        with open(shots_path, encoding="utf-8") as f:
            shots = json.load(f)
        extract_keyframes_from_shots(os.path.join(video_dir, fname), shots, output_dir, num_per_shot)
