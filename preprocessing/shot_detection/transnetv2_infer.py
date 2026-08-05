"""
TransNetV2 inference: phát hiện shot boundary từ video.
Xuất danh sách {start_frame, end_frame} → data/shots/<video_id>.json
"""
import os
import json
from typing import List, Dict


def run_transnetv2(video_path: str, output_dir: str, threshold: float = 0.5) -> List[Dict]:
    """
    Chạy TransNetV2 trên video, lưu kết quả shot boundary ra JSON.
    Trả về danh sách [{start_frame, end_frame}].
    """
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{video_id}.json")

    shots = []
    try:
        from transnetv2 import TransNetV2  # type: ignore
        model = TransNetV2()
        video_frames, single_frame_predictions, _ = model.predict_video(video_path)
        scene_list = model.predictions_to_scenes(single_frame_predictions, threshold=threshold)
        shots = [{"start_frame": int(s), "end_frame": int(e)} for s, e in scene_list]
    except ImportError:
        print("[!] TransNetV2 không khả dụng. Dùng dake_lightweight thay thế.")
        from preprocessing.shot_detection.dake_lightweight import detect_shots_lightweight
        shots = detect_shots_lightweight(video_path)
    except Exception as e:
        print(f"[!] TransNetV2 lỗi: {e}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(shots, f, indent=2)
    print(f"[+] Shot detection [{video_id}]: {len(shots)} shots → {out_path}")
    return shots


def batch_detect_shots(video_dir: str, output_dir: str):
    """Chạy shot detection cho tất cả video trong thư mục."""
    for fname in sorted(os.listdir(video_dir)):
        if not fname.lower().endswith((".mp4", ".avi", ".mkv")):
            continue
        vpath = os.path.join(video_dir, fname)
        vid_id = os.path.splitext(fname)[0]
        out_json = os.path.join(output_dir, f"{vid_id}.json")
        if os.path.exists(out_json):
            print(f"  [skip] {vid_id}")
            continue
        run_transnetv2(vpath, output_dir)
