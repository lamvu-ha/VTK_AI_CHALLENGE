"""
batch_ingest_new_data.py — incremental ingest: chỉ xử lý video mới trong batch 2.
Append vào index hiện có thay vì rebuild từ đầu.

Chạy: python scripts/batch_ingest_new_data.py --new_video_dir data/raw/batch2
"""
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser(description="Incremental ingest new batch videos")
    p.add_argument("--new_video_dir", required=True, help="Thư mục chứa video mới (batch 2)")
    p.add_argument("--shots_dir",     default="data/shots")
    p.add_argument("--keyframes_dir", default="data/keyframes")
    p.add_argument("--asr_dir",       default="data/asr")
    p.add_argument("--ocr_dir",       default="data/ocr")
    p.add_argument("--index_dir",     default="data/faiss_index")
    p.add_argument("--emb_dir",       default="data/clip_features")
    return p.parse_args()


def get_new_videos(video_dir: str, processed_file: str = "data/.processed_videos.json"):
    """Trả về danh sách video chưa xử lý."""
    all_videos = [f for f in os.listdir(video_dir) if f.lower().endswith((".mp4", ".avi", ".mkv"))]
    processed = set()
    if os.path.exists(processed_file):
        with open(processed_file, encoding="utf-8") as f:
            processed = set(json.load(f))
    new_vids = [v for v in all_videos if v not in processed]
    return new_vids, processed, processed_file


def mark_processed(vid_name: str, processed: set, processed_file: str):
    processed.add(vid_name)
    os.makedirs(os.path.dirname(processed_file) or ".", exist_ok=True)
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump(list(processed), f)


def main():
    args = parse_args()
    new_vids, processed, proc_file = get_new_videos(args.new_video_dir)
    print(f"[*] Tìm thấy {len(new_vids)} video mới cần xử lý.")

    if not new_vids:
        print("[+] Không có video mới. Kết thúc.")
        return

    from preprocessing.shot_detection.dake_lightweight import detect_shots_lightweight
    from preprocessing.keyframe_extraction.extract_from_shots import extract_keyframes_from_shots
    from indexing.vector_db.faiss_backup.feature_indexer import FeatureIndexer

    indexer = FeatureIndexer(index_dir=args.index_dir)

    for vid_name in new_vids:
        vid_id = os.path.splitext(vid_name)[0]
        vid_path = os.path.join(args.new_video_dir, vid_name)
        print(f"\n  → Processing: {vid_name}")

        # Shot detection
        shots_path = os.path.join(args.shots_dir, f"{vid_id}.json")
        if not os.path.exists(shots_path):
            shots = detect_shots_lightweight(vid_path)
            os.makedirs(args.shots_dir, exist_ok=True)
            with open(shots_path, "w") as f:
                json.dump(shots, f, indent=2)

        with open(shots_path) as f:
            shots = json.load(f)

        # Keyframe extraction
        extract_keyframes_from_shots(vid_path, shots, args.keyframes_dir)

        # Update FAISS index (append)
        vid_kf_dir = os.path.join(args.keyframes_dir, vid_id)
        if os.path.exists(vid_kf_dir):
            indexer.add_keyframes_dir(vid_kf_dir, vid_id)

        mark_processed(vid_name, processed, proc_file)
        print(f"  [done] {vid_id}")

    print(f"\n[+] Incremental ingest complete. {len(new_vids)} videos added.")


if __name__ == "__main__":
    main()
