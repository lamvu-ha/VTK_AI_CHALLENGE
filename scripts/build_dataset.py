"""
build_dataset.py — orchestrate toàn bộ preprocessing → embedding → indexing.
Thứ tự: shot detection → keyframe extraction → ASR → OCR → embed → index.

Chạy: python scripts/build_dataset.py --video_dir data/raw/videos --output_dir data
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser(description="Build full dataset pipeline")
    p.add_argument("--video_dir",  default="data/raw/videos")
    p.add_argument("--shots_dir",  default="data/shots")
    p.add_argument("--keyframes_dir", default="data/keyframes")
    p.add_argument("--asr_dir",    default="data/asr")
    p.add_argument("--ocr_dir",    default="data/ocr")
    p.add_argument("--emb_dir",    default="data/clip_features")
    p.add_argument("--index_dir",  default="data/faiss_index")
    p.add_argument("--skip_shots", action="store_true")
    p.add_argument("--skip_asr",   action="store_true")
    p.add_argument("--skip_ocr",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.shots_dir,    exist_ok=True)
    os.makedirs(args.keyframes_dir, exist_ok=True)
    os.makedirs(args.asr_dir,      exist_ok=True)
    os.makedirs(args.ocr_dir,      exist_ok=True)

    # 1. Shot detection
    if not args.skip_shots:
        print("\n=== [1/5] Shot Detection ===")
        from preprocessing.shot_detection.dake_lightweight import batch_detect
        batch_detect(args.video_dir, args.shots_dir)

    # 2. Keyframe extraction
    print("\n=== [2/5] Keyframe Extraction ===")
    from preprocessing.keyframe_extraction.extract_from_shots import batch_extract
    batch_extract(args.video_dir, args.shots_dir, args.keyframes_dir, num_per_shot=1)

    # 3. ASR
    if not args.skip_asr:
        print("\n=== [3/5] ASR (faster-whisper) ===")
        from preprocessing.asr_pipeline.whisper_asr import WhisperASRPipeline
        asr = WhisperASRPipeline()
        asr.batch_transcribe(args.video_dir, args.asr_dir)

    # 4. OCR
    if not args.skip_ocr:
        print("\n=== [4/5] OCR (PaddleOCR) ===")
        from preprocessing.ocr_pipeline.paddle_ocr import PaddleOCRPipeline
        ocr = PaddleOCRPipeline()
        ocr.batch_ocr(args.keyframes_dir, os.path.join(args.ocr_dir, "results.json"))

    # 5. Build FAISS index
    print("\n=== [5/5] Build FAISS Index ===")
    from indexing.vector_db.faiss_backup.feature_indexer import FeatureIndexer
    indexer = FeatureIndexer(index_dir=args.index_dir)
    indexer.build_from_keyframes_dir(args.keyframes_dir, args.emb_dir)
    print("\n[+] Dataset build complete!")


if __name__ == "__main__":
    main()
