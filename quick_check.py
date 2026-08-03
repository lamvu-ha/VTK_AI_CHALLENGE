"""
quick_check.py — Chạy nhanh pipeline retrieval và in kết quả ra terminal.
Hỗ trợ cả 3 task: KIS, Q&A, TRAKE.

Cách dùng:
  # Task 1.1: Textual KIS
  python quick_check.py --task kis --query "người mặc áo đỏ phát biểu"

  # Task 1.2: Visual Q&A
  python quick_check.py --task qa --query "Lễ trao giải" --question "Có bao nhiêu người trên sân khấu?"

  # Task 1.3: TRAKE (Temporal Retrieval)
  python quick_check.py --task trake --query "(1) Giậm nhảy, (2) Bay qua xà, (3) Tiếp đất, (4) Đứng dậy"

  # Bật VLM (Qwen2.5-VL) cho QA hoặc Re-ranking (BLIP-2):
  python quick_check.py --task qa --query "Lễ trao giải" --question "Bao nhiêu người?" --use_vlm
  python quick_check.py --task kis --query "người mặc áo đỏ phát biểu" --rerank
"""

import os
import sys
import argparse

# ── Load torch FIRST before FAISS/MKL to prevent WinError 1114 DLL conflict ──
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # no CUDA DLL scan
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import torch as _t  # noqa: F401
except Exception:
    pass

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indexing.dataset_loader import AICDatasetLoader
from indexing.vector_db.faiss_backup.feature_indexer import FeatureIndexer
from indexing.text_search.metadata_indexer import MetadataIndexer
from retrieval.query_processing.query_processor import QueryProcessor
from embedding_models.clip.clip_encoder import CLIPTextEncoder
from retrieval.fusion.hybrid_search import HybridSearchEngine
from task_modules.textual_kis.kis_solver import TextualKISSolver
from task_modules.qa.qa_solver import QASolver
from task_modules.trake.trake_solver import TRAKESolver


def build_pipeline(data_dir: str):
    loader = AICDatasetLoader(data_dir)
    features, keyframe_map = loader.load_video_dataset()
    if features.shape[0] == 0:
        print("[!] Không có dữ liệu trong data/. Hãy kiểm tra thư mục data/clip-features-32/")
        sys.exit(1)

    feature_indexer = FeatureIndexer(embedding_dim=features.shape[1])
    feature_indexer.build_index(features, keyframe_map)

    metadata_indexer = MetadataIndexer()
    objects_dir = os.path.join(data_dir, "objects")
    objects_available = os.path.exists(objects_dir)
    for v_id in loader.get_all_video_ids():
        meta = loader.load_media_info(v_id)
        if meta:
            metadata_indexer.add_video_metadata(
                v_id, meta, objects_dir=objects_dir if objects_available else None
            )

    # Load OCR/ASR nếu có
    ocr_json = os.path.join(data_dir, "ocr", "results.json")
    asr_dir  = os.path.join(data_dir, "asr")
    if os.path.exists(ocr_json):
        metadata_indexer.load_ocr_json(ocr_json)
        print(f"[+] OCR loaded: {ocr_json}")
    if os.path.exists(asr_dir) and any(f.endswith(".json") for f in os.listdir(asr_dir)):
        metadata_indexer.load_asr_dir(asr_dir)
        print(f"[+] ASR loaded: {asr_dir}")

    metadata_indexer.build_bm25_index()

    encoder     = CLIPTextEncoder()
    engine      = HybridSearchEngine(feature_indexer, metadata_indexer)
    processor   = QueryProcessor()

    print(f"[+] Pipeline ready — {features.shape[0]} keyframes | {len(metadata_indexer.video_metadata)} videos")
    return engine, encoder, processor


def run_kis(engine, encoder, processor, query: str, top_k: int = 10, rerank: bool = False, keyframes_dir: str = ""):
    variants = processor.expand_queries(query)
    emb = encoder.encode_text_ensemble(variants)
    
    solver = TextualKISSolver(engine, processor)
    results = solver.solve(query, emb, top_k=top_k * 2)

    if rerank:
        try:
            from retrieval.reranking.blip2_reranker import BLIP2Reranker
            reranker = BLIP2Reranker()
            results = reranker.rerank(results, query_text=query, keyframes_dir=keyframes_dir, top_n=top_k)
            print("[+] BLIP-2 Re-ranking applied.")
        except Exception as e:
            print(f"[!] Re-ranking skipped: {e}")

    results = results[:top_k]
    print(f"\n{'='*65}")
    print(f"[KIS] Query: \"{query}\"")
    print(f"      Variants ({len(variants)}): {variants}")
    print(f"{'='*65}")
    print(f"{'#':>3}  {'Video ID':<16} {'Frame':>7}  {'Score':>8}")
    print(f"{'-'*45}")
    for i, r in enumerate(results, 1):
        print(f"{i:>3}. {r['video_id']:<16} {r['frame_id']:>7}  {r['score']:>8.4f}")
    return results


def run_qa(engine, encoder, processor, event: str, question: str, top_k: int = 10, use_vlm: bool = False, keyframes_dir: str = ""):
    combined = f"{event} {question}"
    variants = processor.expand_queries(combined)
    emb = encoder.encode_text_ensemble(variants)

    vlm_engine = None
    if use_vlm:
        try:
            from task_modules.qa.vlm_engine import Qwen25VLEngine
            vlm_engine = Qwen25VLEngine(model_name="Qwen/Qwen2.5-VL-3B-Instruct")
        except Exception as e:
            print(f"[!] VLM load error: {e}. Falling back to heuristic QA.")

    solver = QASolver(engine, processor, vlm_engine=vlm_engine, keyframes_dir=keyframes_dir)
    results = solver.solve(event, question, emb, top_k=top_k)

    print(f"\n{'='*65}")
    print(f"[Q&A] Event   : \"{event}\"")
    print(f"      Question: \"{question}\"")
    print(f"{'='*65}")
    print(f"{'#':>3}  {'Video ID':<16} {'Frame':>7}  {'Score':>8}  {'Answer'}")
    print(f"{'-'*60}")
    for i, r in enumerate(results[:top_k], 1):
        ans = r.get("answer", "?")
        print(f"{i:>3}. {r['video_id']:<16} {r['frame_id']:>7}  {r['score']:>8.4f}  {ans}")
    return results


def run_trake(engine, encoder, processor, trake_text: str, top_k: int = 10):
    sub_events = processor.parse_trake_query(trake_text)
    event_embs = []
    for ev in sub_events:
        ev_vars = processor.expand_queries(ev)
        ev_emb = encoder.encode_text_ensemble(ev_vars)
        event_embs.append(ev_emb)

    solver = TRAKESolver(engine, processor)
    results = solver.solve(trake_text, event_embs, top_k=top_k)

    print(f"\n{'='*65}")
    print(f"[TRAKE] Sequence: \"{trake_text}\"")
    print(f"        Sub-events ({len(sub_events)}): {sub_events}")
    print(f"{'='*65}")
    print(f"{'#':>3}  {'Video ID':<16} {'Frames (Sequence)':<30}  {'Score':>8}")
    print(f"{'-'*65}")
    for i, r in enumerate(results[:top_k], 1):
        frames_str = " -> ".join(str(f) for f in r.get("frame_ids", []))
        print(f"{i:>3}. {r['video_id']:<16} {frames_str:<30}  {r['score']:>8.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Quick-check AIC 2026 pipeline")
    parser.add_argument("--task",     default="kis", choices=["kis", "qa", "trake"],
                        help="Task type: kis | qa | trake")
    parser.add_argument("--query",    default="Diễn giả mặc áo đỏ phát biểu tại hội nghị",
                        help="Query text for KIS / event description for Q&A / sequence text for TRAKE")
    parser.add_argument("--question", default="Có bao nhiêu người trên sân khấu?",
                        help="Question text (for Q&A only)")
    parser.add_argument("--top_k",   default=10, type=int, help="Number of results to show")
    parser.add_argument("--use_vlm", action="store_true", help="Enable Qwen2.5-VL offline engine for Q&A")
    parser.add_argument("--rerank",  action="store_true", help="Enable BLIP-2 re-ranking")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    keyframes_dir = os.path.join(data_dir, "keyframes")
    engine, encoder, processor = build_pipeline(data_dir)

    if args.task == "qa":
        run_qa(engine, encoder, processor, args.query, args.question, args.top_k, use_vlm=args.use_vlm, keyframes_dir=keyframes_dir)
    elif args.task == "trake":
        run_trake(engine, encoder, processor, args.query, args.top_k)
    else:
        run_kis(engine, encoder, processor, args.query, args.top_k, rerank=args.rerank, keyframes_dir=keyframes_dir)

    print("\n[Done]")


if __name__ == "__main__":
    main()
