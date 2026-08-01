import os
import sys
import numpy as np

# Reconfigure stdout/stderr to utf-8 for Windows console support
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset_loader import AICDatasetLoader
from src.data.feature_indexer import FeatureIndexer
from src.data.metadata_indexer import MetadataIndexer
from src.retrieval.query_processor import QueryProcessor
from src.retrieval.clip_encoder import CLIPTextEncoder
from src.retrieval.hybrid_search import HybridSearchEngine
from src.modules.kis_solver import TextualKISSolver
from src.modules.qa_solver import QASolver
from src.modules.trake_solver import TRAKESolver
from src.submission.ranking_optimizer import RankingOptimizer
from src.submission.format_validator import AICFormatValidator

def main():
    print("=" * 65)
    print("  AIC 2026 VIDEO RETRIEVAL SYSTEM - FULL DATASET RUNNER")
    print("=" * 65)

    # 1. Initialize dataset loader
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    loader = AICDatasetLoader(data_dir)

    # 2. Load full 873 video dataset
    features, keyframe_map = loader.load_video_dataset()
    if features.shape[0] == 0:
        print("[!] No datasets found in data/ directory. Please run download_data.py first.")
        return

    # Build Indexers
    feature_indexer = FeatureIndexer(embedding_dim=features.shape[1])
    feature_indexer.build_index(features, keyframe_map)
    print(f"[+] Feature Indexer indexed {features.shape[0]} keyframes successfully.")

    metadata_indexer = MetadataIndexer()
    v_ids = loader.get_all_video_ids()
    print(f"[+] Loading YouTube metadata for {len(v_ids)} videos...")
    for v_id in v_ids:
        meta = loader.load_media_info(v_id)
        if meta:
            metadata_indexer.add_video_metadata(v_id, meta)

    # 3. Setup Components
    query_processor = QueryProcessor()
    clip_encoder = CLIPTextEncoder()
    search_engine = HybridSearchEngine(feature_indexer, metadata_indexer)
    ranking_optimizer = RankingOptimizer()
    validator = AICFormatValidator()

    # 4. Solvers
    kis_solver = TextualKISSolver(search_engine, query_processor)
    qa_solver = QASolver(search_engine, query_processor)
    trake_solver = TRAKESolver(search_engine, query_processor)

    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), "submissions")
    os.makedirs(output_dir, exist_ok=True)

    # --- Task 1.1: Textual KIS ---
    print("\n" + "=" * 60)
    print("--- Executing Task 1.1: Textual KIS ---")
    query_kis = "Diễn giả mặc áo đỏ phát biểu tại cuộc họp báo ngoài trời"
    print(f"Query: '{query_kis}'")
    emb_kis = clip_encoder.encode_text(query_kis)
    kis_raw = kis_solver.solve(query_kis, emb_kis, top_k=100)
    kis_results = ranking_optimizer.optimize_ranking(kis_raw, max_items=100)
    valid, msg = validator.validate_kis_submission(kis_results)
    out_kis = os.path.join(output_dir, "kis_submission.csv")
    validator.export_csv("query_kis_01", kis_results, out_kis)
    print(f"Status: {msg} | Predictions: {len(kis_results)} items")
    print(f"Top 3 Predictions: {kis_results[:3]}")
    print(f"Exported to: {out_kis}")

    # --- Task 1.2: Visual Question Answering (Q&A) ---
    print("\n" + "=" * 60)
    print("--- Executing Task 1.2: Visual Question Answering (Q&A) ---")
    event_qa = "Lễ trao giải thưởng âm nhạc"
    question_qa = "Trong video có bao nhiêu người lên sân khấu nhận giải?"
    print(f"Event: '{event_qa}' | Question: '{question_qa}'")
    emb_qa = clip_encoder.encode_text(f"{event_qa} {question_qa}")
    qa_raw = qa_solver.solve(event_qa, question_qa, emb_qa, top_k=100)
    qa_results = ranking_optimizer.optimize_ranking(qa_raw, max_items=100)
    valid, msg = validator.validate_qa_submission(qa_results)
    out_qa = os.path.join(output_dir, "qa_submission.csv")
    validator.export_csv("query_qa_01", qa_results, out_qa)
    print(f"Status: {msg} | Predictions: {len(qa_results)} items")
    print(f"Top 3 Predictions: {qa_results[:3]}")
    print(f"Exported to: {out_qa}")

    # --- Task 1.3: TRAKE ---
    print("\n" + "=" * 60)
    print("--- Executing Task 1.3: TRAKE (Temporal Alignment) ---")
    trake_text = "(1) Giậm nhảy, (2) Bay qua xà, (3) Tiếp đất, (4) Đứng dậy"
    print(f"TRAKE Sequence: '{trake_text}'")
    sub_events = query_processor.parse_trake_query(trake_text)
    event_embs = [clip_encoder.encode_text(ev) for ev in sub_events]
    trake_raw = trake_solver.solve(trake_text, event_embs, top_k=100)
    trake_results = ranking_optimizer.optimize_ranking(trake_raw, max_items=100)
    valid, msg = validator.validate_trake_submission(trake_results)
    out_trake = os.path.join(output_dir, "trake_submission.csv")
    validator.export_csv("query_trake_01", trake_results, out_trake)
    print(f"Status: {msg} | Predictions: {len(trake_results)} items")
    print(f"Top 3 Predictions: {trake_results[:3]}")
    print(f"Exported to: {out_trake}")

    print("\n" + "=" * 65)
    print("  ALL 873 VIDEO DATASET PREDICTIONS COMPLETED & EXPORTED!")
    print("=" * 65)

if __name__ == "__main__":
    main()
