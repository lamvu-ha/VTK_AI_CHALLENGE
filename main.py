import os
import sys

# ── Load torch FIRST before FAISS/MKL to prevent WinError 1114 DLL conflict ──
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # no CUDA DLL scan
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import torch as _t  # noqa: F401
except Exception:
    pass


# Reconfigure stdout/stderr to utf-8 for Windows console support
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indexing.dataset_loader import AICDatasetLoader
from indexing.vector_db.faiss_backup.feature_indexer import FeatureIndexer
from indexing.text_search.metadata_indexer import MetadataIndexer
from retrieval.query_processing.query_processor import QueryProcessor
from embedding_models.clip.clip_encoder import CLIPTextEncoder
from retrieval.fusion.hybrid_search import HybridSearchEngine
from task_modules.textual_kis.kis_solver import TextualKISSolver
from task_modules.qa.qa_solver import QASolver
from task_modules.trake.trake_solver import TRAKESolver
from ui.export.ranking_optimizer import RankingOptimizer
from ui.export.format_validator import AICFormatValidator


from typing import List, Dict, Any

class AICRetrievalPipeline:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        objects_dir = os.path.join(data_dir, "objects")
        self.loader = AICDatasetLoader(data_dir)

        features, keyframe_map = self.loader.load_video_dataset()
        dim = features.shape[1] if features.ndim > 1 else 512
        self.feature_indexer = FeatureIndexer(embedding_dim=dim)
        if features.shape[0] > 0:
            self.feature_indexer.build_index(features, keyframe_map)

        self.metadata_indexer = MetadataIndexer()
        v_ids = self.loader.get_all_video_ids()
        objects_available = os.path.exists(objects_dir)

        for v_id in v_ids:
            meta = self.loader.load_media_info(v_id)
            if meta:
                self.metadata_indexer.add_video_metadata(
                    v_id,
                    meta,
                    objects_dir=objects_dir if objects_available else None
                )

        self.metadata_indexer.build_bm25_index()

        self.query_processor = QueryProcessor()
        self.clip_encoder = CLIPTextEncoder()
        self.search_engine = HybridSearchEngine(self.feature_indexer, self.metadata_indexer, clip_encoder=self.clip_encoder)
        self.ranking_optimizer = RankingOptimizer(lambda_mmr=0.7)

        self.kis_solver = TextualKISSolver(self.search_engine, self.query_processor)
        self.qa_solver = QASolver(self.search_engine, self.query_processor)
        self.trake_solver = TRAKESolver(self.search_engine, self.query_processor)

    def search_kis(self, query_text: str, top_k: int = 100) -> List[Dict[str, Any]]:
        variants = self.query_processor.expand_queries(query_text)
        emb = self.clip_encoder.encode_text_ensemble(variants)
        raw = self.kis_solver.solve(query_text, emb, top_k=top_k)
        return self.ranking_optimizer.optimize_ranking(raw, max_items=top_k)

    def search_qa(self, event_text: str, question_text: str, top_k: int = 100) -> List[Dict[str, Any]]:
        combined = f"{event_text} {question_text}"
        variants = self.query_processor.expand_queries(combined)
        emb = self.clip_encoder.encode_text_ensemble(variants)
        raw = self.qa_solver.solve(event_text, question_text, emb, top_k=top_k)
        return self.ranking_optimizer.optimize_ranking(raw, max_items=top_k)

    def search_trake(self, trake_text: str, top_k: int = 100) -> List[Dict[str, Any]]:
        sub_events = self.query_processor.parse_trake_query(trake_text)
        event_embs = [self.clip_encoder.encode_text_ensemble(self.query_processor.expand_queries(ev)) for ev in sub_events]
        raw = self.trake_solver.solve(trake_text, event_embs, top_k=top_k)
        return self.ranking_optimizer.optimize_ranking(raw, max_items=top_k)


def build_pipeline() -> AICRetrievalPipeline:
    return AICRetrievalPipeline()


def main():
    print("=" * 65)
    print("  AIC 2026 VIDEO RETRIEVAL SYSTEM - SOTA PIPELINE")
    print("  Architecture: Hybrid BM25+CLIP → RRF → Object Detection → Two-Stage Rerank")
    print("=" * 65)

    pipeline = build_pipeline()
    output_dir = os.path.join(os.path.dirname(__file__), "submissions")
    os.makedirs(output_dir, exist_ok=True)
    validator = AICFormatValidator()

    # ─────────────────────────────────────────────────────────────────────
    # Task 1.1: Textual KIS
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("--- Task 1.1: Textual KIS ---")
    query_kis = "Diễn giả mặc áo đỏ phát biểu tại cuộc họp báo ngoài trời"
    print(f"Query: '{query_kis}'")

    # Multi-query expansion
    query_variants_kis = query_processor.expand_queries(query_kis)
    print(f"  Query variants ({len(query_variants_kis)}): {query_variants_kis}")

    # Ensemble embedding (average-pool all variants)
    emb_kis = clip_encoder.encode_text_ensemble(query_variants_kis)

    kis_raw = kis_solver.solve(query_kis, emb_kis, top_k=100)
    kis_results = ranking_optimizer.optimize_ranking(kis_raw, max_items=100)
    valid, msg = validator.validate_kis_submission(kis_results)
    out_kis = os.path.join(output_dir, "kis_submission.csv")
    validator.export_csv("query_kis_01", kis_results, out_kis)
    print(f"Status: {msg} | Predictions: {len(kis_results)} items")
    print(f"Top 3: {[(r['video_id'], r['frame_id']) for r in kis_results[:3]]}")
    print(f"Exported to: {out_kis}")

    # ─────────────────────────────────────────────────────────────────────
    # Task 1.2: Visual Question Answering (Q&A)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("--- Task 1.2: Visual Question Answering (Q&A) ---")
    event_qa = "Lễ trao giải thưởng âm nhạc"
    question_qa = "Trong video có bao nhiêu người lên sân khấu nhận giải?"
    print(f"Event: '{event_qa}' | Question: '{question_qa}'")

    combined_qa = f"{event_qa} {question_qa}"
    query_variants_qa = query_processor.expand_queries(combined_qa)
    emb_qa = clip_encoder.encode_text_ensemble(query_variants_qa)

    qa_raw = qa_solver.solve(event_qa, question_qa, emb_qa, top_k=100)
    qa_results = ranking_optimizer.optimize_ranking(qa_raw, max_items=100)
    valid, msg = validator.validate_qa_submission(qa_results)
    out_qa = os.path.join(output_dir, "qa_submission.csv")
    validator.export_csv("query_qa_01", qa_results, out_qa)
    print(f"Status: {msg} | Predictions: {len(qa_results)} items")
    print(f"Top 3: {[(r['video_id'], r['frame_id'], r.get('answer','?')) for r in qa_results[:3]]}")
    print(f"Exported to: {out_qa}")

    # ─────────────────────────────────────────────────────────────────────
    # Task 1.3: TRAKE (Temporal Retrieval and Alignment of Key Events)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("--- Task 1.3: TRAKE (DP Temporal Alignment with Gap Penalty) ---")
    trake_text = "(1) Giậm nhảy, (2) Bay qua xà, (3) Tiếp đất, (4) Đứng dậy"
    print(f"TRAKE Sequence: '{trake_text}'")

    sub_events = query_processor.parse_trake_query(trake_text)
    print(f"  Parsed sub-events ({len(sub_events)}): {sub_events}")

    # Encode each sub-event with multi-query expansion
    event_embs = []
    for ev in sub_events:
        ev_variants = query_processor.expand_queries(ev)
        ev_emb = clip_encoder.encode_text_ensemble(ev_variants)
        event_embs.append(ev_emb)

    trake_raw = trake_solver.solve(trake_text, event_embs, top_k=100)
    trake_results = ranking_optimizer.optimize_ranking(trake_raw, max_items=100)
    valid, msg = validator.validate_trake_submission(trake_results)
    out_trake = os.path.join(output_dir, "trake_submission.csv")
    validator.export_csv("query_trake_01", trake_results, out_trake)
    print(f"Status: {msg} | Predictions: {len(trake_results)} items")
    print(f"Top 3: {[(r['video_id'], r['frame_ids']) for r in trake_results[:3]]}")
    print(f"Exported to: {out_trake}")

    print("\n" + "=" * 65)
    print("  ALL PREDICTIONS COMPLETED & EXPORTED!")
    print("  Pipeline: Hybrid BM25+CLIP → RRF → DP-TRAKE → MMR Rank")
    print("=" * 65)


if __name__ == "__main__":
    main()
