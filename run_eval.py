"""
run_eval.py — chạy toàn bộ pipeline trên local_dev_queries và in Final Score ước tính.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from evaluation.metrics.r_score import calculate_r_score
from evaluation.metrics.final_score import final_score


def load_queries(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dummy_pipeline_run(query: dict) -> list:
    """
    Placeholder: gọi pipeline thật khi có. Hiện trả về empty list.
    Thay thế bằng: main.run_query(query) hoặc từng task module.
    """
    return []


def run_eval(queries_path: str = "evaluation/local_dev_queries/sample_queries.json"):
    queries = load_queries(queries_path)
    all_scores = []

    for q in queries:
        qtype = q["type"]
        gt = q["ground_truth"]
        predictions = dummy_pipeline_run(q)

        scores = final_score(qtype, gt, predictions)
        print(f"[{q['query_id']}] {qtype}: R@1={scores['R@1']:.3f} | Final={scores['final']:.3f}")
        all_scores.append(scores["final"])

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n=== Overall Final Score (estimated): {overall:.4f} ===")
    return overall


if __name__ == "__main__":
    queries_path = sys.argv[1] if len(sys.argv) > 1 else "evaluation/local_dev_queries/sample_queries.json"
    run_eval(queries_path)
