"""
format_submission.py — xuất đúng định dạng nộp bài cho từng loại truy vấn.
- KIS:   video_id + frame_id
- QA:    video_id + frame_id + answer
- TRAKE: video_id + [frame_ids]
Dùng lại AICFormatValidator từ format_validator.py.
"""
import os
import json
from typing import List, Dict, Any
from ui.export.format_validator import AICFormatValidator


def format_and_export(
    query_id: str,
    query_type: str,
    results: List[Dict[str, Any]],
    output_dir: str = "submissions",
) -> str:
    """
    Validate và xuất file nộp bài.
    
    Args:
        query_id: ID câu truy vấn
        query_type: "KIS" | "QA" | "TRAKE"
        results: list kết quả từ task module
        output_dir: thư mục lưu
    Returns:
        Đường dẫn file xuất
    """
    validator = AICFormatValidator()
    qtype = query_type.upper()

    if qtype == "KIS":
        ok, msg = validator.validate_kis_submission(results)
    elif qtype == "QA":
        ok, msg = validator.validate_qa_submission(results)
    elif qtype == "TRAKE":
        ok, msg = validator.validate_trake_submission(results)
    else:
        raise ValueError(f"Loại truy vấn không hợp lệ: {query_type}")

    if not ok:
        raise ValueError(f"Validation lỗi: {msg}")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{query_id}.csv")
    validator.export_csv(query_id, results[:100], csv_path)
    print(f"[+] Exported: {csv_path} ({len(results[:100])} rows)")
    return csv_path


def batch_export(queries: List[Dict], results_map: Dict[str, List], output_dir: str = "submissions"):
    """
    Xuất hàng loạt cho danh sách queries.
    queries: [{"query_id": ..., "type": ...}]
    results_map: {query_id: [results]}
    """
    exported = []
    for q in queries:
        qid = q["query_id"]
        if qid in results_map:
            path = format_and_export(qid, q["type"], results_map[qid], output_dir)
            exported.append(path)
    print(f"[+] Batch export: {len(exported)} files → {output_dir}")
    return exported
