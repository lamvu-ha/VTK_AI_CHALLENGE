"""
R-Score metric theo mục 2.1 thể lệ AIC 2026.
KIS:   R = max(0, 1 - |pred_frame - gt_center| / gt_half_length)
QA:    R = 1 nếu câu trả lời đúng, else 0
TRAKE: R = trung bình R của từng event
"""
from typing import List, Dict, Any, Optional


def _r_score_single(pred_frame: int, gt_start: int, gt_end: int) -> float:
    """
    Tính R-score cho 1 frame so với ground truth [gt_start, gt_end].
    R = max(0, 1 - |pred - center| / half_len)
    """
    center = (gt_start + gt_end) / 2.0
    half_len = max(1, (gt_end - gt_start) / 2.0)
    return max(0.0, 1.0 - abs(pred_frame - center) / half_len)


def r_score_kis(pred_frame: int, gt: Dict[str, int]) -> float:
    """
    R-score cho Textual KIS.
    gt = {"start": s, "end": e}
    """
    return _r_score_single(pred_frame, gt["start"], gt["end"])


def r_score_qa(pred_answer: str, gt_answer: str, pred_frame: int, gt: Dict[str, int]) -> float:
    """
    R-score cho Q&A: phải vừa định vị đúng frame vừa trả lời đúng.
    """
    loc_score = _r_score_single(pred_frame, gt["start"], gt["end"])
    ans_correct = (
        pred_answer.strip().lower() == gt_answer.strip().lower()
        or gt_answer.strip().lower() in pred_answer.strip().lower()
    )
    return loc_score if ans_correct else 0.0


def r_score_trake(
    pred_frames: List[int],
    gt_events: List[Dict[str, int]],
) -> float:
    """
    R-score cho TRAKE: trung bình R của từng event.
    pred_frames: [f1, f2, ...] — frame dự đoán cho từng event
    gt_events: [{"start": s, "end": e}, ...] — ground truth từng event
    """
    if not gt_events:
        return 0.0
    n = min(len(pred_frames), len(gt_events))
    scores = [
        _r_score_single(pred_frames[i], gt_events[i]["start"], gt_events[i]["end"])
        for i in range(n)
    ]
    return sum(scores) / len(gt_events)


def calculate_r_score(
    query_type: str,
    ground_truth: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    rank: int = 1,
) -> float:
    """
    API chung: tính R-score cho một query ở rank cụ thể.
    
    Args:
        query_type: "KIS" | "QA" | "TRAKE"
        ground_truth: {"start": int, "end": int} hoặc {"events": [...], "answer": str}
        predictions: list of result dicts đã sort theo score
        rank: dùng prediction thứ `rank` (1-indexed)
    """
    if not predictions or rank > len(predictions):
        return 0.0

    pred = predictions[rank - 1]
    pred_frame = pred.get("frame_id", 0)

    if query_type.upper() == "KIS":
        return r_score_kis(pred_frame, ground_truth)

    elif query_type.upper() == "QA":
        return r_score_qa(
            pred.get("answer", ""),
            ground_truth.get("answer", ""),
            pred_frame,
            ground_truth,
        )

    elif query_type.upper() == "TRAKE":
        pred_frames = pred.get("frame_ids", [pred_frame])
        return r_score_trake(pred_frames, ground_truth.get("events", [ground_truth]))

    return 0.0
