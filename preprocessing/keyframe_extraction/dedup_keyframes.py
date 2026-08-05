"""
Dedup keyframes: loại bỏ các frame gần trùng bằng cosine similarity threshold.
"""
import os
from typing import List, Dict
import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def dedup_keyframes(
    keyframes: List[Dict],
    embeddings: List[np.ndarray],
    threshold: float = 0.95,
) -> List[Dict]:
    """
    Lọc danh sách keyframes bằng similarity threshold trên embedding.
    Giữ lại các frame đủ khác nhau (sim < threshold so với frame đã chọn).
    
    Args:
        keyframes: list of {video_id, frame_id, path, ...}
        embeddings: embedding tương ứng từng frame (same order)
        threshold: ngưỡng cosine similarity (default=0.95)
    Returns:
        Danh sách keyframes đã dedup.
    """
    if not keyframes or len(keyframes) != len(embeddings):
        return keyframes

    kept = []
    kept_embs = []

    for kf, emb in zip(keyframes, embeddings):
        if not kept or all(cosine_sim(emb, e) < threshold for e in kept_embs):
            kept.append(kf)
            kept_embs.append(emb)

    print(f"[+] Dedup: {len(keyframes)} → {len(kept)} keyframes (threshold={threshold})")
    return kept
