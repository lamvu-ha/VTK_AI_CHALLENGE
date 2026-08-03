# AIC 2026 Retrieval System Methodology

## 1. Overview
Hệ thống kết hợp đa mô hình visual-language embedding (CLIP, SigLIP2, BEiT-3), truy vấn văn bản thưa (BM25 / Elasticsearch) và Re-ranking hai giai đoạn (BLIP-2 / LLM).

## 2. Core Modules
- **First Stage Retrieval**: ANN search + Sparse BM25 text search.
- **Fusion Layer**: Reciprocal Rank Fusion (RRF) giúp kết hợp thứ tự xếp hạng không cần chuẩn hóa điểm số gốc.
- **TRAKE Temporal Alignment**: Beam search kết hợp temporal decay penalty định vị mốc thời gian sự kiện.
- **Frame Refinement**: Tinh chỉnh cửa sổ nhỏ quanh keyframe ứng viên (< 10 frames).
