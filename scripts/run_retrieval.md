# Run Retrieval End-to-End Guide

Quy trình chạy 1 truy vấn end-to-end:
1. Nhập câu query (Textual KIS / Q&A / TRAKE).
2. `query_processing`: Mở rộng query (LLM expansion) hoặc tách chuỗi event (TRAKE decomposition).
3. `first_stage`: ANN Top-100 search qua Vector DB / FAISS + BM25 text search.
4. `fusion`: Kết hợp điểm số bằng Reciprocal Rank Fusion (RRF).
5. `reranking`: BLIP-2 cross-attention re-rank top-100.
6. `export`: Xuất file csv/json kết quả sẵn sàng nộp bài.
