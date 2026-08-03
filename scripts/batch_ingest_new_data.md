# Batch Ingest New Data Guide

Quy trình nạp thêm dữ liệu Batch 2 khi BTC công bố:
1. Đọc danh sách file mới từ `data/raw/batch_manifest.json`.
2. Trích xuất features cho batch mới mà không xóa index hiện tại (Incremental Indexing).
3. Append vector vào FAISS / Milvus index collection.
4. Cập nhật BM25 index với các document mới.
