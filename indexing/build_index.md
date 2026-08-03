# Index Building Workflow

Thứ tự thực hiện build index:
1. **Shots Boundary Detection**: Tách video thành các shot cảnh.
2. **Keyframe Extraction**: Tách/chọn keyframe đại diện.
3. **Feature Extraction**: Sinh embedding bằng CLIP, BEiT-3, SigLIP2.
4. **Text Indexing**: Index OCR, ASR, YouTube Metadata và Captioning vào Elasticsearch / BM25.
5. **Vector Indexing**: Nạp embedding vào FAISS / Milvus vector DB.
