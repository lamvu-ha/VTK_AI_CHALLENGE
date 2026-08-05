"""
Milvus index builder: đọc .npy embeddings → insert vào Milvus theo batch.
"""
import os
import numpy as np
from typing import Optional
from indexing.vector_db.milvus.schema import MILVUS_SCHEMA


class MilvusIndexBuilder:
    def __init__(self, host: str = "localhost", port: int = 19530):
        self.host = host
        self.port = port
        self.col = None
        self._connect()

    def _connect(self):
        try:
            from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility  # type: ignore
            connections.connect(host=self.host, port=str(self.port))
            self._pymilvus = True
            print(f"[+] Milvus connected: {self.host}:{self.port}")
        except Exception as e:
            print(f"[!] Milvus không khả dụng ({e}). Dùng FAISS thay thế.")
            self._pymilvus = False

    def build_from_npy_dir(self, npy_dir: str, model_name: str = "clip", batch_size: int = 1000):
        """
        Quét npy_dir tìm các file <video_id>/<frame_id>.npy,
        insert vào Milvus theo batch.
        """
        if not self._pymilvus:
            print("[!] Milvus chưa kết nối. Bỏ qua.")
            return

        from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility  # type: ignore
        cname = MILVUS_SCHEMA["collection_name"]

        rows = {"video_id": [], "frame_id": [], "model_name": [], "embedding_vector": []}
        for root, _, files in os.walk(npy_dir):
            for f in sorted(files):
                if not f.endswith(".npy"):
                    continue
                video_id = os.path.basename(root)
                frame_id = int(os.path.splitext(f)[0])
                emb = np.load(os.path.join(root, f)).astype(np.float32).flatten()
                rows["video_id"].append(video_id)
                rows["frame_id"].append(frame_id)
                rows["model_name"].append(model_name)
                rows["embedding_vector"].append(emb.tolist())

                if len(rows["video_id"]) >= batch_size:
                    Collection(cname).insert(list(rows.values()))
                    for v in rows.values(): v.clear()

        if rows["video_id"]:
            Collection(cname).insert(list(rows.values()))
        print(f"[+] Milvus insert done: {cname}")
