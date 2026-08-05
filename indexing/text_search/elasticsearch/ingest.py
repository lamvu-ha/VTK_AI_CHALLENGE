"""
Elasticsearch ingest: nạp dữ liệu text (OCR/ASR/caption/metadata) vào ES.
"""
import os
import json
from typing import List, Dict, Optional


class ESIngest:
    def __init__(self, host: str = "localhost", port: int = 9200, index: str = "vtk_frames"):
        self.index = index
        self.es = None
        self._connect(host, port)

    def _connect(self, host: str, port: int):
        try:
            from elasticsearch import Elasticsearch, helpers  # type: ignore
            self.es = Elasticsearch([{"host": host, "port": port, "scheme": "http"}])
            self._helpers = helpers
            print(f"[+] Elasticsearch connected: {host}:{port}")
        except Exception as e:
            print(f"[!] Elasticsearch không khả dụng: {e}")

    def create_index(self, mapping_path: Optional[str] = None):
        if self.es is None:
            return
        mapping = {}
        if mapping_path and os.path.exists(mapping_path):
            with open(mapping_path, encoding="utf-8") as f:
                mapping = json.load(f)
        if not self.es.indices.exists(index=self.index):
            self.es.indices.create(index=self.index, body=mapping)
            print(f"[+] ES index created: {self.index}")

    def ingest_docs(self, docs: List[Dict], batch_size: int = 500):
        """Nạp danh sách doc vào ES."""
        if self.es is None:
            return
        actions = [{"_index": self.index, "_source": doc} for doc in docs]
        for i in range(0, len(actions), batch_size):
            self._helpers.bulk(self.es, actions[i:i + batch_size])
        print(f"[+] ES ingested: {len(docs)} docs → {self.index}")

    def ingest_from_json(self, json_path: str):
        """Load JSON array và nạp vào ES."""
        with open(json_path, encoding="utf-8") as f:
            docs = json.load(f)
        self.ingest_docs(docs)
