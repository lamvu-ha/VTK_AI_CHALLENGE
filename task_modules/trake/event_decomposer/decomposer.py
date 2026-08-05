"""
TRAKE event decomposer: tách truy vấn TRAKE thành chuỗi event có nhãn thứ tự.
Wrapper dùng chung EventSplitter từ query_decomposition.
"""
from typing import List
from retrieval.query_processing.query_decomposition.event_splitter import EventSplitter


class TRAKEDecomposer:
    """
    Tách truy vấn TRAKE thành danh sách event E1, E2, ...
    """
    def __init__(self, api_key=None):
        self.splitter = EventSplitter(api_key=api_key)

    def decompose(self, trake_query: str) -> List[str]:
        """
        Trả về [(E1_description, E2_description, ...)]
        """
        events = self.splitter.split(trake_query)
        print(f"[+] TRAKE decomposed into {len(events)} events.")
        return events
