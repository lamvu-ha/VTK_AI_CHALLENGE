import os
import json
import csv
from typing import List, Dict, Any, Tuple


class AICFormatValidator:
    """
    Validator and Exporter for AIC 2026 submission formats.
    """

    def __init__(self):
        pass

    def validate_kis_submission(self, results: List[Dict[str, Any]]) -> Tuple[bool, str]:
        if len(results) > 100:
            return False, f"Too many predictions ({len(results)} > max 100)"

        for idx, item in enumerate(results):
            if "video_id" not in item or not item["video_id"]:
                return False, f"Item {idx}: missing or empty video_id"
            if "frame_id" not in item or not isinstance(item["frame_id"], (int, str)):
                return False, f"Item {idx}: missing or invalid frame_id"

        return True, "Valid Textual KIS submission format"

    def validate_qa_submission(self, results: List[Dict[str, Any]]) -> Tuple[bool, str]:
        if len(results) > 100:
            return False, f"Too many predictions ({len(results)} > max 100)"

        for idx, item in enumerate(results):
            if "video_id" not in item or not item["video_id"]:
                return False, f"Item {idx}: missing or empty video_id"
            if "frame_id" not in item or not isinstance(item["frame_id"], (int, str)):
                return False, f"Item {idx}: missing or invalid frame_id"
            if "answer" not in item or item["answer"] is None:
                return False, f"Item {idx}: missing answer string"

        return True, "Valid Q&A submission format"

    def validate_trake_submission(self, results: List[Dict[str, Any]]) -> Tuple[bool, str]:
        if len(results) > 100:
            return False, f"Too many predictions ({len(results)} > max 100)"

        for idx, item in enumerate(results):
            if "video_id" not in item or not item["video_id"]:
                return False, f"Item {idx}: missing or empty video_id"
            if "frame_ids" not in item or not isinstance(item["frame_ids"], list) or len(item["frame_ids"]) == 0:
                return False, f"Item {idx}: missing or empty frame_ids sequence"

        return True, "Valid TRAKE submission format"

    def export_csv(self, query_id: str, results: List[Dict[str, Any]], output_filepath: str) -> bool:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for item in results:
                if "answer" in item:
                    writer.writerow([item["video_id"], item["frame_id"], item["answer"]])
                elif "frame_ids" in item:
                    row = [item["video_id"]] + item["frame_ids"]
                    writer.writerow(row)
                else:
                    writer.writerow([item["video_id"], item["frame_id"]])
        return True
