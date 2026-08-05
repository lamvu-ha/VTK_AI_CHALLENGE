import json
import os
from typing import List, Dict, Any, Set, Optional, Tuple

from indexing.text_search.bm25_engine import BM25Engine


class MetadataIndexer:
    """
    Enhanced Indexer for Video Metadata and Object Detections.
    """

    def __init__(self):
        self.video_metadata: Dict[str, Dict[str, Any]] = {}
        self.keyframe_objects: Dict[str, Dict[int, Set[str]]] = {}
        self.ocr_texts: Dict[str, str] = {}   # video_id -> aggregated OCR text
        self.asr_texts: Dict[str, str] = {}   # video_id -> aggregated ASR transcript
        self.bm25_engine = BM25Engine(k1=1.5, b=0.75)
        self._bm25_built = False

    def add_video_metadata(
        self,
        video_id: str,
        metadata: Dict[str, Any],
        objects_dir: Optional[str] = None
    ) -> None:
        self.video_metadata[video_id] = metadata

        object_labels: List[str] = []
        if objects_dir:
            # Check directory-per-video structure: data/objects/<video_id>/<frame_id>.json
            vid_obj_dir = os.path.join(objects_dir, video_id)
            if os.path.isdir(vid_obj_dir):
                try:
                    for fname in os.listdir(vid_obj_dir):
                        if not fname.endswith(".json"):
                            continue
                        fid_str = os.path.splitext(fname)[0]
                        try:
                            fid = int(fid_str)
                        except ValueError:
                            continue
                        fpath = os.path.join(vid_obj_dir, fname)
                        with open(fpath, 'r', encoding='utf-8') as f:
                            f_data = json.load(f)
                        labels = f_data.get("detection_class_entities", f_data.get("labels", f_data.get("objects", [])))
                        if isinstance(labels, list) and labels:
                            label_set = {str(l).lower().strip() for l in labels if l}
                            if video_id not in self.keyframe_objects:
                                self.keyframe_objects[video_id] = {}
                            self.keyframe_objects[video_id][fid] = label_set
                            object_labels.extend(list(label_set))
                except Exception:
                    pass
            else:
                # Legacy single JSON file per video check
                obj_path = os.path.join(objects_dir, f"{video_id}.json")
                if os.path.exists(obj_path):
                    try:
                        with open(obj_path, 'r', encoding='utf-8') as f:
                            obj_data = json.load(f)
                        if isinstance(obj_data, list):
                            for frame_entry in obj_data:
                                if isinstance(frame_entry, dict):
                                    fid = frame_entry.get("frame_idx", frame_entry.get("frame_id", -1))
                                    labels = frame_entry.get("labels", frame_entry.get("objects", []))
                                    if isinstance(labels, list) and fid >= 0:
                                        label_set = {str(l).lower().strip() for l in labels if l}
                                        if video_id not in self.keyframe_objects:
                                            self.keyframe_objects[video_id] = {}
                                        self.keyframe_objects[video_id][int(fid)] = label_set
                                        object_labels.extend(labels)
                    except Exception:
                        pass

        extra_texts = []
        if video_id in self.ocr_texts:
            extra_texts.append(self.ocr_texts[video_id])
        if video_id in self.asr_texts:
            extra_texts.append(self.asr_texts[video_id])
        self.bm25_engine.add_document(video_id, metadata, (object_labels + extra_texts) if (object_labels or extra_texts) else None)
        self._bm25_built = False

    def add_keyframe_objects(self, video_id: str, frame_id: int, detected_objects: List[str]) -> None:
        if video_id not in self.keyframe_objects:
            self.keyframe_objects[video_id] = {}
        labels = {obj.lower().strip() for obj in detected_objects if isinstance(obj, str)}
        self.keyframe_objects[video_id][frame_id] = labels

    def add_ocr_text(self, video_id: str, ocr_text: str) -> None:
        """Add OCR-extracted text for a video (aggregated from all its keyframes)."""
        existing = self.ocr_texts.get(video_id, "")
        self.ocr_texts[video_id] = (existing + " " + ocr_text).strip()
        self._bm25_built = False

    def add_asr_text(self, video_id: str, asr_text: str) -> None:
        """Add ASR transcript text for a video."""
        existing = self.asr_texts.get(video_id, "")
        self.asr_texts[video_id] = (existing + " " + asr_text).strip()
        self._bm25_built = False

    def load_ocr_json(self, ocr_json_path: str) -> None:
        """
        Load OCR results from a JSON file produced by PaddleOCRPipeline.batch_ocr().
        Expected format: {"Lxx_Vxxx/000001.jpg": "detected text", ...}
        Aggregates text per video_id.
        """
        import json
        if not os.path.exists(ocr_json_path):
            return
        with open(ocr_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for rel_path, text in data.items():
            # Extract video_id from path like "L01_V001/000001.jpg"
            parts = rel_path.replace("\\", "/").split("/")
            video_id = parts[0] if parts else ""
            if video_id and text:
                self.add_ocr_text(video_id, text)

    def load_asr_dir(self, asr_dir: str) -> None:
        """
        Load ASR JSON files from directory produced by WhisperASRPipeline.batch_transcribe().
        Each file: {video_id}.json containing [{start, end, text}, ...]
        """
        import json
        if not os.path.exists(asr_dir):
            return
        for fname in os.listdir(asr_dir):
            if not fname.endswith(".json"):
                continue
            video_id = os.path.splitext(fname)[0]
            with open(os.path.join(asr_dir, fname), "r", encoding="utf-8") as f:
                segments = json.load(f)
            asr_text = " ".join(s.get("text", "") for s in segments if isinstance(s, dict))
            if asr_text:
                self.add_asr_text(video_id, asr_text)

    def build_bm25_index(self) -> None:
        if not self._bm25_built:
            self.bm25_engine.build_index()
            self._bm25_built = True

    def search_bm25(self, query: str, top_k: int = 50) -> Dict[str, float]:
        if not self._bm25_built:
            self.build_bm25_index()
        results = self.bm25_engine.search(query, top_k=top_k)
        if not results:
            return {}
        max_score = max(sc for _, sc in results) if results else 1.0
        if max_score == 0:
            max_score = 1.0
        return {vid: sc / max_score for vid, sc in results}

    def search_metadata_by_keywords(self, keywords: List[str]) -> Dict[str, float]:
        if not keywords:
            return {}
        query = " ".join(keywords)
        return self.search_bm25(query, top_k=100)

    def filter_keyframes_by_objects(
        self,
        candidate_video_id: str,
        required_objects: List[str]
    ) -> List[int]:
        if candidate_video_id not in self.keyframe_objects:
            return []

        req_objs = {o.lower().strip() for o in required_objects}
        if not req_objs:
            return []

        matched_frames = []
        partial_frames: List[Tuple[int, int]] = []

        for frame_id, detected in self.keyframe_objects[candidate_video_id].items():
            intersection = req_objs & detected
            if intersection == req_objs:
                matched_frames.append(frame_id)
            elif intersection:
                partial_frames.append((frame_id, len(intersection)))

        if matched_frames:
            return sorted(matched_frames)

        if partial_frames:
            best_count = max(cnt for _, cnt in partial_frames)
            return sorted(fid for fid, cnt in partial_frames if cnt == best_count)

        return []
