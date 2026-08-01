import json
import os
from typing import List, Dict, Any, Set, Optional, Tuple

from src.retrieval.bm25_engine import BM25Engine


class MetadataIndexer:
    """
    Enhanced Indexer for Video Metadata and Object Detections.
    
    Improvements over baseline:
    - Integrates BM25Engine for sparse text search (vs simple keyword counting)
    - Loads Faster R-CNN object detection JSONs from data/objects/
    - Supports weighted keyword scoring (title >> description >> tags >> objects)
    - Provides video-level and frame-level object filtering
    """

    def __init__(self):
        self.video_metadata: Dict[str, Dict[str, Any]] = {}
        self.keyframe_objects: Dict[str, Dict[int, Set[str]]] = {}  # video_id -> {frame_id: set}
        self.bm25_engine = BM25Engine(k1=1.5, b=0.75)
        self._bm25_built = False

    def add_video_metadata(
        self,
        video_id: str,
        metadata: Dict[str, Any],
        objects_dir: Optional[str] = None
    ) -> None:
        """
        Stores video metadata and optionally loads object detections.
        Args:
            video_id: e.g. "L01_V001"
            metadata: YouTube metadata dict (title, description, tags, ...)
            objects_dir: path to directory containing <video_id>.json object files
        """
        self.video_metadata[video_id] = metadata

        # Load object detections if available
        object_labels: List[str] = []
        if objects_dir:
            obj_path = os.path.join(objects_dir, f"{video_id}.json")
            if os.path.exists(obj_path):
                try:
                    with open(obj_path, 'r', encoding='utf-8') as f:
                        obj_data = json.load(f)
                    # Parse Faster R-CNN format: list of frame dicts or dict of frame_id->labels
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
                    elif isinstance(obj_data, dict):
                        for fid_str, labels in obj_data.items():
                            try:
                                fid = int(fid_str)
                                if isinstance(labels, list):
                                    label_set = {str(l).lower().strip() for l in labels if l}
                                    if video_id not in self.keyframe_objects:
                                        self.keyframe_objects[video_id] = {}
                                    self.keyframe_objects[video_id][fid] = label_set
                                    object_labels.extend(labels)
                            except (ValueError, TypeError):
                                continue
                except Exception:
                    pass

        # Register in BM25 engine (will be built lazily)
        self.bm25_engine.add_document(video_id, metadata, object_labels if object_labels else None)
        self._bm25_built = False  # Mark for rebuild

    def add_keyframe_objects(self, video_id: str, frame_id: int, detected_objects: List[str]) -> None:
        """Stores set of object labels detected in a specific keyframe."""
        if video_id not in self.keyframe_objects:
            self.keyframe_objects[video_id] = {}
        labels = {obj.lower().strip() for obj in detected_objects if isinstance(obj, str)}
        self.keyframe_objects[video_id][frame_id] = labels

    def build_bm25_index(self) -> None:
        """Explicitly build the BM25 index (call after all add_video_metadata calls)."""
        if not self._bm25_built:
            self.bm25_engine.build_index()
            self._bm25_built = True

    def search_bm25(self, query: str, top_k: int = 50) -> Dict[str, float]:
        """
        BM25 search over video metadata.
        Returns dict of video_id -> normalized BM25 score.
        """
        if not self._bm25_built:
            self.build_bm25_index()
        results = self.bm25_engine.search(query, top_k=top_k)
        if not results:
            return {}
        # Normalize scores to [0, 1]
        max_score = max(sc for _, sc in results) if results else 1.0
        if max_score == 0:
            max_score = 1.0
        return {vid: sc / max_score for vid, sc in results}

    def search_metadata_by_keywords(self, keywords: List[str]) -> Dict[str, float]:
        """
        Keyword search via BM25 (improved from simple substring matching).
        Returns dict mapping video_id -> relevance score.
        """
        if not keywords:
            return {}
        query = " ".join(keywords)
        return self.search_bm25(query, top_k=100)

    def filter_keyframes_by_objects(
        self,
        candidate_video_id: str,
        required_objects: List[str]
    ) -> List[int]:
        """
        Returns frame_ids in candidate_video_id that contain all required object labels.
        Partial match scoring: returns frames with highest overlap if no full match.
        """
        if candidate_video_id not in self.keyframe_objects:
            return []

        req_objs = {o.lower().strip() for o in required_objects}
        if not req_objs:
            return []

        matched_frames = []
        partial_frames: List[Tuple[int, int]] = []  # (frame_id, match_count)

        for frame_id, detected in self.keyframe_objects[candidate_video_id].items():
            intersection = req_objs & detected
            if intersection == req_objs:
                matched_frames.append(frame_id)
            elif intersection:
                partial_frames.append((frame_id, len(intersection)))

        if matched_frames:
            return sorted(matched_frames)

        # Fallback: return frames with highest partial match
        if partial_frames:
            best_count = max(cnt for _, cnt in partial_frames)
            return sorted(fid for fid, cnt in partial_frames if cnt == best_count)

        return []
