from typing import List, Dict, Any, Set, Optional

class MetadataIndexer:
    """
    Indexer for Video Metadata and Faster R-CNN Detected Objects.
    Provides fast keyword-based filtering and scoring for candidate videos/keyframes.
    """

    def __init__(self):
        self.video_metadata: Dict[str, Dict[str, Any]] = {}
        self.keyframe_objects: Dict[str, Dict[int, Set[str]]] = {} # video_id -> {frame_id: set_of_object_labels}

    def add_video_metadata(self, video_id: str, metadata: Dict[str, Any]) -> None:
        """Stores video metadata (e.g. YouTube title, description, tags)."""
        self.video_metadata[video_id] = metadata

    def add_keyframe_objects(self, video_id: str, frame_id: int, detected_objects: List[str]) -> None:
        """Stores set of object labels detected in a keyframe."""
        if video_id not in self.keyframe_objects:
            self.keyframe_objects[video_id] = {}
        labels = {obj.lower().strip() for obj in detected_objects if isinstance(obj, str)}
        self.keyframe_objects[video_id][frame_id] = labels

    def search_metadata_by_keywords(self, keywords: List[str]) -> Dict[str, float]:
        """
        Calculates a relevance score for videos based on keyword matches in metadata.
        Returns a dict mapping video_id -> relevance_score.
        """
        scores: Dict[str, float] = {}
        keywords = [k.lower().strip() for k in keywords if k.strip()]

        for video_id, meta in self.video_metadata.items():
            text_content = ""
            if "title" in meta and isinstance(meta["title"], str):
                text_content += meta["title"].lower() + " "
            if "description" in meta and isinstance(meta["description"], str):
                text_content += meta["description"].lower() + " "
            if "tags" in meta and isinstance(meta["tags"], list):
                text_content += " ".join([str(t).lower() for t in meta["tags"]]) + " "

            score = 0.0
            for kw in keywords:
                if kw in text_content:
                    score += 1.0

            if score > 0:
                scores[video_id] = score

        return scores

    def filter_keyframes_by_objects(self, candidate_video_id: str, required_objects: List[str]) -> List[int]:
        """
        Returns frame_ids in candidate_video_id that contain all or most required object labels.
        """
        if candidate_video_id not in self.keyframe_objects:
            return []

        req_objs = {o.lower().strip() for o in required_objects}
        matched_frames = []

        for frame_id, detected in self.keyframe_objects[candidate_video_id].items():
            if req_objs.issubset(detected):
                matched_frames.append(frame_id)

        return matched_frames
