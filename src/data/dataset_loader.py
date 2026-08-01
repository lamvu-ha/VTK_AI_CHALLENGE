import os
import json
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class AICDatasetLoader:
    """
    Data loader and management class for AIC 2026 dataset.
    Loads CLIP features (.npy), Keyframe Maps (.csv), YouTube Metadata (.json), and Object JSONs.
    """

    def __init__(self, data_root: str):
        self.data_root = os.path.abspath(data_root)
        self.videos_dir = os.path.join(self.data_root, "videos")
        self.keyframes_dir = os.path.join(self.data_root, "keyframes")
        self.clip_features_dir = os.path.join(self.data_root, "clip-features-32")
        self.map_keyframes_dir = os.path.join(self.data_root, "map-keyframes")
        self.media_info_dir = os.path.join(self.data_root, "media-info")
        self.objects_dir = os.path.join(self.data_root, "objects")

    def get_all_video_ids(self) -> List[str]:
        """Returns sorted list of video IDs found in the clip-features or map-keyframes directory."""
        if os.path.exists(self.clip_features_dir):
            files = os.listdir(self.clip_features_dir)
            v_ids = [os.path.splitext(f)[0] for f in files if f.endswith('.npy')]
            return sorted(v_ids)
        elif os.path.exists(self.map_keyframes_dir):
            files = os.listdir(self.map_keyframes_dir)
            v_ids = [os.path.splitext(f)[0] for f in files if f.endswith('.csv')]
            return sorted(v_ids)
        return []

    def load_video_dataset(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Loads all .npy CLIP feature matrices and maps them to (video_id, frame_id, frame_idx, pts_time).
        Returns aggregated feature matrix (N, 512) and keyframe map list.
        """
        all_features = []
        all_keyframe_maps = []

        v_ids = self.get_all_video_ids()
        print(f"[+] Loading dataset features and maps for {len(v_ids)} videos...")

        for idx, video_id in enumerate(v_ids):
            npy_path = os.path.join(self.clip_features_dir, f"{video_id}.npy")
            csv_path = os.path.join(self.map_keyframes_dir, f"{video_id}.csv")

            if os.path.exists(npy_path) and os.path.exists(csv_path):
                features = np.load(npy_path)
                df = pd.read_csv(csv_path)

                if features.shape[0] == len(df):
                    all_features.append(features)
                    for _, row in df.iterrows():
                        frame_id = int(row["frame_idx"])
                        all_keyframe_maps.append({
                            "video_id": video_id,
                            "frame_id": frame_id,
                            "pts_time": float(row.get("pts_time", 0.0)),
                            "fps": float(row.get("fps", 30.0)),
                            "n": int(row.get("n", 0))
                        })

        if all_features:
            stacked_features = np.vstack(all_features).astype(np.float32)
            print(f"[+] Total keyframe embeddings loaded: {stacked_features.shape[0]} (dim={stacked_features.shape[1]})")
            return stacked_features, all_keyframe_maps
        else:
            return np.empty((0, 512), dtype=np.float32), []

    def load_media_info(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Loads YouTube video metadata JSON for a specific video."""
        json_path = os.path.join(self.media_info_dir, f"{video_id}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
