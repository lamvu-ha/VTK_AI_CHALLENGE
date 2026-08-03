import os
import json
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class AICDatasetLoader:
    """
    Optimized Data loader and management class for AIC 2026 dataset.
    Loads CLIP features (.npy), Keyframe Maps (.csv), YouTube Metadata (.json), and Object JSONs.
    Supports both legacy structure and new structured data/ directory.
    """

    def __init__(self, data_root: str):
        self.data_root = os.path.abspath(data_root)
        self.videos_dir = os.path.join(self.data_root, "raw", "videos") if os.path.exists(os.path.join(self.data_root, "raw", "videos")) else os.path.join(self.data_root, "videos")
        self.keyframes_dir = os.path.join(self.data_root, "keyframes")
        
        # Check new vs legacy paths
        self.clip_features_dir = os.path.join(self.data_root, "clip_features") if os.path.exists(os.path.join(self.data_root, "clip_features")) else os.path.join(self.data_root, "clip-features-32")
        self.map_keyframes_dir = os.path.join(self.data_root, "map-keyframes")
        self.media_info_dir = os.path.join(self.data_root, "raw", "metadata_youtube") if os.path.exists(os.path.join(self.data_root, "raw", "metadata_youtube")) else os.path.join(self.data_root, "media-info")
        self.objects_dir = os.path.join(self.data_root, "objects")

        # Pick media-info dir: prefer whichever actually contains JSON files
        _yt_meta = os.path.join(self.data_root, "raw", "metadata_youtube")
        _media_info = os.path.join(self.data_root, "media-info")
        def _has_json(d): return os.path.isdir(d) and any(f.endswith(".json") for f in os.listdir(d))
        if _has_json(_media_info):
            self.media_info_dir = _media_info
        elif _has_json(_yt_meta):
            self.media_info_dir = _yt_meta
        else:
            self.media_info_dir = _media_info  # default fallback

    def get_all_video_ids(self) -> List[str]:
        """Returns sorted list of video IDs found in the clip-features or map-keyframes directory."""
        if os.path.exists(self.clip_features_dir):
            files = os.listdir(self.clip_features_dir)
            v_ids = [os.path.splitext(f)[0] for f in files if f.endswith('.npy')]
            if v_ids:
                return sorted(v_ids)
        if os.path.exists(self.map_keyframes_dir):
            files = os.listdir(self.map_keyframes_dir)
            v_ids = [os.path.splitext(f)[0] for f in files if f.endswith('.csv')]
            return sorted(v_ids)
        return []

    def load_video_dataset(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Loads all .npy CLIP feature matrices and maps them to keyframes.
        """
        all_features = []
        all_keyframe_maps = []

        v_ids = self.get_all_video_ids()
        print(f"[+] Fast loading features and keyframe maps for {len(v_ids)} videos...")

        for idx, video_id in enumerate(v_ids):
            npy_path = os.path.join(self.clip_features_dir, f"{video_id}.npy")
            csv_path = os.path.join(self.map_keyframes_dir, f"{video_id}.csv")

            if os.path.exists(npy_path) and os.path.exists(csv_path):
                features = np.load(npy_path)
                df = pd.read_csv(csv_path)

                if features.shape[0] == len(df):
                    all_features.append(features)
                    
                    frame_idxs = df["frame_idx"].astype(int).values
                    pts_times = df["pts_time"].astype(float).values if "pts_time" in df else np.zeros(len(df))
                    fps_vals = df["fps"].astype(float).values if "fps" in df else np.full(len(df), 30.0)
                    n_vals = df["n"].astype(int).values if "n" in df else np.zeros(len(df), dtype=int)

                    for fid, pts, fps, n in zip(frame_idxs, pts_times, fps_vals, n_vals):
                        all_keyframe_maps.append({
                            "video_id": video_id,
                            "frame_id": int(fid),
                            "pts_time": float(pts),
                            "fps": float(fps),
                            "n": int(n)
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
        if not os.path.exists(json_path):
            return None
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                with open(json_path, "r", encoding=enc) as f:
                    return json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            except Exception:
                break
        return None
