"""
faster-whisper ASR pipeline for Vietnamese video transcription.
Outputs per-video JSON: [{start, end, text}, ...]
"""
import os
import json
from typing import List, Dict, Optional


class WhisperASRPipeline:
    def __init__(self, model_size: str = "medium", device: str = "cpu", language: str = "vi"):
        self.model_size = model_size
        self.device = device
        self.language = language
        self.model = None
        self._init()

    def _init(self):
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(self.model_size, device=self.device, compute_type="int8")
            print(f"[+] faster-whisper loaded: model={self.model_size}, device={self.device}")
        except Exception as e:
            print(f"[!] faster-whisper not available ({e}). ASR skipped.")

    def transcribe_video(self, video_path: str) -> List[Dict]:
        """Transcribe one video. Returns [{start, end, text}]."""
        if self.model is None or not os.path.exists(video_path):
            return []
        try:
            segments, _ = self.model.transcribe(video_path, language=self.language, beam_size=5)
            return [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments]
        except Exception as e:
            print(f"[!] ASR failed for {video_path}: {e}")
            return []

    def batch_transcribe(self, video_dir: str, output_dir: str, exts: tuple = (".mp4", ".avi", ".mkv")):
        """
        Transcribe all videos in video_dir.
        Saves {video_id}.json → output_dir for each video.
        """
        os.makedirs(output_dir, exist_ok=True)
        for fname in sorted(os.listdir(video_dir)):
            if not fname.lower().endswith(exts):
                continue
            video_id = os.path.splitext(fname)[0]
            out_path = os.path.join(output_dir, f"{video_id}.json")
            if os.path.exists(out_path):
                print(f"  [skip] {video_id} already transcribed.")
                continue
            segments = self.transcribe_video(os.path.join(video_dir, fname))
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)
            print(f"  ASR [{video_id}]: {len(segments)} segments")
        print(f"[+] ASR batch done → {output_dir}")
