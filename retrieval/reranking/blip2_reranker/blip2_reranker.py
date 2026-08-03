"""
BLIP-2 Re-ranker: re-scores top-K candidate frames using
image-text matching (ITM) from BLIP-2.
Requires keyframe images on disk.
"""
import os
import numpy as np
from typing import List, Dict, Any, Optional


class BLIP2Reranker:
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._init()

    def _init(self):
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name, device_map=self.device
            )
            self.model.eval()
            print(f"[+] BLIP-2 loaded: {self.model_name}")
        except Exception as e:
            print(f"[!] BLIP-2 load failed ({e}). Re-ranking skipped.")
            self.model = None

    def _score(self, image_path: str, text: str) -> float:
        """Returns a float ITM-like score for (image, text) pair."""
        try:
            from PIL import Image
            import torch
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                # Use language modelling loss as a proxy: lower loss = better match
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                score = -outputs.loss.item()   # negate: higher is better
            return float(score)
        except Exception:
            return 0.0

    def rerank(
        self,
        candidates: List[Dict[str, Any]],
        query_text: str,
        keyframes_dir: str,
        top_n: int = 100,
        blip_top_k: int = 30,      # only re-rank the first K for speed
    ) -> List[Dict[str, Any]]:
        """
        Re-rank candidates list.
        `keyframes_dir` layout expected: <keyframes_dir>/<video_id>/<frame_id>.jpg
        Candidates outside top blip_top_k keep their original order.
        """
        if self.model is None or not candidates:
            return candidates

        to_rerank = candidates[:blip_top_k]
        rest = candidates[blip_top_k:]

        scored = []
        for cand in to_rerank:
            v_id = cand.get("video_id", "")
            f_id = cand.get("frame_id", 0)
            # Try common keyframe path conventions
            img_path = os.path.join(keyframes_dir, v_id, f"{int(f_id):06d}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(keyframes_dir, v_id, f"{int(f_id)}.jpg")
            blip_score = self._score(img_path, query_text) if os.path.exists(img_path) else 0.0
            # Combine original retrieval score + BLIP-2 score
            combined = cand.get("score", 0.0) + 0.3 * blip_score
            scored.append({**cand, "score": combined, "blip_score": blip_score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return (scored + rest)[:top_n]
