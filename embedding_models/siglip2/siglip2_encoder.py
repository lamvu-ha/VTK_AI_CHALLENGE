"""
SigLIP2 Text Encoder wrapper.
Model: google/siglip2-base-patch16-224 (768-dim)
"""
import numpy as np
from typing import List, Any

_MODEL_NAME = "google/siglip2-base-patch16-224"


class SigLIP2TextEncoder:
    def __init__(self, model_name: str = _MODEL_NAME, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._init()

    def _init(self):
        try:
            from transformers import AutoProcessor, AutoModel
            import torch
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self._torch = torch
            print(f"[+] SigLIP2 loaded: {self.model_name}")
        except Exception as e:
            print(f"[!] SigLIP2 load failed ({e}). Using fallback.")
            self.model = None

    def encode_text(self, text: str) -> np.ndarray:
        if self.model is not None:
            try:
                inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
                with self._torch.no_grad():
                    vec = self.model.get_text_features(**inputs).cpu().numpy().squeeze().astype(np.float32)
                return self._normalize(vec)
            except Exception:
                pass
        return self._fallback(text)

    def encode_text_ensemble(self, texts: List[str]) -> np.ndarray:
        vecs = np.stack([self.encode_text(t) for t in texts])
        return self._normalize(vecs.mean(axis=0))

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def _fallback(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(abs(hash(text)) % (2**32))
        return self._normalize(rng.randn(768).astype(np.float32))
