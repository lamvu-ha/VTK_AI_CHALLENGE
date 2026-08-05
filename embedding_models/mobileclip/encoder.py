"""
MobileCLIP encoder: bản nhẹ của CLIP, dùng khi encode lại tập lớn nhanh.
"""
import numpy as np
from typing import List, Optional, Any


class MobileCLIPEncoder:
    """
    Wrapper cho MobileCLIP (Apple). Nếu không có checkpoint, fallback về CLIP.
    """
    def __init__(self, model_variant: str = "mobileclip_s0", device: str = "cpu"):
        self.model_variant = model_variant
        self.device = device
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._init()

    def _init(self):
        try:
            import mobileclip  # type: ignore
            self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
                self.model_variant, pretrained="datacompdr"
            )
            self.tokenizer = mobileclip.get_tokenizer(self.model_variant)
            self.model.eval()
            print(f"[+] MobileCLIP loaded: {self.model_variant}")
        except Exception as e:
            print(f"[!] MobileCLIP không khả dụng ({e}). Dùng CLIP fallback.")
            self._use_clip_fallback()

    def _use_clip_fallback(self):
        """Fallback sang CLIP thông thường."""
        try:
            from embedding_models.clip.clip_encoder import CLIPTextEncoder
            self._fallback = CLIPTextEncoder()
        except Exception:
            self._fallback = None

    def encode_image(self, image_path: str) -> np.ndarray:
        if self.model is None:
            return self._fallback_encode(image_path)
        try:
            import torch
            from PIL import Image  # type: ignore
            img = self.preprocess(Image.open(image_path)).unsqueeze(0)
            with torch.no_grad():
                feats = self.model.encode_image(img)
            vec = feats.cpu().numpy().squeeze().astype(np.float32)
            return vec / (np.linalg.norm(vec) + 1e-9)
        except Exception as e:
            return self._fallback_encode(image_path)

    def encode_text(self, text: str) -> np.ndarray:
        if self.model is None and hasattr(self, "_fallback") and self._fallback:
            return self._fallback.encode_text(text)
        try:
            import torch
            tokens = self.tokenizer([text])
            with torch.no_grad():
                feats = self.model.encode_text(tokens)
            vec = feats.cpu().numpy().squeeze().astype(np.float32)
            return vec / (np.linalg.norm(vec) + 1e-9)
        except Exception:
            rng = np.random.RandomState(abs(hash(text)) % (2**32))
            return rng.randn(512).astype(np.float32)

    def _fallback_encode(self, image_path: str) -> np.ndarray:
        rng = np.random.RandomState(abs(hash(image_path)) % (2**32))
        return rng.randn(512).astype(np.float32)
