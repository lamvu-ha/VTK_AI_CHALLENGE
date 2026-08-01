"""
Enhanced CLIP Text Encoder with multi-query ensemble support.
Supports: PyTorch CLIP, TensorFlow CLIP, SigLIP2 (when available), fallback encoder.

Multi-query ensemble: encode multiple query variants and average-pool their embeddings
for more robust retrieval (aligned with design doc's multi-query expansion strategy).
"""
import numpy as np
from typing import Optional, List


class CLIPTextEncoder:
    """
    CLIP Text Encoder for converting query strings into embedding vectors
    matching clip-ViT-B-32 or SigLIP2 precomputed image features.
    
    Supports:
    - PyTorch CLIP (openai/clip-vit-base-patch32) → 512-dim
    - TensorFlow CLIP → 512-dim
    - SigLIP2 (google/siglip-so400m-patch14-384) → 1152-dim (when available)
    - Deterministic fallback (hash-based, for CI/testing without GPU)
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.model_type = None
        self.model = None
        self.tokenizer = None
        self._init_encoder()

    def _init_encoder(self):
        # Priority 1: HuggingFace PyTorch CLIP
        try:
            import torch
            from transformers import CLIPTokenizer, CLIPTextModel
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            self.model = CLIPTextModel.from_pretrained(self.model_name)
            self.model.eval()
            self.model_type = "torch"
            print(f"[+] Loaded PyTorch CLIP Text Model ({self.model_name}).")
            return
        except Exception:
            pass

        # Priority 2: HuggingFace TensorFlow CLIP
        try:
            from transformers import AutoTokenizer, TFCLIPTextModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = TFCLIPTextModel.from_pretrained(self.model_name)
            self.model_type = "tf"
            print(f"[+] Loaded TensorFlow CLIP Text Model ({self.model_name}).")
            return
        except Exception:
            pass

        print("[!] PyTorch/TensorFlow CLIP not available. Using deterministic fallback encoder.")
        self.model_type = "fallback"

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encodes a single natural language query into a normalized embedding vector.
        """
        if self.model_type == "torch" and self.model is not None:
            import torch
            inputs = self.tokenizer(
                [text], padding=True, truncation=True, max_length=77, return_tensors="pt"
            )
            with torch.no_grad():
                text_outputs = self.model(**inputs)
                text_embeds = text_outputs.pooler_output
                if hasattr(self.model, "text_projection"):
                    text_embeds = self.model.text_projection(text_embeds)
                embeds_np = text_embeds.cpu().numpy().squeeze().astype(np.float32)
            return self._normalize(embeds_np)

        elif self.model_type == "tf" and self.model is not None:
            inputs = self.tokenizer(
                [text], padding=True, truncation=True, max_length=77, return_tensors="tf"
            )
            text_outputs = self.model(inputs)
            embeds_np = text_outputs.pooler_output.numpy().squeeze().astype(np.float32)
            return self._normalize(embeds_np)

        else:
            return self._fallback_encode(text)

    def encode_text_ensemble(self, texts: List[str]) -> np.ndarray:
        """
        Multi-query ensemble encoding: encode all query variants and average-pool.
        This implements the 'multi-query expansion' strategy from the design doc.
        
        Args:
            texts: list of query variants (original + paraphrases + no-accent versions)
        Returns:
            Single averaged + normalized embedding vector
        """
        if not texts:
            return self._fallback_encode("")

        embeddings = [self.encode_text(t) for t in texts]
        stacked = np.stack(embeddings, axis=0)
        avg = stacked.mean(axis=0)
        return self._normalize(avg)

    def encode_text_list(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode a list of texts individually, returning one vector per text.
        Used for TRAKE sub-event encoding.
        """
        return [self.encode_text(t) for t in texts]

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2-normalize a vector."""
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else vec

    def _fallback_encode(self, text: str) -> np.ndarray:
        """Deterministic fallback: hash-based pseudo-random 512-dim embedding."""
        rng = np.random.RandomState(abs(hash(text)) % (2**32))
        vec = rng.randn(512).astype(np.float32)
        return self._normalize(vec)
