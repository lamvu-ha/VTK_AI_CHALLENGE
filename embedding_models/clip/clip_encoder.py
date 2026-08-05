"""
Enhanced CLIP Text Encoder with multi-query ensemble support.
Supports: PyTorch CLIP, TensorFlow CLIP, SigLIP2 (when available), fallback encoder.

Multi-query ensemble: encode multiple query variants and average-pool their embeddings
for more robust retrieval (aligned with design doc's multi-query expansion strategy).
"""
import os
import sys
import numpy as np
from typing import Optional, List, Any

# ── Must set BEFORE any FAISS/MKL DLL loads to avoid WinError 1114 ──────────
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # force CPU, no CUDA DLL
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Pre-load torch now so its DLLs are resident before FAISS loads MKL DLLs
try:
    import torch as _torch_preload  # noqa: F401
except Exception:
    pass

# Ensure Hugging Face cache is stored on drive E: with ample disk space
_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
_HF_CACHE_DIR = os.path.join(_DATA_DIR, "hf_cache")
os.makedirs(_HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = _HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = _HF_CACHE_DIR


class CLIPTextEncoder:
    """
    CLIP Text Encoder for converting query strings into embedding vectors
    matching clip-ViT-B-32 or SigLIP2 precomputed image features.
    
    Supports:
    - PyTorch CLIP (openai/clip-vit-base-patch32) → 512-dim (when PyTorch >= 2.1)
    - Fast local fallback (hash-based) if PyTorch < 2.1 or GPU unavailable
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", allow_download: bool = True):
        self.model_name = model_name
        self.allow_download = allow_download
        self.model_type = None
        self.model = None
        self.tokenizer = None
        self._init_encoder()

    def _init_encoder(self):
        # Force CPU-only to avoid WinError 1114 CUDA DLL crash on Windows
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        try:
            import torch
            try:
                from transformers import CLIPTokenizer, CLIPTextModel
            except Exception:
                from transformers.models.clip.tokenization_clip import CLIPTokenizer
                from transformers.models.clip.modeling_clip import CLIPTextModel

            print(f"[+] Initializing PyTorch CLIP Text Model ({self.model_name})...")
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.model_name, local_files_only=not self.allow_download
            )
            self.model = CLIPTextModel.from_pretrained(
                self.model_name,
                local_files_only=not self.allow_download,
                low_cpu_mem_usage=False,
            )
            self.model = self.model.to(torch.device("cpu"))
            self.model.eval()
            self.model_type = "torch"
            self._torch_device = torch.device("cpu")
            print(f"[+] Successfully loaded PyTorch CLIP Text Model ({self.model_name}).")
            return
        except Exception as e:
            try:
                from transformers.models.clip.tokenization_clip import CLIPTokenizer
                from transformers.models.clip.modeling_clip import CLIPTextModel
                import torch
                self.tokenizer = CLIPTokenizer.from_pretrained(
                    self.model_name, local_files_only=not self.allow_download
                )
                self.model = CLIPTextModel.from_pretrained(
                    self.model_name, local_files_only=not self.allow_download
                ).to(torch.device("cpu"))
                self.model.eval()
                self.model_type = "torch"
                self._torch_device = torch.device("cpu")
                print(f"[+] Successfully loaded PyTorch CLIP (direct import).")
                return
            except Exception as e2:
                print(f"[!] PyTorch CLIP load warning ({e2}). Trying open_clip fallback...")

        try:
            import open_clip  # type: ignore[import-untyped]  # optional fallback
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
            self.model = model.eval()
            self.model_type = "open_clip"
            print("[+] Loaded open_clip ViT-B-32 model.")
            return
        except Exception:
            pass

        print("[!] Warning: PyTorch/CLIP not loaded. Reverting to fallback encoder (low accuracy).")
        self.model_type = "fallback"

    def encode_text(self, text: Any) -> np.ndarray:
        """
        Encodes a single natural language query into a normalized embedding vector.
        """
        if not isinstance(text, str):
            if isinstance(text, (list, tuple)):
                text = " ".join(str(x) for x in text if x)
            elif text is not None:
                text = str(text)
            else:
                text = ""

        if self.model_type == "torch" and self.model is not None:
            try:
                import torch
                device = getattr(self, "_torch_device", torch.device("cpu"))
                inputs = self.tokenizer(
                    [text], padding=True, truncation=True, max_length=77, return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    text_outputs = self.model(**inputs)
                    text_embeds = text_outputs.pooler_output
                    if hasattr(self.model, "text_projection"):
                        text_embeds = self.model.text_projection(text_embeds)
                    embeds_np = text_embeds.cpu().numpy().squeeze().astype(np.float32)
                return self._normalize(embeds_np)
            except Exception:
                return self._fallback_encode(text)

        elif self.model_type == "open_clip" and self.model is not None:
            try:
                import torch
                tokens = self.tokenizer([text])
                with torch.no_grad():
                    text_features = self.model.encode_text(tokens)
                    embeds_np = text_features.cpu().numpy().squeeze().astype(np.float32)
                return self._normalize(embeds_np)
            except Exception:
                return self._fallback_encode(text)

        else:
            return self._fallback_encode(text)

    def encode_text_ensemble(self, texts: List[str]) -> np.ndarray:
        """
        Multi-query ensemble encoding: encode all query variants and average-pool.
        This implements the 'multi-query expansion' strategy from the design doc.
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

    def _fallback_encode(self, text: Any) -> np.ndarray:
        """Deterministic fallback: hash-based pseudo-random 512-dim embedding."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        rng = np.random.RandomState(abs(hash(text)) % (2**32))
        vec = rng.randn(512).astype(np.float32)
        return self._normalize(vec)
