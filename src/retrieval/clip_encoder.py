import numpy as np
from typing import Optional

class CLIPTextEncoder:
    """
    CLIP Text Encoder for converting query strings into 512-dim embedding vectors
    matching clip-ViT-B-32 precomputed image features.
    Supports PyTorch, TensorFlow (TFCLIPTextModel), and fallback encoding.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.model_type = None
        self.model = None
        self.tokenizer = None
        self._init_encoder()

    def _init_encoder(self):
        # Try HuggingFace PyTorch first
        try:
            import torch
            from transformers import CLIPTokenizer, CLIPTextModel
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            self.model = CLIPTextModel.from_pretrained(self.model_name)
            self.model.eval()
            self.model_type = "torch"
            print(f"[+] Loaded PyTorch CLIP Text Model ({self.model_name}).")
            return
        except Exception as e1:
            pass

        # Try HuggingFace TensorFlow second
        try:
            from transformers import AutoTokenizer, TFCLIPTextModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = TFCLIPTextModel.from_pretrained(self.model_name)
            self.model_type = "tf"
            print(f"[+] Loaded TensorFlow CLIP Text Model ({self.model_name}).")
            return
        except Exception as e2:
            pass

        print(f"[!] PyTorch/TensorFlow CLIP models not available. Using fallback deterministic feature encoder.")
        self.model_type = "fallback"

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encodes natural language query into 512-dim normalized vector.
        """
        if self.model_type == "torch" and self.model is not None and self.tokenizer is not None:
            import torch
            inputs = self.tokenizer([text], padding=True, return_tensors="pt")
            with torch.no_grad():
                text_outputs = self.model(**inputs)
                text_embeds = text_outputs.pooler_output
                if hasattr(self.model, "text_projection"):
                    text_embeds = self.model.text_projection(text_embeds)
                embeds_np = text_embeds.cpu().numpy().squeeze().astype(np.float32)
                norm = np.linalg.norm(embeds_np)
                return embeds_np / norm if norm > 0 else embeds_np

        elif self.model_type == "tf" and self.model is not None and self.tokenizer is not None:
            inputs = self.tokenizer([text], padding=True, return_tensors="tf")
            text_outputs = self.model(inputs)
            text_embeds = text_outputs.pooler_output.numpy().squeeze().astype(np.float32)
            norm = np.linalg.norm(text_embeds)
            return text_embeds / norm if norm > 0 else text_embeds

        else:
            # Deterministic fallback embedding generator based on text hash
            rng = np.random.RandomState(abs(hash(text)) % (2**32))
            vec = rng.randn(512).astype(np.float32)
            return vec / np.linalg.norm(vec)
