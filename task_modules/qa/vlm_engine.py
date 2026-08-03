"""
Qwen2.5-VL offline VLM engine for Visual Question Answering.
Runs locally — no API needed.
Supports Qwen/Qwen2.5-VL-7B-Instruct or smaller -3B variant.
"""
import os
from typing import Optional


class Qwen25VLEngine:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "auto",  # "auto" = use GPU if available
        max_new_tokens: int = 128,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        self._init()

    def _init(self):
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.model.eval()
            print(f"[+] Qwen2.5-VL loaded: {self.model_name}")
        except Exception as e:
            print(f"[!] Qwen2.5-VL load failed ({e}). VLM QA unavailable.")
            self.model = None

    def generate_answer(self, image_path: str, question: str) -> Optional[str]:
        """
        Given an image file path and a question, return a short text answer.
        Returns None if model not loaded or image not found.
        """
        if self.model is None or not os.path.exists(image_path):
            return None
        try:
            from PIL import Image
            import torch

            image = Image.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], return_tensors="pt")
            inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
            with torch.no_grad():
                out_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            # Decode only the generated part
            input_len = inputs["input_ids"].shape[1]
            answer = self.processor.batch_decode(out_ids[:, input_len:], skip_special_tokens=True)[0].strip()
            return answer if answer else None
        except Exception as e:
            print(f"[!] Qwen2.5-VL inference error: {e}")
            return None
