"""
PaddleOCR pipeline: extract text from keyframe images.
Saves results as JSON: {relative_path: "detected text"}
"""
import os
import json
from typing import Optional


class PaddleOCRPipeline:
    def __init__(self, lang: str = "ch", use_gpu: bool = False):
        self.ocr = None
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)
            print("[+] PaddleOCR initialized.")
        except Exception as e:
            print(f"[!] PaddleOCR not available ({e}). OCR skipped.")

    def run_ocr(self, image_path: str) -> str:
        """Return all detected text from a single image as one string."""
        if self.ocr is None or not os.path.exists(image_path):
            return ""
        try:
            result = self.ocr.ocr(image_path, cls=True)
            lines = []
            for block in (result or []):
                for line in (block or []):
                    if line and len(line) >= 2 and line[1]:
                        lines.append(line[1][0])
            return " ".join(lines)
        except Exception:
            return ""

    def batch_ocr(self, image_dir: str, output_json: str, exts: tuple = (".jpg", ".jpeg", ".png")):
        """
        Scan image_dir recursively, run OCR on each image,
        save {relative_path: text} to output_json.
        """
        results = {}
        for root, _, files in os.walk(image_dir):
            for fname in sorted(files):
                if not fname.lower().endswith(exts):
                    continue
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, image_dir)
                text = self.run_ocr(fpath)
                if text:
                    results[rel] = text
                    print(f"  OCR [{rel}]: {text[:60]}")

        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[+] OCR results saved: {output_json} ({len(results)} frames with text)")
        return results
