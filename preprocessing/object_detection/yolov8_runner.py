"""
YOLOv8 / Grounding DINO runner: phát hiện object bổ sung.
Dùng khi cần class vật thể không có trong OpenImages V4 (BTC đã cấp Faster R-CNN).
"""
import os
import json
from typing import List, Dict, Optional


class YOLOv8Runner:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.3, device: str = "cpu"):
        self.conf = conf
        self.device = device
        self.model = None
        try:
            from ultralytics import YOLO  # type: ignore
            self.model = YOLO(model_path)
            print(f"[+] YOLOv8 loaded: {model_path}")
        except Exception as e:
            print(f"[!] YOLOv8 không khả dụng: {e}")

    def detect(self, image_path: str) -> List[Dict]:
        """Trả về [{label, confidence, bbox}]."""
        if self.model is None or not os.path.exists(image_path):
            return []
        try:
            results = self.model(image_path, conf=self.conf, device=self.device, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    detections.append({
                        "label": r.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist(),
                    })
            return detections
        except Exception as e:
            print(f"[!] YOLO detect lỗi: {e}")
            return []


class GroundingDINORunner:
    """Placeholder cho Grounding DINO — open-vocabulary object detection."""
    def __init__(self):
        print("[!] GroundingDINO chưa được tích hợp. Dùng YOLOv8Runner thay thế.")

    def detect(self, image_path: str, text_prompt: str) -> List[Dict]:
        return []
