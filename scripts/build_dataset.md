# Build Dataset Guide

Quy trình chuẩn bị dữ liệu từ raw video đến sẵn sàng cho index:
1. Đặt dữ liệu video gốc do BTC cấp vào `data/raw/videos/`.
2. Chạy module tách keyframe bổ sung (nếu cần) vào `data/keyframes/`.
3. Chạy ASR pipeline (`faster-whisper`) tạo transcript trong `data/asr/`.
4. Chạy OCR pipeline (`PaddleOCR`) trích xuất chữ trong keyframe vào `data/ocr/`.
5. Tạo embedding cho toàn bộ keyframe trong `data/embeddings/` hoặc `data/clip_features/`.
