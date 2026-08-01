# VTK AI Challenge - AIC 2026 Video Retrieval System

Hệ thống truy vấn video đa thức (Multimodal Video Retrieval System) được phát triển cho cuộc thi **AIC 2026**. Hệ thống hỗ trợ tìm kiếm video theo mô tả văn bản (KIS), giải đáp câu hỏi hình ảnh (Visual Q&A), và tìm kiếm chuỗi sự kiện theo thời gian (TRAKE).

---

## 📌 Tính năng chính

- **Automated Dataset Downloader (`download_data.py`)**: Tự động tải và giải nén các tập dữ liệu thiết yếu (`clip-features-32`, `map-keyframes`, `media-info`, `objects`).
- **Feature & Metadata Indexer (`src/data/`)**: Lập chỉ mục không gian đặc trưng vector (CLIP 512-dim) và thông tin metadata video (YouTube API info).
- **Hybrid Search Engine (`src/retrieval/`)**: Kết hợp kết quả tìm kiếm vector và tìm kiếm metadata từ ngữ cảnh.
- **Đa dạng bài toán (`src/modules/`)**:
  - **Task 1.1 - Textual KIS**: Tìm kiếm khung hình cụ thể theo mô tả văn bản.
  - **Task 1.2 - Visual Q&A**: Truy vấn câu hỏi và sự kiện trong video.
  - **Task 1.3 - TRAKE (Temporal Alignment)**: Tìm kiếm chuỗi sự kiện có tính thứ tự thời gian.
- **Ranking Optimizer & Format Validator (`src/submission/`)**: Tối ưu hóa thứ hạng dự đoán và kiểm tra định dạng file nộp bài CSV theo chuẩn AIC.

---

## 🛠️ Yêu cầu hệ thống

- **Python**: 3.8 trở lên
- **RAM**: Tối thiểu 8GB (Khuyến nghị 16GB+)
- **Dung lượng đĩa trống**: Tối thiểu 5GB - 10GB (cho dataset và feature)

---

## 🚀 Hướng dẫn cài đặt & Sử dụng

### 1. Clone Repository

```bash
git clone https://github.com/lamvu-ha/VTK_AI_CHALLENGE.git
cd VTK_AI_CHALLENGE
```

### 2. Cấu hình môi trường ảo (Khuyến nghị)

#### Trên Windows (PowerShell / CMD):
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

#### Trên Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

*(Lưu ý: Nếu không cài đặt `torch` và `transformers`, hệ thống sẽ tự động chuyển sang chế độ **Fallback Deterministic Encoder** để đảm bảo chương trình vẫn khởi chạy bình thường).*

---

### 4. Tải dữ liệu tự động

Chạy script để tải và tự động giải nén các dữ liệu thiết yếu vào thư mục `data/`:

```bash
python download_data.py
```

Sau khi chạy xong, cấu trúc thư mục `data/` sẽ bao gồm:
```text
data/
├── clip-features-32/
├── map-keyframes/
├── media-info/
├── objects/
└── zips/
```

---

### 5. Chạy pipeline truy vấn & Xuất file nộp bài

Chạy file `main.py` để thực thi tìm kiếm trên toàn bộ tập video (873 videos):

```bash
python main.py
```

Kết quả dự đoán sẽ được lưu tại thư mục `submissions/`:
- `submissions/kis_submission.csv`
- `submissions/qa_submission.csv`
- `submissions/trake_submission.csv`

---

## 📂 Cấu trúc dự án

```text
VTK_AI_CHALLENGE/
│
├── data/                         # Thư mục chứa dataset (được ignore trên git)
├── submissions/                  # Thư mục chứa kết quả CSV xuất ra
├── src/
│   ├── data/
│   │   ├── dataset_loader.py     # Load đặc trưng và keyframes
│   │   ├── feature_indexer.py    # Lập chỉ mục vector cho CLIP features
│   │   └── metadata_indexer.py   # Lập chỉ mục thông tin YouTube metadata
│   ├── retrieval/
│   │   ├── clip_encoder.py       # Encode văn bản câu hỏi sang CLIP embedding
│   │   ├── hybrid_search.py      # Động cơ tìm kiếm kết hợp (vector + metadata)
│   │   └── query_processor.py    # Tiền xử lý câu hỏi & phân tích query TRAKE
│   ├── modules/
│   │   ├── kis_solver.py         # Solver cho bài toán Textual KIS
│   │   ├── qa_solver.py          # Solver cho bài toán Visual Q&A
│   │   └── trake_solver.py       # Solver cho bài toán TRAKE
│   └── submission/
│       ├── format_validator.py   # Kiểm tra tính hợp lệ file submission
│       └── ranking_optimizer.py  # Tối ưu hóa thứ tự xếp hạng dự đoán
│
├── download_data.py              # Script tự động tải & giải nén dataset
├── main.py                       # Pipeline chính của dự án
├── spreadsheet_data.csv          # Danh sách URL tải dữ liệu
├── requirements.txt              # Thư viện phụ thuộc
└── README.md                     # Tài liệu hướng dẫn sử dụng
```

---

## 📝 License & Đóng góp

Dự án được duy trì cho cuộc thi **VTK AI Challenge / AIC 2026**. Mọi đóng góp hoặc thắc mắc xin vui lòng mở Issue hoặc Pull Request trên repository.
