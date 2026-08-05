import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import threading
import queue
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from indexing.dataset_loader import AICDatasetLoader
    from indexing.vector_db.faiss_backup.feature_indexer import FeatureIndexer
    from indexing.text_search.metadata_indexer import MetadataIndexer
    from retrieval.query_processing.query_processor import QueryProcessor
    from embedding_models.clip.clip_encoder import CLIPTextEncoder
    from retrieval.fusion.hybrid_search import HybridSearchEngine
    from task_modules.textual_kis.kis_solver import TextualKISSolver
    from task_modules.qa.qa_solver import QASolver
    from task_modules.trake.trake_solver import TRAKESolver
    from ui.export.ranking_optimizer import RankingOptimizer
    from ui.export.format_validator import AICFormatValidator
    from ui.export.format_submission import format_and_export
    from evaluation.metrics.r_score import calculate_r_score
    from evaluation.metrics.final_score import final_score as compute_final_score
except Exception as e:
    import traceback
    print(f"[!] Critical Error importing backend modules: {e}")
    traceback.print_exc()

try:
    from PIL import Image, ImageTk
    import urllib.request
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class AICVideoRetrievalGUI:
    """
    Tkinter Desktop Application for testing 3 Video Retrieval Tasks (AIC 2026).
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AIC 2026 - Video Retrieval & Verification Studio")
        self.root.geometry("1280x820")
        self.root.minsize(1024, 700)

        self.msg_queue = queue.Queue()

        self.is_data_loaded = False
        self.loader = None
        self.feature_indexer = None
        self.metadata_indexer = None
        self.query_processor = None
        self.clip_encoder = None
        self.search_engine = None
        self.ranking_optimizer = RankingOptimizer(lambda_mmr=0.7)
        self.validator = AICFormatValidator()
        self.kis_solver = None
        self.qa_solver = None
        self.trake_solver = None

        self.kis_results = []
        self.qa_results = []
        self.trake_results = []
        self.media_info_cache = {}

        self._setup_styles()
        self._build_ui()

        self.root.after(100, self._process_queue)
        self._log("Chương trình đã khởi động. Đang khởi tạo bộ nạp dữ liệu backend...")
        self.load_dataset_async()

    def _setup_styles(self):
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")

        BG_DARK = "#181825"
        PANEL_BG = "#1E1E2E"
        ACCENT_BLUE = "#89B4FA"
        ACCENT_GREEN = "#A6E3A1"
        TEXT_LIGHT = "#CDD6F4"
        TEXT_MUTED = "#BAC2DE"

        self.root.configure(bg=BG_DARK)

        self.style.configure("TFrame", background=BG_DARK)
        self.style.configure("Card.TFrame", background=PANEL_BG, relief="solid", borderwidth=1)

        self.style.configure("TLabel", background=BG_DARK, foreground=TEXT_LIGHT, font=("Segoe UI", 10))
        self.style.configure("Card.TLabel", background=PANEL_BG, foreground=TEXT_LIGHT, font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", background=PANEL_BG, foreground=ACCENT_BLUE, font=("Segoe UI", 12, "bold"))
        self.style.configure("Title.TLabel", background=BG_DARK, foreground=ACCENT_BLUE, font=("Segoe UI", 16, "bold"))
        self.style.configure("Status.TLabel", background=PANEL_BG, foreground=TEXT_MUTED, font=("Segoe UI", 9))

        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        self.style.configure("Primary.TButton", background="#313244", foreground=TEXT_LIGHT)
        self.style.map("Primary.TButton", background=[("active", "#45475A")])
        self.style.configure("Run.TButton", background="#2563EB", foreground="#FFFFFF")
        self.style.map("Run.TButton", background=[("active", "#1D4ED8")])
        self.style.configure("Export.TButton", background="#059669", foreground="#FFFFFF")
        self.style.map("Export.TButton", background=[("active", "#047857")])

        self.style.configure("TNotebook", background=BG_DARK, borderwidth=0)
        self.style.configure("TNotebook.Tab", background="#313244", foreground=TEXT_LIGHT, font=("Segoe UI", 10, "bold"), padding=[12, 6])
        self.style.map("TNotebook.Tab", background=[("selected", "#45475A")], foreground=[("selected", ACCENT_BLUE)])

        self.style.configure("Treeview", background="#1E1E2E", foreground=TEXT_LIGHT, fieldbackground="#1E1E2E", font=("Segoe UI", 10), rowheight=26)
        self.style.configure("Treeview.Heading", background="#313244", foreground=ACCENT_BLUE, font=("Segoe UI", 10, "bold"))
        self.style.map("Treeview", background=[("selected", "#45475A")], foreground=[("selected", "#FFFFFF")])

    def _build_ui(self):
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")

        title_label = ttk.Label(top_frame, text="🎥 AIC 2026 - TRÌNH KIỂM THỬ VÀ TRUY XUẤT VIDEO", style="Title.TLabel")
        title_label.pack(side="left")

        self.lbl_dataset_status = ttk.Label(top_frame, text="⏳ Đang khởi tạo dữ liệu...", style="Status.TLabel")
        self.lbl_dataset_status.pack(side="right", padx=10)

        main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        main_paned.pack(fill="both", expand=True, padx=10, pady=5)

        left_container = ttk.Frame(main_paned, style="Card.TFrame", padding=12)
        main_paned.add(left_container, weight=1)

        ttk.Label(left_container, text="📝 Nhập Mô Tả 3 Bài Toán", style="Header.TLabel").pack(anchor="w", pady=(0, 10))

        grp_kis = ttk.LabelFrame(left_container, text=" Task 1.1: Textual KIS (Tìm kiếm Keyframe) ", padding=8)
        grp_kis.pack(fill="x", pady=5)
        
        ttk.Label(grp_kis, text="Mô tả sự kiện/đối tượng trong video:").pack(anchor="w")
        self.txt_kis = tk.Text(grp_kis, height=3, width=40, font=("Segoe UI", 9), wrap="word", bg="#313244", fg="#CDD6F4", insertbackground="white")
        self.txt_kis.pack(fill="x", pady=5)
        self.txt_kis.insert("1.0", "Diễn giả mặc áo đỏ phát biểu tại cuộc họp báo ngoài trời")

        self.btn_run_kis = ttk.Button(
            grp_kis,
            text="▶ Chạy Riêng Task 1.1 (KIS)",
            style="Run.TButton",
            command=self.run_kis_async
        )
        self.btn_run_kis.pack(fill="x", pady=(2, 0))

        grp_qa = ttk.LabelFrame(left_container, text=" Task 1.2: Visual Q&A (Hỏi Đáp Video) ", padding=8)
        grp_qa.pack(fill="x", pady=5)

        ttk.Label(grp_qa, text="Mô tả sự kiện (Event Context):").pack(anchor="w")
        self.ent_qa_event = ttk.Entry(grp_qa, font=("Segoe UI", 9))
        self.ent_qa_event.pack(fill="x", pady=(2, 6))
        self.ent_qa_event.insert(0, "Lễ trao giải thưởng âm nhạc")

        ttk.Label(grp_qa, text="Câu hỏi (Question):").pack(anchor="w")
        self.ent_qa_question = ttk.Entry(grp_qa, font=("Segoe UI", 9))
        self.ent_qa_question.pack(fill="x", pady=(2, 4))
        self.ent_qa_question.insert(0, "Trong video có bao nhiêu người lên sân khấu nhận giải?")

        self.btn_run_qa = ttk.Button(
            grp_qa,
            text="▶ Chạy Riêng Task 1.2 (QA)",
            style="Run.TButton",
            command=self.run_qa_async
        )
        self.btn_run_qa.pack(fill="x", pady=(2, 0))

        grp_trake = ttk.LabelFrame(left_container, text=" Task 1.3: TRAKE (Chuỗi Hành Động Thời Gian) ", padding=8)
        grp_trake.pack(fill="x", pady=5)

        ttk.Label(grp_trake, text="Chuỗi các sự kiện nối tiếp:").pack(anchor="w")
        self.txt_trake = tk.Text(grp_trake, height=3, width=40, font=("Segoe UI", 9), wrap="word", bg="#313244", fg="#CDD6F4", insertbackground="white")
        self.txt_trake.pack(fill="x", pady=5)
        self.txt_trake.insert("1.0", "(1) Giậm nhảy, (2) Bay qua xà, (3) Tiếp đất, (4) Đứng dậy")

        self.btn_run_trake = ttk.Button(
            grp_trake,
            text="▶ Chạy Riêng Task 1.3 (TRAKE)",
            style="Run.TButton",
            command=self.run_trake_async
        )
        self.btn_run_trake.pack(fill="x", pady=(2, 0))

        btn_frame = ttk.Frame(left_container, padding=5)
        btn_frame.pack(fill="x", pady=10)

        self.btn_run_all = ttk.Button(
            btn_frame,
            text="🚀 Chạy Cả 3 Bài Toán Đồng Thời",
            style="Primary.TButton",
            command=self.run_all_tasks_async
        )
        self.btn_run_all.pack(fill="x", pady=3)

        self.btn_export = ttk.Button(
            btn_frame,
            text="📁 Xuất Kết Quả CSV Submission",
            style="Export.TButton",
            command=self.export_submissions
        )
        self.btn_export.pack(fill="x", pady=3)

        right_container = ttk.Frame(main_paned, style="Card.TFrame", padding=10)
        main_paned.add(right_container, weight=3)

        self.notebook = ttk.Notebook(right_container)
        self.notebook.pack(fill="both", expand=True)

        self.tab_kis = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.tab_kis, text=" 🏷️ Task 1.1: Textual KIS ")
        self._setup_kis_tab()

        self.tab_qa = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.tab_qa, text=" ❓ Task 1.2: Visual QA ")
        self._setup_qa_tab()

        self.tab_trake = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.tab_trake, text=" ⏱️ Task 1.3: TRAKE Sequence ")
        self._setup_trake_tab()

        self.tab_video = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.tab_video, text=" 🎬 Xem Video & Metadata ")
        self._setup_video_tab()

        self.tab_eval = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.tab_eval, text=" 📊 Đánh Giá (Evaluation) ")
        self._setup_eval_tab()

        self.tab_preprocess = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.tab_preprocess, text=" ⚙️ Tiền Xử Lý (Preprocessing) ")
        self._setup_preprocess_tab()

        bottom_frame = ttk.Frame(self.root, padding=5)
        bottom_frame.pack(fill="x", side="bottom")

        self.progress_bar = ttk.Progressbar(bottom_frame, mode="indeterminate")
        self.progress_bar.pack(fill="x", pady=2)

        self.lbl_status = ttk.Label(bottom_frame, text="Sẵn sàng.", style="Status.TLabel")
        self.lbl_status.pack(side="left")

        log_frame = ttk.LabelFrame(bottom_frame, text=" Log Hệ Thống Pipeline ", padding=5)
        log_frame.pack(fill="x", pady=5)

        self.txt_log = scrolledtext.ScrolledText(
            log_frame, height=4, font=("Consolas", 8), bg="#181825", fg="#CDD6F4", insertbackground="white"
        )
        self.txt_log.pack(fill="x")

        # State cho eval và preprocessing
        self._eval_gt = {}  # {query_id: ground_truth}
        self._preprocess_running = False

    def _setup_kis_tab(self):
        columns = ("rank", "video_id", "frame_id", "pts_time", "score")
        self.tree_kis = ttk.Treeview(self.tab_kis, columns=columns, show="headings", selectmode="browse")
        
        self.tree_kis.heading("rank", text="Hạng")
        self.tree_kis.heading("video_id", text="Video ID")
        self.tree_kis.heading("frame_id", text="Frame ID")
        self.tree_kis.heading("pts_time", text="Thời điểm PTS (s)")
        self.tree_kis.heading("score", text="Điểm Similarity")

        self.tree_kis.column("rank", width=50, anchor="center")
        self.tree_kis.column("video_id", width=120, anchor="center")
        self.tree_kis.column("frame_id", width=100, anchor="center")
        self.tree_kis.column("pts_time", width=140, anchor="center")
        self.tree_kis.column("score", width=120, anchor="center")

        scrollbar = ttk.Scrollbar(self.tab_kis, orient="vertical", command=self.tree_kis.yview)
        self.tree_kis.configure(yscrollcommand=scrollbar.set)

        self.tree_kis.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.tree_kis.bind("<Double-1>", self._on_kis_double_click)

    def _setup_qa_tab(self):
        columns = ("rank", "video_id", "frame_id", "pts_time", "answer", "score")
        self.tree_qa = ttk.Treeview(self.tab_qa, columns=columns, show="headings", selectmode="browse")

        self.tree_qa.heading("rank", text="Hạng")
        self.tree_qa.heading("video_id", text="Video ID")
        self.tree_qa.heading("frame_id", text="Frame ID")
        self.tree_qa.heading("pts_time", text="Thời điểm PTS (s)")
        self.tree_qa.heading("answer", text="Dự đoán Đáp án")
        self.tree_qa.heading("score", text="Điểm Similarity")

        self.tree_qa.column("rank", width=50, anchor="center")
        self.tree_qa.column("video_id", width=120, anchor="center")
        self.tree_qa.column("frame_id", width=100, anchor="center")
        self.tree_qa.column("pts_time", width=130, anchor="center")
        self.tree_qa.column("answer", width=140, anchor="center")
        self.tree_qa.column("score", width=120, anchor="center")

        scrollbar = ttk.Scrollbar(self.tab_qa, orient="vertical", command=self.tree_qa.yview)
        self.tree_qa.configure(yscrollcommand=scrollbar.set)

        self.tree_qa.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.tree_qa.bind("<Double-1>", self._on_qa_double_click)

    def _setup_trake_tab(self):
        columns = ("rank", "video_id", "frame_seq", "pts_seq", "score")
        self.tree_trake = ttk.Treeview(self.tab_trake, columns=columns, show="headings", selectmode="browse")

        self.tree_trake.heading("rank", text="Hạng")
        self.tree_trake.heading("video_id", text="Video ID")
        self.tree_trake.heading("frame_seq", text="Chuỗi Keyframe (f1, f2, ...)")
        self.tree_trake.heading("pts_seq", text="Chuỗi PTS (s)")
        self.tree_trake.heading("score", text="Điểm DP Alignment")

        self.tree_trake.column("rank", width=50, anchor="center")
        self.tree_trake.column("video_id", width=120, anchor="center")
        self.tree_trake.column("frame_seq", width=220, anchor="center")
        self.tree_trake.column("pts_seq", width=220, anchor="center")
        self.tree_trake.column("score", width=130, anchor="center")

        scrollbar = ttk.Scrollbar(self.tab_trake, orient="vertical", command=self.tree_trake.yview)
        self.tree_trake.configure(yscrollcommand=scrollbar.set)

        self.tree_trake.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.tree_trake.bind("<Double-1>", self._on_trake_double_click)

    def _setup_video_tab(self):
        v_container = ttk.Frame(self.tab_video, padding=15)
        v_container.pack(fill="both", expand=True)

        self.lbl_v_title = ttk.Label(v_container, text="Chọn một kết quả từ bảng để xem chi tiết Video", font=("Segoe UI", 12, "bold"), foreground="#89B4FA")
        self.lbl_v_title.pack(anchor="w", pady=(0, 10))

        self.btn_open_youtube = ttk.Button(
            v_container,
            text="▶️ Mở Video trên YouTube (Đúng Thời Điểm PTS)",
            style="Run.TButton",
            command=self._open_youtube_current
        )
        self.btn_open_youtube.pack(anchor="w", pady=5)
        self.btn_open_youtube.config(state="disabled")

        meta_frame = ttk.LabelFrame(v_container, text=" Thông Tin Chi Tiết Media Metadata ", padding=10)
        meta_frame.pack(fill="both", expand=True, pady=10)

        self.lbl_v_id = ttk.Label(meta_frame, text="Video ID: -", font=("Segoe UI", 10, "bold"))
        self.lbl_v_id.grid(row=0, column=0, sticky="w", pady=3, padx=5)

        self.lbl_v_pts = ttk.Label(meta_frame, text="Thời điểm Keyframe (PTS): -")
        self.lbl_v_pts.grid(row=0, column=1, sticky="w", pady=3, padx=15)

        self.lbl_v_channel = ttk.Label(meta_frame, text="Kênh / Tác giả: -")
        self.lbl_v_channel.grid(row=1, column=0, sticky="w", pady=3, padx=5)

        self.lbl_v_duration = ttk.Label(meta_frame, text="Thời lượng Video: -")
        self.lbl_v_duration.grid(row=1, column=1, sticky="w", pady=3, padx=15)
        self.lbl_v_url = ttk.Label(meta_frame, text="YouTube URL: -", foreground="#89B4FA")
        self.lbl_v_url.grid(row=2, column=0, columnspan=2, sticky="w", pady=3, padx=5)

        ttk.Label(meta_frame, text="Mô tả Video:").grid(row=3, column=0, sticky="w", pady=(10, 2), padx=5)
        self.txt_v_desc = scrolledtext.ScrolledText(meta_frame, height=8, font=("Segoe UI", 9), bg="#181825", fg="#CDD6F4")
        self.txt_v_desc.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=5, pady=2)
        meta_frame.rowconfigure(4, weight=1)
        meta_frame.columnconfigure(1, weight=1)

        self.current_selected_watch_url = None
        self.current_selected_pts = 0.0

    # ─────────────────────────────────────────────
    #  TAB: EVALUATION
    # ─────────────────────────────────────────────
    def _setup_eval_tab(self):
        f = ttk.Frame(self.tab_eval)
        f.pack(fill="both", expand=True)

        ttk.Label(f, text="📊 Đánh giá R-Score theo thể lệ AIC 2026",
                  font=("Segoe UI", 12, "bold"), foreground="#89B4FA").pack(anchor="w", pady=(0, 4))

        # Hướng dẫn nhanh
        note = (
            "💡 Cách dùng: (1) Chạy tìm kiếm ở tab KIS/QA/TRAKE  →  "
            "(2) Xem Dự đoán Top-N bên dưới  →  "
            "(3) Nhập GT đúng  →  (4) Bấm Tính R-Score"
        )
        ttk.Label(f, text=note, foreground="#A6E3A1", font=("Segoe UI", 9), wraplength=780).pack(
            anchor="w", pady=(0, 6))

        # ── Khu vực diagnostic: Top predictions ──────────────────────────
        diag_outer = ttk.Frame(f)
        diag_outer.pack(fill="x", pady=(0, 4))

        diag_frame = ttk.LabelFrame(
            diag_outer,
            text=" 🔍 Kết quả dự đoán hiện tại (top-5) — dùng để điền GT cho đúng ",
            padding=6
        )
        diag_frame.pack(fill="x")

        diag_cols = ("rank", "video_id", "frame_id", "pts_time", "score")
        self.tree_diag = ttk.Treeview(diag_frame, columns=diag_cols, show="headings", height=5)
        for col, hd, w in zip(diag_cols,
                              ("Hạng", "Video ID", "Frame ID (dùng làm GT)", "PTS (s)", "Score"),
                              (50, 140, 180, 100, 110)):
            self.tree_diag.heading(col, text=hd)
            self.tree_diag.column(col, width=w, anchor="center")
        self.tree_diag.pack(fill="x")

        btn_diag_row = ttk.Frame(diag_frame)
        btn_diag_row.pack(fill="x", pady=(4, 0))
        ttk.Button(btn_diag_row, text="🔄 Làm mới Top-N",
                   style="Primary.TButton", command=self._refresh_diag).pack(side="left", padx=2)
        ttk.Button(btn_diag_row, text="⬇ Tự điền GT từ kết quả #1",
                   style="Primary.TButton", command=self._autofill_gt).pack(side="left", padx=2)

        self._diag_type = tk.StringVar(value="KIS")
        for t in ("KIS", "QA", "TRAKE"):
            ttk.Radiobutton(btn_diag_row, text=t, variable=self._diag_type,
                            value=t, command=self._refresh_diag).pack(side="left", padx=4)

        # ── Ground Truth input ──────────────────────────────────────────
        gt_frame = ttk.LabelFrame(f, text=" Ground Truth (nhập frame_id thực tế từ bảng bên trên) ", padding=8)
        gt_frame.pack(fill="x", pady=5)

        row1 = ttk.Frame(gt_frame); row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="Loại query:").pack(side="left")
        self._eval_type = tk.StringVar(value="KIS")
        for t in ("KIS", "QA", "TRAKE"):
            ttk.Radiobutton(row1, text=t, variable=self._eval_type, value=t).pack(side="left", padx=8)

        row2 = ttk.Frame(gt_frame); row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="GT start frame:", width=16).pack(side="left")
        self._eval_gt_start = ttk.Entry(row2, width=10); self._eval_gt_start.pack(side="left", padx=4)
        self._eval_gt_start.insert(0, "0")
        ttk.Label(row2, text="GT end frame:").pack(side="left", padx=(10, 0))
        self._eval_gt_end = ttk.Entry(row2, width=10); self._eval_gt_end.pack(side="left", padx=4)
        self._eval_gt_end.insert(0, "0")
        ttk.Label(row2, text="← frame_id từ dataset thực tế",
                  foreground="#F9E2AF", font=("Segoe UI", 8)).pack(side="left", padx=6)

        row3 = ttk.Frame(gt_frame); row3.pack(fill="x", pady=2)
        ttk.Label(row3, text="GT answer (QA):", width=16).pack(side="left")
        self._eval_gt_ans = ttk.Entry(row3, width=20); self._eval_gt_ans.pack(side="left", padx=4)

        # ── Nút tính ──────────────────────────────────────────────────
        btn_row = ttk.Frame(f); btn_row.pack(fill="x", pady=4)
        ttk.Button(btn_row, text="▶  Tính R-Score & Final Score",
                   style="Run.TButton", command=self._run_evaluation).pack(side="left", padx=4)
        ttk.Button(btn_row, text="📂 Chạy Sample Queries (JSON)",
                   style="Primary.TButton", command=self._run_sample_queries).pack(side="left", padx=4)

        # ── Bảng kết quả ────────────────────────────────────────────────
        res_frame = ttk.LabelFrame(f, text=" Kết quả R-Score ", padding=8)
        res_frame.pack(fill="both", expand=True, pady=5)

        cols = ("metric", "k1", "k5", "k20", "k50", "k100", "final")
        self.tree_eval = ttk.Treeview(res_frame, columns=cols, show="headings", height=5)
        for col, hd, w in zip(cols,
                               ("Task / Query", "R@1", "R@5", "R@20", "R@50", "R@100", "Final Score"),
                               (180, 75, 75, 75, 75, 75, 90)):
            self.tree_eval.heading(col, text=hd)
            self.tree_eval.column(col, width=w, anchor="center")
        self.tree_eval.column("metric", anchor="w")

        sb = ttk.Scrollbar(res_frame, orient="vertical", command=self.tree_eval.yview)
        self.tree_eval.configure(yscrollcommand=sb.set)
        self.tree_eval.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Ghi chú giải thích
        ttk.Label(f,
                  text="⚠ R-Score = 0 khi frame_id dự đoán nằm ngoài khoảng [GT_start, GT_end]. "
                       "Nhập đúng GT từ bảng top-N ở trên để kiểm thử chính xác.",
                  foreground="#F38BA8", font=("Segoe UI", 8), wraplength=780
                  ).pack(anchor="w", pady=2)

    def _refresh_diag(self):
        """Làm mới bảng diagnostic top-5 prediction theo loại query."""
        qtype = self._diag_type.get()
        if qtype == "KIS":
            preds = self.kis_results
        elif qtype == "QA":
            preds = self.qa_results
        else:
            preds = self.trake_results

        for item in self.tree_diag.get_children():
            self.tree_diag.delete(item)

        if not preds:
            self.tree_diag.insert("", "end", values=("—", f"Chưa có kết quả {qtype}", "—", "—", "—"))
            return

        def get_pts(v_id, f_id):
            if self.feature_indexer:
                for kmap in self.feature_indexer.keyframe_map:
                    if kmap["video_id"] == v_id and kmap["frame_id"] == f_id:
                        return kmap.get("pts_time", 0.0)
            return 0.0

        for i, r in enumerate(preds[:5], 1):
            v_id = r.get("video_id", "?")
            f_id = r.get("frame_id", 0)
            pts = get_pts(v_id, f_id)
            score = r.get("score", 0.0)
            self.tree_diag.insert("", "end",
                                  values=(i, v_id, f_id, f"{pts:.2f}s", f"{score:.4f}"))

    def _autofill_gt(self):
        """Tự điền GT start/end từ frame_id của kết quả hạng 1."""
        self._refresh_diag()
        children = self.tree_diag.get_children()
        if not children:
            return
        vals = self.tree_diag.item(children[0], "values")
        try:
            frame_id = int(vals[2])
            # Đặt window ±30 frame quanh frame_id đó
            self._eval_gt_start.delete(0, tk.END)
            self._eval_gt_start.insert(0, str(max(0, frame_id - 30)))
            self._eval_gt_end.delete(0, tk.END)
            self._eval_gt_end.insert(0, str(frame_id + 30))
            # Đồng bộ loại query
            self._eval_type.set(self._diag_type.get())
            self._log(f"[Eval] Auto GT: frame_id={frame_id} → [{frame_id-30}, {frame_id+30}]")
        except (ValueError, IndexError):
            pass

    def _run_evaluation(self):
        """Tính R-Score từ kết quả hiện có và GT nhập tay."""
        qtype = self._eval_type.get()
        try:
            gt_s = int(self._eval_gt_start.get())
            gt_e = int(self._eval_gt_end.get())
        except ValueError:
            messagebox.showerror("Lỗi", "GT start/end frame phải là số nguyên.")
            return

        gt_ans = self._eval_gt_ans.get().strip()
        gt = {"start": gt_s, "end": gt_e}
        if gt_ans:
            gt["answer"] = gt_ans

        if qtype == "KIS":
            preds = self.kis_results
        elif qtype == "QA":
            preds = self.qa_results
        else:
            preds = self.trake_results

        if not preds:
            messagebox.showwarning("Cảnh báo", f"Chưa có kết quả {qtype}. Hãy chạy tìm kiếm trước.")
            return

        try:
            scores = compute_final_score(qtype, gt, preds)
        except Exception as e:
            messagebox.showerror("Lỗi tính score", str(e))
            return

        for item in self.tree_eval.get_children():
            self.tree_eval.delete(item)

        tag = "good" if scores.get("final", 0) > 0.3 else "zero"
        self.tree_eval.insert("", "end", tag=tag, values=(
            f"{qtype}  [GT: {gt_s}–{gt_e}]",
            f"{scores.get('R@1', 0):.4f}",
            f"{scores.get('R@5', 0):.4f}",
            f"{scores.get('R@20', 0):.4f}",
            f"{scores.get('R@50', 0):.4f}",
            f"{scores.get('R@100', 0):.4f}",
            f"{scores.get('final', 0):.4f}",
        ))
        self.tree_eval.tag_configure("good", foreground="#A6E3A1")
        self.tree_eval.tag_configure("zero", foreground="#F38BA8")

        # Diagnostic log
        top1_fid = preds[0].get("frame_id", "?") if preds else "?"
        self._log(
            f"[Eval] {qtype}: GT=[{gt_s},{gt_e}] | Pred#1 frame_id={top1_fid} | "
            f"R@1={scores.get('R@1',0):.4f} | Final={scores.get('final',0):.4f}"
        )
        if scores.get("final", 0) == 0:
            self._log(
                f"  ⚠ Score=0 vì frame_id={top1_fid} không nằm trong GT [{gt_s},{gt_e}]. "
                f"Hãy dùng nút '⬇ Tự điền GT' để test với kết quả thực tế."
            )

    def _run_sample_queries(self):
        """Load và chạy sample_queries.json."""
        import json
        sample_path = os.path.join(PROJECT_ROOT, "evaluation", "local_dev_queries", "sample_queries.json")
        if not os.path.exists(sample_path):
            messagebox.showerror("Lỗi", f"Không tìm thấy:\n{sample_path}")
            return
        with open(sample_path, encoding="utf-8") as f:
            queries = json.load(f)

        self._log(f"[Eval] Đang chạy {len(queries)} sample queries...")
        for item in self.tree_eval.get_children():
            self.tree_eval.delete(item)

        for q in queries:
            qtype = q["type"]
            gt = q["ground_truth"]
            if qtype == "KIS":
                preds = self.kis_results
            elif qtype == "QA":
                preds = self.qa_results
            else:
                preds = self.trake_results
            try:
                scores = compute_final_score(qtype, gt, preds)
            except Exception:
                scores = {"R@1": 0, "R@5": 0, "R@20": 0, "R@50": 0, "R@100": 0, "final": 0}

            top1_fid = preds[0].get("frame_id", "?") if preds else "không có kết quả"
            gt_center = (gt.get("start", 0) + gt.get("end", 0)) // 2
            self.tree_eval.insert("", "end", values=(
                f"{qtype} | {q['query_id']}",
                f"{scores.get('R@1', 0):.4f}",
                f"{scores.get('R@5', 0):.4f}",
                f"{scores.get('R@20', 0):.4f}",
                f"{scores.get('R@50', 0):.4f}",
                f"{scores.get('R@100', 0):.4f}",
                f"{scores.get('final', 0):.4f}",
            ))
            self._log(f"  [{q['query_id']}] {qtype}: Final={scores.get('final', 0):.4f}")

    # ─────────────────────────────────────────────
    #  TAB: PREPROCESSING
    # ─────────────────────────────────────────────
    def _setup_preprocess_tab(self):
        f = ttk.Frame(self.tab_preprocess)
        f.pack(fill="both", expand=True)

        ttk.Label(f, text="⚙️ Tiền Xử Lý Dữ Liệu Video",
                  font=("Segoe UI", 12, "bold"), foreground="#89B4FA").pack(anchor="w", pady=(0, 8))

        # Đường dẫn
        path_frame = ttk.LabelFrame(f, text=" Đường dẫn ", padding=8)
        path_frame.pack(fill="x", pady=5)

        for lbl, attr, default in [
            ("Thư mục video:",    "_pp_video_dir",    "data/raw/videos"),
            ("Thư mục shots:",    "_pp_shots_dir",    "data/shots"),
            ("Thư mục keyframes:","_pp_kf_dir",       "data/keyframes"),
        ]:
            row = ttk.Frame(path_frame); row.pack(fill="x", pady=2)
            ttk.Label(row, text=lbl, width=20).pack(side="left")
            entry = ttk.Entry(row)
            entry.insert(0, os.path.join(PROJECT_ROOT, default))
            entry.pack(side="left", fill="x", expand=True)
            setattr(self, attr, entry)

        opt_frame = ttk.Frame(f); opt_frame.pack(fill="x", pady=5)
        self._pp_num_kf = tk.IntVar(value=1)
        ttk.Label(opt_frame, text="Keyframe/shot:").pack(side="left")
        ttk.Spinbox(opt_frame, from_=1, to=5, textvariable=self._pp_num_kf, width=5).pack(side="left", padx=5)

        self._pp_use_transnet = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text="Dùng TransNetV2 (nếu có)",
                        variable=self._pp_use_transnet).pack(side="left", padx=10)

        btn_frame = ttk.Frame(f); btn_frame.pack(fill="x", pady=8)
        ttk.Button(btn_frame, text="🔍 Shot Detection",
                   style="Run.TButton", command=self._run_shot_detection).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="🖼️ Keyframe Extraction",
                   style="Run.TButton", command=self._run_keyframe_extraction).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="🚀 Chạy Toàn Bộ Pipeline",
                   style="Export.TButton", command=self._run_full_preprocess).pack(side="left", padx=4)

        # Log riêng cho preprocessing
        log_pp = ttk.LabelFrame(f, text=" Log Preprocessing ", padding=5)
        log_pp.pack(fill="both", expand=True, pady=5)
        self.txt_pp_log = scrolledtext.ScrolledText(
            log_pp, height=10, font=("Consolas", 8), bg="#181825", fg="#A6E3A1"
        )
        self.txt_pp_log.pack(fill="both", expand=True)

    def _pp_log(self, msg: str):
        self.txt_pp_log.insert(tk.END, msg + "\n")
        self.txt_pp_log.see(tk.END)
        self._log(msg)

    def _run_shot_detection(self):
        if self._preprocess_running:
            return
        video_dir = self._pp_video_dir.get()
        shots_dir = self._pp_shots_dir.get()
        self._preprocess_running = True
        self.progress_bar.start()
        self._pp_log(f"[Shot] Bắt đầu detection: {video_dir}")

        def _thread():
            try:
                if self._pp_use_transnet.get():
                    from preprocessing.shot_detection.transnetv2_infer import batch_detect_shots
                    batch_detect_shots(video_dir, shots_dir)
                else:
                    from preprocessing.shot_detection.dake_lightweight import batch_detect
                    batch_detect(video_dir, shots_dir)
                self.msg_queue.put({"type": "log", "text": f"[Shot] Xong! → {shots_dir}"})
            except Exception as e:
                self.msg_queue.put({"type": "log", "text": f"[Shot] Lỗi: {e}"})
            finally:
                self._preprocess_running = False
                self.root.after(0, self.progress_bar.stop)

        threading.Thread(target=_thread, daemon=True).start()

    def _run_keyframe_extraction(self):
        if self._preprocess_running:
            return
        video_dir = self._pp_video_dir.get()
        shots_dir = self._pp_shots_dir.get()
        kf_dir = self._pp_kf_dir.get()
        num_kf = self._pp_num_kf.get()
        self._preprocess_running = True
        self.progress_bar.start()
        self._pp_log(f"[KF] Bắt đầu extraction: {video_dir}")

        def _thread():
            try:
                from preprocessing.keyframe_extraction.extract_from_shots import batch_extract
                batch_extract(video_dir, shots_dir, kf_dir, num_per_shot=num_kf)
                self.msg_queue.put({"type": "log", "text": f"[KF] Xong! → {kf_dir}"})
            except Exception as e:
                self.msg_queue.put({"type": "log", "text": f"[KF] Lỗi: {e}"})
            finally:
                self._preprocess_running = False
                self.root.after(0, self.progress_bar.stop)

        threading.Thread(target=_thread, daemon=True).start()

    def _run_full_preprocess(self):
        """Chạy toàn bộ: Shot → Keyframe → Reload index."""
        self._pp_log("[Pipeline] Bắt đầu full preprocessing...")
        self._run_shot_detection()
        # Keyframe sẽ chạy sau khi shot xong (dùng after polling đơn giản)
        self.root.after(500, lambda: self._run_keyframe_extraction()
                        if not self._preprocess_running else None)

    def _log(self, text: str):
        self.txt_log.insert(tk.END, text + "\n")
        self.txt_log.see(tk.END)

    def _process_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                msg_type = msg.get("type")
                
                if msg_type == "status":
                    self.lbl_status.config(text=msg.get("text", ""))
                elif msg_type == "log":
                    self._log(msg.get("text", ""))
                elif msg_type == "dataset_ready":
                    self.is_data_loaded = True
                    self.lbl_dataset_status.config(text="✅ Đã nạp dữ liệu xong!", foreground="#A6E3A1")
                    self.progress_bar.stop()
                    self.lbl_status.config(text="Hệ thống sẵn sàng tìm kiếm.")
                    messagebox.showinfo("Thông báo", "Dữ liệu và các Index (FAISS + BM25) đã được khởi tạo thành công!")
                elif msg_type == "dataset_error":
                    self.lbl_dataset_status.config(text="❌ Lỗi dữ liệu", foreground="#F38BA8")
                    self.progress_bar.stop()
                    messagebox.showerror("Lỗi dữ liệu", msg.get("text", "Không thể nạp dữ liệu."))
                elif msg_type == "search_complete":
                    self.progress_bar.stop()
                    self._set_buttons_state("normal")
                    self._update_results_tables()
                    target_tab = msg.get("target_tab")
                    if target_tab == "kis":
                        self.notebook.select(self.tab_kis)
                        self.lbl_status.config(text="Đã hoàn thành Task 1.1 KIS!")
                        messagebox.showinfo("Thành công", "Đã tìm kiếm xong Task 1.1: Textual KIS!")
                    elif target_tab == "qa":
                        self.notebook.select(self.tab_qa)
                        self.lbl_status.config(text="Đã hoàn thành Task 1.2 QA!")
                        messagebox.showinfo("Thành công", "Đã tìm kiếm xong Task 1.2: Visual QA!")
                    elif target_tab == "trake":
                        self.notebook.select(self.tab_trake)
                        self.lbl_status.config(text="Đã hoàn thành Task 1.3 TRAKE!")
                        messagebox.showinfo("Thành công", "Đã tìm kiếm xong Task 1.3: TRAKE Sequence!")
                    else:
                        self.lbl_status.config(text="Đã hoàn thành tìm kiếm cả 3 bài toán!")
                        messagebox.showinfo("Thành công", "Đã tìm kiếm xong kết quả cho cả 3 bài toán!")
                elif msg_type == "search_error":
                    self.progress_bar.stop()
                    self._set_buttons_state("normal")
                    messagebox.showerror("Lỗi thực thi", msg.get("text", "Có lỗi xảy ra khi thực thi tìm kiếm."))

        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._process_queue)

    def _set_buttons_state(self, state: str):
        for btn in (getattr(self, "btn_run_kis", None),
                    getattr(self, "btn_run_qa", None),
                    getattr(self, "btn_run_trake", None),
                    getattr(self, "btn_run_all", None)):
            if btn:
                btn.config(state=state)

    def load_dataset_async(self):
        self.progress_bar.start()
        self.msg_queue.put({"type": "status", "text": "Đang nạp dữ liệu và xây dựng FAISS/BM25 index..."})

        def thread_target():
            try:
                data_dir = os.path.join(PROJECT_ROOT, "data")
                objects_dir = os.path.join(data_dir, "objects")

                self.loader = AICDatasetLoader(data_dir)
                features, keyframe_map = self.loader.load_video_dataset()

                if features.shape[0] == 0:
                    self.msg_queue.put({"type": "dataset_error", "text": "Không tìm thấy dữ liệu trong thư mục data/. Hãy chạy download_data.py trước."})
                    return

                self.feature_indexer = FeatureIndexer(embedding_dim=features.shape[1])
                self.feature_indexer.build_index(features, keyframe_map)

                self.metadata_indexer = MetadataIndexer()
                objects_cache_path = os.path.join(data_dir, "objects_cache.pkl")
                cache_loaded = self.metadata_indexer.load_objects_cache(objects_cache_path)

                v_ids = self.loader.get_all_video_ids()
                objects_available = os.path.exists(objects_dir)

                for v_id in v_ids:
                    meta = self.loader.load_media_info(v_id)
                    if meta:
                        self.media_info_cache[v_id] = meta
                        self.metadata_indexer.add_video_metadata(
                            v_id,
                            meta,
                            objects_dir=objects_dir if (objects_available and not cache_loaded) else None
                        )

                if not cache_loaded and objects_available:
                    self.metadata_indexer.save_objects_cache(objects_cache_path)

                self.metadata_indexer.build_bm25_index()

                self.query_processor = QueryProcessor()
                self.clip_encoder = CLIPTextEncoder()
                self.search_engine = HybridSearchEngine(self.feature_indexer, self.metadata_indexer, clip_encoder=self.clip_encoder)

                self.kis_solver = TextualKISSolver(self.search_engine, self.query_processor)
                self.qa_solver = QASolver(self.search_engine, self.query_processor)
                self.trake_solver = TRAKESolver(self.search_engine, self.query_processor)

                self.msg_queue.put({"type": "log", "text": f"[+] Đã index {features.shape[0]} keyframes và {len(v_ids)} video metadata."})
                self.msg_queue.put({"type": "dataset_ready"})
            except Exception as err:
                self.msg_queue.put({"type": "dataset_error", "text": str(err)})

        threading.Thread(target=thread_target, daemon=True).start()

    def run_kis_async(self):
        if not self.is_data_loaded:
            messagebox.showwarning("Cảnh báo", "Dữ liệu chưa nạp xong. Vui lòng chờ vài giây!")
            return
        query_kis = self.txt_kis.get("1.0", tk.END).strip()
        if not query_kis:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập mô tả cho Task 1.1 KIS!")
            return
        self._set_buttons_state("disabled")
        self.progress_bar.start()
        self.msg_queue.put({"type": "status", "text": "Đang thực thi Task 1.1 KIS..."})
        self._log(f"\n--- BẮT ĐẦU CHẠY TASK 1.1 KIS: '{query_kis}' ---")

        def thread_target():
            try:
                variants_kis = self.query_processor.expand_queries(query_kis)
                emb_kis = self.clip_encoder.encode_text_ensemble(variants_kis)
                kis_raw = self.kis_solver.solve(query_kis, emb_kis, top_k=100)
                self.kis_results = self.ranking_optimizer.optimize_ranking(kis_raw, max_items=100)
                self.msg_queue.put({"type": "log", "text": "[+] Task 1.1 KIS hoàn tất!"})
                self.msg_queue.put({"type": "search_complete", "target_tab": "kis"})
            except Exception as err:
                self.msg_queue.put({"type": "log", "text": f"[-] Lỗi KIS: {err}"})
                self.msg_queue.put({"type": "search_error", "text": str(err)})

        threading.Thread(target=thread_target, daemon=True).start()

    def run_qa_async(self):
        if not self.is_data_loaded:
            messagebox.showwarning("Cảnh báo", "Dữ liệu chưa nạp xong. Vui lòng chờ vài giây!")
            return
        event_qa = self.ent_qa_event.get().strip()
        question_qa = self.ent_qa_question.get().strip()
        if not question_qa:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập câu hỏi cho Task 1.2 QA!")
            return
        self._set_buttons_state("disabled")
        self.progress_bar.start()
        self.msg_queue.put({"type": "status", "text": "Đang thực thi Task 1.2 QA..."})
        self._log(f"\n--- BẮT ĐẦU CHẠY TASK 1.2 QA: Event='{event_qa}', Question='{question_qa}' ---")

        def thread_target():
            try:
                combined_qa = f"{event_qa} {question_qa}"
                variants_qa = self.query_processor.expand_queries(combined_qa)
                emb_qa = self.clip_encoder.encode_text_ensemble(variants_qa)
                qa_raw = self.qa_solver.solve(event_qa, question_qa, emb_qa, top_k=100)
                self.qa_results = self.ranking_optimizer.optimize_ranking(qa_raw, max_items=100)
                self.msg_queue.put({"type": "log", "text": "[+] Task 1.2 QA hoàn tất!"})
                self.msg_queue.put({"type": "search_complete", "target_tab": "qa"})
            except Exception as err:
                self.msg_queue.put({"type": "log", "text": f"[-] Lỗi QA: {err}"})
                self.msg_queue.put({"type": "search_error", "text": str(err)})

        threading.Thread(target=thread_target, daemon=True).start()

    def run_trake_async(self):
        if not self.is_data_loaded:
            messagebox.showwarning("Cảnh báo", "Dữ liệu chưa nạp xong. Vui lòng chờ vài giây!")
            return
        trake_text = self.txt_trake.get("1.0", tk.END).strip()
        if not trake_text:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập mô tả chuỗi cho Task 1.3 TRAKE!")
            return
        self._set_buttons_state("disabled")
        self.progress_bar.start()
        self.msg_queue.put({"type": "status", "text": "Đang thực thi Task 1.3 TRAKE..."})
        self._log(f"\n--- BẮT ĐẦU CHẠY TASK 1.3 TRAKE: '{trake_text}' ---")

        def thread_target():
            try:
                sub_events = self.query_processor.parse_trake_query(trake_text)
                event_embs = [self.clip_encoder.encode_text_ensemble(self.query_processor.expand_queries(ev)) for ev in sub_events]
                self.trake_results = self.trake_solver.solve(trake_text, event_embs, top_k=100)
                self.msg_queue.put({"type": "log", "text": "[+] Task 1.3 TRAKE hoàn tất!"})
                self.msg_queue.put({"type": "search_complete", "target_tab": "trake"})
            except Exception as err:
                self.msg_queue.put({"type": "log", "text": f"[-] Lỗi TRAKE: {err}"})
                self.msg_queue.put({"type": "search_error", "text": str(err)})

        threading.Thread(target=thread_target, daemon=True).start()

    def run_all_tasks_async(self):
        if not self.is_data_loaded:
            messagebox.showwarning("Cảnh báo", "Dữ liệu chưa nạp xong. Vui lòng chờ vài giây!")
            return

        query_kis = self.txt_kis.get("1.0", tk.END).strip()
        event_qa = self.ent_qa_event.get().strip()
        question_qa = self.ent_qa_question.get().strip()
        trake_text = self.txt_trake.get("1.0", tk.END).strip()

        if not query_kis or not question_qa or not trake_text:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập đầy đủ mô tả cho cả 3 bài toán!")
            return

        self._set_buttons_state("disabled")
        self.progress_bar.start()
        self.msg_queue.put({"type": "status", "text": "Đang thực thi mô hình cho cả 3 bài toán..."})
        self._log("\n--- BẮT ĐẦU CHẠY THỬ MÔ HÌNH (CẢ 3 BÀI TOÁN) ---")

        def thread_target():
            try:
                self.msg_queue.put({"type": "log", "text": f"[1/3] Đang xử lý Task 1.1 KIS: '{query_kis}'..."})
                variants_kis = self.query_processor.expand_queries(query_kis)
                emb_kis = self.clip_encoder.encode_text_ensemble(variants_kis)
                kis_raw = self.kis_solver.solve(query_kis, emb_kis, top_k=100)
                self.kis_results = self.ranking_optimizer.optimize_ranking(kis_raw, max_items=100)

                self.msg_queue.put({"type": "log", "text": f"[2/3] Đang xử lý Task 1.2 QA: Event='{event_qa}', Question='{question_qa}'..."})
                combined_qa = f"{event_qa} {question_qa}"
                variants_qa = self.query_processor.expand_queries(combined_qa)
                emb_qa = self.clip_encoder.encode_text_ensemble(variants_qa)
                qa_raw = self.qa_solver.solve(event_qa, question_qa, emb_qa, top_k=100)
                self.qa_results = self.ranking_optimizer.optimize_ranking(qa_raw, max_items=100)

                self.msg_queue.put({"type": "log", "text": f"[3/3] Đang xử lý Task 1.3 TRAKE Sequence: '{trake_text}'..."})
                sub_events = self.query_processor.parse_trake_query(trake_text)
                event_embs = [self.clip_encoder.encode_text_ensemble(self.query_processor.expand_queries(ev)) for ev in sub_events]
                self.trake_results = self.trake_solver.solve(sub_events, event_embs, top_k=100)

                self.msg_queue.put({"type": "log", "text": "[+] Tìm kiếm hoàn tất cho cả 3 bài toán!"})
                self.msg_queue.put({"type": "search_complete"})
            except Exception as err:
                self.msg_queue.put({"type": "log", "text": f"[-] Lỗi: {err}"})
                self.msg_queue.put({"type": "search_error", "text": str(err)})

        threading.Thread(target=thread_target, daemon=True).start()

    def _update_results_tables(self):
        for tree in (self.tree_kis, self.tree_qa, self.tree_trake):
            for item in tree.get_children():
                tree.delete(item)

        def get_pts(v_id, f_id):
            if self.feature_indexer:
                for kmap in self.feature_indexer.keyframe_map:
                    if kmap["video_id"] == v_id and kmap["frame_id"] == f_id:
                        return kmap.get("pts_time", 0.0)
            return 0.0

        for i, res in enumerate(self.kis_results[:50], 1):
            v_id = res["video_id"]
            f_id = res["frame_id"]
            pts = get_pts(v_id, f_id)
            score = f"{res.get('score', 0.0):.4f}"
            self.tree_kis.insert("", "end", values=(i, v_id, f_id, f"{pts:.2f}s", score))

        for i, res in enumerate(self.qa_results[:50], 1):
            v_id = res["video_id"]
            f_id = res["frame_id"]
            pts = get_pts(v_id, f_id)
            ans = res.get("answer", "có")
            score = f"{res.get('score', 0.0):.4f}"
            self.tree_qa.insert("", "end", values=(i, v_id, f_id, f"{pts:.2f}s", ans, score))

        for i, res in enumerate(self.trake_results[:50], 1):
            v_id = res["video_id"]
            f_ids = res.get("frame_ids", [])
            f_str = f"({', '.join(str(f) for f in f_ids)})"
            pts_list = [f"{get_pts(v_id, f):.1f}s" for f in f_ids]
            pts_str = f"({', '.join(pts_list)})"
            score = f"{res.get('score', 0.0):.4f}"
            self.tree_trake.insert("", "end", values=(i, v_id, f_str, pts_str, score))

    def _on_kis_double_click(self, event):
        item = self.tree_kis.selection()
        if item:
            vals = self.tree_kis.item(item[0], "values")
            video_id = vals[1]
            pts_str = vals[3].replace("s", "")
            pts_time = float(pts_str) if pts_str != "-" else 0.0
            self._display_video_preview(video_id, pts_time)

    def _on_qa_double_click(self, event):
        item = self.tree_qa.selection()
        if item:
            vals = self.tree_qa.item(item[0], "values")
            video_id = vals[1]
            pts_str = vals[3].replace("s", "")
            pts_time = float(pts_str) if pts_str != "-" else 0.0
            self._display_video_preview(video_id, pts_time)

    def _on_trake_double_click(self, event):
        item = self.tree_trake.selection()
        if item:
            vals = self.tree_trake.item(item[0], "values")
            video_id = vals[1]
            pts_str = vals[3].strip("()").split(",")[0].replace("s", "").strip()
            pts_time = float(pts_str) if pts_str and pts_str != "-" else 0.0
            self._display_video_preview(video_id, pts_time)

    def _display_video_preview(self, video_id: str, pts_time: float):
        self.notebook.select(self.tab_video)

        meta = self.media_info_cache.get(video_id)
        if not meta and self.loader:
            meta = self.loader.load_media_info(video_id)
            if meta:
                self.media_info_cache[video_id] = meta

        self.lbl_v_id.config(text=f"Video ID: {video_id}")
        self.lbl_v_pts.config(text=f"Thời điểm Keyframe (PTS): {pts_time:.2f} giây")

        if meta:
            title = meta.get("title", f"Video {video_id}")
            channel = meta.get("author", "N/A")
            duration = f"{meta.get('length', 0)} giây"
            watch_url = meta.get("watch_url", "")

            self.lbl_v_title.config(text=f"🎥 {title}")
            self.lbl_v_channel.config(text=f"Kênh / Tác giả: {channel}")
            self.lbl_v_duration.config(text=f"Thời lượng Video: {duration}")
            self.lbl_v_url.config(text=f"YouTube URL: {watch_url}")

            self.txt_v_desc.delete("1.0", tk.END)
            self.txt_v_desc.insert("1.0", meta.get("description", "Không có mô tả."))

            self.current_selected_watch_url = watch_url
            self.current_selected_pts = pts_time
            self.btn_open_youtube.config(state="normal")
        else:
            self.lbl_v_title.config(text=f"Video {video_id} (Chưa có Media Metadata)")
            self.lbl_v_channel.config(text="Kênh / Tác giả: -")
            self.lbl_v_duration.config(text="Thời lượng Video: -")
            self.lbl_v_url.config(text="YouTube URL: -")
            self.txt_v_desc.delete("1.0", tk.END)

            self.current_selected_watch_url = None
            self.current_selected_pts = 0.0
            self.btn_open_youtube.config(state="disabled")

    def _open_youtube_current(self):
        if self.current_selected_watch_url:
            pts_int = int(self.current_selected_pts)
            url_with_timestamp = f"{self.current_selected_watch_url}&t={pts_int}s"
            self._log(f"[+] Mở trình duyệt xem video: {url_with_timestamp}")
            webbrowser.open(url_with_timestamp)

    def export_submissions(self):
        if not self.kis_results and not self.qa_results and not self.trake_results:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả tìm kiếm để xuất file!")
            return

        out_dir = os.path.join(PROJECT_ROOT, "submissions")
        os.makedirs(out_dir, exist_ok=True)
        exported = []

        try:
            if self.kis_results:
                p = format_and_export("query_kis_gui", "KIS", self.kis_results, out_dir)
                exported.append(p)
            if self.qa_results:
                p = format_and_export("query_qa_gui", "QA", self.qa_results, out_dir)
                exported.append(p)
            if self.trake_results:
                p = format_and_export("query_trake_gui", "TRAKE", self.trake_results, out_dir)
                exported.append(p)
            msg = "Đã xuất submission:\n" + "\n".join(f"- {p}" for p in exported)
            messagebox.showinfo("Thành công", msg)
            self._log(f"[Export] {len(exported)} file(s) → {out_dir}")
        except Exception as e:
            messagebox.showerror("Lỗi xuất file", str(e))
            self._log(f"[Export] Lỗi: {e}")


def main():
    root = tk.Tk()
    app = AICVideoRetrievalGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
