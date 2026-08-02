import os
import sys

# Prevent OpenMP duplicate runtime initialization crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import threading
import queue
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import backend modules
try:
    from src.data.dataset_loader import AICDatasetLoader
    from src.data.feature_indexer import FeatureIndexer
    from src.data.metadata_indexer import MetadataIndexer
    from src.retrieval.query_processor import QueryProcessor
    from src.retrieval.clip_encoder import CLIPTextEncoder
    from src.retrieval.hybrid_search import HybridSearchEngine
    from src.modules.kis_solver import TextualKISSolver
    from src.modules.qa_solver import QASolver
    from src.modules.trake_solver import TRAKESolver
    from src.submission.ranking_optimizer import RankingOptimizer
    from src.submission.format_validator import AICFormatValidator
except Exception as e:
    import traceback
    print(f"[!] Critical Error importing backend modules: {e}")
    traceback.print_exc()

# Optional PIL for thumbnail viewing
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
    - Task 1.1: Textual KIS
    - Task 1.2: Visual Q&A
    - Task 1.3: TRAKE Temporal Alignment
    Supports dataset indexing, search execution, submission export, and video playback on YouTube at exact PTS timestamps.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AIC 2026 - Video Retrieval & Verification Studio")
        self.root.geometry("1280x820")
        self.root.minsize(1024, 700)

        # Thread queue for background tasks
        self.msg_queue = queue.Queue()

        # Backend Engine References
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

        # Data cache for results
        self.kis_results = []
        self.qa_results = []
        self.trake_results = []
        self.media_info_cache = {}

        # Setup modern styles
        self._setup_styles()

        # Build UI layout
        self._build_ui()

        # Poll message queue
        self.root.after(100, self._process_queue)

        # Auto-load dataset in background thread upon launch
        self._log("Chương trình đã khởi động. Đang khởi tạo bộ nạp dữ liệu backend...")
        self.load_dataset_async()

    def _setup_styles(self):
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")

        # Color Palette - Professional Dark Mode Theme
        BG_DARK = "#181825"
        PANEL_BG = "#1E1E2E"
        ACCENT_BLUE = "#89B4FA"
        ACCENT_GREEN = "#A6E3A1"
        TEXT_LIGHT = "#CDD6F4"
        TEXT_MUTED = "#BAC2DE"

        self.root.configure(bg=BG_DARK)

        # Frame styles
        self.style.configure("TFrame", background=BG_DARK)
        self.style.configure("Card.TFrame", background=PANEL_BG, relief="solid", borderwidth=1)

        # Label styles
        self.style.configure("TLabel", background=BG_DARK, foreground=TEXT_LIGHT, font=("Segoe UI", 10))
        self.style.configure("Card.TLabel", background=PANEL_BG, foreground=TEXT_LIGHT, font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", background=PANEL_BG, foreground=ACCENT_BLUE, font=("Segoe UI", 12, "bold"))
        self.style.configure("Title.TLabel", background=BG_DARK, foreground=ACCENT_BLUE, font=("Segoe UI", 16, "bold"))
        self.style.configure("Status.TLabel", background=PANEL_BG, foreground=TEXT_MUTED, font=("Segoe UI", 9))

        # Button styles
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        self.style.configure("Primary.TButton", background="#313244", foreground=TEXT_LIGHT)
        self.style.map("Primary.TButton", background=[("active", "#45475A")])
        self.style.configure("Run.TButton", background="#2563EB", foreground="#FFFFFF")
        self.style.map("Run.TButton", background=[("active", "#1D4ED8")])
        self.style.configure("Export.TButton", background="#059669", foreground="#FFFFFF")
        self.style.map("Export.TButton", background=[("active", "#047857")])

        # Notebook tabs
        self.style.configure("TNotebook", background=BG_DARK, borderwidth=0)
        self.style.configure("TNotebook.Tab", background="#313244", foreground=TEXT_LIGHT, font=("Segoe UI", 10, "bold"), padding=[12, 6])
        self.style.map("TNotebook.Tab", background=[("selected", "#45475A")], foreground=[("selected", ACCENT_BLUE)])

        # Treeview (Tables)
        self.style.configure("Treeview", background="#1E1E2E", foreground=TEXT_LIGHT, fieldbackground="#1E1E2E", font=("Segoe UI", 10), rowheight=26)
        self.style.configure("Treeview.Heading", background="#313244", foreground=ACCENT_BLUE, font=("Segoe UI", 10, "bold"))
        self.style.map("Treeview", background=[("selected", "#45475A")], foreground=[("selected", "#FFFFFF")])

    def _build_ui(self):
        # Main Top Header
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")

        title_label = ttk.Label(top_frame, text="🎥 AIC 2026 - TRÌNH KIỂM THỬ VÀ TRUY XUẤT VIDEO", style="Title.TLabel")
        title_label.pack(side="left")

        self.lbl_dataset_status = ttk.Label(top_frame, text="⏳ Đang khởi tạo dữ liệu...", style="Status.TLabel")
        self.lbl_dataset_status.pack(side="right", padx=10)

        # Split Main Container (Left Panel: Inputs, Right Panel: Results & Preview)
        main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        main_paned.pack(fill="both", expand=True, padx=10, pady=5)

        # ── LEFT PANEL: INPUT FORM ──────────────────────────────────────────────
        left_container = ttk.Frame(main_paned, style="Card.TFrame", padding=12)
        main_paned.add(left_container, weight=1)

        # Panel Header
        ttk.Label(left_container, text="📝 Nhập Mô Tả 3 Bài Toán", style="Header.TLabel").pack(anchor="w", pady=(0, 10))

        # --- Task 1: Textual KIS ---
        grp_kis = ttk.LabelFrame(left_container, text=" Task 1.1: Textual KIS (Tìm kiếm Keyframe) ", padding=8)
        grp_kis.pack(fill="x", pady=5)
        
        ttk.Label(grp_kis, text="Mô tả sự kiện/đối tượng trong video:").pack(anchor="w")
        self.txt_kis = tk.Text(grp_kis, height=3, width=40, font=("Segoe UI", 9), wrap="word", bg="#313244", fg="#CDD6F4", insertbackground="white")
        self.txt_kis.pack(fill="x", pady=5)
        self.txt_kis.insert("1.0", "Diễn giả mặc áo đỏ phát biểu tại cuộc họp báo ngoài trời")

        # --- Task 2: Visual Q&A ---
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

        # --- Task 3: TRAKE ---
        grp_trake = ttk.LabelFrame(left_container, text=" Task 1.3: TRAKE (Chuỗi Hành Động Thời Gian) ", padding=8)
        grp_trake.pack(fill="x", pady=5)

        ttk.Label(grp_trake, text="Chuỗi các sự kiện nối tiếp:").pack(anchor="w")
        self.txt_trake = tk.Text(grp_trake, height=3, width=40, font=("Segoe UI", 9), wrap="word", bg="#313244", fg="#CDD6F4", insertbackground="white")
        self.txt_trake.pack(fill="x", pady=5)
        self.txt_trake.insert("1.0", "(1) Giậm nhảy, (2) Bay qua xà, (3) Tiếp đất, (4) Đứng dậy")

        # --- Execution Action Buttons ---
        btn_frame = ttk.Frame(left_container, padding=5)
        btn_frame.pack(fill="x", pady=10)

        self.btn_run_all = ttk.Button(
            btn_frame,
            text="🚀 Chạy Mô Hình (Cả 3 Bài Toán)",
            style="Run.TButton",
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

        # ── RIGHT PANEL: RESULTS & VIDEO PREVIEW ────────────────────────────────
        right_container = ttk.Frame(main_paned, style="Card.TFrame", padding=10)
        main_paned.add(right_container, weight=3)

        # Notebook Tabs
        self.notebook = ttk.Notebook(right_container)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: KIS Results
        self.tab_kis = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.tab_kis, text=" 🏷️ Task 1.1: Textual KIS ")
        self._setup_kis_tab()

        # Tab 2: QA Results
        self.tab_qa = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.tab_qa, text=" ❓ Task 1.2: Visual QA ")
        self._setup_qa_tab()

        # Tab 3: TRAKE Results
        self.tab_trake = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.tab_trake, text=" ⏱️ Task 1.3: TRAKE Sequence ")
        self._setup_trake_tab()

        # Tab 4: Video Preview & Metadata
        self.tab_video = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.tab_video, text=" 🎬 Xem Video & Metadata ")
        self._setup_video_tab()

        # ── BOTTOM STATUS BAR & LOG ─────────────────────────────────────────────
        bottom_frame = ttk.Frame(self.root, padding=5)
        bottom_frame.pack(fill="x", side="bottom")

        self.progress_bar = ttk.Progressbar(bottom_frame, mode="indeterminate")
        self.progress_bar.pack(fill="x", pady=2)

        self.lbl_status = ttk.Label(bottom_frame, text="Sẵn sàng.", style="Status.TLabel")
        self.lbl_status.pack(side="left")

        # Collapsible Log Box
        log_frame = ttk.LabelFrame(bottom_frame, text=" Log Hệ Thống Pipeline ", padding=5)
        log_frame.pack(fill="x", pady=5)

        self.txt_log = scrolledtext.ScrolledText(
            log_frame, height=4, font=("Consolas", 8), bg="#181825", fg="#CDD6F4", insertbackground="white"
        )
        self.txt_log.pack(fill="x")

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
        # Video Details Container
        v_container = ttk.Frame(self.tab_video, padding=15)
        v_container.pack(fill="both", expand=True)

        # Header Info Card
        self.lbl_v_title = ttk.Label(v_container, text="Chọn một kết quả từ bảng để xem chi tiết Video", font=("Segoe UI", 12, "bold"), foreground="#89B4FA")
        self.lbl_v_title.pack(anchor="w", pady=(0, 10))

        # YouTube Watch Action Button
        self.btn_open_youtube = ttk.Button(
            v_container,
            text="▶️ Mở Video trên YouTube (Đúng Thời Điểm PTS)",
            style="Run.TButton",
            command=self._open_youtube_current
        )
        self.btn_open_youtube.pack(anchor="w", pady=5)
        self.btn_open_youtube.config(state="disabled")

        # Video Metadata Grid
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

        # Description Box
        ttk.Label(meta_frame, text="Mô tả Video:").grid(row=3, column=0, sticky="w", pady=(10, 2), padx=5)
        self.txt_v_desc = scrolledtext.ScrolledText(meta_frame, height=8, font=("Segoe UI", 9), bg="#181825", fg="#CDD6F4")
        self.txt_v_desc.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=5, pady=2)
        meta_frame.rowconfigure(4, weight=1)
        meta_frame.columnconfigure(1, weight=1)

        self.current_selected_watch_url = None
        self.current_selected_pts = 0.0

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
                    self.btn_run_all.config(state="normal")
                    self._update_results_tables()
                    self.lbl_status.config(text="Đã hoàn thành tìm kiếm 3 bài toán!")
                    messagebox.showinfo("Thành công", "Đã tìm kiếm xong kết quả cho cả 3 bài toán! Hãy xem chi tiết ở các Tab.")
                elif msg_type == "search_error":
                    self.progress_bar.stop()
                    self.btn_run_all.config(state="normal")
                    messagebox.showerror("Lỗi thực thi", msg.get("text", "Có lỗi xảy ra khi thực thi tìm kiếm."))

        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._process_queue)

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

                # Build Feature Indexer
                self.feature_indexer = FeatureIndexer(embedding_dim=features.shape[1])
                self.feature_indexer.build_index(features, keyframe_map)

                # Build Metadata Indexer
                self.metadata_indexer = MetadataIndexer()
                v_ids = self.loader.get_all_video_ids()
                objects_available = os.path.exists(objects_dir)

                for v_id in v_ids:
                    meta = self.loader.load_media_info(v_id)
                    if meta:
                        self.media_info_cache[v_id] = meta
                        self.metadata_indexer.add_video_metadata(
                            v_id,
                            meta,
                            objects_dir=objects_dir if objects_available else None
                        )

                self.metadata_indexer.build_bm25_index()

                # Build Solvers
                self.query_processor = QueryProcessor()
                self.clip_encoder = CLIPTextEncoder()
                self.search_engine = HybridSearchEngine(self.feature_indexer, self.metadata_indexer)

                self.kis_solver = TextualKISSolver(self.search_engine, self.query_processor)
                self.qa_solver = QASolver(self.search_engine, self.query_processor)
                self.trake_solver = TRAKESolver(self.search_engine, self.query_processor)

                self.msg_queue.put({"type": "log", "text": f"[+] Đã index {features.shape[0]} keyframes và {len(v_ids)} video metadata."})
                self.msg_queue.put({"type": "dataset_ready"})
            except Exception as err:
                self.msg_queue.put({"type": "dataset_error", "text": str(err)})

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

        self.btn_run_all.config(state="disabled")
        self.progress_bar.start()
        self.msg_queue.put({"type": "status", "text": "Đang thực thi mô hình cho 3 bài toán..."})
        self._log("\n--- BẮT ĐẦU CHẠY THỬ MÔ HÌNH (3 BÀI TOÁN) ---")

        def thread_target():
            try:
                # ── 1. Task 1.1: Textual KIS ──
                self.msg_queue.put({"type": "log", "text": f"[1/3] Đang xử lý Task 1.1 KIS: '{query_kis}'..."})
                variants_kis = self.query_processor.expand_queries(query_kis)
                emb_kis = self.clip_encoder.encode_text_ensemble(variants_kis)
                kis_raw = self.kis_solver.solve(query_kis, emb_kis, top_k=100)
                self.kis_results = self.ranking_optimizer.optimize_ranking(kis_raw, max_items=100)

                # ── 2. Task 1.2: Visual QA ──
                self.msg_queue.put({"type": "log", "text": f"[2/3] Đang xử lý Task 1.2 QA: Event='{event_qa}', Question='{question_qa}'..."})
                combined_qa = f"{event_qa} {question_qa}"
                variants_qa = self.query_processor.expand_queries(combined_qa)
                emb_qa = self.clip_encoder.encode_text_ensemble(variants_qa)
                qa_raw = self.qa_solver.solve(event_qa, question_qa, emb_qa, top_k=100)
                self.qa_results = self.ranking_optimizer.optimize_ranking(qa_raw, max_items=100)

                # ── 3. Task 1.3: TRAKE ──
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
        # Clear tables
        for tree in (self.tree_kis, self.tree_qa, self.tree_trake):
            for item in tree.get_children():
                tree.delete(item)

        # Helper to get PTS time
        def get_pts(v_id, f_id):
            if self.feature_indexer:
                for kmap in self.feature_indexer.keyframe_map:
                    if kmap["video_id"] == v_id and kmap["frame_id"] == f_id:
                        return kmap.get("pts_time", 0.0)
            return 0.0

        # Fill KIS Table
        for i, res in enumerate(self.kis_results[:50], 1):
            v_id = res["video_id"]
            f_id = res["frame_id"]
            pts = get_pts(v_id, f_id)
            score = f"{res.get('score', 0.0):.4f}"
            self.tree_kis.insert("", "end", values=(i, v_id, f_id, f"{pts:.2f}s", score))

        # Fill QA Table
        for i, res in enumerate(self.qa_results[:50], 1):
            v_id = res["video_id"]
            f_id = res["frame_id"]
            pts = get_pts(v_id, f_id)
            ans = res.get("answer", "có")
            score = f"{res.get('score', 0.0):.4f}"
            self.tree_qa.insert("", "end", values=(i, v_id, f_id, f"{pts:.2f}s", ans, score))

        # Fill TRAKE Table
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
            # Parse first frame's PTS
            pts_str = vals[3].strip("()").split(",")[0].replace("s", "").strip()
            pts_time = float(pts_str) if pts_str and pts_str != "-" else 0.0
            self._display_video_preview(video_id, pts_time)

    def _display_video_preview(self, video_id: str, pts_time: float):
        # Switch to Video Tab
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
        if not self.kis_results and not self.qa_results:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả tìm kiếm để xuất file!")
            return

        out_dir = os.path.join(PROJECT_ROOT, "submissions")
        os.makedirs(out_dir, exist_ok=True)

        out_kis = os.path.join(out_dir, "kis_submission_gui.csv")
        out_qa = os.path.join(out_dir, "qa_submission_gui.csv")

        self.validator.export_csv("query_kis_gui", self.kis_results, out_kis)
        self.validator.export_csv("query_qa_gui", self.qa_results, out_qa)

        messagebox.showinfo("Thành công", f"Đã xuất các file submission tại:\n- {out_kis}\n- {out_qa}")


def main():
    root = tk.Tk()
    app = AICVideoRetrievalGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
