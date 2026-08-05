"""
TRAKE viewer: hiển thị nhiều frame ứng với nhiều event, cho chọn frame từng event.
Dùng trong giao diện chính (Tkinter) khi cần submit TRAKE.
"""
import os
from typing import List, Dict, Any, Callable, Optional


class MultiFrameSelector:
    """
    Hiển thị danh sách event và cho phép người dùng chọn frame cho từng event.
    Dùng Tkinter.
    """
    def __init__(self, root, events: List[str], candidates_per_event: List[List[Dict]],
                 keyframes_dir: str, on_submit: Optional[Callable] = None):
        """
        Args:
            root: Tkinter root window hoặc Toplevel
            events: mô tả từng event
            candidates_per_event: [[{video_id, frame_id, score}]] cho từng event
            keyframes_dir: đường dẫn gốc tới keyframes
            on_submit: callback(selected_frames: List[Dict])
        """
        self.root = root
        self.events = events
        self.candidates = candidates_per_event
        self.keyframes_dir = keyframes_dir
        self.on_submit = on_submit
        self.selected: List[Optional[Dict]] = [None] * len(events)
        self._build_ui()

    def _build_ui(self):
        try:
            import tkinter as tk
            from tkinter import ttk
            from PIL import Image, ImageTk  # type: ignore
        except ImportError:
            print("[!] tkinter/Pillow không khả dụng.")
            return

        import tkinter as tk
        from tkinter import ttk

        win = tk.Toplevel(self.root)
        win.title("TRAKE Frame Selector")
        win.geometry("900x600")

        for event_idx, event_desc in enumerate(self.events):
            frame = tk.LabelFrame(win, text=f"E{event_idx+1}: {event_desc[:80]}", padx=5, pady=5)
            frame.pack(fill="x", padx=10, pady=5)

            var = tk.IntVar(value=0)
            cands = self.candidates[event_idx] if event_idx < len(self.candidates) else []
            for i, cand in enumerate(cands[:5]):
                label = f"{cand.get('video_id')} @ {cand.get('frame_id')} (score={cand.get('score', 0):.3f})"
                rb = tk.Radiobutton(frame, text=label, variable=var, value=i,
                                    command=lambda idx=event_idx, i=i, c=cands: self._select(idx, c[i] if i < len(c) else None))
                rb.pack(anchor="w")
            if cands:
                self._select(event_idx, cands[0])

        btn = tk.Button(win, text="✅ Submit", command=self._submit, bg="#4CAF50", fg="white")
        btn.pack(pady=10)

    def _select(self, event_idx: int, candidate: Optional[Dict]):
        self.selected[event_idx] = candidate

    def _submit(self):
        if self.on_submit:
            self.on_submit(self.selected)
