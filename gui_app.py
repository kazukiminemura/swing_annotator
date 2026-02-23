#!/usr/bin/env python3
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2

from dataset_builder import (
    Anchor,
    build_dataset_from_anchors,
    load_anchors,
    parse_split_ratios,
    save_anchors,
)


class SwingAnnotatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Swing Annotator")
        self.root.geometry("1180x780")

        self.video_path: Path | None = None
        self.cap: cv2.VideoCapture | None = None
        self.total_frames = 0
        self.current_frame_index = 0
        self.preview_image: tk.PhotoImage | None = None
        self.anchors: list[Anchor] = []

        self.jump_var = tk.StringVar(value="0")
        self.label_var = tk.StringVar(value="")
        self.step_var = tk.StringVar(value="1")
        self.quality_var = tk.StringVar(value="95")
        self.output_var = tk.StringVar(value="dataset")
        self.format_var = tk.StringVar(value="roboflow")
        self.split_var = tk.StringVar(value="80,10,10")
        self.label_db_var = tk.StringVar(value="label_store.json")
        self.status_var = tk.StringVar(value="Open a video to start.")

        self._build_layout()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root, padding=8)
        top.grid(row=0, column=0, columnspan=2, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Button(top, text="Open Video", command=self.open_video).grid(
            row=0, column=0, padx=(0, 8)
        )
        self.video_path_label = ttk.Label(top, text="(no video)")
        self.video_path_label.grid(row=0, column=1, sticky="w")

        viewer = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        viewer.grid(row=1, column=0, sticky="nsew")
        viewer.rowconfigure(0, weight=1)
        viewer.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(viewer, bg="#111111", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        controls = ttk.Frame(viewer)
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        controls.columnconfigure(8, weight=1)

        ttk.Button(controls, text="<< -30", command=lambda: self.seek_relative(-30)).grid(
            row=0, column=0, padx=2
        )
        ttk.Button(controls, text="< -1", command=lambda: self.seek_relative(-1)).grid(
            row=0, column=1, padx=2
        )
        ttk.Button(controls, text="+1 >", command=lambda: self.seek_relative(1)).grid(
            row=0, column=2, padx=2
        )
        ttk.Button(controls, text="+30 >>", command=lambda: self.seek_relative(30)).grid(
            row=0, column=3, padx=2
        )

        ttk.Label(controls, text="Jump").grid(row=0, column=4, padx=(14, 2))
        ttk.Entry(controls, width=10, textvariable=self.jump_var).grid(
            row=0, column=5, padx=2
        )
        ttk.Button(controls, text="Go", command=self.seek_absolute).grid(
            row=0, column=6, padx=2
        )
        self.frame_info_label = ttk.Label(controls, text="frame: - / -")
        self.frame_info_label.grid(row=0, column=8, sticky="e")

        side = ttk.Frame(self.root, padding=(0, 0, 8, 8))
        side.grid(row=1, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)
        side.rowconfigure(3, weight=1)

        db_box = ttk.LabelFrame(side, text="Label DB (append across videos)", padding=8)
        db_box.grid(row=0, column=0, sticky="ew")
        db_box.columnconfigure(1, weight=1)
        ttk.Label(db_box, text="file").grid(row=0, column=0, sticky="w")
        ttk.Entry(db_box, textvariable=self.label_db_var).grid(
            row=0, column=1, sticky="ew", padx=(8, 4)
        )
        ttk.Button(db_box, text="Select", command=self.select_label_db).grid(row=0, column=2)
        ttk.Button(db_box, text="Load this video", command=self.load_current_video_labels).grid(
            row=1, column=0, pady=(8, 0), sticky="ew"
        )
        ttk.Button(db_box, text="Append save this video", command=self.append_current_video_labels).grid(
            row=1, column=1, columnspan=2, padx=(8, 0), pady=(8, 0), sticky="ew"
        )

        anchor_box = ttk.LabelFrame(side, text="Anchors", padding=8)
        anchor_box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        anchor_box.columnconfigure(1, weight=1)
        ttk.Label(anchor_box, text="label").grid(row=0, column=0, sticky="w")
        ttk.Entry(anchor_box, textvariable=self.label_var).grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )
        ttk.Button(anchor_box, text="Add/update at current frame", command=self.add_or_update_anchor).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )
        row = ttk.Frame(anchor_box)
        row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(row, text="Remove selected", command=self.remove_selected_anchor).grid(
            row=0, column=0, padx=(0, 4)
        )
        ttk.Button(row, text="Load anchors JSON", command=self.load_anchor_file).grid(
            row=0, column=1, padx=4
        )
        ttk.Button(row, text="Save anchors JSON", command=self.save_anchor_file).grid(
            row=0, column=2, padx=(4, 0)
        )

        list_box = ttk.LabelFrame(side, text="Anchor list (start_frame: label)", padding=8)
        list_box.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        list_box.rowconfigure(0, weight=1)
        list_box.columnconfigure(0, weight=1)
        self.anchor_list = tk.Listbox(list_box, height=10)
        self.anchor_list.grid(row=0, column=0, sticky="nsew")
        self.anchor_list.bind("<<ListboxSelect>>", self.on_anchor_select)
        sb = ttk.Scrollbar(list_box, orient="vertical", command=self.anchor_list.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.anchor_list.configure(yscrollcommand=sb.set)

        export_box = ttk.LabelFrame(side, text="Dataset export", padding=8)
        export_box.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        export_box.columnconfigure(1, weight=1)

        ttk.Label(export_box, text="output").grid(row=0, column=0, sticky="w")
        ttk.Entry(export_box, textvariable=self.output_var).grid(
            row=0, column=1, sticky="ew", padx=(8, 4)
        )
        ttk.Button(export_box, text="Select", command=self.select_output_dir).grid(row=0, column=2)

        ttk.Label(export_box, text="format").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            export_box,
            textvariable=self.format_var,
            state="readonly",
            values=["roboflow", "native"],
            width=14,
        ).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(export_box, text="split").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(export_box, textvariable=self.split_var, width=14).grid(
            row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0)
        )
        ttk.Label(export_box, text="step").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(export_box, textvariable=self.step_var, width=14).grid(
            row=3, column=1, sticky="w", padx=(8, 0), pady=(8, 0)
        )
        ttk.Label(export_box, text="jpeg quality").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(export_box, textvariable=self.quality_var, width=14).grid(
            row=4, column=1, sticky="w", padx=(8, 0), pady=(8, 0)
        )
        ttk.Button(export_box, text="Export dataset", command=self.export_dataset).grid(
            row=5, column=0, columnspan=3, sticky="ew", pady=(10, 0)
        )

        status = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        status.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(status, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

    def set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.root.update_idletasks()

    def release_video(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.total_frames = 0
        self.current_frame_index = 0

    def open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All files", "*.*")],
        )
        if not path:
            return

        self.release_video()
        self.video_path = Path(path)
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            self.cap = None
            messagebox.showerror("Error", "Failed to open video.")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0
        self.video_path_label.configure(text=str(self.video_path))
        self.set_status("Video loaded.")
        self.show_frame(0)
        self.load_current_video_labels(silent=True)

    def read_frame(self, frame_index: int):
        if self.cap is None:
            return None
        if frame_index < 0:
            return None
        if self.total_frames > 0 and frame_index >= self.total_frames:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def show_frame(self, frame_index: int) -> None:
        frame = self.read_frame(frame_index)
        if frame is None:
            return
        self.current_frame_index = frame_index

        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        h, w = frame.shape[:2]
        scale = min(canvas_w / w, canvas_h / h)
        target_w = max(1, int(w * scale))
        target_h = max(1, int(h * scale))
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        ok, buf = cv2.imencode(".ppm", rgb)
        if not ok:
            return

        self.preview_image = tk.PhotoImage(data=buf.tobytes())
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.preview_image)

        self.frame_info_label.configure(
            text=f"frame: {self.current_frame_index} / {max(self.total_frames - 1, 0)}"
        )
        self.jump_var.set(str(self.current_frame_index))
        current_label = self.resolve_current_label() or "(none)"
        self.set_status(f"frame={self.current_frame_index} current_label={current_label}")

    def resolve_current_label(self) -> str | None:
        sorted_anchors = sorted(self.anchors, key=lambda a: a.start_frame)
        for idx, anchor in enumerate(sorted_anchors):
            next_start = sorted_anchors[idx + 1].start_frame if idx + 1 < len(sorted_anchors) else None
            if next_start is None and self.current_frame_index >= anchor.start_frame:
                return anchor.label
            if next_start is not None and anchor.start_frame <= self.current_frame_index < next_start:
                return anchor.label
        return None

    def seek_relative(self, delta: int) -> None:
        if self.cap is None:
            return
        target = max(0, min(self.current_frame_index + delta, max(self.total_frames - 1, 0)))
        self.show_frame(target)

    def seek_absolute(self) -> None:
        if self.cap is None:
            return
        try:
            target = int(self.jump_var.get())
        except ValueError:
            messagebox.showerror("Error", "Jump frame must be an integer.")
            return
        target = max(0, min(target, max(self.total_frames - 1, 0)))
        self.show_frame(target)

    def refresh_anchor_list(self) -> None:
        self.anchors.sort(key=lambda a: a.start_frame)
        self.anchor_list.delete(0, tk.END)
        for anchor in self.anchors:
            self.anchor_list.insert(tk.END, f"{anchor.start_frame}: {anchor.label}")

    def add_or_update_anchor(self) -> None:
        if self.cap is None:
            messagebox.showerror("Error", "Load a video first.")
            return
        label = self.label_var.get().strip()
        if not label:
            messagebox.showerror("Error", "Input label.")
            return

        frame = self.current_frame_index
        replaced = False
        new_anchors: list[Anchor] = []
        for anchor in self.anchors:
            if anchor.start_frame == frame:
                new_anchors.append(Anchor(start_frame=frame, label=label))
                replaced = True
            else:
                new_anchors.append(anchor)
        if not replaced:
            new_anchors.append(Anchor(start_frame=frame, label=label))
        self.anchors = new_anchors
        self.refresh_anchor_list()
        self.set_status(
            f"Anchor {'updated' if replaced else 'added'}: frame={frame}, label={label}"
        )

    def remove_selected_anchor(self) -> None:
        sel = self.anchor_list.curselection()
        if not sel:
            return
        idx = sel[0]
        anchor = sorted(self.anchors, key=lambda a: a.start_frame)[idx]
        self.anchors = [a for a in self.anchors if a.start_frame != anchor.start_frame]
        self.refresh_anchor_list()
        self.set_status(f"Anchor removed: frame={anchor.start_frame}")

    def on_anchor_select(self, _event=None) -> None:
        sel = self.anchor_list.curselection()
        if not sel:
            return
        idx = sel[0]
        anchor = sorted(self.anchors, key=lambda a: a.start_frame)[idx]
        self.label_var.set(anchor.label)
        self.show_frame(anchor.start_frame)

    def load_anchor_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select anchors JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            anchors = load_anchors(Path(path))
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load anchors: {exc}")
            return
        self.anchors = anchors
        self.refresh_anchor_list()
        self.set_status(f"Anchors loaded: {path}")

    def save_anchor_file(self) -> None:
        if not self.anchors:
            messagebox.showerror("Error", "No anchors to save.")
            return
        path = filedialog.asksaveasfilename(
            title="Save anchors JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            save_anchors(Path(path), self.anchors)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save anchors: {exc}")
            return
        self.set_status(f"Anchors saved: {path}")

    def select_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_var.set(path)

    def select_label_db(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Select or create label DB",
            defaultextension=".json",
            initialfile=Path(self.label_db_var.get()).name,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.label_db_var.set(path)

    def _label_db_path(self) -> Path:
        raw = self.label_db_var.get().strip() or "label_store.json"
        return Path(raw)

    def _video_key(self) -> str:
        if self.video_path is None:
            raise ValueError("No video loaded.")
        return str(self.video_path.resolve())

    def _load_label_db(self) -> dict:
        path = self._label_db_path()
        if not path.exists():
            return {"videos": {}}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("label DB root must be an object")
        videos = data.get("videos", {})
        if not isinstance(videos, dict):
            raise ValueError("label DB 'videos' must be an object")
        return {"videos": videos}

    def append_current_video_labels(self) -> None:
        if self.video_path is None:
            messagebox.showerror("Error", "Load a video first.")
            return
        if not self.anchors:
            messagebox.showerror("Error", "No anchors to save.")
            return
        try:
            db = self._load_label_db()
            db["videos"][self._video_key()] = [
                {"start_frame": a.start_frame, "label": a.label} for a in sorted(self.anchors, key=lambda x: x.start_frame)
            ]
            db_path = self._label_db_path()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db_path.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to append label DB: {exc}")
            return
        self.set_status(f"Saved to label DB: {self._label_db_path()}")

    def load_current_video_labels(self, silent: bool = False) -> None:
        if self.video_path is None:
            if not silent:
                messagebox.showerror("Error", "Load a video first.")
            return
        try:
            db = self._load_label_db()
            items = db["videos"].get(self._video_key())
            if items is None:
                if not silent:
                    messagebox.showinfo("Info", "No labels for this video in DB.")
                return
            anchors = [Anchor(start_frame=int(it["start_frame"]), label=str(it["label"])) for it in items]
            self.anchors = sorted(anchors, key=lambda a: a.start_frame)
            self.refresh_anchor_list()
            self.set_status(f"Loaded labels from DB for current video: {self._label_db_path()}")
        except Exception as exc:
            if not silent:
                messagebox.showerror("Error", f"Failed to load label DB: {exc}")

    def export_dataset(self) -> None:
        if self.video_path is None:
            messagebox.showerror("Error", "Load a video first.")
            return
        if not self.anchors:
            messagebox.showerror("Error", "Add at least one anchor.")
            return
        try:
            step = int(self.step_var.get())
            quality = int(self.quality_var.get())
            split_ratios = parse_split_ratios(self.split_var.get())
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))
            return
        if step <= 0:
            messagebox.showerror("Error", "step must be >= 1.")
            return
        if not (1 <= quality <= 100):
            messagebox.showerror("Error", "jpeg quality must be in [1, 100].")
            return

        output_dir = Path(self.output_var.get().strip() or "dataset")
        self.set_status("Exporting dataset...")
        self.root.update_idletasks()
        try:
            manifest = build_dataset_from_anchors(
                video_path=self.video_path,
                anchors=self.anchors,
                output_dir=output_dir,
                step=step,
                jpeg_quality=quality,
                output_format=self.format_var.get().strip(),
                split_ratios=split_ratios,
            )
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to export dataset: {exc}")
            self.set_status("Export failed.")
            return

        self.set_status(f"Done: {manifest}")
        messagebox.showinfo("Done", f"Dataset exported.\n{manifest}")


def main() -> int:
    root = tk.Tk()
    app = SwingAnnotatorApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.release_video(), root.destroy()))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
