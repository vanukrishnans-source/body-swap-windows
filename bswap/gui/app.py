"""Touch-first UI for the ROG Ally X (7\" 1080p, ~150% scaling)."""
from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider, QStackedWidget,
    QVBoxLayout, QWidget,
)

from .. import __version__
from ..engine import Cancelled, Settings, open_models, swap_video
from ..media import default_out_dir, probe
from ..models import MOVENET, REQUIRED_BYTES, RVM, ModelStore
from .theme import DARK_QSS

log = logging.getLogger("bswap")
PAGE_SETUP, PAGE_MAIN, PAGE_OPTIONS, PAGE_PROGRESS, PAGE_DONE = range(5)
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def pix(bgr, w, h):
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return QPixmap()
    dpr = QApplication.instance().devicePixelRatio() if QApplication.instance() else 1.0
    W, H = max(1, int(w * dpr)), max(1, int(h * dpr))
    ih, iw = bgr.shape[:2]
    s = min(W / iw, H / ih)
    img = cv2.resize(bgr, (max(1, int(iw * s)), max(1, int(ih * s))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
    pm = QPixmap.fromImage(q)
    pm.setDevicePixelRatio(dpr)
    return pm


def button(text, kind=None):
    b = QPushButton(text)
    if kind:
        b.setObjectName(kind)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setMinimumHeight(56)
    return b


class Worker(QThread):
    progressed = Signal(object)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.cancelled = False

    def run(self):
        try:
            self.done.emit(self.fn(lambda: self.cancelled, self.progressed.emit))
        except Cancelled:
            self.failed.emit("cancelled")
        except Exception as e:
            log.error("worker\n%s", traceback.format_exc())
            self.failed.emit(str(e) or type(e).__name__)


class MainWindow(QMainWindow):
    def __init__(self, args=None):
        super().__init__()
        self.setWindowTitle(f"Body Swap {__version__}")
        self.resize(1100, 700)
        self.store = ModelStore(Path(args.models) if args and args.models else None)
        self.target = ""
        self.source = ""
        self.out_file = ""
        self.device = "auto"
        self.device_label = "…"
        self.note = ""
        self.max_short = 1080
        self.fps = 0
        self.color = 45
        self.length = 20
        self.worker = None
        self._build()
        self.stack.setCurrentIndex(PAGE_SETUP if not self.store.all_installed() else PAGE_MAIN)
        if args and getattr(args, "video", None):
            self._set_target(args.video)
        if args and getattr(args, "source", None):
            self._set_source(args.source)
        QTimer.singleShot(200, self._refresh_device)

    def _build(self):
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.stack.addWidget(self._setup_page())
        self.stack.addWidget(self._main_page())
        self.stack.addWidget(self._options_page())
        self.stack.addWidget(self._progress_page())
        self.stack.addWidget(self._done_page())

    def _card(self):
        f = QFrame()
        f.setObjectName("card")
        return f

    def _setup_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(28, 28, 28, 28)
        lay.addWidget(QLabel("Body Swap"))
        title = lay.itemAt(0).widget()
        title.setObjectName("title")
        sub = QLabel(
            f"First launch downloads about {REQUIRED_BYTES/1e6:.0f} MB of models "
            "(pose + person matte). After that the app works offline.\n"
            "Models go in %LOCALAPPDATA%\\BodySwap\\models."
        )
        sub.setObjectName("subtitle")
        sub.setWordWrap(True)
        lay.addWidget(sub)
        self.dl_bar = QProgressBar()
        self.dl_label = QLabel("Ready when you are.")
        self.dl_label.setWordWrap(True)
        lay.addWidget(self.dl_label)
        lay.addWidget(self.dl_bar)
        self.dl_btn = button("Download models", "primary")
        self.dl_btn.clicked.connect(self._download)
        lay.addWidget(self.dl_btn)
        lay.addStretch(1)
        return w

    def _download(self):
        self.dl_btn.setEnabled(False)
        missing = self.store.missing()

        def work(cancel, progress):
            def prog(name, done, total, bps, verifying):
                progress({"name": name, "done": done, "total": total, "verifying": verifying})
            self.store.ensure(missing, progress=prog, cancelled=cancel)
            return True

        self.worker = Worker(work)
        self.worker.progressed.connect(self._on_dl)
        self.worker.done.connect(lambda _: self.stack.setCurrentIndex(PAGE_MAIN))
        self.worker.failed.connect(self._fail)
        self.worker.start()

    def _on_dl(self, d):
        total = max(1, d["total"])
        self.dl_bar.setMaximum(total)
        self.dl_bar.setValue(min(d["done"], total))
        self.dl_label.setText(("Checking " if d["verifying"] else "Downloading ") + d["name"])

    def _main_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(12)
        head = QHBoxLayout()
        t = QLabel("Body Swap")
        t.setObjectName("title")
        self.dev_chip = QLabel("device …")
        self.dev_chip.setObjectName("hint")
        head.addWidget(t)
        head.addStretch(1)
        head.addWidget(self.dev_chip)
        lay.addLayout(head)
        hint = QLabel("Replaces the whole visible person in a video with another person’s body. Not a face-only swap, and not cloud quality.")
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        row = QHBoxLayout()
        self.target_view = QLabel("1  ·  Target video\nPerson to replace")
        self.target_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.target_view.setMinimumHeight(220)
        self.source_view = QLabel("2  ·  Source body\nPhoto or short clip")
        self.source_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_view.setMinimumHeight(220)
        for lab, slot in ((self.target_view, self._pick_target), (self.source_view, self._pick_source)):
            card = self._card()
            c = QVBoxLayout(card)
            lab.setWordWrap(True)
            c.addWidget(lab)
            b = button("Choose", "ghost")
            b.clicked.connect(slot)
            c.addWidget(b)
            row.addWidget(card)
        lay.addLayout(row)

        self.summary = QLabel("Pick a target video and a source body.")
        self.summary.setObjectName("hint")
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        actions = QHBoxLayout()
        opt = button("Options")
        opt.clicked.connect(lambda: self.stack.setCurrentIndex(PAGE_OPTIONS))
        self.go = button("Swap body", "primary")
        self.go.setEnabled(False)
        self.go.clicked.connect(self._start)
        actions.addWidget(opt)
        actions.addWidget(self.go, 1)
        lay.addLayout(actions)
        return w

    def _options_page(self):
        w = QWidget()
        outer = QVBoxLayout(w)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(22, 18, 22, 18)
        title = QLabel("Options")
        title.setObjectName("title")
        lay.addWidget(title)
        lay.addWidget(QLabel("Longest clip this run will process"))
        self.len_slider = QSlider(Qt.Orientation.Horizontal)
        self.len_slider.setRange(2, 90)
        self.len_slider.setValue(self.length)
        self.len_label = QLabel()
        self.len_slider.valueChanged.connect(lambda v: self.len_label.setText(f"{v} seconds (cap 90s)"))
        self.len_slider.valueChanged.emit(self.length)
        lay.addWidget(self.len_label)
        lay.addWidget(self.len_slider)
        lay.addWidget(QLabel("Resolution cap (short side). 1080 keeps the original sharp. Lower is faster."))
        self.res_slider = QSlider(Qt.Orientation.Horizontal)
        self.res_slider.setRange(480, 1080)
        self.res_slider.setSingleStep(120)
        self.res_slider.setValue(1080)
        self.res_label = QLabel()
        self.res_slider.valueChanged.connect(lambda v: self.res_label.setText(f"Up to {v}p — never upscales"))
        self.res_slider.valueChanged.emit(1080)
        lay.addWidget(self.res_label)
        lay.addWidget(self.res_slider)
        self.cpu_box = QCheckBox("Force CPU (skip the GPU)")
        self.cpu_box.setMinimumHeight(48)
        lay.addWidget(self.cpu_box)
        note = QLabel(
            "What this actually does: cuts the person out, lines their joints up with the target pose, "
            "and repaints their clothes and skin onto that pose. It is not a neural body generator. "
            "Wild pose changes, crossed limbs, and feet missing from the source photo will look wrong. "
            "One main person per clip."
        )
        note.setWordWrap(True)
        note.setObjectName("hint")
        lay.addWidget(note)
        retry = button("Retry GPU probe")
        retry.clicked.connect(self._retry_gpu)
        lay.addWidget(retry)
        back = button("Done", "primary")
        back.clicked.connect(self._apply_options)
        lay.addWidget(back)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        return w

    def _apply_options(self):
        self.length = self.len_slider.value()
        self.max_short = self.res_slider.value()
        self.device = "cpu" if self.cpu_box.isChecked() else "auto"
        self.stack.setCurrentIndex(PAGE_MAIN)
        self._refresh_device()

    def _retry_gpu(self):
        from .. import dml_probe
        dml_probe.clear_status()
        os.environ["BSWAP_DML_REPROBE"] = "1"
        self.cpu_box.setChecked(False)
        self.device = "auto"
        self._refresh_device()

    def _refresh_device(self):
        if not self.store.all_installed():
            self.dev_chip.setText("models needed")
            return
        try:
            models, label, note = open_models(str(self.store.path(MOVENET)), str(self.store.path(RVM)), self.device)
            self.device_label = label
            self.note = note or ""
            extra = " · GPU failed, using CPU" if label == "CPU" and self.device != "cpu" else ""
            self.dev_chip.setText(label + extra)
            del models
        except Exception as e:
            self.dev_chip.setText("CPU")
            self.note = str(e)

    def _pick_target(self):
        path, _ = QFileDialog.getOpenFileName(self, "Target video", "", "Video (*.mp4 *.mov *.mkv *.webm *.avi)")
        if path:
            self._set_target(path)

    def _pick_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Source body", "", "Image or video (*.jpg *.jpeg *.png *.webp *.mp4 *.mov *.mkv *.webm)")
        if path:
            self._set_source(path)

    def _thumb(self, path):
        ext = Path(path).suffix.lower()
        if ext in IMAGE_EXT:
            return cv2.imread(path)
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, 400)
        ok, fr = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, fr = cap.read()
        cap.release()
        return fr if ok else None

    def _set_target(self, path):
        self.target = path
        fr = self._thumb(path)
        if fr is not None:
            self.target_view.setPixmap(pix(fr, 460, 240))
        try:
            info = probe(path)
            self._target_summary = info.summary()
        except Exception as e:
            self._target_summary = str(e)
        self._sync()

    def _set_source(self, path):
        self.source = path
        fr = self._thumb(path)
        if fr is not None:
            self.source_view.setPixmap(pix(fr, 460, 240))
        self._sync()

    def _sync(self):
        self.go.setEnabled(bool(self.target and self.source and self.store.all_installed()))
        bits = []
        if self.target:
            bits.append("Target: " + getattr(self, "_target_summary", Path(self.target).name))
        if self.source:
            bits.append("Source: " + Path(self.source).name)
        bits.append(f"Up to {self.length}s · {self.max_short}p")
        self.summary.setText("\n".join(bits))

    def _progress_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(28, 28, 28, 28)
        self.prog_title = QLabel("Swapping…")
        self.prog_title.setObjectName("title")
        self.prog_label = QLabel("")
        self.prog_label.setWordWrap(True)
        self.prog_bar = QProgressBar()
        self.cancel_btn = button("Cancel", "danger")
        self.cancel_btn.clicked.connect(self._cancel)
        lay.addWidget(self.prog_title)
        lay.addWidget(self.prog_label)
        lay.addWidget(self.prog_bar)
        lay.addWidget(self.cancel_btn)
        lay.addStretch(1)
        return w

    def _done_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(28, 28, 28, 28)
        self.done_title = QLabel("Done")
        self.done_title.setObjectName("title")
        self.done_label = QLabel("")
        self.done_label.setWordWrap(True)
        self.done_view = QLabel()
        self.done_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.done_view.setMinimumHeight(240)
        again = button("Swap another", "primary")
        again.clicked.connect(lambda: self.stack.setCurrentIndex(PAGE_MAIN))
        lay.addWidget(self.done_title)
        lay.addWidget(self.done_view)
        lay.addWidget(self.done_label)
        lay.addWidget(again)
        return w

    def _start(self):
        if not self.store.all_installed():
            self.stack.setCurrentIndex(PAGE_SETUP)
            return
        self.stack.setCurrentIndex(PAGE_PROGRESS)
        self.prog_bar.setRange(0, 100)
        out = default_out_dir() / (time_name())
        self.out_file = str(out)
        settings = Settings(
            max_short=self.max_short, fps=0, start=0, length=float(self.length),
            max_seconds=90, color=self.color / 100.0, device=self.device,
        )

        def work(cancel, progress):
            models, label, note = open_models(
                str(self.store.path(MOVENET)), str(self.store.path(RVM)), self.device)
            self.device_label = label
            self.note = note or ""
            return swap_video(models, self.target, self.source, self.out_file, settings,
                              progress=progress, cancel=cancel, device_label=label)

        self.worker = Worker(work)
        self.worker.progressed.connect(self._on_prog)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._fail)
        self.worker.start()

    def _on_prog(self, d):
        total = max(1, int(d.get("total") or 1))
        done = int(d.get("done") or 0)
        self.prog_bar.setMaximum(total)
        self.prog_bar.setValue(min(done, total))
        self.prog_title.setText({"source": "Reading source", "swap": "Swapping body",
                                 "encode": "Encoding", "done": "Finishing"}.get(d.get("stage"), "Working"))
        self.prog_label.setText(d.get("detail") or "")

    def _on_done(self, rep):
        self.dev_chip.setText(rep.get("device") or self.device_label)
        cap = cv2.VideoCapture(rep["out"])
        mid = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1) // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, fr = cap.read()
        cap.release()
        if ok:
            self.done_view.setPixmap(pix(fr, 900, 360))
        warn = ""
        if rep.get("missed"):
            warn = f"\n{rep['missed']} frame(s) had no usable pose and were left unchanged."
        gpu = ("\n" + self.note) if self.note else ""
        self.done_label.setText(
            f"Saved {rep['out']}\n{rep['width']}×{rep['height']} · {rep['fps']:.2f} fps · "
            f"{rep['encoder']} · {rep['audio']} · {rep['device']} · {rep['seconds']}s"
            f"{warn}{gpu}\n\nLimits: geometric pose transfer, one person, no invented limbs."
        )
        self.stack.setCurrentIndex(PAGE_DONE)

    def _cancel(self):
        if self.worker:
            self.worker.cancelled = True
            self.prog_label.setText("Cancelling…")

    def _fail(self, msg):
        if msg == "cancelled":
            self.stack.setCurrentIndex(PAGE_MAIN)
            return
        QMessageBox.warning(self, "Body Swap", msg[:800])
        self.stack.setCurrentIndex(PAGE_MAIN)


def time_name():
    import time
    return time.strftime("BodySwap_%Y%m%d_%H%M%S.mp4")


def launch(args=None):
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(DARK_QSS)
    win = MainWindow(args)
    win.show()
    return app.exec()


def save_screenshots(folder, video=None, source=None):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(DARK_QSS)
    class A:
        models = None
        video = video
        source = source
    win = MainWindow(A())
    win.resize(1280, 800)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(("setup", "main", "options")):
        win.stack.setCurrentIndex(min(i, PAGE_OPTIONS) if not win.store.all_installed() and i == 0 else (PAGE_MAIN if i == 1 else PAGE_OPTIONS))
        if i == 0 and win.store.all_installed():
            win.stack.setCurrentIndex(PAGE_MAIN)
        app.processEvents()
        pixm = win.grab()
        pixm.save(str(folder / f"{name}.png"))
    return 0
