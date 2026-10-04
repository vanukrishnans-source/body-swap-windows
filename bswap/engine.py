"""Whole-person video swap: matte + pose mesh warp + high-bitrate H.264."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import dml_probe, media
from .infer import Models
from .warp import ema_kps, pose_distance, transfer

log = logging.getLogger("bswap")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class Cancelled(RuntimeError):
    pass


@dataclass
class Settings:
    max_short: int = 1080
    fps: float = 0          # 0 = min(source, 30)
    start: float = 0
    length: float = 0       # seconds, 0 = until the cap
    max_seconds: float = 90
    color: float = 0.45
    device: str = "auto"    # auto | cpu | directml


def _fit(bgr, max_short):
    h, w = bgr.shape[:2]
    W, H = media.scaled_size(w, h, max_short)
    if (W, H) != (w, h):
        bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
    return bgr


def open_models(pose_path: str, rvm_path: str, device: str = "auto"):
    """Returns (Models, device_label, note). DirectML is probed in a child first."""
    import onnxruntime as ort
    want_dml = device in ("auto", "directml", "dml", "gpu")
    have = "DmlExecutionProvider" in ort.get_available_providers()
    if device == "cpu" or not want_dml or not have:
        note = "" if device == "cpu" or not want_dml else "DirectML not in this build — using CPU"
        return Models(pose_path, rvm_path, ["CPUExecutionProvider"]), "CPU", note
    ok, reason = dml_probe.probe_directml(pose_path, rvm_path)
    if not ok:
        return Models(pose_path, rvm_path, ["CPUExecutionProvider"]), "CPU", "GPU failed, using CPU. " + reason
    try:
        m = Models(pose_path, rvm_path, [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
        used = m.pose_sess.get_providers()
        if used and used[0] == "DmlExecutionProvider":
            return m, "DirectML", ""
    except Exception as e:
        log.warning("in-process DML failed: %s", e)
        reason = str(e)
    return Models(pose_path, rvm_path, ["CPUExecutionProvider"]), "CPU", "GPU failed, using CPU. " + (reason or "")


def _analyze(models: Models, bgr, reset_matte: bool):
    alpha = models.matte(bgr, reset=reset_matte)
    kps = models.pose_refined(bgr, alpha)
    return alpha, kps


def load_source(models: Models, path: str, max_short: int, limit_frames: int = 24):
    """One still, or up to `limit_frames` spread across a short clip."""
    ext = Path(path).suffix.lower()
    bank = []
    if ext in IMAGE_EXT or ext not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Couldn't read that source image.")
        bgr = _fit(bgr, max_short)
        models.reset_matting()
        alpha, kps = _analyze(models, bgr, True)
        if float(kps[:, 2].mean()) < 0.15:
            raise ValueError("No person pose in the source. Use a clear, full-body photo.")
        bank.append({"bgr": bgr, "alpha": alpha, "kps": kps, "t": 0.0})
        return bank
    info = media.probe(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Couldn't open the source clip.")
    span = min(info.duration or 1, 8.0)
    n = max(1, min(limit_frames, int(span * min(info.fps or 8, 8))))
    models.reset_matting()
    try:
        for i in range(n):
            t = (span * i) / max(1, n)
            fr = media.read_at(cap, t)
            if fr is None:
                continue
            fr = _fit(fr, max_short)
            alpha, kps = _analyze(models, fr, reset_matte=(i == 0))
            if float(np.mean(kps[:, 2])) < 0.12:
                continue
            bank.append({"bgr": fr, "alpha": alpha, "kps": kps, "t": t})
    finally:
        cap.release()
    if not bank:
        raise ValueError("No usable person in that source clip.")
    return bank


def _pick(bank, kps):
    best, score = bank[0], 1e9
    if len(bank) == 1:
        return best, 0.0
    for item in bank:
        d = pose_distance(item["kps"], kps)
        if d < score:
            best, score = item, d
    return best, score


def swap_video(models: Models, target: str, source: str, out_path: str, settings: Settings,
               progress=None, cancel=None, device_label="CPU"):
    def tick(stage, done, total, detail=""):
        if cancel and cancel():
            raise Cancelled()
        if progress:
            progress({"stage": stage, "done": done, "total": total, "detail": detail, "device": device_label})

    info = media.probe(target)
    length = settings.length
    cap_len = min(settings.max_seconds, info.duration - settings.start)
    if length <= 0 or length > cap_len:
        length = max(0.2, cap_len)
    W, H = media.scaled_size(info.width, info.height, settings.max_short)
    times, fps = media.frame_times(info.duration, info.fps, settings.start, length, settings.fps)
    if not times:
        raise ValueError("Nothing to export — check the start time and length.")
    tick("source", 0, 1, "Reading the source body…")
    bank = load_source(models, source, settings.max_short)
    tick("source", 1, 1, f"{len(bank)} source frame{'s' if len(bank) != 1 else ''}")

    cap = cv2.VideoCapture(str(target))
    if not cap.isOpened():
        raise ValueError("Couldn't open the target video.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".video.mp4")
    enc = media.Encoder(tmp, W, H, fps)
    models.reset_matting()
    ema = None
    swapped = 0
    missed = 0
    t0 = time.time()
    try:
        for i, t in enumerate(times):
            tick("swap", i, len(times), f"Frame {i + 1} / {len(times)}")
            fr = media.read_at(cap, t)
            if fr is None:
                missed += 1
                continue
            if fr.shape[1] != W or fr.shape[0] != H:
                fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
            alpha, kps = _analyze(models, fr, reset_matte=(i == 0))
            kps = ema_kps(ema, kps, 0.62)
            ema = kps
            src, _dist = _pick(bank, kps)
            out, soft, stats = transfer(
                src["bgr"], src["alpha"], src["kps"], fr, alpha, kps, settings.color,
            )
            if out is None:
                missed += 1
                enc.write(fr)
            else:
                swapped += 1
                enc.write(out)
        tick("encode", len(times), len(times), f"Finishing {enc.encoder}…")
        enc.close()
    except Exception:
        enc.kill()
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        cap.release()
    note = media.mux_audio(tmp, target, settings.start, length, out_path)
    tick("done", 1, 1, note)
    return {
        "out": str(out_path),
        "width": W,
        "height": H,
        "fps": fps,
        "frames": enc.frames,
        "swapped": swapped,
        "missed": missed,
        "seconds": round(time.time() - t0, 2),
        "encoder": enc.encoder,
        "audio": note,
        "device": device_label,
        "source_frames": len(bank),
    }
