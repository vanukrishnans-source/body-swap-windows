"""Video decode (OpenCV) and H.264 encode + audio copy (FFmpeg).

Windows zip ships an LGPL FFmpeg (no libx264): AMD AMF on the Ally X, then
Media Foundation / OpenH264. A dev machine with libx264 uses CRF so the file
is not a soft, pixelated encode.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("bswap")
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _bin(name: str) -> str:
    exe = name + (".exe" if os.name == "nt" else "")
    env = os.environ.get("BSWAP_" + name.upper())
    if env and Path(env).is_file():
        return env
    roots = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    me = getattr(sys, "_MEIPASS", None)
    if me:
        roots.append(Path(me))
    roots.append(Path(__file__).resolve().parent.parent)
    for r in roots:
        for c in (r / "ffmpeg" / exe, r / "ffmpeg" / "bin" / exe):
            if c.is_file():
                return str(c)
    found = shutil.which(name) or shutil.which(exe)
    if found:
        return found
    raise FileNotFoundError(f"{exe} not found (expected beside the app or on PATH)")


def ffmpeg():
    return _bin("ffmpeg")


def ffprobe():
    return _bin("ffprobe")


def run(cmd, timeout=None):
    return subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=NO_WINDOW)


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    duration: float
    frames: int
    has_audio: bool
    acodec: str

    def summary(self):
        a = f"sound: {self.acodec}" if self.has_audio else "no sound"
        return f"{self.width}×{self.height} · {self.fps:.3g} fps · {self.duration:.1f}s · {a}"


def probe(path) -> VideoInfo:
    cp = run([ffprobe(), "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], timeout=60)
    if cp.returncode != 0:
        raise ValueError("Couldn't read that video: " + cp.stderr.decode("utf-8", "replace")[-240:])
    meta = json.loads(cp.stdout.decode("utf-8", "replace") or "{}")
    streams = meta.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v is None:
        raise ValueError("That file has no video track.")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("This video codec isn't supported.")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or v.get("width") or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or v.get("height") or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    dur = float((meta.get("format") or {}).get("duration") or 0)
    if dur <= 0 and n and fps:
        dur = n / fps
    return VideoInfo(str(path), w, h, fps, dur, n, a is not None, (a or {}).get("codec_name") or "")


def scaled_size(w, h, max_short: int):
    short = min(w, h)
    if max_short and short > max_short:
        s = max_short / short
        w, h = int(round(w * s)), int(round(h * s))
    w -= w % 2
    h -= h % 2
    return max(2, w), max(2, h)


def frame_times(duration, src_fps, start, length, out_fps):
    start = max(0.0, float(start or 0))
    end = duration if not length else min(duration, start + float(length))
    if end <= start + 1e-3:
        end = min(duration, start + 1.0 / max(src_fps, 1))
    fps = float(out_fps or 0) or min(float(src_fps or 30), 30.0)
    fps = max(1.0, min(fps, 60.0))
    n = max(1, int(round((end - start) * fps)))
    return [start + i / fps for i in range(n) if start + i / fps < end - 1e-4] or [start], fps


def read_at(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
    ok, fr = cap.read()
    return fr if ok else None


ENCODER_ORDER = ["h264_amf", "h264_nvenc", "h264_qsv", "h264_mf", "libopenh264", "libx264"]
_encoder_cache = {}


def encoder_args(name, W, H, fps):
    # ~0.20 bits/pixel/frame keeps 720p/1080p from looking soft. libx264 (dev PCs) uses CRF 16.
    br = int(max(4.0e6, 0.20 * W * H * max(fps, 1)))
    g = str(max(8, int(round(fps * 2))))
    if name == "h264_amf":
        return ["-c:v", name, "-usage", "transcoding", "-quality", "quality", "-rc", "vbr_peak",
                "-b:v", str(br), "-maxrate", str(int(br * 1.5)), "-profile:v", "high", "-g", g]
    if name == "h264_nvenc":
        return ["-c:v", name, "-preset", "p5", "-rc", "vbr", "-b:v", str(br), "-maxrate", str(int(br * 1.5)), "-g", g]
    if name == "h264_qsv":
        return ["-c:v", name, "-preset", "slow", "-b:v", str(br), "-g", g]
    if name == "h264_mf":
        return ["-c:v", name, "-rate_control", "pc_vbr", "-b:v", str(br), "-hw_encoding", "1", "-g", g]
    if name == "libopenh264":
        return ["-c:v", name, "-rc_mode", "bitrate", "-b:v", str(br), "-profile:v", "high", "-g", g]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-g", g]


def pick_encoder(prefer=None):
    key = prefer or "auto"
    if key in _encoder_cache:
        return _encoder_cache[key]
    names = run([ffmpeg(), "-hide_banner", "-encoders"], timeout=30).stdout.decode("utf-8", "replace")
    order = ([prefer] if prefer else []) + [e for e in ENCODER_ORDER if e != prefer]
    tried = []
    for e in order:
        if e not in names:
            continue
        cmd = [ffmpeg(), "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=320x240:r=10", "-frames:v", "3",
               *encoder_args(e, 320, 240, 10), "-f", "null", "-"]
        try:
            ok = run(cmd, timeout=30).returncode == 0
        except Exception:
            ok = False
        tried.append((e, ok))
        if ok:
            _encoder_cache[key] = e
            return e
    raise RuntimeError(f"No working H.264 encoder ({tried})")


class Encoder:
    def __init__(self, out_path, W, H, fps, encoder=None):
        self.encoder = encoder or pick_encoder()
        self.W, self.H, self.fps = int(W), int(H), float(fps)
        rate = Fraction(self.fps).limit_denominator(100000)
        cmd = [ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{self.W}x{self.H}", "-r", f"{rate.numerator}/{rate.denominator}",
               "-i", "-", "-an", *encoder_args(self.encoder, self.W, self.H, self.fps),
               "-movflags", "+faststart", str(out_path)]
        self.err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=self.err, creationflags=NO_WINDOW)
        self.frames = 0

    def write(self, frame):
        arr = np.ascontiguousarray(frame)
        if arr.shape[0] != self.H or arr.shape[1] != self.W:
            arr = cv2.resize(arr, (self.W, self.H), interpolation=cv2.INTER_AREA)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        try:
            self.proc.stdin.write(np.ascontiguousarray(arr).tobytes())
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError("Encoder stopped: " + self._err()) from e
        self.frames += 1

    def _err(self):
        try:
            self.err.seek(0)
            return self.err.read().decode("utf-8", "replace")[-400:]
        except Exception:
            return ""

    def close(self, timeout=180):
        def _close():
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except OSError:
                pass
        th = threading.Thread(target=_close, daemon=True)
        th.start()
        th.join(timeout)
        if th.is_alive():
            self.proc.kill()
            raise RuntimeError("Encoder hung while finishing the MP4")
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            raise RuntimeError("Encoder timed out writing the MP4: " + self._err())
        if self.proc.returncode != 0:
            raise RuntimeError(f"Encoder {self.encoder} failed ({self.proc.returncode}): {self._err()}")

    def kill(self):
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.kill()
        except OSError:
            pass


def mux_audio(video_only, source, start, length, out_path):
    out_path = Path(out_path)
    info = probe(source)
    if not info.has_audio:
        if Path(video_only) != out_path:
            shutil.move(str(video_only), str(out_path))
        return "no audio in the original"
    dur = float(length) if length else max(0.1, info.duration - float(start))
    cmd = [ffmpeg(), "-y", "-v", "error", "-i", str(video_only),
           "-ss", f"{float(start):.3f}", "-t", f"{dur:.3f}", "-i", str(source),
           "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-shortest", "-movflags", "+faststart", str(out_path)]
    cp = run(cmd, timeout=180)
    if cp.returncode != 0 or not out_path.is_file():
        log.warning("audio mux failed: %s", cp.stderr.decode("utf-8", "replace")[-200:])
        shutil.move(str(video_only), str(out_path))
        return "video only (audio copy failed)"
    try:
        Path(video_only).unlink(missing_ok=True)
    except OSError:
        pass
    return "audio copied"


def default_out_dir() -> Path:
    home = Path.home()
    videos = home / "Videos"
    if os.name == "nt":
        videos = Path(os.environ.get("USERPROFILE", str(home))) / "Videos"
    d = videos / "BodySwap"
    d.mkdir(parents=True, exist_ok=True)
    return d
