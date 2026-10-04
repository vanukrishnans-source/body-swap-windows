"""ONNX weights for pose (MoveNet) and person matting (Robust Video Matting).

Not bundled in the app zip. First launch downloads them and checks SHA-256.
MoveNet Lightning: Apache-2.0 (Google), ONNX conversion by Xenova.
RVM MobileNetV3: GPL-3.0 weights / code (Peter Lin). The app source is MIT;
using the RVM weights is under GPL-3.0. Personal / offline use.
"""
from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

@dataclass(frozen=True)
class ModelSpec:
    file: str
    label: str
    bytes: int
    sha256: str
    urls: tuple[str, ...]

MOVENET = ModelSpec(
    "movenet_singlepose_lightning.onnx",
    "Pose (MoveNet Lightning)",
    9_413_268,
    "1ad4f8d6c2f776a9967db3993c9ca740bc350104f9d37c151dc183fc29a464ad",
    (
        "https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/main/onnx/model.onnx",
    ),
)
RVM = ModelSpec(
    "rvm_mobilenetv3_fp32.onnx",
    "Person matte (RVM MobileNetV3)",
    14_975_696,
    "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828",
    (
        "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx",
    ),
)

REQUIRED = [MOVENET, RVM]
REQUIRED_BYTES = sum(s.bytes for s in REQUIRED)


def default_models_dir() -> Path:
    env = os.environ.get("BSWAP_MODELS")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else (Path.home() / ".local" / "share")
    return root / "BodySwap" / "models"


def link_or_copy(src: Path, dest: Path) -> None:
    try:
        if dest.exists():
            dest.unlink()
        os.link(src, dest)
    except OSError:
        dest.write_bytes(src.read_bytes())


class ModelStore:
    def __init__(self, directory: Path | None = None, specs: Iterable[ModelSpec] = REQUIRED):
        self.dir = Path(directory or default_models_dir())
        self.dir.mkdir(parents=True, exist_ok=True)
        self.specs = list(specs)

    def path(self, s: ModelSpec) -> Path:
        return self.dir / s.file

    def _part(self, s: ModelSpec) -> Path:
        return self.dir / (s.file + ".part")

    def _marker(self, s: ModelSpec) -> Path:
        return self.dir / (s.file + ".ok")

    def is_installed(self, s: ModelSpec) -> bool:
        f, m = self.path(s), self._marker(s)
        if not f.is_file() or f.stat().st_size != s.bytes or not m.is_file():
            return False
        return m.read_text(encoding="utf-8").strip() == s.sha256

    def all_installed(self) -> bool:
        return all(self.is_installed(s) for s in self.specs)

    def missing(self) -> list[ModelSpec]:
        return [s for s in self.specs if not self.is_installed(s)]

    def bytes_present(self) -> int:
        n = 0
        for s in self.specs:
            if self.is_installed(s):
                n += s.bytes
            else:
                p = self._part(s)
                if p.is_file():
                    n += min(p.stat().st_size, s.bytes)
        return n

    def import_from(self, folder, specs=None) -> list[str]:
        got = []
        for s in (list(specs) if specs is not None else self.specs):
            src = Path(folder) / s.file
            if self.is_installed(s) or not src.is_file() or src.stat().st_size != s.bytes:
                continue
            digest = hashlib.sha256(src.read_bytes()).hexdigest()
            if digest != s.sha256:
                continue
            link_or_copy(src, self.path(s))
            self._marker(s).write_text(digest + "\n", encoding="utf-8")
            got.append(s.file)
        return got

    def ensure(self, specs=None, progress: Optional[Callable] = None, cancelled: Optional[Callable[[], bool]] = None):
        for s in (list(specs) if specs is not None else self.specs):
            if self.is_installed(s):
                continue
            self._download(s, progress, cancelled)

    def _sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def _download(self, s: ModelSpec, progress, cancelled):
        part = self._part(s)
        last_err = None
        for url in s.urls:
            if cancelled and cancelled():
                raise RuntimeError("cancelled")
            have = part.stat().st_size if part.is_file() else 0
            if have > s.bytes:
                part.unlink(missing_ok=True)
                have = 0
            try:
                self._fetch(url, part, have, s, progress, cancelled)
                digest = self._sha256(part)
                if digest != s.sha256 or part.stat().st_size != s.bytes:
                    part.unlink(missing_ok=True)
                    raise RuntimeError(f"bad checksum for {s.file}")
                dest = self.path(s)
                if dest.exists():
                    dest.unlink()
                part.replace(dest)
                self._marker(s).write_text(s.sha256 + "\n", encoding="utf-8")
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
        raise RuntimeError(f"Could not download {s.file}: {last_err}")

    def _fetch(self, url, part, have, spec: ModelSpec, progress, cancelled):
        headers = {"User-Agent": "BodySwap/1.0"}
        if have > 0:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        t0 = time.time()
        got0 = have
        with urllib.request.urlopen(req, timeout=60) as r:
            code = getattr(r, "status", 200)
            mode = "ab" if code == 206 and have else "wb"
            if code != 206:
                have = 0
                got0 = 0
            with part.open(mode) as f:
                while True:
                    if cancelled and cancelled():
                        raise RuntimeError("cancelled")
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
                    if progress and time.time() - t0 > 0.05:
                        dt = max(1e-3, time.time() - t0)
                        progress(spec.file, have, spec.bytes, (have - got0) / dt, False)
        if progress:
            progress(spec.file, have, spec.bytes, 0, True)
