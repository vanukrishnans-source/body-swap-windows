"""Body Swap CLI: GUI by default, plus --run / --selftest / --dml-probe."""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path

from . import __version__

log = logging.getLogger("bswap")


def _setup_logging():
    from . import crashlog
    crashlog.install()
    crashlog.log_dir().mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(crashlog.log_dir() / "bodyswap.log", encoding="utf-8")]
    if sys.stdout is not None:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers, force=True)


def versions():
    import cv2
    import numpy as np
    import onnxruntime as ort
    from . import media
    try:
        ff = media.run([media.ffmpeg(), "-hide_banner", "-version"], timeout=20).stdout.decode().splitlines()[0]
    except Exception as e:
        ff = f"error: {e}"
    return {
        "app": __version__, "python": sys.version.split()[0], "platform": platform.platform(),
        "onnxruntime": ort.__version__, "providers": ort.get_available_providers(),
        "opencv": cv2.__version__, "numpy": np.__version__, "ffmpeg": ff,
        "frozen": bool(getattr(sys, "frozen", False)),
    }


def _store(path):
    from .models import ModelStore
    store = ModelStore(Path(path) if path else None)
    missing = store.missing()
    if missing:
        log.info("downloading %s", [s.file for s in missing])

        def prog(name, done, total, bps, verifying):
            log.info("%s %s %.1f/%.1f MB", "verify" if verifying else "get", name, done / 1e6, total / 1e6)

        store.ensure(missing, progress=prog)
    return store


def _run(args):
    from .engine import Settings, open_models, swap_video
    from .models import MOVENET, RVM
    store = _store(args.models)
    models, label, note = open_models(str(store.path(MOVENET)), str(store.path(RVM)), args.device)
    if note:
        log.info(note)
    settings = Settings(
        max_short=args.max_short, fps=args.fps, start=args.start, length=args.length,
        max_seconds=args.max_seconds, color=args.color, device=args.device,
    )
    out = args.out or str(Path(args.video).with_name("bodyswap_out.mp4"))
    last = [0]

    def progress(d):
        now = time.time()
        if d.get("stage") in ("done", "encode") or now - last[0] > 1.5:
            last[0] = now
            log.info("%s %s/%s %s", d.get("stage"), d.get("done"), d.get("total"), d.get("detail") or "")

    rep = swap_video(models, args.video, args.source, out, settings, progress=progress, device_label=label)
    rep["note"] = note
    log.info("wrote %s", rep["out"])
    print(json.dumps(rep, indent=2))
    return 0


def _selftest(args):
    from .engine import Settings, open_models, swap_video
    from .models import MOVENET, RVM
    import cv2
    report = {"ok": False, "versions": versions(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_dir = Path(args.out or tempfile.mkdtemp(prefix="bswap_selftest_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        store = _store(args.models)
        models, label, note = open_models(str(store.path(MOVENET)), str(store.path(RVM)), args.device or "cpu")
        settings = Settings(max_short=args.max_short or 480, fps=args.fps or 8, start=args.start or 4,
                            length=args.length or 1.2, max_seconds=30, color=0.4, device=args.device or "cpu")
        out = str(out_dir / "selftest.mp4")
        rep = swap_video(models, args.video, args.source, out, settings, device_label=label)
        cap = cv2.VideoCapture(out)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, fr = cap.read()
        cap.release()
        size = Path(out).stat().st_size if Path(out).is_file() else 0
        checks = {
            "file": size > 20_000,
            "frames": n >= 4 and rep["swapped"] >= 3,
            "resolution": w >= 2 and h >= 2 and min(w, h) <= (args.max_short or 480) + 2,
            "decoded": bool(ok),
        }
        report.update(ok=all(checks.values()), checks=checks, result=rep, note=note,
                      width=w, height=h, bytes=size)
        if args.dml_smoke:
            from . import dml_probe
            dml_ok, dml_reason = dml_probe.probe_directml(str(store.path(MOVENET)), str(store.path(RVM)))
            report["directml"] = {"ok": dml_ok, "reason": dml_reason}
    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}"
        report["trace"] = traceback.format_exc()[-2000:]
    (out_dir / "selftest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 1


def _dml_fallback(args):
    """Process must stay alive when DirectML is forced to fail, and report CPU."""
    os.environ["BSWAP_FORCE_DML_FAIL"] = "1"
    from . import dml_probe
    from .engine import open_models
    from .models import MOVENET, RVM, ModelStore
    dml_probe.clear_status()
    store = ModelStore(Path(args.models) if args.models else None)
    if store.missing():
        store.ensure(store.missing())
    models, label, note = open_models(str(store.path(MOVENET)), str(store.path(RVM)), "auto")
    out_dir = Path(args.out or tempfile.mkdtemp(prefix="bswap_dml_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"ok": label == "CPU" and "GPU failed" in (note or ""), "device": label, "note": note}
    (out_dir / "dml_fallback_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog="BodySwap", description="Whole-person video swap for the ROG Ally X")
    p.add_argument("--video", help="Target video (person to replace)")
    p.add_argument("--source", help="Source body image or short clip")
    p.add_argument("--out", help="Output mp4 or folder")
    p.add_argument("--models", help="Model directory")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "directml"])
    p.add_argument("--max-short", type=int, default=1080)
    p.add_argument("--fps", type=float, default=0)
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--length", type=float, default=0)
    p.add_argument("--max-seconds", type=float, default=90)
    p.add_argument("--color", type=float, default=0.45)
    p.add_argument("--run", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--dml-smoke", action="store_true")
    p.add_argument("--test-dml-fallback", action="store_true")
    p.add_argument("--dml-probe", nargs="*", help=argparse.SUPPRESS)
    p.add_argument("--screenshots", help="Write UI screenshots into this folder and exit")
    args = p.parse_args(argv)
    if args.dml_probe is not None:
        from .dml_probe import run_probe_in_this_process
        pose = args.dml_probe[0] if args.dml_probe else ""
        rvm = args.dml_probe[1] if len(args.dml_probe) > 1 else ""
        return run_probe_in_this_process(pose, rvm)
    _setup_logging()
    log.info("Body Swap %s %s", __version__, versions())
    if args.test_dml_fallback:
        return _dml_fallback(args)
    if args.selftest:
        if not args.video or not args.source:
            print("selftest needs --video and --source", file=sys.stderr)
            return 2
        return _selftest(args)
    if args.run:
        if not args.video or not args.source:
            print("--run needs --video and --source", file=sys.stderr)
            return 2
        return _run(args)
    if args.screenshots:
        from .gui.app import save_screenshots
        save_screenshots(args.screenshots, args.video, args.source)
        return 0
    from .gui.app import launch
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
