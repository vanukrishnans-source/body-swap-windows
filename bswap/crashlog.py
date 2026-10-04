"""Crash log for the windowed app. Native DirectML faults never hit Python."""
from __future__ import annotations

import faulthandler
import os
import sys
import threading
import traceback
from pathlib import Path

_fh = None
_installed = False


def log_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else (Path.home() / ".local" / "share")
    return root / "BodySwap"


def crash_path() -> Path:
    return log_dir() / "crash.log"


def install() -> Path:
    global _fh, _installed
    path = crash_path()
    if _installed:
        return path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _fh = open(path, "a", buffering=1, encoding="utf-8")
        _fh.write("\n----- crash log armed -----\n")
        _fh.flush()
        faulthandler.enable(_fh, all_threads=True)
    except Exception:
        return path
    _installed = True

    def _hook(exc_type, exc, tb):
        try:
            _fh.write("unhandled:\n")
            traceback.print_exception(exc_type, exc, tb, file=_fh)
            _fh.flush()
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook

    def _thread(args):
        try:
            _fh.write(f"thread {args.thread.name if args.thread else ''} failed:\n")
            traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=_fh)
            _fh.flush()
        except Exception:
            pass

    threading.excepthook = _thread
    return path


def record_current(where=""):
    try:
        install()
        _fh.write(f"record {where}\n")
        traceback.print_exc(file=_fh)
        _fh.flush()
    except Exception:
        pass
