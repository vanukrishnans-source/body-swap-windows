"""Out-of-process DirectML probe.

A native fault inside onnxruntime-directml on the Radeon 780M kills the process.
Python try/except cannot catch it. The first DML sessions are created in a child.
If the child dies, the UI stays up and the app uses CPU.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("bswap")
STATUS_FILE = "dml_status.json"
PROBE_TIMEOUT_S = 90


def _state_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else (Path.home() / ".local" / "share")
    d = root / "BodySwap"
    d.mkdir(parents=True, exist_ok=True)
    return d


def status_path() -> Path:
    return _state_dir() / STATUS_FILE


def read_status() -> dict:
    p = status_path()
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_status(ok: bool, reason: str = "") -> None:
    data = {"ok": bool(ok), "reason": (reason or "")[:400],
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid()}
    try:
        status_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning("could not write dml status: %s", e)


def force_fail_env() -> bool:
    return os.environ.get("BSWAP_FORCE_DML_FAIL", "").strip() in ("1", "true", "yes")


def clear_status() -> None:
    try:
        status_path().unlink(missing_ok=True)
    except Exception:
        pass


def run_probe_in_this_process(pose_path: str, rvm_path: str = "") -> int:
    if force_fail_env():
        print("DML_PROBE fail: forced", flush=True)
        return 2
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as e:
        print(f"DML_PROBE import fail: {e}", flush=True)
        return 2
    if "DmlExecutionProvider" not in ort.get_available_providers():
        print("DML_PROBE fail: provider missing", flush=True)
        return 2
    if not Path(pose_path).is_file():
        print(f"DML_PROBE fail: missing {pose_path}", flush=True)
        return 2

    def _try(path, feed_fn):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        last = None
        for opts in ({"device_id": 0}, {"device_id": 0, "disable_metacommands": True}):
            try:
                sess = ort.InferenceSession(
                    str(path), so,
                    providers=[("DmlExecutionProvider", opts), "CPUExecutionProvider"],
                )
                if sess.get_providers()[0] != "DmlExecutionProvider":
                    last = f"not DML {sess.get_providers()}"
                    continue
                sess.run(None, feed_fn(sess))
                return None
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
        return last or "no session"

    def pose_feed(sess):
        inp = sess.get_inputs()[0]
        return {inp.name: np.zeros((1, 192, 192, 3), np.int32)}

    err = _try(pose_path, pose_feed)
    if err:
        print(f"DML_PROBE fail: pose {err}", flush=True)
        return 2
    if rvm_path and Path(rvm_path).is_file():
        def rvm_feed(sess):
            z = np.zeros((1, 1, 1, 1), np.float32)
            return {
                "src": np.zeros((1, 3, 64, 64), np.float32),
                "r1i": z, "r2i": z, "r3i": z, "r4i": z,
                "downsample_ratio": np.array([1.0], np.float32),
            }
        err = _try(rvm_path, rvm_feed)
        if err:
            print(f"DML_PROBE fail: matte {err}", flush=True)
            return 2
    print("DML_PROBE ok", flush=True)
    return 0


def _probe_command(pose_path: str, rvm_path: str) -> list[str]:
    if getattr(sys, "frozen", False):
        exe = Path(sys.executable)
        cli = exe.with_name("BodySwap_cli.exe")
        target = str(cli if cli.is_file() else exe)
        return [target, "--dml-probe", pose_path, rvm_path]
    return [sys.executable, "-m", "bswap", "--dml-probe", pose_path, rvm_path]


def probe_directml(pose_path: str, rvm_path: str = "", timeout: float = PROBE_TIMEOUT_S):
    if force_fail_env():
        reason = "BSWAP_FORCE_DML_FAIL=1 (simulated GPU failure)"
        write_status(False, reason)
        return False, reason
    if os.environ.get("BSWAP_FORCE_CPU", "").strip() in ("1", "true", "yes"):
        reason = "BSWAP_FORCE_CPU=1"
        write_status(False, reason)
        return False, reason
    st = read_status()
    reprobe = os.environ.get("BSWAP_DML_REPROBE", "").strip() in ("1", "true", "yes")
    if st.get("ok") is True and not reprobe:
        return True, ""
    if st.get("ok") is False and not reprobe:
        return False, st.get("reason") or "previous DirectML probe failed"
    cmd = _probe_command(pose_path, rvm_path)
    log.info("DirectML probe: %s", " ".join(cmd))
    creation = 0x08000000 if os.name == "nt" else 0
    try:
        cp = subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=creation)
    except subprocess.TimeoutExpired:
        reason = f"DirectML probe timed out after {timeout:.0f}s"
        write_status(False, reason)
        return False, reason
    except Exception as e:
        reason = f"DirectML probe spawn failed: {type(e).__name__}: {e}"
        write_status(False, reason)
        return False, reason
    out = ((cp.stdout or b"") + b"\n" + (cp.stderr or b"")).decode("utf-8", "replace")
    if cp.returncode == 0 and "DML_PROBE ok" in out:
        write_status(True, "")
        log.info("DirectML probe succeeded")
        return True, ""
    reason = f"DirectML probe failed (exit {cp.returncode}): {out.strip()[-300:] or 'no output / native crash'}"
    log.error(reason)
    write_status(False, reason)
    return False, reason
