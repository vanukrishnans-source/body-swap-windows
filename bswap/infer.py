"""MoveNet pose + Robust Video Matting on ONNX Runtime (DirectML or CPU)."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("bswap")

# COCO-17, same order MoveNet uses.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_ELB, R_ELB, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16
N_KP = 17


def _session(path: str, providers: list):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=providers)


class Models:
    def __init__(self, pose_path: str, rvm_path: str, providers: list[str] | None = None):
        import onnxruntime as ort
        prov = providers or ["CPUExecutionProvider"]
        self.providers = prov
        self.pose_sess = _session(pose_path, prov)
        self.rvm_sess = _session(rvm_path, prov)
        self._rec = None
        self._rec_hw = None
        log.info("sessions ready providers=%s", self.pose_sess.get_providers())

    def reset_matting(self):
        self._rec = None
        self._rec_hw = None

    def pose(self, bgr: np.ndarray) -> np.ndarray:
        """Return (17, 3) float32 keypoints: x, y, confidence in full-frame pixels."""
        h, w = bgr.shape[:2]
        side = 192
        scale = side / max(h, w)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        canvas = np.zeros((side, side, 3), np.int32)
        y0 = (side - nh) // 2
        x0 = (side - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = rgb.astype(np.int32)
        out = self.pose_sess.run(None, {"input": canvas[None, ...]})[0]
        kps = np.zeros((N_KP, 3), np.float32)
        pts = out[0, 0]
        for i in range(N_KP):
            y, x, c = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
            kps[i, 0] = (x * side - x0) / scale
            kps[i, 1] = (y * side - y0) / scale
            kps[i, 2] = c
        return kps

    def pose_refined(self, bgr: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
        """MoveNet on the whole frame, and again on the matte crop. Keep the clearer one."""
        full = self.pose(bgr)
        if alpha is None:
            return full
        ys, xs = np.where(alpha > 0.45)
        if len(xs) < 80:
            return full
        h, w = bgr.shape[:2]
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        if bw > 0.92 * w and bh > 0.92 * h:
            return full
        pad = int(0.18 * max(bw, bh))
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w - 1, x1 + pad), min(h - 1, y1 + pad)
        if x1 - x0 < 24 or y1 - y0 < 24:
            return full
        crop = bgr[y0:y1 + 1, x0:x1 + 1]
        local = self.pose(crop)
        local[:, 0] += x0
        local[:, 1] += y0
        if float(local[:, 2].mean()) + 0.02 >= float(full[:, 2].mean()):
            return local
        return full

    def matte(self, bgr: np.ndarray, reset: bool = False) -> np.ndarray:
        """Foreground alpha float32 HxW in 0..1. Recurrent across calls until reset/size change."""
        h, w = bgr.shape[:2]
        long = max(h, w)
        if long > 720:
            s = 720.0 / long
            nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
            small = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            small = bgr
            nh, nw = h, w
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        src = np.transpose(rgb, (2, 0, 1))[None]
        hw = (nh, nw)
        if reset or self._rec is None or self._rec_hw != hw:
            self._rec = [np.zeros((1, 1, 1, 1), np.float32) for _ in range(4)]
            self._rec_hw = hw
        ratio = np.array([min(512.0 / max(nh, nw), 1.0)], np.float32)
        fgr, pha, *rec = self.rvm_sess.run(None, {
            "src": src,
            "r1i": self._rec[0], "r2i": self._rec[1], "r3i": self._rec[2], "r4i": self._rec[3],
            "downsample_ratio": ratio,
        })
        self._rec = rec
        alpha = pha[0, 0]
        if alpha.shape[0] != h or alpha.shape[1] != w:
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(alpha, 0, 1).astype(np.float32)


def provider_list(device: str) -> list:
    """device: auto | cpu | directml. auto prefers DML only if the caller already probed it."""
    import onnxruntime as ort
    avail = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device in ("directml", "dml", "gpu"):
        if "DmlExecutionProvider" not in avail:
            raise RuntimeError("DirectML is not available in this ONNX Runtime build.")
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    if "DmlExecutionProvider" in avail:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
