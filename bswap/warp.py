"""Pose-aligned body transfer (piecewise-affine mesh).

Not a diffusion model. MoveNet joints (plus a few torso anchors) are triangulated
on the target frame. Each triangle copies the matching triangle of the source
person, so the source clothes / skin / hair follow the target pose as one surface
instead of a hard cut-out sticker. Robust Video Matting separates the person from
the background. The original person is inpainted out, then the warped source is
feathered on top with a light brightness match.

Limits: extreme pose or camera-angle differences stretch the texture; missing
joints (out of frame, occluded) leave that limb unreplaced; tiny people go soft.
"""
from __future__ import annotations

import cv2
import numpy as np

from .infer import (
    L_ANK, L_ELB, L_HIP, L_KNEE, L_SH, L_WR, NOSE,
    R_ANK, R_ELB, R_HIP, R_KNEE, R_SH, R_WR,
)

NECK, PELVIS, HEAD = 17, 18, 19
N_AUG = 20

# Bones used only to drop extra surface points (left/right of the limb).
SURFACE = (
    (HEAD, NECK), (NECK, PELVIS),
    (L_SH, R_SH), (L_HIP, R_HIP),
    (L_SH, L_ELB), (L_ELB, L_WR),
    (R_SH, R_ELB), (R_ELB, R_WR),
    (L_HIP, L_KNEE), (L_KNEE, L_ANK),
    (R_HIP, R_KNEE), (R_KNEE, R_ANK),
)
EXTEND = (
    (L_ELB, L_WR, 0.42), (R_ELB, R_WR, 0.42),
    (L_KNEE, L_ANK, 0.38), (R_KNEE, R_ANK, 0.38),
)


def _ok(kps, i, thr=0.22) -> bool:
    return 0 <= i < len(kps) and float(kps[i, 2]) >= thr and np.isfinite(kps[i, 0])


def shoulder_width(kps) -> float:
    if _ok(kps, L_SH, 0.15) and _ok(kps, R_SH, 0.15):
        d = float(np.hypot(kps[L_SH, 0] - kps[R_SH, 0], kps[L_SH, 1] - kps[R_SH, 1]))
        if d > 6:
            return d
    return 0.0


def torso_len(kps) -> float:
    if _ok(kps, NECK, 0.15) and _ok(kps, PELVIS, 0.15):
        return float(np.hypot(kps[NECK, 0] - kps[PELVIS, 0], kps[NECK, 1] - kps[PELVIS, 1]))
    return 0.0


def body_size(kps) -> float:
    """View-stable size. Profile shots collapse shoulder width, so torso length leads."""
    t = torso_len(kps)
    s = shoulder_width(kps)
    if t > 8 and s > 8:
        return max(t, s * 0.85)
    return t or s


def augment(kps: np.ndarray) -> np.ndarray:
    a = np.zeros((N_AUG, 3), np.float32)
    n = min(17, len(kps))
    a[:n] = kps[:n]
    if _ok(kps, L_SH, 0.15) and _ok(kps, R_SH, 0.15):
        a[NECK, :2] = (kps[L_SH, :2] + kps[R_SH, :2]) * 0.5
        a[NECK, 2] = min(float(kps[L_SH, 2]), float(kps[R_SH, 2]))
    if _ok(kps, L_HIP, 0.15) and _ok(kps, R_HIP, 0.15):
        a[PELVIS, :2] = (kps[L_HIP, :2] + kps[R_HIP, :2]) * 0.5
        a[PELVIS, 2] = min(float(kps[L_HIP, 2]), float(kps[R_HIP, 2]))
    size = body_size(a)
    if _ok(a, NECK, 0.15) and size > 8:
        neck = a[NECK, :2]
        if _ok(kps, NOSE, 0.15):
            direction = kps[NOSE, :2] - neck
            nrm = float(np.linalg.norm(direction))
            direction = np.array([0.0, -1.0], np.float32) if nrm < 1 else direction / nrm
            a[HEAD, :2] = kps[NOSE, :2] + direction * (0.42 * size)
            a[HEAD, 2] = min(float(kps[NOSE, 2]), float(a[NECK, 2]))
        else:
            a[HEAD, 0] = neck[0]
            a[HEAD, 1] = neck[1] - 0.55 * size
            a[HEAD, 2] = float(a[NECK, 2]) * 0.5
    return a


def ema_kps(prev, cur, a=0.62):
    if prev is None or prev.shape != cur.shape:
        return cur.copy()
    out = cur.copy()
    for i in range(len(cur)):
        if cur[i, 2] < 0.15:
            if prev[i, 2] > 0.2:
                out[i] = prev[i]
                out[i, 2] *= 0.85
            continue
        if prev[i, 2] < 0.15:
            continue
        out[i, 0] = a * cur[i, 0] + (1 - a) * prev[i, 0]
        out[i, 1] = a * cur[i, 1] + (1 - a) * prev[i, 1]
        out[i, 2] = a * cur[i, 2] + (1 - a) * prev[i, 2]
    return out


def pose_distance(a, b) -> float:
    A, B = augment(a), augment(b)
    sa, sb = body_size(A), body_size(B)
    if sa < 8 or sb < 8:
        return 1e6
    ca = A[NECK, :2] if _ok(A, NECK) else A[NOSE, :2]
    cb = B[NECK, :2] if _ok(B, NECK) else B[NOSE, :2]
    pa = (A[:, :2] - ca) / sa
    pb = (B[:, :2] - cb) / sb
    w = (A[:, 2] > 0.25) & (B[:, 2] > 0.25)
    if int(w.sum()) < 6:
        return 1e6
    return float(np.mean(np.linalg.norm(pa[w] - pb[w], axis=1)))


def _radius(alpha, p0, p1, prior):
    h, w = alpha.shape[:2]
    tang = np.asarray(p1, np.float32) - np.asarray(p0, np.float32)
    L = float(np.hypot(tang[0], tang[1])) + 1e-6
    tang = tang / L
    nrm = np.array([-tang[1], tang[0]], np.float32)
    samples = []
    steps = int(max(6, min(160, prior * 3)))
    for t in (0.35, 0.55, 0.75):
        c = np.asarray(p0, np.float32) + tang * (L * t)
        for sgn in (1.0, -1.0):
            last = 0
            for step in range(1, steps):
                q = c + nrm * (sgn * step)
                xi, yi = int(round(float(q[0]))), int(round(float(q[1])))
                if xi < 0 or yi < 0 or xi >= w or yi >= h or alpha[yi, xi] < 0.42:
                    break
                last = step
            samples.append(last)
    med = float(np.median(samples)) if samples else 0.0
    if med < 2:
        return float(max(3.0, prior * 0.35))
    return float(np.clip(med, 3.0, max(prior * 1.6, prior + 6)))


def _person_component(alpha, cover):
    binm = (alpha > 0.40).astype(np.uint8)
    if int(binm.sum()) < 30 or int((cover > 0).sum()) < 30:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        return cv2.dilate((cover > 0).astype(np.uint8) * 255, k)
    n, labels = cv2.connectedComponents(binm)
    hit = np.unique(labels[(cover > 0) & (labels > 0)])
    remove = np.zeros(alpha.shape, np.uint8)
    for lab in hit:
        if lab == 0:
            continue
        remove[labels == lab] = 255
    if int(remove.sum()) < 30:
        remove = (cover > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.dilate(remove, k)


def _inpaint(frame, mask):
    if int((mask > 0).sum()) < 20:
        return frame
    h, w = frame.shape[:2]
    long = max(h, w)
    if long > 960:
        scale = 960.0 / long
        sw, sh = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        sm = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
        filled = cv2.inpaint(small, sm, 3, cv2.INPAINT_TELEA)
        up = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
        out = frame.copy()
        m = mask > 0
        out[m] = up[m]
        return out
    return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)


def _match_light(src_bgr, ref_bgr, mask, strength):
    if strength <= 0.01:
        return src_bgr
    m = mask > 0.5
    if int(m.sum()) < 40:
        return src_bgr
    s = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c, g in enumerate((0.60 * strength, 0.18 * strength, 0.18 * strength)):
        s[:, :, c] += (float(r[:, :, c][m].mean()) - float(s[:, :, c][m].mean())) * g
    return cv2.cvtColor(np.clip(s, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _push(pairs, sxy, dxy, min_dist, W, H):
    sxy = np.asarray(sxy, np.float32)
    dxy = np.asarray(dxy, np.float32)
    if not np.isfinite(sxy).all() or not np.isfinite(dxy).all():
        return
    x = float(np.clip(dxy[0], 2, W - 3))
    y = float(np.clip(dxy[1], 2, H - 3))
    for _, (px, py) in pairs:
        if (px - x) ** 2 + (py - y) ** 2 < min_dist ** 2:
            return
    pairs.append((sxy.copy(), np.array([x, y], np.float32)))


def _collect(sk, dk, src_a, dst_a, W, H):
    pairs = []
    size_s = max(body_size(sk), 8.0)
    size_d = max(body_size(dk), 8.0)
    min_d = max(4.0, size_d * 0.03)
    for i in range(N_AUG):
        if _ok(sk, i, 0.25) and _ok(dk, i, 0.25):
            _push(pairs, sk[i, :2], dk[i, :2], min_d, W, H)
    for a, b, ext in EXTEND:
        if all(_ok(k, j, 0.28) for k in (sk, dk) for j in (a, b)):
            for kps, bucket in ((sk, "s"), (dk, "d")):
                pass
            ss = sk[b, :2] + (sk[b, :2] - sk[a, :2]) * ext
            dd = dk[b, :2] + (dk[b, :2] - dk[a, :2]) * ext
            _push(pairs, ss, dd, min_d, W, H)
    # Surface "width" points were triangulated across the background and shattered
    # the body. Joints + hand/foot tips stay inside the person and stay continuous.
    return pairs


def _tri_area(p1, p2, p3):
    return abs(float(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))) * 0.5



def transfer(src_bgr, src_alpha, src_kps, dst_bgr, dst_alpha, dst_kps, color_strength=0.45):
    """Soft bone warp over the whole target person. Continuous, no hard triangle seams."""
    sk, dk = augment(src_kps), augment(dst_kps)
    if body_size(sk) < 10 or body_size(dk) < 10 or not (_ok(sk, NECK) and _ok(dk, NECK)):
        return None, None, "no torso (need both shoulders)"
    H, W = dst_bgr.shape[:2]
    sh, sw = src_alpha.shape[:2]
    size_s = max(body_size(sk), 8.0)
    size_d = max(body_size(dk), 8.0)
    # Work on the person bounding box, not the whole frame.
    ys, xs = np.where(dst_alpha > 0.28)
    if len(xs) < 40:
        return None, None, "no person in the target frame"
    pad = int(max(12, size_d * 0.35))
    x0, x1 = max(0, int(xs.min()) - pad), min(W, int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(H, int(ys.max()) + pad + 1)
    roi_x, roi_y = np.meshgrid(
        np.arange(x0, x1, dtype=np.float32),
        np.arange(y0, y1, dtype=np.float32),
    )
    acc_x = np.zeros(roi_x.shape, np.float32)
    acc_y = np.zeros(roi_x.shape, np.float32)
    acc_w = np.zeros(roi_x.shape, np.float32)
    used = 0
    bones = (
        (HEAD, NECK, 1.25), (NECK, PELVIS, 1.15),
        (L_SH, R_SH, 0.55), (L_HIP, R_HIP, 0.70),
        (L_SH, L_ELB, 1.0), (L_ELB, L_WR, 1.05),
        (R_SH, R_ELB, 1.0), (R_ELB, R_WR, 1.05),
        (L_HIP, L_KNEE, 1.05), (L_KNEE, L_ANK, 1.0),
        (R_HIP, R_KNEE, 1.05), (R_KNEE, R_ANK, 1.0),
    )
    for a, b, gain_b in bones:
        if not all(_ok(k, j, 0.25) for k in (sk, dk) for j in (a, b)):
            continue
        s0, s1 = sk[a, :2].astype(np.float32), sk[b, :2].astype(np.float32)
        d0, d1 = dk[a, :2].astype(np.float32), dk[b, :2].astype(np.float32)
        if float(np.hypot(*(s1 - s0))) < 3 or float(np.hypot(*(d1 - d0))) < 3:
            continue
        rs = _radius(src_alpha, s0, s1, 0.20 * size_s) * gain_b
        rd = _radius(dst_alpha, d0, d1, 0.20 * size_d) * gain_b
        rs = max(rs, 4.0)
        rd = max(rd, 4.0)
        sv = s1 - s0
        sL = float(np.hypot(sv[0], sv[1])) + 1e-6
        st, sn = sv / sL, np.array([-sv[1] / sL, sv[0] / sL], np.float32)
        dv = d1 - d0
        vx, vy = float(dv[0]), float(dv[1])
        L2 = vx * vx + vy * vy + 1e-6
        t = ((roi_x - d0[0]) * vx + (roi_y - d0[1]) * vy) / L2
        tclip = np.clip(t, 0.0, 1.0)
        cx = d0[0] + tclip * vx
        cy = d0[1] + tclip * vy
        offx, offy = roi_x - cx, roi_y - cy
        dist = np.hypot(offx, offy)
        # Signed normal offset (dest pixels).
        dn_x, dn_y = -vy / (L2 ** 0.5), vx / (L2 ** 0.5)
        comp_n = offx * dn_x + offy * dn_y
        comp_t = offx * (vx / (L2 ** 0.5)) + offy * (vy / (L2 ** 0.5))
        sigma = rd
        w = np.exp(-0.5 * (dist / sigma) ** 2).astype(np.float32)
        # Fade influence past the bone ends so hands don't pull the whole torso.
        over = np.maximum(np.maximum(-t, t - 1.0), 0)
        w *= np.exp(-0.5 * (over / 0.25) ** 2)
        if float(w.max()) < 1e-4:
            continue
        scale = rs / rd
        sx = (s0[0] + tclip * (s1[0] - s0[0])) + (comp_n * scale) * sn[0] + (comp_t * scale) * st[0]
        sy = (s0[1] + tclip * (s1[1] - s0[1])) + (comp_n * scale) * sn[1] + (comp_t * scale) * st[1]
        acc_x += w * sx
        acc_y += w * sy
        acc_w += w
        used += 1
    if used < 2:
        return None, None, "not enough confident joints"
    good = acc_w > 0.08
    mapx = np.full((H, W), -1, np.float32)
    mapy = np.full((H, W), -1, np.float32)
    subx = np.full(roi_x.shape, -1, np.float32)
    suby = np.full(roi_x.shape, -1, np.float32)
    subx[good] = (acc_x[good] / acc_w[good]).astype(np.float32)
    suby[good] = (acc_y[good] / acc_w[good]).astype(np.float32)
    mapx[y0:y1, x0:x1] = subx
    mapy[y0:y1, x0:x1] = suby
    # Only keep samples that land on the source person.
    warped = cv2.remap(src_bgr, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    src_a = cv2.remap(src_alpha, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # weight confidence: how strongly a bone claimed this pixel
    conf = np.zeros((H, W), np.float32)
    conf[y0:y1, x0:x1] = np.clip(acc_w / (acc_w.max() + 1e-6) * 3.0, 0, 1)
    soft = np.clip(src_a, 0, 1) * conf
    soft = cv2.GaussianBlur(soft, (0, 0), 1.2)
    hard = soft > 0.25
    if int(hard.sum()) < 40:
        return None, None, "warp missed the source person"
    person = _person_component(dst_alpha, (dst_alpha > 0.45).astype(np.uint8) * 255)
    rad = int(max(5, min(41, round(size_d * 0.08))))
    if rad % 2 == 0:
        rad += 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad, rad))
    covered = cv2.dilate((soft > 0.42).astype(np.uint8) * 255, kern)
    # Erase only the target pixels the new body actually covers, so missing
    # limbs (feet out of the source frame) are not replaced by a white hole.
    remove = cv2.bitwise_and(person, covered)
    bg = _inpaint(dst_bgr, remove)
    warped = _match_light(warped, dst_bgr, soft, color_strength)
    a = soft[..., None]
    out = np.clip(warped.astype(np.float32) * a + bg.astype(np.float32) * (1 - a), 0, 255).astype(np.uint8)
    stats = {"bones": used, "src_size": round(size_s, 1), "dst_size": round(size_d, 1),
             "cover_px": int((soft > 0.3).sum()), "remove_px": int((remove > 0).sum())}
    return out, soft, stats
