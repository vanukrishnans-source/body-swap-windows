"""Mixkit stock clips used by CI (people walking, not the maintainer). SHA-checked."""
import hashlib, os, sys, urllib.request
CLIPS = {
    "garden.mp4": (
        "https://assets.mixkit.co/videos/35361/35361-720.mp4",
        "3a9e03f486e85fe6f0d889b797976a01fe8caaa4eb9a0ac5057698340120094a",
    ),
    "braids.mp4": (
        "https://assets.mixkit.co/videos/46494/46494-720.mp4",
        "a7e106001ff941c64992c90930b8561c353b2611776092b0aef432cad7d4ab55",
    ),
}
out = sys.argv[1] if len(sys.argv) > 1 else "clips"
os.makedirs(out, exist_ok=True)
for name, (url, sha) in CLIPS.items():
    p = os.path.join(out, name)
    if not (os.path.exists(p) and hashlib.sha256(open(p, "rb").read()).hexdigest() == sha):
        req = urllib.request.Request(url, headers={"User-Agent": "BodySwap-CI"})
        with urllib.request.urlopen(req, timeout=180) as r, open(p, "wb") as f:
            f.write(r.read())
    got = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if got != sha:
        raise SystemExit(f"{name} sha mismatch {got}")
    print(name, os.path.getsize(p), "OK")
